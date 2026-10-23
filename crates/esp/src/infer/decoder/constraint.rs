// Copyright 2026 Mark Alan Boykin
// SPDX-License-Identifier: MPL-2.0

//! Incremental byte grammar over a verified byte-level tokenizer. No decoded
//! fragments are repaired. Syntax-only spans have canonical tokenization and
//! can be evaluated as a block; semantic choices still use masked logits.
use crate::infer::{InferError, OutputGrammar};
use tokenizers::Tokenizer;

#[derive(Clone)]
enum Node {
    Byte(u8, usize),
    Split(Vec<usize>),
    String(usize, usize),
    End,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Position {
    node: usize,
    used: usize,
    phase: u8,
    low: u8,
    high: u8,
}
impl Position {
    fn at(node: usize) -> Self {
        Self {
            node,
            used: 0,
            phase: 0,
            low: 0,
            high: 0,
        }
    }
}

pub(super) struct Constraint {
    nodes: Vec<Node>,
    active: Vec<Position>,
    tokens: Vec<Vec<u8>>,
}
impl Drop for Constraint {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for node in &mut self.nodes {
            if let Node::Byte(byte, _) = node {
                byte.zeroize();
            }
        }
    }
}
fn invalid(s: &str) -> InferError {
    InferError::InvalidRequest(s.into())
}
impl Constraint {
    pub fn new(
        grammar: &OutputGrammar,
        tokenizer: &Tokenizer,
        vocabulary: usize,
    ) -> Result<Self, InferError> {
        fn compile(
            g: &OutputGrammar,
            next: usize,
            nodes: &mut Vec<Node>,
            depth: usize,
            bytes: &mut usize,
        ) -> Result<usize, InferError> {
            if depth > 24 || nodes.len() > 100_000 || *bytes > 262_144 {
                return Err(invalid("output grammar exceeds its bounds"));
            }
            let node = match g {
                OutputGrammar::Literal { value } => {
                    *bytes += value.len();
                    if *bytes > 262_144 {
                        return Err(invalid("output grammar literals exceed their bound"));
                    }
                    let mut at = next;
                    for b in value.bytes().rev() {
                        nodes.push(Node::Byte(b, at));
                        at = nodes.len() - 1;
                    }
                    return Ok(at);
                }
                OutputGrammar::JsonString { max_bytes } => {
                    if !(1..=4096).contains(max_bytes) {
                        return Err(invalid("JSON string bound must be 1..4096"));
                    }
                    Node::String(*max_bytes, next)
                }
                OutputGrammar::Sequence { items } => {
                    let mut at = next;
                    for item in items.iter().rev() {
                        at = compile(item, at, nodes, depth + 1, bytes)?;
                    }
                    return Ok(at);
                }
                OutputGrammar::Choice { alternatives } => {
                    if alternatives.is_empty() || alternatives.len() > 64 {
                        return Err(invalid("grammar choice must have 1..64 alternatives"));
                    }
                    Node::Split(
                        alternatives
                            .iter()
                            .map(|g| compile(g, next, nodes, depth + 1, bytes))
                            .collect::<Result<_, _>>()?,
                    )
                }
            };
            nodes.push(node);
            Ok(nodes.len() - 1)
        }
        let mut nodes = vec![Node::End];
        let start = compile(grammar, 0, &mut nodes, 0, &mut 0)?;
        if nodes.len() > 100_000 {
            return Err(invalid("compiled output grammar exceeds its state bound"));
        }
        let description: serde_json::Value = serde_json::from_str(
            &tokenizer
                .to_string(false)
                .map_err(|e| invalid(&e.to_string()))?,
        )
        .map_err(|e| invalid(&e.to_string()))?;
        if description["decoder"]["type"] != "ByteLevel" {
            return Err(invalid("constrained decoding requires a ByteLevel decoder"));
        }
        // GPT-2 / Qwen byte alphabet, including nonprinting bytes represented
        // by Unicode scalars above 255. Decode token pieces without lossy UTF-8.
        let mut alphabet: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut extra = 0;
        let mut inverse = std::collections::HashMap::new();
        for b in 0..=255u32 {
            let scalar = if alphabet.contains(&b) {
                b
            } else {
                let x = 256 + extra;
                extra += 1;
                alphabet.push(b);
                x
            };
            inverse.insert(char::from_u32(scalar).unwrap(), b as u8);
        }
        let added: std::collections::HashSet<u32> = description["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v["id"].as_u64().map(|id| id as u32))
            .collect();
        let mut tokens = vec![Vec::new(); vocabulary];
        for (piece, id) in tokenizer.get_vocab(false) {
            if id as usize >= vocabulary || added.contains(&id) {
                continue;
            }
            if let Some(bytes) = piece
                .chars()
                .map(|c| inverse.get(&c).copied())
                .collect::<Option<Vec<_>>>()
            {
                tokens[id as usize] = bytes;
            }
        }
        let mut this = Self {
            nodes,
            active: vec![],
            tokens,
        };
        this.active = this.close(vec![Position::at(start)]);
        Ok(this)
    }
    fn close(&self, mut positions: Vec<Position>) -> Vec<Position> {
        let mut result = Vec::new();
        while let Some(p) = positions.pop() {
            match &self.nodes[p.node] {
                Node::Split(ids) => positions.extend(ids.iter().map(|id| Position::at(*id))),
                _ => result.push(p),
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }
    fn step(&self, active: &[Position], byte: u8) -> Vec<Position> {
        let mut out = Vec::new();
        for &p in active {
            match self.nodes[p.node] {
                Node::Byte(b, next) if b == byte => out.push(Position::at(next)),
                Node::String(max, next) => {
                    let mut q = p;
                    if p.phase == 0 {
                        if byte == b'"' {
                            q.phase = 1;
                            out.push(q);
                        }
                        continue;
                    }
                    if p.phase == 1 && byte == b'"' {
                        if p.used > 0 {
                            out.push(Position::at(next));
                        }
                        continue;
                    }
                    if p.used >= max {
                        continue;
                    }
                    q.used += 1;
                    match p.phase {
                        1 => match byte {
                            b'\\' if p.used + 2 <= max => q.phase = 2,
                            0x20..=0x7f if byte != b'"' && byte != b'\\' => {}
                            0xc2..=0xdf if p.used + 2 <= max => {
                                q.phase = 3;
                                q.low = 0x80;
                                q.high = 0xbf;
                            }
                            0xe0..=0xef if p.used + 3 <= max => {
                                q.phase = 4;
                                q.low = if byte == 0xe0 { 0xa0 } else { 0x80 };
                                q.high = if byte == 0xed { 0x9f } else { 0xbf };
                            }
                            0xf0..=0xf4 if p.used + 4 <= max => {
                                q.phase = 5;
                                q.low = if byte == 0xf0 { 0x90 } else { 0x80 };
                                q.high = if byte == 0xf4 { 0x8f } else { 0xbf };
                            }
                            _ => continue,
                        },
                        2 => {
                            if !b"\"\\/bfnrt".contains(&byte) {
                                continue;
                            }
                            q.phase = 1;
                        }
                        3..=5 => {
                            if byte < p.low || byte > p.high {
                                continue;
                            }
                            q.phase = if p.phase == 3 { 1 } else { p.phase - 1 };
                            q.low = 0x80;
                            q.high = 0xbf;
                        }
                        _ => continue,
                    }
                    out.push(q);
                }
                _ => {}
            }
        }
        self.close(out)
    }
    fn after(&self, bytes: &[u8]) -> Vec<Position> {
        let mut state = self.active.clone();
        for &b in bytes {
            state = self.step(&state, b);
            if state.is_empty() {
                break;
            }
        }
        state
    }
    pub fn complete(&self) -> bool {
        self.active
            .iter()
            .any(|p| matches!(self.nodes[p.node], Node::End))
    }
    pub fn accept(&mut self, id: u32) -> Result<(), InferError> {
        let bytes = self
            .tokens
            .get(id as usize)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("constraint encountered an unsupported token"))?;
        let state = self.after(bytes);
        if state.is_empty() {
            return Err(invalid("token violates output grammar"));
        }
        self.active = state;
        Ok(())
    }
    pub fn mask(&self, row: &mut [f32]) -> Result<(), InferError> {
        let mut any = false;
        for (id, logit) in row.iter_mut().enumerate() {
            let allowed = self
                .tokens
                .get(id)
                .is_some_and(|bytes| !bytes.is_empty() && !self.after(bytes).is_empty());
            if !allowed {
                *logit = f32::NEG_INFINITY;
            } else if logit.is_finite() {
                any = true;
            }
        }
        if !any {
            return Err(invalid("output grammar has no finite eligible token"));
        }
        Ok(())
    }
    pub fn forced(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>, InferError> {
        let mut state = self.active.clone();
        let mut bytes = Vec::new();
        for _ in 0..4096 {
            if state
                .iter()
                .any(|p| matches!(self.nodes[p.node], Node::End))
            {
                break;
            }
            let mut possible = (0..=255u8).filter_map(|b| {
                let s = self.step(&state, b);
                (!s.is_empty()).then_some((b, s))
            });
            let Some((b, next)) = possible.next() else {
                break;
            };
            if possible.next().is_some() {
                break;
            }
            bytes.push(b);
            state = next;
        }
        // A forced prefix can stop before a non-ASCII branch; never feed a
        // lossy replacement through the tokenizer.
        while std::str::from_utf8(&bytes).is_err() {
            bytes.pop();
        }
        if bytes.is_empty() {
            return Ok(vec![]);
        }
        let ids = tokenizer
            .encode(std::str::from_utf8(&bytes).unwrap(), false)
            .map_err(|e| invalid(&e.to_string()))?
            .get_ids()
            .to_vec();
        let actual: Vec<_> = ids
            .iter()
            .flat_map(|id| self.tokens.get(*id as usize).into_iter().flatten().copied())
            .collect();
        if actual != bytes {
            return Err(invalid(
                "forced grammar span does not round-trip through tokenizer",
            ));
        }
        Ok(ids)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub fn tokenizer() -> Tokenizer {
        let printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut n = 0;
        let vocab: serde_json::Map<String, serde_json::Value> = (0..=255u32)
            .map(|b| {
                let c = if printable.contains(&b) {
                    b
                } else {
                    let c = 256 + n;
                    n += 1;
                    c
                };
                (char::from_u32(c).unwrap().to_string(), serde_json::json!(b))
            })
            .collect();
        Tokenizer::from_bytes(serde_json::to_vec(&serde_json::json!({"version":"1.0","decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","vocab":vocab,"merges":[]}})).unwrap()).unwrap()
    }
    fn string(max_bytes: usize) -> OutputGrammar {
        OutputGrammar::JsonString { max_bytes }
    }
    #[test]
    fn every_accepted_small_string_is_valid_json() {
        fn walk(c: &Constraint, state: Vec<Position>, raw: &mut Vec<u8>, accepted: &mut usize) {
            if state.iter().any(|p| matches!(c.nodes[p.node], Node::End)) {
                let value: String = serde_json::from_slice(raw)
                    .expect("accepted bytes must be a whole JSON string");
                assert!(!value.is_empty());
                *accepted += 1;
                return;
            }
            for byte in [
                32, 34, 47, 92, 97, 110, 114, 116, 117, 120, 10, 0xc3, 0xa9, 0xe0, 0xed, 0xa0,
                0x80, 0xf4, 0x8f, 0xbf,
            ] {
                let next = c.step(&state, byte);
                if !next.is_empty() {
                    raw.push(byte);
                    walk(c, next, raw, accepted);
                    raw.pop();
                }
            }
        }
        let t = tokenizer();
        let mut accepted = 0;
        for max in 1..=4 {
            let c = Constraint::new(&string(max), &t, 256).unwrap();
            walk(&c, c.active.clone(), &mut vec![], &mut accepted);
        }
        assert!(accepted > 1000);
    }

    #[test]
    fn utf8_split_tokens_and_escapes_are_exact_and_bounded() {
        let t = tokenizer();
        let mut c = Constraint::new(&string(32), &t, 256).unwrap();
        for &b in "\"café → \\".as_bytes() {
            c.accept(b as u32).unwrap();
        }
        assert!(c.accept(b'x' as u32).is_err());
        for &b in b"n\"" {
            c.accept(b as u32).unwrap();
        }
        assert!(c.complete());
        assert!(c.accept(b'"' as u32).is_err());
        for bad in [
            vec![b'"', 0xed, 0xa0, 0x80, b'"'],
            vec![b'"', 0xc0, 0x80, b'"'],
            b"\"a\nb\"".to_vec(),
            b"\"\\uD800\"".to_vec(),
            b"\"\"".to_vec(),
        ] {
            let mut c = Constraint::new(&string(32), &t, 256).unwrap();
            assert!(bad.into_iter().any(|b| c.accept(b as u32).is_err()));
        }
        let mut c = Constraint::new(&string(1), &t, 256).unwrap();
        c.accept(34).unwrap();
        assert!(c.accept(92).is_err());
        assert!(c.accept(0xc3).is_err());
        c.accept(97).unwrap();
        assert!(c.accept(98).is_err());
        c.accept(34).unwrap();
        assert!(c.complete());
    }
    #[test]
    fn masking_and_forced_spans_refuse_extra_syntax() {
        let t = tokenizer();
        let g = OutputGrammar::Sequence {
            items: vec![
                OutputGrammar::Literal {
                    value: "{\"v\":".into(),
                },
                string(8),
                OutputGrammar::Literal { value: "}".into() },
            ],
        };
        let mut c = Constraint::new(&g, &t, 256).unwrap();
        let ids = c.forced(&t).unwrap();
        assert_eq!(
            ids,
            b"{\"v\":\"".iter().map(|b| *b as u32).collect::<Vec<_>>()
        );
        for id in ids {
            c.accept(id).unwrap();
        }
        let mut row = vec![0.; 256];
        c.mask(&mut row).unwrap();
        assert!(row[10].is_infinite());
        assert!(row[34].is_infinite());
        assert!(row[97].is_finite());
        for &b in b"ok\"}" {
            c.accept(b as u32).unwrap();
        }
        assert!(c.complete());
        assert!(c.accept(b'\n' as u32).is_err());
        assert!(Constraint::new(&string(4097), &t, 256).is_err());
    }
}
