// Copyright 2026 Mark AB (markik)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The trainer forcing fixture (distillery v0 plan §9).
//!
//! One local, deterministic ranking receipt that materializes a canonical
//! `TrainingCorpus`, derives real PEFT LoRA adapter weights from the corpus's
//! training partition alone, publishes the adapter manifest with that corpus
//! as its provenance, and writes an `EvalReport` in which the adapter strictly
//! beats the unchanged baseline on the same fixed held-out cases.
//!
//! The receipt fixes the first trainer resource input/output shape without
//! founding a trainer framework: inputs are the base-model manifest, the
//! tokenizer blob, and the corpus's training partition plus explicit
//! hyperparameters; outputs are the adapter weight/config blobs, the adapter
//! manifest, and the evaluation report. Everything runs through public ESP and
//! Eidetic API: loss evaluations load merged weights through the real
//! safetensors loader, and both evaluated sessions are built by the real
//! `PeftLoraAdapterLoader` from bytes resolved back out of the store. ESP owns
//! the tensor execution, Eidetic owns the artifacts; no Mesh job, lease,
//! checkpoint, or Distillery authority is involved.

#![cfg(feature = "decoder-lora")]

use burn::tensor::{Device, Int, Tensor, TensorData};
use eidetic::models::{EvalMetric, EvalReport, EvalTally, OpaqueBlob, TrainingCorpus};
use eidetic::typed::{load_typed, save_typed};
use eidetic::{
    AdapterRuntimeCompat, Hash, ManifestId, ModelAdapterManifest, ModelLibrary, NoFetcher,
    PrivacyClass, ProvenanceRecord, Timestamp, TrustEnvelope,
};
use esp::infer::decoder::{
    DecoderConfig, PEFT_LORA_NDARRAY_LOADER, PeftLoraAdapterLoader, load_decoder_from_bytes,
};
use esp::infer::{AdapterArtifact, AdapterLoader, AdapterSelection, ModelSession};
use muniment::MemoryBackend;
use safetensors::tensor::{Dtype, TensorView};
use tokenizers::Tokenizer;

// ── Fixture identity ────────────────────────────────────────────────────────

const MODEL_ID: &str = "fixture/trainer-forcing";
const PROMPT_TEMPLATE: &[u8] = b"{{ prompt }}";
const ADAPTER_NAME: &str = "trainer-forcing-fixture";

/// The deterministic task family: every prompt ends with the trigger token
/// and the expected next token is fixed, while the held-out prefixes are
/// disjoint from the training prefixes. The receipt's point is the artifact
/// dataflow and a strict held-out improvement, not benchmark difficulty.
const TRIGGER: &str = "t29";
const EXPECTED: &str = "t7";
const TRAIN_PREFIXES: [&str; 6] = [
    "t3 t11 t5",
    "t18 t2 t26",
    "t9 t14 t1",
    "t22 t6 t13",
    "t4 t27 t10",
    "t15 t8 t21",
];
const EVAL_PREFIXES: [&str; 6] = [
    "t12 t25 t3",
    "t7 t19 t30",
    "t24 t1 t16",
    "t5 t28 t9",
    "t17 t20 t2",
    "t31 t10 t23",
];

// ── Trainer hyperparameters (part of the published training_method) ─────────

const LORA_RANK: usize = 1;
const LORA_ALPHA: f32 = 8.0;
const TARGET_MODULE: &str = "v_proj";
const TRAIN_STEPS: usize = 40;
/// Initial trial step for the backtracking line search: scale-free against
/// the tiny model's strongly attenuated gradients, and still deterministic.
const INITIAL_STEP_LENGTH: f32 = 1.0;
const MINIMUM_STEP_LENGTH: f32 = 1.0e-4;
const DIFFERENCE_EPSILON: f32 = 0.02;

// ── Tiny synthetic base model (the eidetic_corridor artifact shape) ─────────

const CONFIG_JSON: &str = r#"{
    "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "max_position_embeddings": 16,
    "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

fn tokenizer_json() -> String {
    let vocab: Vec<String> = (0..32).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{
            "version": "1.0",
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
        }}"#,
        vocab.join(", ")
    )
}

fn det_vec(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt * 7919) as f32) * 0.618_034).sin() * 0.05)
        .collect()
}

/// Named HF-layout tensor table: safetensors row-major `[out, in]` matrices.
type TensorTable = Vec<(String, Vec<usize>, Vec<f32>)>;

fn base_tensor_table() -> TensorTable {
    let (h, kv, inter, vocab) = (8usize, 4usize, 16usize, 32usize);
    let mut table: TensorTable = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, salt: usize| {
        let n: usize = shape.iter().product();
        table.push((name, shape, det_vec(n, salt)));
    };
    push("model.embed_tokens.weight".into(), vec![vocab, h], 7);
    for i in 0..2usize {
        let p = format!("model.layers.{i}");
        let s = 100 * (i + 1);
        push(format!("{p}.input_layernorm.weight"), vec![h], s);
        push(format!("{p}.self_attn.q_proj.weight"), vec![h, h], s + 1);
        push(format!("{p}.self_attn.k_proj.weight"), vec![kv, h], s + 2);
        push(format!("{p}.self_attn.v_proj.weight"), vec![kv, h], s + 3);
        push(format!("{p}.self_attn.o_proj.weight"), vec![h, h], s + 4);
        push(
            format!("{p}.post_attention_layernorm.weight"),
            vec![h],
            s + 8,
        );
        push(format!("{p}.mlp.gate_proj.weight"), vec![inter, h], s + 5);
        push(format!("{p}.mlp.up_proj.weight"), vec![inter, h], s + 6);
        push(format!("{p}.mlp.down_proj.weight"), vec![h, inter], s + 7);
    }
    push("model.norm.weight".into(), vec![h], 9);
    push("lm_head.weight".into(), vec![vocab, h], 8);
    table
}

fn serialize_table(table: &TensorTable) -> Vec<u8> {
    let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = table
        .iter()
        .map(|(name, shape, values)| {
            (
                name.clone(),
                shape.clone(),
                values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize(views, &None).unwrap()
}

// ── Corpus cases ────────────────────────────────────────────────────────────

/// One stored task case: a rendered prompt and the expected next token.
#[derive(serde::Serialize, serde::Deserialize)]
struct RankingCase {
    prompt: String,
    expected_token: String,
}

fn case_blob(prefix: &str) -> OpaqueBlob {
    let case = RankingCase {
        prompt: format!("{prefix} {TRIGGER}"),
        expected_token: EXPECTED.to_string(),
    };
    OpaqueBlob(serde_json::to_vec(&case).unwrap())
}

fn save_cases(store: &mut MemoryBackend, prefixes: &[&str]) -> Vec<ManifestId> {
    let mut ids: Vec<ManifestId> = prefixes
        .iter()
        .map(|prefix| {
            pollster::block_on(save_typed(
                store,
                &case_blob(prefix),
                vec![],
                PrivacyClass::LocalOnly,
                ProvenanceRecord::self_imported(ADAPTER_NAME),
                TrustEnvelope::self_asserted(),
                Timestamp(0),
            ))
            .expect("save case engram")
        })
        .collect();
    ids.sort_by_key(ToString::to_string);
    ids
}

fn load_cases(store: &mut MemoryBackend, ids: &[ManifestId]) -> Vec<RankingCase> {
    ids.iter()
        .map(|id| {
            let blob = pollster::block_on(load_typed::<OpaqueBlob>(store, &mut NoFetcher, *id))
                .expect("load case engram")
                .expect("case engram present");
            serde_json::from_slice(&blob.0).expect("case JSON")
        })
        .collect()
}

// ── The deterministic trainer ───────────────────────────────────────────────

/// LoRA factors for one target module across every decoder layer, in PEFT
/// layout: `a` is `[rank, in]`, `b` is `[out, rank]`, row-major.
struct LoraFactors {
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl LoraFactors {
    /// PEFT-style start: `a` deterministic non-zero, `b` zero, so the
    /// step-zero model is exactly the baseline.
    fn initial(layers: usize, in_features: usize, out_features: usize) -> Self {
        Self {
            a: (0..layers)
                .map(|layer| {
                    det_vec(LORA_RANK * in_features, 9000 + layer)
                        .iter()
                        .map(|value| value * 2.0)
                        .collect()
                })
                .collect(),
            b: (0..layers)
                .map(|_| vec![0.0; out_features * LORA_RANK])
                .collect(),
            in_features,
            out_features,
        }
    }

    fn params(&self) -> Vec<f32> {
        let mut params = Vec::new();
        for layer in 0..self.a.len() {
            params.extend_from_slice(&self.a[layer]);
            params.extend_from_slice(&self.b[layer]);
        }
        params
    }

    fn set_params(&mut self, params: &[f32]) {
        let mut offset = 0;
        for layer in 0..self.a.len() {
            let a_len = self.a[layer].len();
            self.a[layer].copy_from_slice(&params[offset..offset + a_len]);
            offset += a_len;
            let b_len = self.b[layer].len();
            self.b[layer].copy_from_slice(&params[offset..offset + b_len]);
            offset += b_len;
        }
        assert_eq!(offset, params.len(), "parameter vector length");
    }

    /// HF-layout delta for one layer: `scale * (B @ A)`, `[out, in]`
    /// row-major — the exact PEFT merge formula.
    fn merged_delta(&self, layer: usize, scale: f32) -> Vec<f32> {
        let (r, i_n, out) = (LORA_RANK, self.in_features, self.out_features);
        let mut delta = vec![0.0f32; out * i_n];
        for o in 0..out {
            for rank in 0..r {
                let b = self.b[layer][o * r + rank];
                for i in 0..i_n {
                    delta[o * i_n + i] += scale * b * self.a[layer][rank * i_n + i];
                }
            }
        }
        delta
    }
}

/// Merge the LoRA delta into the base table and serialize the result.
fn merged_weights(base: &TensorTable, lora: &LoraFactors, scale: f32) -> Vec<u8> {
    let mut table = base.clone();
    for (layer, entry) in table
        .iter_mut()
        .filter(|(name, _, _)| name.ends_with(&format!("self_attn.{TARGET_MODULE}.weight")))
        .enumerate()
    {
        let delta = lora.merged_delta(layer, scale);
        for (weight, d) in entry.2.iter_mut().zip(delta.iter()) {
            *weight += d;
        }
    }
    serialize_table(&table)
}

/// The fixed training objective: the base table, the tokenized full batch of
/// training cases, and the expected token, evaluated on one device.
struct TrainingTask<'a> {
    config: &'a DecoderConfig,
    base: &'a TensorTable,
    scale: f32,
    batch_ids: Vec<i32>,
    batch: usize,
    seq: usize,
    expected_id: usize,
    device: Device,
}

impl TrainingTask<'_> {
    /// Mean next-token cross-entropy of the expected token over the training
    /// cases, computed through the real safetensors loader and decoder
    /// forward.
    fn loss(&self, lora: &LoraFactors) -> f64 {
        let weights = merged_weights(self.base, lora, self.scale);
        let model =
            load_decoder_from_bytes(self.config, &weights, &self.device).expect("load merged");
        let logits = model
            .logits(
                Tensor::<2, Int>::from_data(
                    TensorData::new(self.batch_ids.clone(), [self.batch, self.seq]),
                    &self.device,
                ),
                0,
            )
            .into_data()
            .to_vec::<f32>()
            .expect("logits data");
        let (vocab, seq) = (self.config.vocab_size, self.seq);
        let mut loss = 0.0f64;
        for case in 0..self.batch {
            let row = &logits[(case * seq + (seq - 1)) * vocab..(case * seq + seq) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let log_sum_exp = row
                .iter()
                .map(|&logit| (f64::from(logit) - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            loss += log_sum_exp - f64::from(row[self.expected_id]);
        }
        loss / self.batch as f64
    }
}

// ── Ranking evaluation ──────────────────────────────────────────────────────

/// 1-based rank of the expected token in one next-token logit row; ties
/// resolve in the expected token's favor, deterministically.
fn rank_of(logits: &[f32], expected_id: usize) -> usize {
    1 + logits
        .iter()
        .filter(|&&logit| logit > logits[expected_id])
        .count()
}

fn tally(
    provider: &esp::infer::decoder::DecoderProvider,
    cases: &[RankingCase],
    expected_id: usize,
    limit: usize,
) -> EvalTally {
    let passed = cases
        .iter()
        .filter(|case| {
            let logits = provider
                .next_token_logits(&case.prompt)
                .expect("next-token logits");
            rank_of(&logits, expected_id) <= limit
        })
        .count() as u64;
    EvalTally {
        passed,
        total: cases.len() as u64,
    }
}

// ── The receipt ─────────────────────────────────────────────────────────────

#[test]
fn adapter_trained_from_corpus_beats_baseline_on_held_out_ranking() {
    let device = Device::ndarray();
    let mut store = MemoryBackend::default();
    let provenance = || ProvenanceRecord::self_imported(ADAPTER_NAME);

    // 1. Base model triple through the eidetic model corridor.
    let base_table = base_tensor_table();
    let base_weights = serialize_table(&base_table);
    let tokenizer_bytes = tokenizer_json();
    let base_model_ref = pollster::block_on(ModelLibrary::save_model_with_components(
        &mut store,
        MODEL_ID,
        "llama",
        "MIT",
        serde_json::from_str(CONFIG_JSON).unwrap(),
        &base_weights,
        tokenizer_bytes.as_bytes(),
        Vec::new(),
        Vec::new(),
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(0),
    ))
    .expect("save base model");
    let resolved = pollster::block_on(ModelLibrary::resolve_components(
        &mut store,
        &mut NoFetcher,
        base_model_ref,
    ))
    .expect("resolve base model")
    .expect("base model present");
    let tokenizer_ref = resolved.manifest.tokenizer_blob;

    // 2. Canonical corpus: disjoint, strictly ordered source partitions.
    let corpus = TrainingCorpus {
        training_source_engrams: save_cases(&mut store, &TRAIN_PREFIXES),
        evaluation_source_engrams: save_cases(&mut store, &EVAL_PREFIXES),
    };
    corpus.validate().expect("corpus valid");
    let corpus_ref = pollster::block_on(save_typed(
        &mut store,
        &corpus,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(1),
    ))
    .expect("save corpus");

    // 3. Deterministic trainer over the training partition only: full-batch
    //    central-finite-difference gradient descent on the LoRA factors.
    let config = DecoderConfig::from_json_bytes(&resolved.components.config_bytes).unwrap();
    let tokenizer = Tokenizer::from_bytes(resolved.components.tokenizer_bytes.as_slice()).unwrap();
    let expected_id = tokenizer.token_to_id(EXPECTED).expect("expected token") as usize;
    let train_cases = load_cases(&mut store, &corpus.training_source_engrams);
    let seq = 4usize;
    let batch_ids: Vec<i32> = train_cases
        .iter()
        .flat_map(|case| {
            let ids = tokenizer.encode(case.prompt.as_str(), true).unwrap();
            assert_eq!(ids.get_ids().len(), seq, "uniform case length");
            ids.get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>()
        })
        .collect();
    for case in &train_cases {
        assert_eq!(case.expected_token, EXPECTED);
    }

    let scale = LORA_ALPHA / LORA_RANK as f32;
    let kv = config.kv_heads() * config.head_dim();
    let mut lora = LoraFactors::initial(config.num_hidden_layers, config.hidden_size, kv);
    let task = TrainingTask {
        config: &config,
        base: &base_table,
        scale,
        batch_ids,
        batch: train_cases.len(),
        seq,
        expected_id,
        device: device.clone(),
    };
    let loss = |factors: &LoraFactors| task.loss(factors);
    let initial_loss = loss(&lora);
    let mut params = lora.params();
    let mut current_loss = initial_loss;
    let mut step_length = INITIAL_STEP_LENGTH;
    for step in 0..TRAIN_STEPS {
        let mut gradient = vec![0.0f32; params.len()];
        for j in 0..params.len() {
            let original = params[j];
            params[j] = original + DIFFERENCE_EPSILON;
            lora.set_params(&params);
            let plus = loss(&lora);
            params[j] = original - DIFFERENCE_EPSILON;
            lora.set_params(&params);
            let minus = loss(&lora);
            params[j] = original;
            gradient[j] = ((plus - minus) / (2.0 * f64::from(DIFFERENCE_EPSILON))) as f32;
        }
        let norm = gradient
            .iter()
            .map(|g| f64::from(*g) * f64::from(*g))
            .sum::<f64>()
            .sqrt() as f32;
        assert!(norm > 0.0, "vanished training gradient at step {step}");

        // Backtracking line search along the normalized descent direction:
        // grow the last accepted step once, halve on failure, and keep the
        // parameters when no trial length descends.
        let mut trial = step_length * 2.0;
        while trial >= MINIMUM_STEP_LENGTH {
            let candidate: Vec<f32> = params
                .iter()
                .zip(gradient.iter())
                .map(|(param, g)| param - trial * g / norm)
                .collect();
            lora.set_params(&candidate);
            let candidate_loss = loss(&lora);
            if candidate_loss < current_loss {
                params = candidate;
                current_loss = candidate_loss;
                step_length = trial;
                break;
            }
            trial *= 0.5;
        }
        lora.set_params(&params);
        if step % 10 == 9 {
            println!(
                "step {:>3}: training loss {current_loss:.6}, step length {step_length:.5}",
                step + 1
            );
        }
    }
    let trained_loss = loss(&lora);
    assert!(
        trained_loss < initial_loss,
        "training must reduce the training loss: {initial_loss} -> {trained_loss}"
    );

    // 4. Publish the adapter: PEFT-layout safetensors + config blobs and the
    //    manifest whose provenance names the corpus.
    let adapter_weight_bytes = {
        let mut table: TensorTable = Vec::new();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("base_model.model.model.layers.{layer}.self_attn.{TARGET_MODULE}");
            table.push((
                format!("{prefix}.lora_A.weight"),
                vec![LORA_RANK, config.hidden_size],
                lora.a[layer].clone(),
            ));
            table.push((
                format!("{prefix}.lora_B.weight"),
                vec![kv, LORA_RANK],
                lora.b[layer].clone(),
            ));
        }
        serialize_table(&table)
    };
    let adapter_config_bytes = serde_json::to_vec(&serde_json::json!({
        "base_model_name_or_path": MODEL_ID,
        "peft_type": "LORA",
        "peft_version": "fixture",
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": [TARGET_MODULE],
        "bias": "none",
    }))
    .unwrap();
    let save_blob = |store: &mut MemoryBackend, bytes: &[u8]| {
        pollster::block_on(save_typed(
            store,
            &OpaqueBlob(bytes.to_vec()),
            vec![],
            PrivacyClass::LocalOnly,
            provenance(),
            TrustEnvelope::self_asserted(),
            Timestamp(2),
        ))
        .expect("save adapter blob")
    };
    let adapter_blob = save_blob(&mut store, &adapter_weight_bytes);
    let adapter_config_blob = save_blob(&mut store, &adapter_config_bytes);
    let manifest = ModelAdapterManifest {
        name: ADAPTER_NAME.into(),
        base_model_ref,
        adapter_blob,
        adapter_config_blob,
        adapter_format: "peft-lora".into(),
        adapter_format_version: "peft-fixture".into(),
        runtime_compat: AdapterRuntimeCompat {
            minimum_capabilities: vec!["peft-lora".into()],
            known_loaders: vec![PEFT_LORA_NDARRAY_LOADER.into()],
            converter_lineage: vec![],
        },
        rank: LORA_RANK as u16,
        alpha: LORA_ALPHA,
        target_modules: vec![TARGET_MODULE.into()],
        tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization_assumption: None,
        training_corpus_root: Some(corpus_ref),
        training_method: serde_json::json!({
            "trainer": "trainer-forcing-fixture/v0",
            "objective": "next-token cross-entropy",
            "optimizer": {
                "kind": "backtracking-line-search-finite-difference-descent",
                "initial_step_length": INITIAL_STEP_LENGTH,
                "minimum_step_length": MINIMUM_STEP_LENGTH,
                "steps": TRAIN_STEPS,
                "epsilon": DIFFERENCE_EPSILON,
            },
            "inputs": {
                "base_model_ref": base_model_ref.to_string(),
                "tokenizer_ref": tokenizer_ref.to_string(),
                "corpus_partition": "training_source_engrams",
            },
            "outputs": ["adapter_blob", "adapter_config_blob", "eval_report"],
        }),
        eval_results: None,
    };
    let adapter_ref = pollster::block_on(save_typed(
        &mut store,
        &manifest,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(3),
    ))
    .expect("save adapter manifest");

    // 5. Evaluate baseline and adapter through the real session loader, from
    //    bytes resolved back out of the store.
    let stored_manifest = pollster::block_on(load_typed::<ModelAdapterManifest>(
        &mut store,
        &mut NoFetcher,
        adapter_ref,
    ))
    .expect("load adapter manifest")
    .expect("adapter manifest present");
    let load_blob = |store: &mut MemoryBackend, id: ManifestId| {
        pollster::block_on(load_typed::<OpaqueBlob>(store, &mut NoFetcher, id))
            .expect("load adapter blob")
            .expect("adapter blob present")
            .0
    };
    let stored_weights = load_blob(&mut store, stored_manifest.adapter_blob);
    let stored_config = load_blob(&mut store, stored_manifest.adapter_config_blob);
    let loader = PeftLoraAdapterLoader::new(
        base_model_ref,
        tokenizer_ref,
        &resolved.components.config_bytes,
        &resolved.components.tokenizer_bytes,
        &resolved.components.weight_bytes,
        MODEL_ID,
        PEFT_LORA_NDARRAY_LOADER,
        &device,
    );
    let session = |adapters: Vec<AdapterSelection>| ModelSession {
        base_model_ref,
        model_id: MODEL_ID.into(),
        tokenizer_ref,
        prompt_template_hash: Hash::of(PROMPT_TEMPLATE),
        quantization: None,
        loader: PEFT_LORA_NDARRAY_LOADER.into(),
        adapters,
    };
    let baseline_session = loader
        .load_session(session(vec![]), &[])
        .expect("baseline session");
    let adapted = || {
        loader
            .load_session(
                session(vec![AdapterSelection {
                    manifest_ref: adapter_ref,
                    scale: 1.0,
                }]),
                &[AdapterArtifact {
                    manifest_ref: adapter_ref,
                    manifest: &stored_manifest,
                    config_bytes: &stored_config,
                    weight_bytes: &stored_weights,
                }],
            )
            .expect("adapted session")
    };
    let adapted_session = adapted();

    let eval_cases = load_cases(&mut store, &corpus.evaluation_source_engrams);
    for (label, session) in [
        ("baseline", &baseline_session),
        ("adapter", &adapted_session),
    ] {
        let ranks: Vec<usize> = eval_cases
            .iter()
            .map(|case| {
                let logits = session.provider().next_token_logits(&case.prompt).unwrap();
                rank_of(&logits, expected_id)
            })
            .collect();
        println!("{label} held-out expected-token ranks: {ranks:?}");
    }
    // Ranking cutoff 3: the rank-1 value-projection adapter tilts the tiny
    // model's near-uniform logits decisively into the top 3 (baseline ranks
    // sit at 7-8) without demanding a fragile single-case top-1 margin.
    let limit = 3u32;
    let baseline = tally(
        baseline_session.provider(),
        &eval_cases,
        expected_id,
        limit as usize,
    );
    let adapter = tally(
        adapted_session.provider(),
        &eval_cases,
        expected_id,
        limit as usize,
    );
    assert_eq!(baseline.total, eval_cases.len() as u64);
    assert!(
        baseline.passed < baseline.total,
        "receipt is trivial: the baseline already passes every held-out case ({baseline:?})"
    );
    assert!(
        adapter.passed > baseline.passed,
        "adapter must strictly beat the baseline: baseline {baseline:?}, adapter {adapter:?}"
    );

    // Determinism: a freshly loaded adapted session reproduces the tally.
    assert_eq!(
        tally(
            adapted().provider(),
            &eval_cases,
            expected_id,
            limit as usize
        ),
        adapter,
        "adapter tally must be reproducible from a fresh session"
    );

    // 6. The immutable receipt, round-tripped and provenance-checked.
    let report = EvalReport {
        base_model_ref,
        adapter_ref,
        corpus_ref,
        metric: EvalMetric::RankingAt { limit },
        baseline,
        adapter,
    };
    let report_ref = pollster::block_on(save_typed(
        &mut store,
        &report,
        vec![],
        PrivacyClass::LocalOnly,
        provenance(),
        TrustEnvelope::self_asserted(),
        Timestamp(4),
    ))
    .expect("save eval report");
    let stored_report = pollster::block_on(load_typed::<EvalReport>(
        &mut store,
        &mut NoFetcher,
        report_ref,
    ))
    .expect("load eval report")
    .expect("eval report present");
    assert_eq!(stored_report, report);
    assert!(
        stored_report
            .adapter_beats_baseline()
            .expect("comparable report"),
        "stored receipt must show the strict improvement"
    );
    stored_report
        .validate_for_adapter(adapter_ref, &stored_manifest)
        .expect("receipt provenance links match the adapter manifest");

    println!(
        "trainer forcing receipt: loss {initial_loss:.4} -> {trained_loss:.4}, \
         baseline {}/{} vs adapter {}/{} at ranking@{limit}",
        report.baseline.passed, report.baseline.total, report.adapter.passed, report.adapter.total,
    );
}
