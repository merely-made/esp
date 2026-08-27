// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Temperature / top-p sampling over a host-side logits row.
//!
//! **Seeding policy** (the deliberate part): a request with
//! `seed: Some(s)` produces a bit-reproducible token stream — same
//! seed, same model, same prompt, same tokens. `seed: None` draws one
//! fresh seed per generation (from `RandomState`'s per-process entropy;
//! no rand dependency) and the drawn seed is what a caller would log to
//! reproduce a run. The RNG is splitmix64: tiny, well-understood, and
//! deterministic across platforms — statistical perfection is not a
//! requirement here, reproducibility is.
//!
//! The logits row is read back to the host per token (~128KB at 32k
//! vocab), which is noise next to the forward pass.

/// splitmix64 — the standard 64-bit mixing PRNG.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A fresh seed from per-process entropy (`RandomState`), for
    /// `seed: None` requests.
    pub fn from_entropy() -> (Self, u64) {
        use std::hash::{BuildHasher, Hasher};
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        (Self::new(seed), seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Temperature + optional nucleus sampler with its own RNG state.
#[derive(Debug, Clone)]
pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    rng: SplitMix64,
}

impl Sampler {
    /// `temperature` must be positive (greedy is a different code path,
    /// not temperature zero here).
    pub fn new(temperature: f32, top_p: Option<f32>, seed: u64) -> Self {
        debug_assert!(temperature > 0.0);
        Self {
            temperature,
            top_p,
            rng: SplitMix64::new(seed),
        }
    }

    /// Draw a token id from a logits row.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Softmax over logits/T, computed stably against the max.
        let inv_t = 1.0 / self.temperature;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weighted: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, ((l - max) * inv_t).exp()))
            .collect();

        // Nucleus cut: keep the smallest prefix of the sorted
        // distribution whose mass reaches top_p (always at least one).
        weighted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if let Some(top_p) = self.top_p {
            let total: f32 = weighted.iter().map(|(_, w)| w).sum();
            let target = total * top_p.clamp(0.0, 1.0);
            let mut mass = 0.0;
            let mut keep = 0;
            for (i, (_, w)) in weighted.iter().enumerate() {
                mass += w;
                keep = i + 1;
                if mass >= target {
                    break;
                }
            }
            weighted.truncate(keep.max(1));
        }

        // Inverse-CDF draw over the (renormalized-by-construction) kept set.
        let total: f32 = weighted.iter().map(|(_, w)| w).sum();
        let mut u = self.rng.next_f64() as f32 * total;
        for (id, w) in &weighted {
            u -= w;
            if u <= 0.0 {
                return *id;
            }
        }
        weighted.last().map(|(id, _)| *id).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> {
        // Token 2 strongly favored, 0 close behind, 1/3 far behind.
        vec![2.0, -1.0, 2.5, -3.0]
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = Sampler::new(0.8, Some(0.9), 42);
        let mut b = Sampler::new(0.8, Some(0.9), 42);
        let sa: Vec<u32> = (0..32).map(|_| a.sample(&logits())).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.sample(&logits())).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn tiny_top_p_collapses_to_greedy() {
        let mut s = Sampler::new(1.0, Some(1.0e-6), 7);
        for _ in 0..16 {
            assert_eq!(s.sample(&logits()), 2, "nucleus of one = argmax");
        }
    }

    #[test]
    fn low_temperature_is_effectively_greedy() {
        let mut s = Sampler::new(0.01, None, 7);
        for _ in 0..16 {
            assert_eq!(s.sample(&logits()), 2);
        }
    }

    #[test]
    fn frequencies_track_the_distribution() {
        // At T=1 with logits [2.0, -1.0, 2.5, -3.0], tokens 2 and 0 carry
        // ~97% of the mass with 2 ≈ 1.6x of 0. Over many seeded draws the
        // ordering of observed counts must match.
        let mut s = Sampler::new(1.0, None, 1234);
        let mut counts = [0usize; 4];
        for _ in 0..2000 {
            counts[s.sample(&logits()) as usize] += 1;
        }
        assert!(counts[2] > counts[0], "counts: {counts:?}");
        assert!(counts[0] > counts[1], "counts: {counts:?}");
        assert!(counts[1] >= counts[3], "counts: {counts:?}");
        assert!(counts[2] + counts[0] > 1800, "counts: {counts:?}");
    }

    #[test]
    fn entropy_seeds_vary_across_draws() {
        let (_, s1) = SplitMix64::from_entropy();
        let (_, s2) = SplitMix64::from_entropy();
        // RandomState is per-instance-keyed; two draws colliding would be
        // a 1-in-2^64 event.
        assert_ne!(s1, s2);
    }
}
