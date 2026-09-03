// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Which adapter the wgpu runtime *would* bind, asked without binding it.
//!
//! [`DecoderDevice::wgpu`](super::DecoderDevice::wgpu) is infallible by
//! construction: it names a [`DecoderGpuKind`] and the adapter is not resolved
//! until the first kernel launches, inside cubecl, where the failure mode is a
//! panic. And the panic is not the dangerous part. cubecl's
//! `select_from_adapter_list` partitions the enumerated adapters into "matches
//! the requested device type" and "reports `DeviceType::Other`", and when the
//! requested list is too short it **falls through to the `Other` list** at the
//! same index, panicking only when that is short too. A host that asked for a
//! discrete GPU can therefore be handed something else and never be told.
//!
//! So a host that means to advertise a GPU has to know *which* adapter it got,
//! before it advertises anything. This module answers that question by
//! mirroring cubecl's selection — the same graphics backend, the same
//! `enumerate_adapters`, the same device-type filter, the same index, the same
//! `CUBECL_WGPU_DEFAULT_DEVICE` rule — and reporting the adapter's own
//! `wgpu::AdapterInfo` as plain data. It creates an instance and enumerates; it
//! never requests a device or a queue, so nothing here binds the GPU.
//!
//! It is a *mirror*, which means it can drift. The mirrored code is
//! `cubecl-wgpu`'s `runtime.rs` (`request_adapter`, `select_from_adapter_list`,
//! `get_device_override`) and `graphics.rs` (`AutoGraphicsApi`), read at
//! cubecl-wgpu 0.11.0-pre.2. A caller that composes on the strength of this
//! answer is asserting the mirror is current; a receipt that runs real work on
//! the reported adapter is what keeps that assertion honest.
//!
//! ## The `CUBECL_WGPU_DEFAULT_DEVICE` rule, copied rather than invented
//!
//! cubecl consults that variable **only** when the requested device is
//! `DefaultDevice`/`BestAvailable`. An explicit `DiscreteGpu(0)` ignores it
//! outright. This module does the same, so the answer stays an answer about
//! what cubecl would do. A value the variable carries that cubecl cannot parse
//! is refused by name instead of being silently dropped: cubecl logs and
//! ignores it, and an owner who set it deserves better than a log line nobody
//! reads.

use burn::tensor::DeviceKind;

use crate::infer::provider::InferError;

/// The device vocabulary the wgpu backend selects from, re-exported so a
/// composition layer never names `burn` itself — the same courtesy
/// [`DecoderDevice`](super::DecoderDevice) already pays.
pub use burn::tensor::DeviceKind as DecoderGpuKind;

/// What kind of adapter the runtime found, in this crate's own vocabulary.
///
/// A restatement of `wgpu::DeviceType` rather than a re-export, for the reason
/// the whole module exists: a host asks this question in order to *decide*, and
/// a decision that had to name `wgpu` would put the host back inside the
/// backend it is trying to hold at arm's length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GpuDeviceType {
    /// A discrete GPU on its own board and its own memory.
    Discrete,
    /// A GPU sharing the CPU package and system memory.
    Integrated,
    /// A virtualised GPU, as presented by a hypervisor.
    Virtual,
    /// A software rasteriser running on the CPU.
    Cpu,
    /// An adapter the driver declines to classify. **This is the trap**:
    /// cubecl hands these out in place of a requested class rather than
    /// failing, so a host must check for it rather than assume it got what it
    /// asked for.
    Other,
}

impl std::fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let word = match self {
            GpuDeviceType::Discrete => "discrete",
            GpuDeviceType::Integrated => "integrated",
            GpuDeviceType::Virtual => "virtual",
            GpuDeviceType::Cpu => "cpu",
            GpuDeviceType::Other => "other",
        };
        f.write_str(word)
    }
}

#[cfg(not(target_family = "wasm"))]
impl From<wgpu::DeviceType> for GpuDeviceType {
    fn from(value: wgpu::DeviceType) -> Self {
        match value {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Cpu => GpuDeviceType::Cpu,
            wgpu::DeviceType::Other => GpuDeviceType::Other,
        }
    }
}

/// The adapter the wgpu runtime would bind for one [`DecoderGpuKind`], as
/// plain data.
///
/// Deliberately small and owned: it is meant to be logged, asserted on, and
/// carried into a host's own facts, none of which should keep a live wgpu
/// handle alive.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuAdapterFacts {
    /// The adapter's own name, as the driver reports it.
    pub name: String,
    /// The graphics API this selection ran through — the one
    /// `AutoGraphicsApi` picks on this platform, because that is the one the
    /// runtime will use.
    pub backend: String,
    /// What the driver says the adapter *is*, which is not always what was
    /// asked for.
    pub device_type: GpuDeviceType,
    /// Whether this adapter came out of the requested class's list, or out of
    /// the unclassified `Other` list cubecl falls through to. `false` is the
    /// silent substitution this module exists to make visible.
    pub matched_requested_class: bool,
}

impl std::fmt::Display for GpuAdapterFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {} adapter)",
            self.name, self.backend, self.device_type
        )
    }
}

/// The graphics API `AutoGraphicsApi` resolves to on this platform.
///
/// Copied from cubecl-wgpu's `graphics.rs` rather than called, because the
/// trait is not in esp's dependency graph: cubecl arrives through burn, whose
/// re-exports do not carry it. cubecl's `AUTO_GRAPHICS_BACKEND` override is
/// deliberately not mirrored — it is `#[cfg(test)]` inside cubecl's own crate
/// and is never active in a build that consumes it.
#[cfg(not(target_family = "wasm"))]
fn auto_backend() -> wgpu::Backend {
    if cfg!(target_os = "macos") {
        wgpu::Backend::Metal
    } else {
        wgpu::Backend::Vulkan
    }
}

/// cubecl's `get_device_override`, with its unparseable case made loud.
///
/// Returns `Ok(None)` when the variable is unset, `Ok(Some(kind))` when it
/// carries a value cubecl would honour, and an error when it carries one
/// cubecl would warn about and drop.
#[cfg(not(target_family = "wasm"))]
fn device_override() -> Result<Option<DeviceKind>, InferError> {
    const VAR: &str = "CUBECL_WGPU_DEFAULT_DEVICE";

    let Ok(value) = std::env::var(VAR) else {
        return Ok(None);
    };
    let parsed = if let Some(inner) = value.strip_prefix("DiscreteGpu(") {
        inner
            .strip_suffix(')')
            .and_then(|index| index.parse().ok())
            .map(DeviceKind::DiscreteGpu)
    } else if let Some(inner) = value.strip_prefix("IntegratedGpu(") {
        inner
            .strip_suffix(')')
            .and_then(|index| index.parse().ok())
            .map(DeviceKind::IntegratedGpu)
    } else if let Some(inner) = value.strip_prefix("VirtualGpu(") {
        inner
            .strip_suffix(')')
            .and_then(|index| index.parse().ok())
            .map(DeviceKind::VirtualGpu)
    } else if value == "Cpu" {
        Some(DeviceKind::Cpu)
    } else {
        None
    };

    parsed.map(Some).ok_or_else(|| {
        InferError::Backend(format!(
            "{VAR} is set to {value:?}, which the wgpu runtime cannot parse and would \
             silently ignore. Set it to DiscreteGpu(n), IntegratedGpu(n), VirtualGpu(n) \
             or Cpu, or unset it."
        ))
    })
}

/// Report the adapter the wgpu runtime would bind for `kind`, without binding
/// it and without panicking when there is none.
///
/// The failure cases are exactly cubecl's, turned from panics into refusals:
/// no adapter of the requested class *and* none in the unclassified fallback
/// list at that index. The *quiet* case — a fallback adapter standing in for
/// the requested class — is reported rather than refused, through
/// [`GpuAdapterFacts::matched_requested_class`], because whether an
/// unclassified adapter is acceptable is the caller's policy and not this
/// module's.
///
/// `DeviceKind::Existing` is refused: an externally created setup is not a
/// selection this module can predict, and answering for it would be a guess.
#[cfg(not(target_family = "wasm"))]
pub fn probe_gpu_adapter(kind: DecoderGpuKind) -> Result<GpuAdapterFacts, InferError> {
    let backend = auto_backend();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    // cubecl consults the override only for the "best available" request; an
    // explicit class ignores it. Mirrored exactly, including the asymmetry.
    let requested = match kind {
        DeviceKind::DefaultDevice => device_override()?.unwrap_or(DeviceKind::DefaultDevice),
        other => other,
    };

    let (wanted, index) = match requested {
        DeviceKind::DiscreteGpu(index) => (wgpu::DeviceType::DiscreteGpu, index),
        DeviceKind::IntegratedGpu(index) => (wgpu::DeviceType::IntegratedGpu, index),
        DeviceKind::VirtualGpu(index) => (wgpu::DeviceType::VirtualGpu, index),
        DeviceKind::Cpu => (wgpu::DeviceType::Cpu, 0),
        DeviceKind::DefaultDevice => return best_available(&instance, backend),
        DeviceKind::Existing(id) => {
            return Err(InferError::Backend(format!(
                "an existing wgpu setup (device {id}) is supplied by its creator, so no \
                 adapter selection happens that this probe could report"
            )));
        }
    };

    let adapters = pollster::block_on(instance.enumerate_adapters(backend.into()));
    let mut matching = Vec::new();
    let mut unclassified = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Other {
            unclassified.push(info);
        } else if info.device_type == wanted {
            matching.push(info);
        }
    }

    if index < matching.len() {
        return Ok(facts(matching.swap_remove(index), true));
    }
    if index < unclassified.len() {
        return Ok(facts(unclassified.swap_remove(index), false));
    }
    Err(InferError::Backend(format!(
        "no {wanted:?} adapter at index {index} on the {backend} backend: {} of that class \
         and {} unclassified adapter(s) are present",
        matching.len(),
        unclassified.len()
    )))
}

/// The `DefaultDevice` arm: cubecl asks wgpu for a high-power adapter rather
/// than enumerating, so the mirror does the same.
#[cfg(not(target_family = "wasm"))]
fn best_available(
    instance: &wgpu::Instance,
    backend: wgpu::Backend,
) -> Result<GpuAdapterFacts, InferError> {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..wgpu::RequestAdapterOptions::default()
    }))
    .map_err(|error| {
        InferError::Backend(format!(
            "no adapter available on the {backend} backend for a best-available \
             request: {error}"
        ))
    })?;
    Ok(facts(adapter.get_info(), true))
}

#[cfg(not(target_family = "wasm"))]
fn facts(info: wgpu::AdapterInfo, matched_requested_class: bool) -> GpuAdapterFacts {
    GpuAdapterFacts {
        name: info.name,
        backend: info.backend.to_string(),
        device_type: info.device_type.into(),
        matched_requested_class,
    }
}

/// On the browser there is one device and no enumeration, so there is nothing
/// to mirror and nothing this probe could add over simply asking for it.
#[cfg(target_family = "wasm")]
pub fn probe_gpu_adapter(_kind: DecoderGpuKind) -> Result<GpuAdapterFacts, InferError> {
    Err(InferError::Backend(
        "WebGPU exposes a single device and no adapter enumeration, so there is no \
         selection to report"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The forcing question, on real hardware: the discrete GPU this machine
    /// has is the one the runtime would bind, and it is reported as discrete
    /// rather than substituted from the unclassified list.
    #[test]
    fn the_discrete_probe_names_a_real_discrete_adapter() {
        let facts = probe_gpu_adapter(DecoderGpuKind::DiscreteGpu(0))
            .expect("this machine has a discrete GPU on the auto backend");
        println!("discrete adapter 0: {facts}");
        assert_eq!(
            facts.device_type,
            GpuDeviceType::Discrete,
            "an adapter reported as anything but discrete means cubecl substituted one: {facts:?}"
        );
        assert!(
            facts.matched_requested_class,
            "a discrete answer that came out of the unclassified fallback list is the \
             silent substitution this probe exists to catch: {facts:?}"
        );
        assert!(!facts.name.trim().is_empty(), "{facts:?}");
        assert!(!facts.backend.trim().is_empty(), "{facts:?}");
    }

    /// The absent case is a refusal carrying the index it could not satisfy —
    /// not the panic cubecl would raise, and not a quietly substituted
    /// adapter.
    #[test]
    fn an_absent_index_is_refused_rather_than_panicking_or_substituting() {
        let refusal = probe_gpu_adapter(DecoderGpuKind::DiscreteGpu(99))
            .expect_err("no machine has a hundred discrete GPUs");
        let message = refusal.to_string();
        println!("absent index refusal: {message}");
        assert!(message.contains("99"), "{message}");
    }
}
