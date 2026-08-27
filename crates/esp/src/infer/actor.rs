// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The armillary inference actor (local-models harness §3).
//!
//! Holds a loaded [`InferenceProvider`] on its own thread, answers
//! generation requests, and streams fragments back as typed updates —
//! the host never blocks its loop on a token. The provider is built *on
//! the actor thread* via the build closure, per armillary doctrine, so
//! model load cost never lands on the kernel thread either.
//!
//! Cancellation: mid-generation, the actor drains its command channel
//! from inside the streaming callback. A [`InferCommand::Cancel`] for the
//! running request stops it via the provider callback's `ControlFlow`
//! return (the in-flight fragment is dropped, not forwarded); a `Cancel`
//! for a queued request drops it before it starts; new `Generate`
//! commands seen mid-stream are queued, not lost. Cancelled requests
//! emit [`InferUpdate::Cancelled`] — real status, never silence.
//!
//! Feature-gated (`actor`) because armillary spawns OS threads; the
//! seam's core stays thread-free so it keeps compiling for
//! wasm32-unknown-unknown.

use std::collections::{HashSet, VecDeque};
use std::ops::ControlFlow;
use std::sync::mpsc::Receiver;

use armillary::actor::spawn_named;
use armillary::{ActorHandle, Wake};

use crate::infer::provider::{GenerationRequest, InferError, InferenceProvider, ModelCapability};

/// Commands the kernel sends the inference actor.
#[derive(Debug, Clone, PartialEq)]
pub enum InferCommand {
    /// Run one generation. `id` is the kernel's correlation key; every
    /// update for this request carries it back.
    Generate { id: u64, request: GenerationRequest },
    /// Cancel request `id`: stops it mid-stream if running, drops it if
    /// still queued. Unknown/finished ids are ignored.
    Cancel { id: u64 },
}

/// Updates the actor streams back to the kernel.
#[derive(Debug, Clone, PartialEq)]
pub enum InferUpdate {
    /// The provider is loaded and ready; sent once at actor startup so
    /// status surfaces can show the real capability, not a guess.
    Ready { capability: ModelCapability },
    /// Generation `id` began.
    Started { id: u64 },
    /// One streamed fragment of generation `id`, in production order.
    Fragment { id: u64, text: String },
    /// Generation `id` completed; `text` is the full assembled output.
    Finished { id: u64, text: String },
    /// Generation `id` was cancelled (mid-stream or while queued).
    Cancelled { id: u64 },
    /// Generation `id` failed.
    Failed { id: u64, error: InferError },
}

/// Spawn the inference actor. `build` runs on the actor thread and
/// constructs (or loads) the provider there; the kernel gets the usual
/// armillary pair of a `Send` handle and an update receiver.
pub fn spawn_inference_actor<F>(
    wake: Wake,
    build: F,
) -> (ActorHandle<InferCommand>, Receiver<InferUpdate>)
where
    F: FnOnce() -> Box<dyn InferenceProvider> + Send + 'static,
{
    spawn_named::<InferCommand, InferUpdate, _>("infer", wake, move |commands, out| {
        let provider = build();
        out.emit(InferUpdate::Ready {
            capability: provider.capability().clone(),
        });

        let mut pending: VecDeque<(u64, GenerationRequest)> = VecDeque::new();
        let mut cancelled: HashSet<u64> = HashSet::new();

        loop {
            // Refill the queue: block for one command when idle (a closed
            // channel with nothing pending ends the actor), then drain
            // whatever else is waiting. A disconnect seen while pending
            // work exists must NOT exit — queued requests still run; the
            // blocking recv above ends the loop afterwards.
            if pending.is_empty() {
                match commands.recv() {
                    Ok(c) => absorb(c, &mut pending, &mut cancelled),
                    Err(_) => break,
                }
            }
            while let Ok(c) = commands.try_recv() {
                absorb(c, &mut pending, &mut cancelled);
            }

            let Some((id, request)) = pending.pop_front() else {
                continue;
            };
            if cancelled.remove(&id) {
                out.emit(InferUpdate::Cancelled { id });
                continue;
            }

            out.emit(InferUpdate::Started { id });
            let streamed = out.clone();
            let mut was_cancelled = false;
            let result = provider.generate_streaming(&request, &mut |fragment| {
                // Drain commands mid-stream so a Cancel can land while
                // tokens flow; other commands queue for later.
                loop {
                    match commands.try_recv() {
                        Ok(c) => absorb(c, &mut pending, &mut cancelled),
                        Err(_) => break,
                    }
                }
                if cancelled.remove(&id) {
                    was_cancelled = true;
                    return ControlFlow::Break(());
                }
                streamed.emit(InferUpdate::Fragment {
                    id,
                    text: fragment.to_string(),
                });
                ControlFlow::Continue(())
            });
            match (was_cancelled, result) {
                (true, _) => out.emit(InferUpdate::Cancelled { id }),
                (false, Ok(text)) => out.emit(InferUpdate::Finished { id, text }),
                (false, Err(error)) => out.emit(InferUpdate::Failed { id, error }),
            }
        }
    })
}

fn absorb(
    command: InferCommand,
    pending: &mut VecDeque<(u64, GenerationRequest)>,
    cancelled: &mut HashSet<u64>,
) {
    match command {
        InferCommand::Generate { id, request } => pending.push_back((id, request)),
        InferCommand::Cancel { id } => {
            cancelled.insert(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::infer::stub::StubInferenceProvider;

    fn no_wake() -> Wake {
        Arc::new(|| {})
    }

    #[test]
    fn actor_streams_ready_started_fragments_finished_in_order() {
        let (handle, updates) = spawn_inference_actor(no_wake(), || {
            Box::new(StubInferenceProvider::new().with_response("ping", "pong pang"))
        });
        assert!(handle.command(InferCommand::Generate {
            id: 7,
            request: GenerationRequest {
                prompt: "ping".to_string(),
                ..Default::default()
            },
        }));
        handle.join();

        let got: Vec<InferUpdate> = updates.iter().collect();
        assert!(
            matches!(&got[0], InferUpdate::Ready { capability } if capability.loader == "stub")
        );
        assert_eq!(got[1], InferUpdate::Started { id: 7 });
        assert_eq!(
            got[2],
            InferUpdate::Fragment {
                id: 7,
                text: "pong".to_string()
            }
        );
        assert_eq!(
            got[3],
            InferUpdate::Fragment {
                id: 7,
                text: " pang".to_string()
            }
        );
        assert_eq!(
            got[4],
            InferUpdate::Finished {
                id: 7,
                text: "pong pang".to_string()
            }
        );
        assert_eq!(got.len(), 5);
    }

    #[test]
    fn failed_generation_reports_error_with_id() {
        let (handle, updates) =
            spawn_inference_actor(no_wake(), || Box::new(StubInferenceProvider::new()));
        assert!(handle.command(InferCommand::Generate {
            id: 1,
            request: GenerationRequest::default(), // empty prompt → InvalidRequest
        }));
        handle.join();

        let got: Vec<InferUpdate> = updates.iter().collect();
        assert!(matches!(
            got.last(),
            Some(InferUpdate::Failed {
                id: 1,
                error: InferError::InvalidRequest(_)
            })
        ));
    }

    #[test]
    fn sequential_requests_keep_their_correlation_ids() {
        let (handle, updates) = spawn_inference_actor(no_wake(), || {
            Box::new(
                StubInferenceProvider::new()
                    .with_response("a", "alpha")
                    .with_response("b", "beta"),
            )
        });
        for (id, prompt) in [(1u64, "a"), (2u64, "b")] {
            handle.command(InferCommand::Generate {
                id,
                request: GenerationRequest {
                    prompt: prompt.to_string(),
                    ..Default::default()
                },
            });
        }
        handle.join();

        let got: Vec<InferUpdate> = updates.iter().collect();
        let finished: Vec<(u64, String)> = got
            .iter()
            .filter_map(|u| match u {
                InferUpdate::Finished { id, text } => Some((*id, text.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            finished,
            vec![(1, "alpha".to_string()), (2, "beta".to_string())]
        );
    }

    /// Generate + Cancel are queued before the actor starts working. The
    /// cancel lands either in the pre-pop drain (dropped before start) or
    /// in the first callback (stopped mid-stream) depending on thread
    /// interleaving — both are correct; what must hold is: Cancelled is
    /// reported and no fragment or finish leaks.
    #[test]
    fn cancel_lands_mid_stream_and_suppresses_fragments() {
        let (handle, updates) = spawn_inference_actor(no_wake(), || {
            Box::new(StubInferenceProvider::new().with_response("q", "one two three four five"))
        });
        handle.command(InferCommand::Generate {
            id: 3,
            request: GenerationRequest {
                prompt: "q".to_string(),
                ..Default::default()
            },
        });
        handle.command(InferCommand::Cancel { id: 3 });
        handle.join();

        let got: Vec<InferUpdate> = updates.iter().collect();
        assert!(got.contains(&InferUpdate::Cancelled { id: 3 }));
        assert!(
            !got.iter().any(|u| matches!(
                u,
                InferUpdate::Fragment { .. } | InferUpdate::Finished { .. }
            )),
            "no fragments or finish may leak past a cancel: {got:?}"
        );
    }

    /// A cancel for a queued (not yet started) request drops it before
    /// any work happens, while the other request still completes.
    #[test]
    fn cancel_drops_queued_request() {
        let (handle, updates) = spawn_inference_actor(no_wake(), || {
            Box::new(
                StubInferenceProvider::new()
                    .with_response("a", "alpha")
                    .with_response("b", "beta"),
            )
        });
        handle.command(InferCommand::Generate {
            id: 1,
            request: GenerationRequest {
                prompt: "a".to_string(),
                ..Default::default()
            },
        });
        handle.command(InferCommand::Generate {
            id: 2,
            request: GenerationRequest {
                prompt: "b".to_string(),
                ..Default::default()
            },
        });
        handle.command(InferCommand::Cancel { id: 2 });
        handle.join();

        let got: Vec<InferUpdate> = updates.iter().collect();
        assert!(got.contains(&InferUpdate::Finished {
            id: 1,
            text: "alpha".to_string()
        }));
        assert!(got.contains(&InferUpdate::Cancelled { id: 2 }));
        assert!(
            !got.iter().any(|u| matches!(
                u,
                InferUpdate::Fragment { id: 2, .. } | InferUpdate::Finished { id: 2, .. }
            )),
            "a cancelled request must produce no fragments or finish: {got:?}"
        );
    }
}
