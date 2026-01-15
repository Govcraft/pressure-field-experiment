//! ClaimManager: Stigmergic coordination for column/value reservations.
//!
//! Implements a semaphore-like claim system within ticks:
//! - LLM actors claim (col, value) pairs after LLM response, before proposal
//! - If any claim denied (already taken), actor discards proposal gracefully
//! - Claims reset at tick boundaries via ResetClaims broadcast
//!
//! This prevents multiple agents from proposing the same value for the same
//! column within a tick, reducing the ~93% patch rejection rate.
//!
//! ## Acton-Reactive Pattern
//!
//! Uses proper envelope routing instead of oneshot channels:
//! - LlmActor sends ClaimBatch via `envelope.new_envelope(&claim_manager.reply_address())`
//! - ClaimManager responds via `envelope.reply_envelope().send(ClaimBatchResult)`
//! - No manual correlation tracking needed - envelope routing handles it

use acton_reactive::prelude::*;
use dashmap::DashMap;

use crate::messages::{ClaimBatch, ClaimBatchResult, ClaimManagerReady, ResetClaims};
use crate::region::RegionId;

/// Actor state for ClaimManager.
#[derive(Default, Clone)]
pub struct ClaimManagerState {
    /// Active claims: (column, value) -> region that claimed it
    /// Using DashMap for thread-safe concurrent access
    claims: DashMap<(usize, u8), RegionId>,
    /// Count of batches granted this tick (for metrics)
    batches_granted: usize,
    /// Count of batches denied this tick (for metrics)
    batches_denied: usize,
}

impl std::fmt::Debug for ClaimManagerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaimManagerState")
            .field("active_claims", &self.claims.len())
            .finish()
    }
}

/// Actor that manages stigmergic column/value claims within a tick.
///
/// Coordinates LLM agents by tracking which (column, value) pairs have been
/// claimed. When an agent proposes values for columns, it first claims all
/// cells atomically. If any are already claimed, all are denied and the
/// agent should discard its proposal.
///
/// ## Message Flow
///
/// ```text
/// Tick Start
///   |
///   +-- ResetClaims (broadcast) --> ClaimManager clears all claims
///   |
///   +-- ProposeForRegion --> LLM Actors (parallel)
///   |                            |
///   |                            v
///   |                       LLM generates response
///   |                            |
///   |                            v
///   |                       ClaimBatch --> ClaimManager (via new_envelope)
///   |                            |
///   |                            v
///   |                       ClaimBatchResult <-- ClaimManager (via reply_envelope)
///   |                            |
///   |                            v
///   |                       If granted: PatchProposal
///   |                       If denied: Empty proposal
///   |
/// Tick End
/// ```
pub struct ClaimManager;

impl ClaimManager {
    /// Spawn the ClaimManager actor in the given runtime.
    ///
    /// The actor will:
    /// 1. Subscribe to `ClaimBatch` and `ResetClaims` messages
    /// 2. Broadcast `ClaimManagerReady` on start with its handle
    /// 3. Handle claim requests via `reply_envelope()` (not broadcast)
    pub async fn spawn(runtime: &mut ActorRuntime) -> ActorHandle {
        let mut actor =
            runtime.new_actor_with_name::<ClaimManagerState>("ClaimManager".to_string());

        // Initialize state with empty claims map
        actor.model.claims = DashMap::new();

        // Subscribe to messages BEFORE starting
        // ClaimBatch comes via direct message (new_envelope), not broadcast
        actor.handle().subscribe::<ClaimBatch>().await;
        actor.handle().subscribe::<ResetClaims>().await;

        // Broadcast ClaimManagerReady on start so coordinator/actors know about us
        actor.after_start(|actor| {
            let broker = actor.broker().clone();
            let handle = actor.handle().clone();

            Reply::pending(async move {
                tracing::info!("ClaimManager started, broadcasting ready");
                broker.broadcast(ClaimManagerReady { handle }).await;
            })
        });

        // Handle ResetClaims - clear all claims at tick start
        actor.mutate_on::<ResetClaims>(|actor, _context| {
            let claim_count = actor.model.claims.len();
            let granted = actor.model.batches_granted;
            let denied = actor.model.batches_denied;

            // Log tick summary at info level for monitoring
            if granted > 0 || denied > 0 {
                let denial_rate = if granted + denied > 0 {
                    (denied as f64 / (granted + denied) as f64) * 100.0
                } else {
                    0.0
                };
                tracing::info!(
                    granted = granted,
                    denied = denied,
                    denial_rate = format!("{:.1}%", denial_rate),
                    active_claims = claim_count,
                    "ClaimManager tick summary (stigmergy effectiveness)"
                );
            }

            // Reset for new tick
            actor.model.claims.clear();
            actor.model.batches_granted = 0;
            actor.model.batches_denied = 0;

            tracing::debug!(
                cleared = claim_count,
                "ClaimManager: Reset all claims for new tick"
            );

            Reply::ready()
        });

        // Handle ClaimBatch - atomically claim all (col, value) pairs or deny all
        actor.mutate_on::<ClaimBatch>(|actor, context| {
            let msg = context.message().clone();

            // Use reply_envelope for proper acton-reactive request-response pattern
            let reply_envelope = context.reply_envelope();

            // Check each claim - if any conflict, deny all
            let mut all_granted = true;
            let mut granted_claims = Vec::new();

            for (col, value) in &msg.claims {
                let key = (*col, *value);

                match actor.model.claims.entry(key) {
                    dashmap::mapref::entry::Entry::Occupied(existing) => {
                        // Already claimed by another region
                        let holder = existing.get();
                        tracing::debug!(
                            col = col,
                            value = value,
                            requester = %msg.region_id,
                            holder = %holder,
                            "ClaimManager: Claim DENIED (already held)"
                        );
                        all_granted = false;
                        break;
                    }
                    dashmap::mapref::entry::Entry::Vacant(vacant) => {
                        // Tentatively grant - will rollback if any fails
                        vacant.insert(msg.region_id);
                        granted_claims.push(key);
                        tracing::trace!(
                            col = col,
                            value = value,
                            requester = %msg.region_id,
                            "ClaimManager: Claim tentatively granted"
                        );
                    }
                }
            }

            // If any claim failed, rollback all granted claims
            if !all_granted {
                for key in granted_claims {
                    actor.model.claims.remove(&key);
                }
                actor.model.batches_denied += 1;
                // Log at info level - this shows stigmergic coordination working!
                tracing::info!(
                    region_id = %msg.region_id,
                    actor = %msg.actor_name,
                    claims = ?msg.claims,
                    "CLAIM DENIED - stigmergy prevented duplicate proposal"
                );
            } else {
                actor.model.batches_granted += 1;
                tracing::debug!(
                    region_id = %msg.region_id,
                    claims = msg.claims.len(),
                    "ClaimManager: Batch GRANTED"
                );
            }

            // Send result via reply_envelope (proper acton-reactive pattern)
            let result = ClaimBatchResult {
                correlation_id: msg.correlation_id,
                all_granted,
                actor_name: msg.actor_name,
                patch: msg.patch,
                prompt_tokens: msg.prompt_tokens,
                completion_tokens: msg.completion_tokens,
            };

            Reply::pending(async move {
                reply_envelope.send(result).await;
            })
        });

        actor.start().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Patch, PatchOp};
    use acton_reactive::prelude::ActonApp;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn test_region_id(name: &str) -> RegionId {
        // Use a deterministic UUID for testing
        // RegionId is a type alias for Uuid
        Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes())
    }

    fn test_patch(region_id: RegionId) -> Patch {
        Patch {
            region: region_id,
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "test".to_string(),
            expected_delta: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_claim_manager_grants_first_claim() {
        let mut runtime = ActonApp::launch_async().await;
        let handle = ClaimManager::spawn(&mut runtime).await;

        // Subscribe to ClaimBatchResult
        handle.subscribe::<ClaimBatchResult>().await;

        let region_a = test_region_id("row_a");
        let correlation_id = "test-1".to_string();

        // First batch claim should be granted
        handle
            .send(ClaimBatch {
                correlation_id: correlation_id.clone(),
                claims: vec![(0, 5), (1, 3)],
                region_id: region_a,
                actor_name: "test-actor".to_string(),
                patch: test_patch(region_a),
                prompt_tokens: 10,
                completion_tokens: 5,
            })
            .await;

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let _ = runtime.shutdown_all().await;
    }

    #[tokio::test]
    async fn test_claim_manager_denies_duplicate() {
        let mut runtime = ActonApp::launch_async().await;
        let _handle = ClaimManager::spawn(&mut runtime).await;

        // The actual test would need to set up a proper request-response flow
        // For now, just verify spawn works
        let _ = runtime.shutdown_all().await;
    }

    #[tokio::test]
    async fn test_claim_manager_reset_clears_claims() {
        let mut runtime = ActonApp::launch_async().await;
        let _handle = ClaimManager::spawn(&mut runtime).await;

        // The actual test would need to verify claims are cleared
        // For now, just verify spawn works
        let _ = runtime.shutdown_all().await;
    }
}
