//! Integration tests for the stigmergic claim system.
//!
//! Tests the full flow of:
//! - ClaimBatch -> ClaimBatchResult
//! - Multiple actors competing for the same cells
//! - Claim rollback on conflict
//!
//! Uses the proper acton-reactive trigger pattern:
//! 1. Test sends TriggerClaim to mock actor
//! 2. Mock actor sends ClaimBatch via envelope.new_envelope()
//! 3. ClaimManager responds via reply_envelope()
//! 4. Mock actor receives ClaimBatchResult

use std::collections::HashMap;
use std::sync::Arc;

use acton_reactive::prelude::*;
use tokio::sync::RwLock;
use tokio::time::Duration;
use uuid::Uuid;

use survival_kernel::actors::ClaimManager;
use survival_kernel::messages::{ClaimBatch, ClaimBatchResult, ResetClaims};
use survival_kernel::region::{Patch, PatchOp, RegionId};

/// Test helper to create a region ID from a name
fn test_region_id(name: &str) -> RegionId {
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes())
}

/// Test helper to create a mock Patch
fn mock_patch(region_id: RegionId, new_content: &str) -> Patch {
    Patch {
        region: region_id,
        op: PatchOp::Replace(new_content.to_string()),
        rationale: "test patch".to_string(),
        expected_delta: HashMap::new(),
    }
}

/// Trigger message to initiate a claim from the mock actor
#[derive(Debug, Clone)]
struct TriggerClaim {
    correlation_id: String,
    claims: Vec<(usize, u8)>,
    region_id: RegionId,
    claim_manager: ActorHandle,
}

/// Mock actor state for testing claim flows
#[derive(Default, Clone)]
struct MockActorState {
    name: String,
    received_results: Arc<RwLock<Vec<ClaimBatchResult>>>,
}

impl std::fmt::Debug for MockActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockActorState")
            .field("name", &self.name)
            .finish()
    }
}

/// Spawn a mock actor that handles TriggerClaim and records ClaimBatchResult
async fn spawn_mock_claimer(
    runtime: &mut ActorRuntime,
    name: &str,
    results: Arc<RwLock<Vec<ClaimBatchResult>>>,
) -> ActorHandle {
    let mut actor = runtime.new_actor_with_name::<MockActorState>(name.to_string());
    actor.model.name = name.to_string();
    actor.model.received_results = results;

    // Subscribe to messages BEFORE starting
    actor.handle().subscribe::<TriggerClaim>().await;
    actor.handle().subscribe::<ClaimBatchResult>().await;

    let actor_name = name.to_string();

    // Handle TriggerClaim - send ClaimBatch to ClaimManager via proper envelope routing
    actor.act_on::<TriggerClaim>(move |_actor, context| {
        let msg = context.message().clone();
        let name = actor_name.clone();

        // Create envelope with proper reply chain using new_envelope
        let request_envelope = context.new_envelope(&msg.claim_manager.reply_address());

        Reply::pending(async move {
            let claim_batch = ClaimBatch {
                correlation_id: msg.correlation_id,
                claims: msg.claims,
                region_id: msg.region_id,
                actor_name: name,
                patch: mock_patch(msg.region_id, "test"),
                prompt_tokens: 10,
                completion_tokens: 5,
            };

            request_envelope.send(claim_batch).await;
        })
    });

    // Handle ClaimBatchResult - record it
    actor.mutate_on::<ClaimBatchResult>(|actor, context| {
        let msg = context.message().clone();
        let results = actor.model.received_results.clone();

        Reply::pending(async move {
            results.write().await.push(msg);
        })
    });

    actor.start().await
}

#[tokio::test]
async fn test_claim_batch_granted_when_no_conflict() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn ClaimManager
    let claim_manager_handle = ClaimManager::spawn(&mut runtime).await;

    // Wait for ClaimManagerReady
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create mock actor to record results
    let results = Arc::new(RwLock::new(Vec::new()));
    let mock_actor = spawn_mock_claimer(&mut runtime, "MockActor1", results.clone()).await;

    // Trigger a claim via the mock actor (proper envelope routing)
    let region_id = test_region_id("row_0");
    mock_actor
        .send(TriggerClaim {
            correlation_id: "test-correlation-1".to_string(),
            claims: vec![(0, 1), (1, 2), (2, 3)],
            region_id,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;

    // Wait for response
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check results
    let received = results.read().await;
    assert_eq!(received.len(), 1, "Should receive exactly one result");
    assert!(received[0].all_granted, "All claims should be granted");
    assert_eq!(received[0].correlation_id, "test-correlation-1");

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn test_claim_batch_denied_on_conflict() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn ClaimManager
    let claim_manager_handle = ClaimManager::spawn(&mut runtime).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create mock actors to record results
    let results1 = Arc::new(RwLock::new(Vec::new()));
    let results2 = Arc::new(RwLock::new(Vec::new()));
    let mock_actor1 = spawn_mock_claimer(&mut runtime, "MockActor1", results1.clone()).await;
    let mock_actor2 = spawn_mock_claimer(&mut runtime, "MockActor2", results2.clone()).await;

    let region_id1 = test_region_id("row_0");
    let region_id2 = test_region_id("row_1");

    // First claim - should succeed
    mock_actor1
        .send(TriggerClaim {
            correlation_id: "test-1".to_string(),
            claims: vec![(0, 5)], // Claim col 0, value 5
            region_id: region_id1,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;

    // Wait for first claim to process
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Second claim with same (col, value) - should be denied
    mock_actor2
        .send(TriggerClaim {
            correlation_id: "test-2".to_string(),
            claims: vec![(0, 5)], // Same col 0, value 5 - conflict!
            region_id: region_id2,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;

    // Wait for second claim to process
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check results
    let received1 = results1.read().await;
    let received2 = results2.read().await;

    assert_eq!(received1.len(), 1, "Actor 1 should receive result");
    assert!(received1[0].all_granted, "First claim should be granted");

    assert_eq!(received2.len(), 1, "Actor 2 should receive result");
    assert!(
        !received2[0].all_granted,
        "Second claim should be denied due to conflict"
    );

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn test_reset_claims_clears_all() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn ClaimManager
    let claim_manager_handle = ClaimManager::spawn(&mut runtime).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let results = Arc::new(RwLock::new(Vec::new()));
    let mock_actor = spawn_mock_claimer(&mut runtime, "MockActor", results.clone()).await;

    let region_id = test_region_id("row_0");

    // First claim
    mock_actor
        .send(TriggerClaim {
            correlation_id: "test-1".to_string(),
            claims: vec![(0, 5)],
            region_id,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Reset claims
    claim_manager_handle.send(ResetClaims).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Same claim again should now succeed (was reset)
    mock_actor
        .send(TriggerClaim {
            correlation_id: "test-2".to_string(),
            claims: vec![(0, 5)], // Same cell as before
            region_id,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Both claims should be granted (second one after reset)
    let received = results.read().await;
    assert_eq!(received.len(), 2, "Should receive two results");
    assert!(received[0].all_granted, "First claim should be granted");
    assert!(
        received[1].all_granted,
        "Second claim should be granted after reset"
    );

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn test_batch_claim_atomic_rollback() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn ClaimManager
    let claim_manager_handle = ClaimManager::spawn(&mut runtime).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let results1 = Arc::new(RwLock::new(Vec::new()));
    let results2 = Arc::new(RwLock::new(Vec::new()));
    let mock_actor1 = spawn_mock_claimer(&mut runtime, "MockActor1", results1.clone()).await;
    let mock_actor2 = spawn_mock_claimer(&mut runtime, "MockActor2", results2.clone()).await;

    let region_id1 = test_region_id("row_0");
    let region_id2 = test_region_id("row_1");

    // First actor claims (0, 1)
    mock_actor1
        .send(TriggerClaim {
            correlation_id: "test-1".to_string(),
            claims: vec![(0, 1)],
            region_id: region_id1,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Second actor tries batch with (0, 1) as second claim
    // First cell (1, 2) would be granted, but (0, 1) conflicts
    // Both should be rolled back
    mock_actor2
        .send(TriggerClaim {
            correlation_id: "test-2".to_string(),
            claims: vec![(1, 2), (0, 1)], // (1, 2) ok, but (0, 1) conflicts
            region_id: region_id2,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let received1 = results1.read().await;
    let received2 = results2.read().await;

    assert!(received1[0].all_granted, "First claim should be granted");
    assert!(
        !received2[0].all_granted,
        "Second batch should be denied (atomic)"
    );

    // Now verify (1, 2) was rolled back by trying to claim it
    let results3 = Arc::new(RwLock::new(Vec::new()));
    let mock_actor3 = spawn_mock_claimer(&mut runtime, "MockActor3", results3.clone()).await;

    let region_id3 = test_region_id("row_2");
    mock_actor3
        .send(TriggerClaim {
            correlation_id: "test-3".to_string(),
            claims: vec![(1, 2)], // Should be available due to rollback
            region_id: region_id3,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let received3 = results3.read().await;
    assert!(
        received3[0].all_granted,
        "(1, 2) should be available after rollback"
    );

    runtime.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn test_different_values_same_column_allowed() {
    let mut runtime = ActonApp::launch_async().await;

    // Spawn ClaimManager
    let claim_manager_handle = ClaimManager::spawn(&mut runtime).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let results1 = Arc::new(RwLock::new(Vec::new()));
    let results2 = Arc::new(RwLock::new(Vec::new()));
    let mock_actor1 = spawn_mock_claimer(&mut runtime, "MockActor1", results1.clone()).await;
    let mock_actor2 = spawn_mock_claimer(&mut runtime, "MockActor2", results2.clone()).await;

    let region_id1 = test_region_id("row_0");
    let region_id2 = test_region_id("row_1");

    // Actor 1 claims (col=0, value=5)
    mock_actor1
        .send(TriggerClaim {
            correlation_id: "test-1".to_string(),
            claims: vec![(0, 5)],
            region_id: region_id1,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Actor 2 claims (col=0, value=3) - different value, same column
    // In Latin Square, different rows can have different values in same column
    mock_actor2
        .send(TriggerClaim {
            correlation_id: "test-2".to_string(),
            claims: vec![(0, 3)], // Same column, different value
            region_id: region_id2,
            claim_manager: claim_manager_handle.clone(),
        })
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let received1 = results1.read().await;
    let received2 = results2.read().await;

    assert!(received1[0].all_granted, "First claim should be granted");
    assert!(
        received2[0].all_granted,
        "Second claim with different value should also be granted"
    );

    runtime.shutdown_all().await.unwrap();
}
