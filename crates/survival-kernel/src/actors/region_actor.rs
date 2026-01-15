//! RegionActor: autonomous owner of region state with patch validation.
//!
//! Each region is represented by its own actor, providing:
//! - Natural conflict resolution via mailbox serialization
//! - Local state ownership (stigmergy model from paper)
//! - Post-patch validation to ensure δ_min > 0 (convergence theorem)

use std::collections::HashMap;
use std::sync::Arc;

use acton_reactive::prelude::*;
use tracing::{info, warn};

use crate::config::PressureAxisConfig;
use crate::messages::{
    ApplyDecay, EvaluatePatch, EvaluatePatchResponse, MeasurementResult, PressureResponse,
    QueryPressure, RefreshContent, RegionApplyPatch, RegionPatchResult,
};
use crate::pressure::{Sensor, Signals};
use crate::region::{RegionId, RegionState, RegionView};

/// Pending patch validation state.
#[derive(Clone)]
pub struct PendingValidation {
    /// Correlation ID for the original RegionApplyPatch
    pub correlation_id: String,
    /// Proposed new content
    pub new_content: String,
    /// Timestamp
    pub now_ms: u64,
    /// Inhibition window
    pub inhibit_ms: u64,
    /// Patch rationale for provenance
    pub rationale: String,
}

/// Actor state for a single region.
///
/// Owns the region's content and state, providing natural conflict resolution
/// through actor mailbox serialization - only one message processed at a time.
#[derive(Default, Clone)]
pub struct RegionActorState {
    /// Unique region identifier
    pub region_id: RegionId,
    /// Region kind (e.g., "function", "struct")
    pub kind: String,
    /// Current content of the region
    pub content: String,
    /// Arbitrary metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Mutable state: fitness, confidence, pressure EMA, inhibition
    pub state: RegionState,
    /// Handle to coordinator for sending responses
    pub coordinator: Option<ActorHandle>,
    /// Sensor for validating patch pressure reduction (legacy, may be None)
    pub sensor: Option<Arc<dyn Sensor>>,
    /// Pressure axis configuration for weighted pressure calculation
    pub pressure_axes: Vec<PressureAxisConfig>,
    /// Current signals from last measurement
    pub signals: Signals,
    /// Pending validation requests (correlation_id -> validation state)
    pub pending_validations: HashMap<String, PendingValidation>,
}

impl std::fmt::Debug for RegionActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionActorState")
            .field("region_id", &self.region_id)
            .field("kind", &self.kind)
            .field("content_len", &self.content.len())
            .field("coordinator", &self.coordinator.is_some())
            .field("sensor", &self.sensor.as_ref().map(|s| s.name()))
            .finish()
    }
}

/// Actor representing a single region in the artifact.
///
/// Handles:
/// - `ApplyDecay` - decay fitness/confidence at tick start
/// - `MeasurementResult` - update signals and pressure EMA
/// - `QueryPressure` - respond with current pressure state
/// - `RegionApplyPatch` - validate and apply patches
/// - `RefreshContent` - update content after artifact modification
pub struct RegionActor {
    /// Unique region identifier
    pub region_id: RegionId,
    /// Region kind
    pub kind: String,
    /// Initial content
    pub content: String,
    /// Initial metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Handle to coordinator
    pub coordinator: ActorHandle,
    /// Sensor for validation
    pub sensor: Arc<dyn Sensor>,
    /// Pressure axis configuration
    pub pressure_axes: Vec<PressureAxisConfig>,
}

impl RegionActor {
    /// Create a new RegionActor.
    pub fn new(
        region_id: RegionId,
        kind: String,
        content: String,
        metadata: HashMap<String, serde_json::Value>,
        coordinator: ActorHandle,
        sensor: Arc<dyn Sensor>,
        pressure_axes: Vec<PressureAxisConfig>,
    ) -> Self {
        Self {
            region_id,
            kind,
            content,
            metadata,
            coordinator,
            sensor,
            pressure_axes,
        }
    }

    /// Spawn this region actor in the given runtime.
    ///
    /// The actor will:
    /// 1. Subscribe to `ApplyDecay` and `QueryPressure` broadcasts
    /// 2. Handle messages and broadcast `PressureResponse` results
    pub async fn spawn(self, runtime: &mut ActorRuntime, now_ms: u64) -> ActorHandle {
        let region_id = self.region_id.clone();
        let region_id_short = format!("{:.8}", region_id);

        let mut actor = runtime
            .new_actor_with_name::<RegionActorState>(format!("Region:{}", region_id_short));

        // Initialize state
        actor.model.region_id = self.region_id;
        actor.model.kind = self.kind;
        actor.model.content = self.content;
        actor.model.metadata = self.metadata;
        actor.model.state = RegionState::new(now_ms);
        actor.model.coordinator = Some(self.coordinator);
        actor.model.sensor = Some(self.sensor);
        actor.model.pressure_axes = self.pressure_axes;
        actor.model.signals = HashMap::new();

        // Subscribe to broadcast messages BEFORE starting
        actor.handle().subscribe::<ApplyDecay>().await;
        actor.handle().subscribe::<QueryPressure>().await;

        // Configure handlers
        configure_region_actor(&mut actor);

        actor.start().await
    }
}

/// Configure message handlers for the RegionActor.
fn configure_region_actor(actor: &mut ManagedActor<Idle, RegionActorState>) {
    // Handle ApplyDecay - mutate_on because we modify state
    actor.mutate_on::<ApplyDecay>(|actor, context| {
        let msg = context.message();

        // Apply decay to fitness
        if msg.fitness_half_life_ms > 0 {
            let dt_ms = msg.now_ms.saturating_sub(actor.model.state.last_updated_ms);
            if dt_ms > 0 {
                let lambda = std::f64::consts::LN_2 / msg.fitness_half_life_ms as f64;
                actor.model.state.fitness *= (-lambda * dt_ms as f64).exp();
            }
        }

        // Apply decay to confidence
        if msg.confidence_half_life_ms > 0 {
            let dt_ms = msg.now_ms.saturating_sub(actor.model.state.last_updated_ms);
            if dt_ms > 0 {
                let lambda = std::f64::consts::LN_2 / msg.confidence_half_life_ms as f64;
                actor.model.state.confidence *= (-lambda * dt_ms as f64).exp();
            }
        }

        actor.model.state.last_updated_ms = msg.now_ms;

        Reply::ready()
    });

    // Handle MeasurementResult - update signals and pressure EMA
    actor.mutate_on::<MeasurementResult>(|actor, context| {
        let msg = context.message();

        // Merge new signals into our signal map
        for (key, value) in &msg.signals {
            actor.model.signals.insert(key.clone(), *value);
        }

        // Update pressure EMA for each axis
        let alpha = 0.2; // EMA smoothing factor
        for axis in &actor.model.pressure_axes {
            if let Some(signal_value) = msg.signals.get(&axis.expr) {
                let weight = axis
                    .kind_weights
                    .get(&actor.model.kind)
                    .copied()
                    .unwrap_or(axis.weight);
                let weighted_pressure = signal_value * weight;

                let current = actor
                    .model
                    .state
                    .pressure_ema
                    .get(&axis.name)
                    .copied()
                    .unwrap_or(weighted_pressure);
                let new_ema = alpha * weighted_pressure + (1.0 - alpha) * current;
                actor.model.state.pressure_ema.insert(axis.name.clone(), new_ema);
            }
        }

        Reply::ready()
    });

    // Handle QueryPressure - respond with current state via broker broadcast
    actor.act_on::<QueryPressure>(|actor, context| {
        let msg = context.message().clone();
        let broker = actor.broker().clone();
        let region_id = actor.model.region_id.clone();
        let kind = actor.model.kind.clone();
        let content = actor.model.content.clone();
        let metadata = actor.model.metadata.clone();
        let state = actor.model.state.clone();
        let signals = actor.model.signals.clone();

        // Calculate total pressure
        let total_pressure: f64 = state.pressure_ema.values().sum();
        let is_inhibited = state.is_inhibited(msg.now_ms);

        let response = PressureResponse {
            correlation_id: msg.correlation_id,
            region_id: region_id.clone(),
            total_pressure,
            is_inhibited,
            state,
            view: RegionView {
                id: region_id,
                kind,
                content,
                metadata,
            },
            signals,
        };

        // Broadcast response (coordinator subscribes to PressureResponse)
        Reply::pending(async move {
            broker.broadcast(response).await;
        })
    });

    // Handle RegionApplyPatch - request evaluation from coordinator
    actor.mutate_on::<RegionApplyPatch>(|actor, context| {
        let msg = context.message().clone();
        let coordinator = actor.model.coordinator.clone();
        let region_id = actor.model.region_id.clone();

        let Some(coordinator) = coordinator else {
            warn!(region_id = %region_id, "RegionApplyPatch: coordinator not set");
            return Reply::ready();
        };

        // Save pending validation state
        let validation_id = msg.correlation_id.clone();
        actor.model.pending_validations.insert(
            validation_id.clone(),
            PendingValidation {
                correlation_id: msg.correlation_id.clone(),
                new_content: String::new(), // Will be filled by coordinator response
                now_ms: msg.now_ms,
                inhibit_ms: msg.inhibit_ms,
                rationale: msg.patch.rationale.clone(),
            },
        );

        // Send EvaluatePatch to coordinator for clone-based validation
        let evaluate_msg = EvaluatePatch {
            correlation_id: validation_id,
            patch: msg.patch,
            now_ms: msg.now_ms,
            inhibit_ms: msg.inhibit_ms,
        };

        Reply::pending(async move {
            coordinator.send(evaluate_msg).await;
        })
    });

    // Handle EvaluatePatchResponse - use coordinator's clone-based validation result
    actor.mutate_on::<EvaluatePatchResponse>(|actor, context| {
        let msg = context.message().clone();
        let coordinator = actor.model.coordinator.clone();
        let region_id = actor.model.region_id.clone();

        let Some(coordinator) = coordinator else {
            warn!(region_id = %region_id, "EvaluatePatchResponse: coordinator not set");
            return Reply::ready();
        };

        // Retrieve pending validation state
        let Some(pending) = actor.model.pending_validations.remove(&msg.correlation_id) else {
            warn!(
                region_id = %region_id,
                correlation_id = %msg.correlation_id,
                "EvaluatePatchResponse: no pending validation found"
            );
            return Reply::ready();
        };

        // Use coordinator's clone-based evaluation result
        if !msg.should_accept {
            warn!(
                region_id = %region_id,
                delta = msg.pressure_delta,
                "Patch rejected - no improvement (coordinator validation)"
            );
            let result = RegionPatchResult {
                correlation_id: msg.correlation_id,
                region_id: region_id.clone(),
                success: false,
                new_content: None,
                pressure_delta: msg.pressure_delta,
                error: Some(format!(
                    "Patch provides no improvement (delta={:.2})",
                    msg.pressure_delta
                )),
            };
            return Reply::pending(async move {
                coordinator.send(result).await;
            });
        }

        // Accept patch
        info!(
            region_id = %region_id,
            pressure_delta = msg.pressure_delta,
            "Patch accepted (coordinator validation)"
        );

        actor.model.content = msg.new_content.clone();
        actor.model.state.fitness = (actor.model.state.fitness + 0.03).min(1.0);
        actor.model.state.confidence = (actor.model.state.confidence + 0.05).min(1.0);
        actor.model.state.suppress_until_ms = Some(pending.now_ms + pending.inhibit_ms);
        actor.model.state.last_updated_ms = pending.now_ms;
        actor.model.state.provenance.push(pending.rationale);

        let result = RegionPatchResult {
            correlation_id: msg.correlation_id,
            region_id,
            success: true,
            new_content: Some(msg.new_content),
            pressure_delta: msg.pressure_delta,
            error: None,
        };

        Reply::pending(async move {
            coordinator.send(result).await;
        })
    });

    // Handle RefreshContent - update content from artifact
    actor.mutate_on::<RefreshContent>(|actor, context| {
        let msg = context.message();
        actor.model.content = msg.new_content.clone();
        actor.model.metadata = msg.metadata.clone();
        Reply::ready()
    });
}
