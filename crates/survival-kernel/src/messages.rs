//! Message types for acton-reactive actor communication.
//!
//! Messages use correlation IDs (via mti crate) to track request-response
//! patterns across multiple concurrent actors.

use std::collections::HashMap;

use crate::pressure::{PressureVector, Signals};
use crate::region::{Patch, RegionId, RegionState, RegionView};

/// Notification that a sensor is ready - broadcast by SensorActors on start.
///
/// The coordinator gets the sensor's ERN from the message envelope.
/// No fields needed - the actor name contains the sensor identity.
#[derive(Debug, Clone)]
pub struct SensorReady;

/// Notification that a patch actor is ready - broadcast on start.
///
/// The coordinator gets the actor's ERN from the message envelope.
#[derive(Debug, Clone)]
pub struct PatchActorReady;

/// Request to measure signals for a region - broadcast to SensorActors.
#[derive(Debug, Clone)]
pub struct MeasureRegion {
    /// Correlation ID for this tick's measurements
    pub correlation_id: String,
    /// The region to measure
    pub region_id: RegionId,
    /// View of the region content
    pub region_view: RegionView,
    /// Current timestamp
    pub now_ms: u64,
}

/// Result of measuring a region - sent back to Coordinator.
#[derive(Debug, Clone)]
pub struct MeasurementResult {
    /// Correlation ID matching the original request
    pub correlation_id: String,
    /// The region that was measured
    pub region_id: RegionId,
    /// Name of the sensor that produced this result
    pub sensor_name: String,
    /// Measured signals (axis -> value)
    pub signals: Signals,
}

/// Request to propose patches for a high-pressure region - sent to PatchActors.
#[derive(Debug, Clone)]
pub struct ProposeForRegion {
    /// Correlation ID for this tick's proposals
    pub correlation_id: String,
    /// The region to propose patches for
    pub region_id: RegionId,
    /// View of the region content
    pub region_view: RegionView,
    /// Current signals for this region
    pub signals: Signals,
    /// Current pressures for this region
    pub pressures: PressureVector,
    /// Current state for this region
    pub state: RegionState,
}

/// Patch proposal result - sent back to Coordinator.
#[derive(Debug, Clone)]
pub struct PatchProposal {
    /// Correlation ID matching the original request
    pub correlation_id: String,
    /// Name of the actor that produced this proposal
    pub actor_name: String,
    /// Proposed patches with scores (higher = better)
    pub patches: Vec<(f64, Patch)>,
}

/// Tick trigger - sent to KernelCoordinator to start a tick cycle.
#[derive(Debug, Clone)]
pub struct Tick {
    /// Current timestamp
    pub now_ms: u64,
}

/// Tick completion notification - sent to registered tick driver.
#[derive(Debug, Clone)]
pub struct TickComplete {
    /// Result of the tick
    pub result: crate::kernel::TickResult,
}

/// Register the tick driver handle with the coordinator.
///
/// The coordinator will send TickComplete messages to this handle.
#[derive(Debug, Clone)]
pub struct RegisterTickDriver {
    /// The tick driver's actor handle
    pub handle: acton_reactive::prelude::ActorHandle,
}

// ============================================================================
// RegionActor Messages
// ============================================================================

/// Apply temporal decay to region fitness and confidence.
///
/// Broadcast to all RegionActors at the start of each tick.
#[derive(Debug, Clone)]
pub struct ApplyDecay {
    /// Current timestamp for decay calculation
    pub now_ms: u64,
    /// Half-life for fitness decay (milliseconds)
    pub fitness_half_life_ms: u64,
    /// Half-life for confidence decay (milliseconds)
    pub confidence_half_life_ms: u64,
}

/// Query a region's current pressure state.
///
/// Sent to RegionActor, expects PressureResponse.
#[derive(Debug, Clone)]
pub struct QueryPressure {
    /// Correlation ID for this query
    pub correlation_id: String,
    /// Current timestamp
    pub now_ms: u64,
}

/// Response with region's pressure state.
///
/// Sent from RegionActor back to Coordinator.
#[derive(Debug, Clone)]
pub struct PressureResponse {
    /// Correlation ID matching the original request
    pub correlation_id: String,
    /// The region that was queried
    pub region_id: RegionId,
    /// Total weighted pressure
    pub total_pressure: f64,
    /// Whether the region is currently inhibited
    pub is_inhibited: bool,
    /// Current state snapshot
    pub state: RegionState,
    /// View of the region content
    pub view: RegionView,
    /// Current signals for this region
    pub signals: Signals,
}

/// Apply a patch to a region with validation.
///
/// Sent to RegionActor. The actor validates that the patch actually reduces
/// pressure before accepting it (ensures δ_min > 0 per convergence theorem).
#[derive(Debug, Clone)]
pub struct RegionApplyPatch {
    /// Correlation ID for this patch operation
    pub correlation_id: String,
    /// The patch to apply
    pub patch: Patch,
    /// Current timestamp
    pub now_ms: u64,
    /// Current tick number (for file naming)
    pub tick: usize,
    /// Inhibition window after applying (milliseconds)
    pub inhibit_ms: u64,
    /// Minimum pressure reduction required to accept
    pub min_expected_improvement: f64,
}

/// Result of patch application attempt.
///
/// Sent from RegionActor back to Coordinator.
#[derive(Debug, Clone)]
pub struct RegionPatchResult {
    /// Correlation ID matching the original request
    pub correlation_id: String,
    /// The region that was patched (or attempted)
    pub region_id: RegionId,
    /// Whether the patch was accepted
    pub success: bool,
    /// New content if patch was applied
    pub new_content: Option<String>,
    /// Actual measured pressure improvement (positive = better)
    pub pressure_delta: f64,
    /// Error message if patch was rejected
    pub error: Option<String>,
}

/// Update region content after artifact re-parse.
///
/// Sent when the artifact is modified and regions need to refresh.
#[derive(Debug, Clone)]
pub struct RefreshContent {
    /// New content for the region
    pub new_content: String,
    /// Updated metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request coordinator to write artifact with proposed patch for validation.
///
/// Sent from RegionActor to Coordinator. The coordinator applies the patch
/// to a copy of the artifact and writes it to disk for external validation.
#[derive(Debug, Clone)]
pub struct ValidatePatch {
    /// Correlation ID for this validation request
    pub correlation_id: String,
    /// Region being patched
    pub region_id: RegionId,
    /// Proposed new content for the region
    pub new_content: String,
    /// Current tick number (for file naming)
    pub tick: usize,
}

/// Response with path to written artifact for validation.
///
/// Sent from Coordinator back to RegionActor with the file path.
#[derive(Debug, Clone)]
pub struct ValidatePatchResponse {
    /// Correlation ID matching the original request
    pub correlation_id: String,
    /// Region being validated
    pub region_id: RegionId,
    /// Path to the written artifact file
    pub artifact_path: std::path::PathBuf,
    /// Path to the original artifact (for comparison)
    pub original_path: std::path::PathBuf,
}

/// Register region actors with the coordinator.
///
/// Sent after spawning all RegionActors during kernel initialization.
#[derive(Debug, Clone)]
pub struct RegisterRegionActors {
    /// Map of region IDs to their actor handles
    pub actors: HashMap<RegionId, acton_reactive::prelude::ActorHandle>,
}

/// Save the current artifact state to a file.
///
/// The coordinator collects all region contents and writes them to the specified path.
#[derive(Debug, Clone)]
pub struct SaveArtifact {
    /// Path to write the artifact to
    pub path: std::path::PathBuf,
}

/// Set the output directory for validation artifacts.
///
/// Sent to coordinator to configure where patched artifacts are written for validation.
#[derive(Debug, Clone)]
pub struct SetOutputDir {
    /// Directory to write validation artifacts
    pub path: std::path::PathBuf,
}

