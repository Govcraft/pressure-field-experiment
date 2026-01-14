//! KernelCoordinator: orchestrates the tick loop with RegionActors.
//!
//! Each region owns its state via a RegionActor, providing natural conflict
//! resolution through mailbox serialization. The coordinator orchestrates:
//! 1. Decay broadcast → RegionActors
//! 2. Measurement → SensorActors → RegionActors
//! 3. Pressure query → RegionActors → Coordinator (finds high-pressure)
//! 4. Proposals → PatchActors → Coordinator
//! 5. Patch application → RegionActors (with validation)
//! 6. TickComplete

use std::collections::HashMap;

use acton_reactive::prelude::*;
use dashmap::DashMap;
use mti::prelude::*;
use tracing::{debug, info, trace, warn};

// RegionActor import used by AsyncKernelBuilder (spawns RegionActors externally)
use std::collections::HashSet;

use crate::artifact::Artifact;
use crate::config::KernelConfig;
use crate::kernel::TickResult;
use crate::messages::{
    ApplyDecay, MeasureRegion, MeasurementResult, PatchActorReady, PatchProposal, PressureResponse,
    ProposeForRegion, QueryPressure, RegionApplyPatch, RegionPatchResult, RegisterRegionActors,
    RegisterTickDriver, SaveArtifact, SensorReady, SetOutputDir, Tick, TickComplete, ValidatePatch,
    ValidatePatchResponse,
};
use crate::pressure::PressureVector;
use crate::region::{Patch, RegionId, RegionView};

/// Compute velocity (dP/dt) from pressure history.
fn compute_velocity(current_pressure: f64, pressure_history: &[f64]) -> f64 {
    pressure_history
        .last()
        .map(|prev| current_pressure - prev)
        .unwrap_or(0.0)
}

/// Compute acceleration (d²P/dt²) from velocity history.
fn compute_acceleration(current_velocity: f64, velocity_history: &[f64]) -> f64 {
    velocity_history
        .last()
        .map(|prev| current_velocity - prev)
        .unwrap_or(0.0)
}

/// Tracks pending measurements for a tick.
#[derive(Debug, Clone)]
struct PendingMeasurements {
    /// Expected number of responses (sensors × regions)
    expected_count: usize,
    /// Received measurement results
    results: Vec<MeasurementResult>,
    /// Timestamp for this tick
    now_ms: u64,
}

impl PendingMeasurements {
    fn new(expected_count: usize, now_ms: u64) -> Self {
        Self {
            expected_count,
            results: Vec::new(),
            now_ms,
        }
    }

    fn is_complete(&self) -> bool {
        self.results.len() >= self.expected_count
    }
}

/// Tracks pending pressure queries.
#[derive(Debug, Clone)]
struct PendingPressureQueries {
    /// Expected number of responses (one per region)
    expected_count: usize,
    /// Received responses
    responses: Vec<PressureResponse>,
    /// Timestamp for this tick
    now_ms: u64,
}

impl PendingPressureQueries {
    fn new(expected_count: usize, now_ms: u64) -> Self {
        Self {
            expected_count,
            responses: Vec::new(),
            now_ms,
        }
    }

    fn is_complete(&self) -> bool {
        self.responses.len() >= self.expected_count
    }
}

/// Tracks pending proposals for a tick.
#[derive(Debug, Clone)]
struct PendingProposals {
    /// Expected number of responses (actors × high-pressure regions)
    expected_count: usize,
    /// Received proposals
    proposals: Vec<PatchProposal>,
    /// Regions that were proposed for
    high_pressure_regions: Vec<(RegionId, RegionView, PressureVector)>,
    /// Timestamp for this tick
    now_ms: u64,
    /// Total pressure at time of proposal (for calculating final pressure after patches)
    total_pressure: f64,
}

impl PendingProposals {
    fn new(
        expected_count: usize,
        high_pressure_regions: Vec<(RegionId, RegionView, PressureVector)>,
        now_ms: u64,
        total_pressure: f64,
    ) -> Self {
        Self {
            expected_count,
            proposals: Vec::new(),
            high_pressure_regions,
            now_ms,
            total_pressure,
        }
    }

    fn is_complete(&self) -> bool {
        self.proposals.len() >= self.expected_count
    }
}

/// Tracks pending patch applications.
#[derive(Debug, Clone)]
struct PendingPatches {
    /// Expected number of responses
    expected_count: usize,
    /// Received results
    results: Vec<RegionPatchResult>,
    /// Last known total pressure (from query phase)
    last_total_pressure: f64,
    /// Count of evaluated regions
    evaluated_count: usize,
    /// Count of skipped regions
    skipped_count: usize,
}

impl PendingPatches {
    fn is_complete(&self) -> bool {
        self.results.len() >= self.expected_count
    }
}

/// Actor state for KernelCoordinator.
pub struct KernelCoordinatorState {
    /// Kernel configuration
    config: Option<KernelConfig>,
    /// The artifact being coordinated
    artifact: Option<Box<dyn Artifact>>,
    /// Handles to RegionActors (one per region)
    region_actors: DashMap<RegionId, ActorHandle>,
    /// Registered sensor IDs (sensors self-register via SensorReady broadcast)
    registered_sensors: HashSet<String>,
    /// Registered patch actor IDs (patch actors self-register via PatchActorReady broadcast)
    registered_patch_actors: HashSet<String>,
    /// Pending measurement requests by correlation ID
    pending_measurements: DashMap<String, PendingMeasurements>,
    /// Pending pressure queries by correlation ID
    pending_pressure_queries: DashMap<String, PendingPressureQueries>,
    /// Pending proposal requests by correlation ID
    pending_proposals: DashMap<String, PendingProposals>,
    /// Pending patch applications by correlation ID
    pending_patches: DashMap<String, PendingPatches>,
    /// Handle to tick driver for sending TickComplete
    tick_driver: Option<ActorHandle>,
    /// Consecutive ticks with no patches (for stability)
    stable_ticks: usize,
    /// Current tick number
    current_tick: usize,
    /// Output directory for validation artifacts (set via SetOutputDir message)
    output_dir: Option<std::path::PathBuf>,
    /// History of pressure values for derivative calculation
    pressure_history: Vec<f64>,
    /// History of velocity values for acceleration calculation
    velocity_history: Vec<f64>,
}

impl Default for KernelCoordinatorState {
    fn default() -> Self {
        Self {
            config: None,
            artifact: None,
            region_actors: DashMap::new(),
            registered_sensors: HashSet::new(),
            registered_patch_actors: HashSet::new(),
            pending_measurements: DashMap::new(),
            pending_pressure_queries: DashMap::new(),
            pending_proposals: DashMap::new(),
            pending_patches: DashMap::new(),
            tick_driver: None,
            stable_ticks: 0,
            current_tick: 0,
            output_dir: None,
            pressure_history: Vec::new(),
            velocity_history: Vec::new(),
        }
    }
}

impl Clone for KernelCoordinatorState {
    fn clone(&self) -> Self {
        // Clone DashMaps by iterating and collecting
        let region_actors = DashMap::new();
        for entry in self.region_actors.iter() {
            region_actors.insert(*entry.key(), entry.value().clone());
        }

        let pending_measurements = DashMap::new();
        for entry in self.pending_measurements.iter() {
            pending_measurements.insert(entry.key().clone(), entry.value().clone());
        }

        let pending_pressure_queries = DashMap::new();
        for entry in self.pending_pressure_queries.iter() {
            pending_pressure_queries.insert(entry.key().clone(), entry.value().clone());
        }

        let pending_proposals = DashMap::new();
        for entry in self.pending_proposals.iter() {
            pending_proposals.insert(entry.key().clone(), entry.value().clone());
        }

        let pending_patches = DashMap::new();
        for entry in self.pending_patches.iter() {
            pending_patches.insert(entry.key().clone(), entry.value().clone());
        }

        Self {
            config: self.config.clone(),
            artifact: None, // Can't clone trait object
            region_actors,
            tick_driver: self.tick_driver.clone(),
            registered_sensors: self.registered_sensors.clone(),
            registered_patch_actors: self.registered_patch_actors.clone(),
            pending_measurements,
            pending_pressure_queries,
            pending_proposals,
            pending_patches,
            stable_ticks: self.stable_ticks,
            current_tick: self.current_tick,
            output_dir: self.output_dir.clone(),
            pressure_history: self.pressure_history.clone(),
            velocity_history: self.velocity_history.clone(),
        }
    }
}

impl std::fmt::Debug for KernelCoordinatorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelCoordinatorState")
            .field("config", &self.config.is_some())
            .field("region_actors", &self.region_actors.len())
            .field("artifact", &self.artifact.is_some())
            .field("registered_sensors", &self.registered_sensors.len())
            .field("registered_patch_actors", &self.registered_patch_actors.len())
            .field("pending_measurements", &self.pending_measurements.len())
            .field(
                "pending_pressure_queries",
                &self.pending_pressure_queries.len(),
            )
            .field("pending_proposals", &self.pending_proposals.len())
            .field("pending_patches", &self.pending_patches.len())
            .field("stable_ticks", &self.stable_ticks)
            .finish()
    }
}

/// Central coordinator actor for the pressure-field kernel.
///
/// Orchestrates the tick loop with RegionActors:
/// 1. On Tick: broadcast ApplyDecay to RegionActors
/// 2. Broadcast MeasureRegion (sensors subscribe via broker)
/// 3. Query QueryPressure from all RegionActors
/// 4. On PressureResponse: find high-pressure regions, broadcast ProposeForRegion
/// 5. On PatchProposal: send RegionApplyPatch to target RegionActors
/// 6. On RegionPatchResult: update artifact, send TickComplete
///
/// Sensors and patch actors register themselves by broadcasting `SensorReady`
/// and `PatchActorReady` on start. The coordinator subscribes to these messages.
pub struct KernelCoordinator {
    /// Kernel configuration
    pub config: KernelConfig,
    /// The artifact being coordinated
    pub artifact: Box<dyn Artifact>,
}

impl KernelCoordinator {
    /// Create a new KernelCoordinator.
    pub fn new(config: KernelConfig, artifact: Box<dyn Artifact>) -> Self {
        Self { config, artifact }
    }

    /// Spawn this coordinator.
    ///
    /// Sensors and patch actors should be spawned separately.
    /// They will self-register via broker broadcasts.
    pub async fn spawn(self, runtime: &mut ActorRuntime) -> ActorHandle {
        let mut actor =
            runtime.new_actor_with_name::<KernelCoordinatorState>("KernelCoordinator".to_string());

        // Set initial state
        actor.model.config = Some(self.config.clone());
        actor.model.artifact = Some(self.artifact);

        // Subscribe to actor registration and response broadcasts BEFORE starting
        actor.handle().subscribe::<SensorReady>().await;
        actor.handle().subscribe::<MeasurementResult>().await;
        actor.handle().subscribe::<PatchActorReady>().await;
        actor.handle().subscribe::<PatchProposal>().await;
        actor.handle().subscribe::<PressureResponse>().await;

        // Configure handlers before starting
        configure_handlers(&mut actor);

        actor.start().await
    }
}

/// Configure all message handlers for the coordinator.
fn configure_handlers(actor: &mut ManagedActor<Idle, KernelCoordinatorState>) {
    // Handle sensor self-registration via broker
    actor.mutate_on::<SensorReady>(|actor, context| {
        let reply_address = context.origin_envelope().reply_to();
        let sender_ern = reply_address.name();

        actor
            .model
            .registered_sensors
            .insert(sender_ern.to_string());
        debug!(
            sensor_ern = %sender_ern,
            total_sensors = actor.model.registered_sensors.len(),
            "Sensor registered"
        );
        Reply::ready()
    });

    // Handle patch actor self-registration via broker
    actor.mutate_on::<PatchActorReady>(|actor, context| {
        let reply_address = context.origin_envelope().reply_to();
        let sender_ern = reply_address.name();

        actor
            .model
            .registered_patch_actors
            .insert(sender_ern.to_string());
        debug!(
            patch_actor_ern = %sender_ern,
            total_patch_actors = actor.model.registered_patch_actors.len(),
            "Patch actor registered"
        );
        Reply::ready()
    });

    // Handle RegionActor registration
    actor.mutate_on::<RegisterRegionActors>(|actor, context| {
        let msg = context.message();
        actor.model.region_actors.clear();
        for (region_id, handle) in &msg.actors {
            actor.model.region_actors.insert(*region_id, handle.clone());
        }
        trace!(
            regions = actor.model.region_actors.len(),
            "Registered region actors"
        );
        Reply::ready()
    });

    // Handle tick driver registration
    actor.mutate_on::<RegisterTickDriver>(|actor, context| {
        let handle = context.message().handle.clone();
        actor.model.tick_driver = Some(handle);
        debug!("Registered tick driver");
        Reply::ready()
    });

    // Handle Tick - start decay and measurement phase
    actor.mutate_on::<Tick>(|actor, context| {
        let now_ms = context.message().now_ms;

        // Increment tick counter
        actor.model.current_tick += 1;
        let tick_num = actor.model.current_tick;

        let Some(config) = actor.model.config.as_ref() else {
            warn!("KernelCoordinator: config not initialized");
            return Reply::ready();
        };

        let Some(artifact) = actor.model.artifact.as_ref() else {
            warn!("KernelCoordinator: artifact not initialized");
            return Reply::ready();
        };

        info!(
            tick = tick_num,
            regions = actor.model.region_actors.len(),
            "Tick started"
        );

        // Phase 1: Broadcast ApplyDecay to all RegionActors
        let decay_msg = ApplyDecay {
            now_ms,
            fitness_half_life_ms: config.decay.fitness_half_life_ms,
            confidence_half_life_ms: config.decay.confidence_half_life_ms,
        };

        // Generate correlation ID for measurement phase
        let correlation_id = "tick".create_type_id::<V7>().to_string();

        // Collect regions for broadcast
        let region_data: Vec<_> = artifact
            .region_ids()
            .iter()
            .filter_map(|rid| artifact.read_region(*rid).ok().map(|view| (*rid, view)))
            .collect();

        let sensor_count = actor.model.registered_sensors.len();
        let expected_count = sensor_count * region_data.len();

        // Track pending measurements
        actor.model.pending_measurements.insert(
            correlation_id.clone(),
            PendingMeasurements::new(expected_count, now_ms),
        );

        trace!(
            correlation_id = %correlation_id,
            regions = region_data.len(),
            sensors = sensor_count,
            expected = expected_count,
            "Starting tick: decay + measurement phase"
        );

        // Get broker for broadcasting
        let broker = actor.broker().clone();

        // Broadcast decay and measurements via broker
        Reply::pending(async move {
            // Broadcast decay to all region actors via broker
            // RegionActors subscribe to ApplyDecay
            broker.broadcast(decay_msg).await;

            // Broadcast MeasureRegion to all sensors via broker
            // Sensors subscribe to MeasureRegion and respond with MeasurementResult
            for (rid, region_view) in region_data {
                let msg = MeasureRegion {
                    correlation_id: correlation_id.clone(),
                    region_id: rid,
                    region_view,
                    now_ms,
                };

                broker.broadcast(msg).await;
            }
        })
    });

    // Handle MeasurementResult - route to RegionActor and check completion
    actor.mutate_on::<MeasurementResult>(|actor, context| {
        let result = context.message().clone();
        let correlation_id = result.correlation_id.clone();
        let region_id = result.region_id;

        let Some(mut pending) = actor.model.pending_measurements.get_mut(&correlation_id) else {
            warn!(
                correlation_id = %correlation_id,
                "Received measurement for unknown correlation ID"
            );
            return Reply::ready();
        };

        // Store result
        pending.results.push(result.clone());

        // Route measurement to the target RegionActor
        if let Some(region_handle) = actor.model.region_actors.get(&region_id) {
            let handle = region_handle.clone();
            let result_clone = result;
            tokio::spawn(async move {
                handle.send(result_clone).await;
            });
        }

        // Check if all measurements received
        if !pending.is_complete() {
            return Reply::ready();
        }
        drop(pending); // Release the lock before removing

        let (_, pending) = actor
            .model
            .pending_measurements
            .remove(&correlation_id)
            .unwrap();
        let now_ms = pending.now_ms;

        trace!(
            correlation_id = %correlation_id,
            results = pending.results.len(),
            "Measurement phase complete, starting pressure query"
        );

        // Start pressure query phase
        let query_correlation_id = "query".create_type_id::<V7>().to_string();
        let expected_count = actor.model.region_actors.len();

        actor.model.pending_pressure_queries.insert(
            query_correlation_id.clone(),
            PendingPressureQueries::new(expected_count, now_ms),
        );

        // Broadcast QueryPressure to all region actors via broker
        let broker = actor.broker().clone();

        Reply::pending(async move {
            let msg = QueryPressure {
                correlation_id: query_correlation_id,
                now_ms,
            };
            broker.broadcast(msg).await;
        })
    });

    // Handle PressureResponse - find high-pressure regions and start proposals
    actor.mutate_on::<PressureResponse>(|actor, context| {
        let response = context.message().clone();
        let correlation_id = response.correlation_id.clone();

        let Some(mut pending) = actor
            .model
            .pending_pressure_queries
            .get_mut(&correlation_id)
        else {
            warn!(
                correlation_id = %correlation_id,
                "Received pressure response for unknown correlation ID"
            );
            return Reply::ready();
        };

        // Store response
        pending.responses.push(response);

        // Check if all queries complete
        if !pending.is_complete() {
            return Reply::ready();
        }
        drop(pending); // Release the lock before removing

        let (_, pending) = actor
            .model
            .pending_pressure_queries
            .remove(&correlation_id)
            .unwrap();
        let now_ms = pending.now_ms;

        let Some(config) = actor.model.config.as_ref() else {
            return Reply::ready();
        };

        // Find high-pressure, non-inhibited regions
        let threshold = config.activation.min_total_pressure;

        let high_pressure_regions: Vec<_> = pending
            .responses
            .iter()
            .filter(|r| !r.is_inhibited && r.total_pressure >= threshold)
            .map(|r| {
                let pressures: PressureVector = r.state.pressure_ema.clone();
                (r.region_id, r.view.clone(), pressures)
            })
            .collect();

        let total_pressure: f64 = pending.responses.iter().map(|r| r.total_pressure).sum();
        let evaluated = pending.responses.len();
        let skipped = pending.responses.iter().filter(|r| r.is_inhibited).count();

        trace!(
            high_pressure = high_pressure_regions.len(),
            total_pressure = %total_pressure,
            "Pressure query complete"
        );

        if high_pressure_regions.is_empty() {
            // No regions to propose for - complete tick immediately

            // Compute derivatives
            let velocity = compute_velocity(total_pressure, &actor.model.pressure_history);
            let acceleration = compute_acceleration(velocity, &actor.model.velocity_history);

            // Update history
            actor.model.pressure_history.push(total_pressure);
            actor.model.velocity_history.push(velocity);

            let result = TickResult {
                applied: Vec::new(),
                evaluated,
                skipped,
                total_pressure,
                velocity,
                acceleration,
            };

            actor.model.stable_ticks += 1;

            info!(
                tick = actor.model.current_tick,
                pressure = format!("{:.2}", total_pressure),
                velocity = format!("{:.3}", velocity),
                acceleration = format!("{:.3}", acceleration),
                applied = 0,
                "Tick complete - stable (no high-pressure regions)"
            );

            if let Some(tick_driver) = actor.model.tick_driver.clone() {
                return Reply::pending(async move {
                    tick_driver.send(TickComplete { result }).await;
                });
            }
            return Reply::ready();
        }

        // Generate correlation ID for proposal phase
        let proposal_correlation_id = "propose".create_type_id::<V7>().to_string();

        let actor_count = actor.model.registered_patch_actors.len();
        let expected_count = actor_count * high_pressure_regions.len();

        // Track pending proposals
        actor.model.pending_proposals.insert(
            proposal_correlation_id.clone(),
            PendingProposals::new(
                expected_count,
                high_pressure_regions.clone(),
                now_ms,
                total_pressure,
            ),
        );

        trace!(
            correlation_id = %proposal_correlation_id,
            regions = high_pressure_regions.len(),
            actors = actor_count,
            expected = expected_count,
            "Starting proposal phase"
        );

        // Collect proposal data including signals from pressure response
        let proposal_data: Vec<_> = pending
            .responses
            .into_iter()
            .filter(|r| !r.is_inhibited && r.total_pressure >= threshold)
            .map(|r| {
                let pressures: PressureVector = r.state.pressure_ema.clone();
                (r.region_id, r.view, r.signals, pressures, r.state)
            })
            .collect();

        // Get broker for broadcasting
        let broker = actor.broker().clone();

        // Broadcast ProposeForRegion to all patch actors via broker
        Reply::pending(async move {
            for (rid, view, signals, pressures, state) in proposal_data {
                let msg = ProposeForRegion {
                    correlation_id: proposal_correlation_id.clone(),
                    region_id: rid,
                    region_view: view,
                    signals,
                    pressures,
                    state,
                };

                broker.broadcast(msg).await;
            }
        })
    });

    // Handle PatchProposal - collect and start patch application
    actor.mutate_on::<PatchProposal>(|actor, context| {
        let proposal = context.message().clone();
        let correlation_id = proposal.correlation_id.clone();

        let Some(mut pending) = actor.model.pending_proposals.get_mut(&correlation_id) else {
            warn!(
                correlation_id = %correlation_id,
                "Received proposal for unknown correlation ID"
            );
            return Reply::ready();
        };

        // Store proposal
        pending.proposals.push(proposal);

        // Check if all proposals received
        if !pending.is_complete() {
            return Reply::ready();
        }
        drop(pending); // Release the lock before removing

        let (_, pending) = actor
            .model
            .pending_proposals
            .remove(&correlation_id)
            .unwrap();
        let now_ms = pending.now_ms;

        trace!(
            correlation_id = %correlation_id,
            proposals = pending.proposals.len(),
            "Proposal phase complete"
        );

        let Some(config) = actor.model.config.as_ref() else {
            return Reply::ready();
        };

        // Collect and sort all patches by score
        let mut all_patches: Vec<(f64, Patch)> = pending
            .proposals
            .into_iter()
            .flat_map(|p| p.patches)
            .collect();

        all_patches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top patches
        let max_patches = config.selection.max_patches_per_tick;
        let top_patches: Vec<_> = all_patches.into_iter().take(max_patches).collect();

        if top_patches.is_empty() {
            // No patches to apply - calculate total pressure from high_pressure_regions
            let total_pressure: f64 = pending
                .high_pressure_regions
                .iter()
                .map(|(_rid, _, pressures)| pressures.values().sum::<f64>())
                .sum();

            // Compute derivatives
            let velocity = compute_velocity(total_pressure, &actor.model.pressure_history);
            let acceleration = compute_acceleration(velocity, &actor.model.velocity_history);

            // Update history
            actor.model.pressure_history.push(total_pressure);
            actor.model.velocity_history.push(velocity);

            let result = TickResult {
                applied: Vec::new(),
                evaluated: actor.model.region_actors.len(),
                skipped: 0,
                total_pressure,
                velocity,
                acceleration,
            };

            actor.model.stable_ticks += 1;

            info!(
                tick = actor.model.current_tick,
                pressure = format!("{:.2}", total_pressure),
                velocity = format!("{:.3}", velocity),
                acceleration = format!("{:.3}", acceleration),
                applied = 0,
                "Tick complete - stable (no patches proposed)"
            );

            if let Some(tick_driver) = actor.model.tick_driver.clone() {
                return Reply::pending(async move {
                    tick_driver.send(TickComplete { result }).await;
                });
            }
            return Reply::ready();
        }

        // Generate correlation ID for patch application
        let patch_correlation_id = "patch".create_type_id::<V7>().to_string();
        let expected_count = top_patches.len();

        // Track pending patch results
        actor.model.pending_patches.insert(
            patch_correlation_id.clone(),
            PendingPatches {
                expected_count,
                results: Vec::new(),
                last_total_pressure: pending.total_pressure,
                evaluated_count: actor.model.region_actors.len(),
                skipped_count: 0,
            },
        );

        let inhibit_ms = config.activation.inhibit_ms;
        let min_improvement = config.selection.min_expected_improvement;
        let region_actors: HashMap<RegionId, ActorHandle> = actor
            .model
            .region_actors
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        let current_tick = actor.model.current_tick;

        trace!(
            correlation_id = %patch_correlation_id,
            patches = top_patches.len(),
            "Sending patches to RegionActors for validation"
        );

        // Send patches to RegionActors
        Reply::pending(async move {
            for (_, patch) in top_patches {
                let rid = patch.region;
                if let Some(region_handle) = region_actors.get(&rid) {
                    let msg = RegionApplyPatch {
                        correlation_id: patch_correlation_id.clone(),
                        patch,
                        now_ms,
                        tick: current_tick,
                        inhibit_ms,
                        min_expected_improvement: min_improvement,
                    };
                    region_handle.send(msg).await;
                }
            }
        })
    });

    // Handle RegionPatchResult - collect results and complete tick
    actor.mutate_on::<RegionPatchResult>(|actor, context| {
        let result = context.message().clone();
        let correlation_id = result.correlation_id.clone();

        let Some(mut pending) = actor.model.pending_patches.get_mut(&correlation_id) else {
            warn!(
                correlation_id = %correlation_id,
                "Received patch result for unknown correlation ID"
            );
            return Reply::ready();
        };

        // Store result
        pending.results.push(result.clone());

        // Check if all results received (check before dropping lock)
        let is_complete = pending.is_complete();
        drop(pending); // Release the lock before potentially modifying artifact

        // If patch was successful, update the artifact
        if result.success
            && let (Some(artifact), Some(new_content)) =
                (actor.model.artifact.as_mut(), &result.new_content)
        {
            // Create a patch to update the artifact
            let patch = Patch {
                region: result.region_id,
                op: crate::region::PatchOp::Replace(new_content.clone()),
                rationale: format!("Validated patch (δ={:.3})", result.pressure_delta),
                expected_delta: HashMap::new(),
            };

            if let Err(e) = artifact.apply_patch(patch) {
                warn!(
                    region = %result.region_id,
                    error = %e,
                    "Failed to apply validated patch to artifact"
                );
            }
        }

        // Check if all results received
        if !is_complete {
            return Reply::ready();
        }

        let (_, pending) = actor.model.pending_patches.remove(&correlation_id).unwrap();

        // Compile tick result
        let applied: Vec<_> = pending
            .results
            .iter()
            .filter(|r| r.success)
            .map(|r| Patch {
                region: r.region_id,
                op: crate::region::PatchOp::Replace(r.new_content.clone().unwrap_or_default()),
                rationale: format!("δ={:.3}", r.pressure_delta),
                expected_delta: HashMap::new(),
            })
            .collect();

        let rejected_count = pending.results.iter().filter(|r| !r.success).count();

        // Track stability
        if applied.is_empty() {
            actor.model.stable_ticks += 1;
        } else {
            actor.model.stable_ticks = 0;
        }

        // Calculate total pressure delta
        let total_delta: f64 = pending
            .results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.pressure_delta)
            .sum();

        let new_pressure = pending.last_total_pressure - total_delta;

        // Get previous tick's final pressure for display (before updating history)
        // This matches the comparison used for velocity calculation
        let previous_tick_pressure = actor.model.pressure_history.last().copied();

        // Compute derivatives
        let velocity = compute_velocity(new_pressure, &actor.model.pressure_history);
        let acceleration = compute_acceleration(velocity, &actor.model.velocity_history);

        // Update history
        actor.model.pressure_history.push(new_pressure);
        actor.model.velocity_history.push(velocity);

        let tick_result = TickResult {
            applied: applied.clone(),
            evaluated: pending.evaluated_count,
            skipped: pending.skipped_count,
            total_pressure: new_pressure,
            velocity,
            acceleration,
        };

        info!(
            tick = actor.model.current_tick,
            pressure = format!(
                "{:.2} -> {:.2}",
                previous_tick_pressure.unwrap_or(new_pressure),
                new_pressure
            ),
            velocity = format!("{:.3}", velocity),
            acceleration = format!("{:.3}", acceleration),
            applied = applied.len(),
            rejected = rejected_count,
            delta = format!("-{:.2}", total_delta),
            "Tick complete"
        );

        if let Some(tick_driver) = actor.model.tick_driver.clone() {
            return Reply::pending(async move {
                tick_driver
                    .send(TickComplete {
                        result: tick_result,
                    })
                    .await;
            });
        }

        Reply::ready()
    });

    // Handle SaveArtifact - write the current artifact state to a file
    actor.act_on::<SaveArtifact>(|actor, context| {
        let msg = context.message();

        if let Some(artifact) = actor.model.artifact.as_ref() {
            if let Some(source) = artifact.source() {
                match std::fs::write(&msg.path, &source) {
                    Ok(_) => {
                        info!(path = %msg.path.display(), bytes = source.len(), "Artifact saved");
                    }
                    Err(e) => {
                        warn!(path = %msg.path.display(), error = %e, "Failed to save artifact");
                    }
                }
            } else {
                warn!("Artifact does not support source() method");
            }
        } else {
            warn!("No artifact to save");
        }

        Reply::ready()
    });

    // Handle SetOutputDir - configure validation artifact output directory
    actor.mutate_on::<SetOutputDir>(|actor, context| {
        let path = context.message().path.clone();
        // Create the directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&path) {
            warn!(path = %path.display(), error = %e, "Failed to create output directory");
        } else {
            debug!(path = %path.display(), "Output directory set");
        }
        actor.model.output_dir = Some(path);
        Reply::ready()
    });

    // Handle ValidatePatch - write artifact with proposed patch for validation
    actor.act_on::<ValidatePatch>(|actor, context| {
        let msg = context.message().clone();
        let region_actors: HashMap<RegionId, ActorHandle> = actor
            .model
            .region_actors
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();

        let Some(output_dir) = actor.model.output_dir.clone() else {
            warn!("ValidatePatch: output_dir not set");
            // Send error response
            if let Some(handle) = region_actors.get(&msg.region_id).cloned() {
                return Reply::pending(async move {
                    handle
                        .send(ValidatePatchResponse {
                            correlation_id: msg.correlation_id,
                            region_id: msg.region_id,
                            artifact_path: std::path::PathBuf::new(),
                            original_path: std::path::PathBuf::new(),
                        })
                        .await;
                });
            }
            return Reply::ready();
        };

        let Some(artifact) = actor.model.artifact.as_ref() else {
            warn!("ValidatePatch: artifact not initialized");
            return Reply::ready();
        };

        // Get current artifact source
        let Some(original_source) = artifact.source() else {
            warn!("ValidatePatch: artifact does not support source()");
            return Reply::ready();
        };

        // Get the current content for the region being patched
        let current_content = match artifact.read_region(msg.region_id) {
            Ok(view) => view.content,
            Err(e) => {
                warn!(error = %e, "Failed to read region for validation");
                return Reply::ready();
            }
        };

        // Replace the old content with new content in the source
        let patched_source = original_source.replace(&current_content, &msg.new_content);

        // Write original artifact (once per tick)
        let original_path = output_dir.join(format!("tick_{}_original.rs", msg.tick));
        if !original_path.exists()
            && let Err(e) = std::fs::write(&original_path, &original_source)
        {
            warn!(path = %original_path.display(), error = %e, "Failed to write original artifact");
        }

        // Write patched artifact
        let region_short = format!("{:.8}", msg.region_id);
        let artifact_path =
            output_dir.join(format!("tick_{}_region_{}.rs", msg.tick, region_short));

        if let Err(e) = std::fs::write(&artifact_path, &patched_source) {
            warn!(path = %artifact_path.display(), error = %e, "Failed to write patched artifact");
        } else {
            trace!(path = %artifact_path.display(), "Wrote patched artifact for validation");
        }

        // Send response to RegionActor
        let response = ValidatePatchResponse {
            correlation_id: msg.correlation_id,
            region_id: msg.region_id,
            artifact_path,
            original_path,
        };

        if let Some(handle) = region_actors.get(&msg.region_id).cloned() {
            Reply::pending(async move {
                handle.send(response).await;
            })
        } else {
            warn!(region = %msg.region_id, "No actor handle for region");
            Reply::ready()
        }
    });
}
