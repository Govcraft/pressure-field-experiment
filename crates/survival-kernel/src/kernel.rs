//! The coordination kernel: tick-based pressure descent with decay.
//!
//! This module provides the async actor-based kernel using acton-reactive.
//!
//! ## Usage
//!
//! ```ignore
//! use survival_kernel::{AsyncKernelBuilder, KernelConfig};
//! use acton_reactive::prelude::*;
//!
//! // Create runtime and spawn actors
//! let mut runtime = ActonApp::launch_async().await;
//!
//! // Build kernel - sensors self-register via broker
//! let kernel = AsyncKernelBuilder::new(config, artifact)
//!     .add_sensor(Box::new(MySensor))
//!     .spawn(&mut runtime)
//!     .await;
//!
//! // Spawn patch actors separately - they self-register via PatchActorReady
//! let llm_actor = LlmActor::new(config);
//! llm_actor.spawn(&mut runtime).await;
//!
//! // Send Tick messages to drive the kernel
//! kernel.send(Tick { now_ms: 0 }).await;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use acton_reactive::prelude::*;

use crate::actors::{KernelCoordinator, RegionActor};
use crate::artifact::Artifact;
use crate::config::KernelConfig;
use crate::messages::RegisterRegionActors;
use crate::pressure::Sensor;
use crate::region::{Patch, RegionId};

/// Result of a single tick.
#[derive(Debug, Default, Clone)]
pub struct TickResult {
    /// Patches that were applied
    pub applied: Vec<Patch>,
    /// Regions that were evaluated but not patched
    pub evaluated: usize,
    /// Regions that were skipped (inhibited)
    pub skipped: usize,
    /// Total pressure across all regions
    pub total_pressure: f64,
    /// Velocity: rate of pressure change (dP/dt)
    pub velocity: f64,
    /// Acceleration: rate of velocity change (d²P/dt²)
    pub acceleration: f64,
}

/// Apply exponential decay with the given half-life.
pub fn half_life_decay(value: &mut f64, dt_ms: u64, half_life_ms: u64) {
    if half_life_ms == 0 {
        return;
    }
    let lambda = std::f64::consts::LN_2 / half_life_ms as f64;
    *value *= (-lambda * dt_ms as f64).exp();
}

/// Builder for creating an async kernel with parallel execution.
///
/// Spawns RegionActors for each region in the artifact, providing:
/// - Natural conflict resolution via mailbox serialization
/// - Post-patch validation to ensure pressure reduction (δ_min > 0)
/// - Local state ownership (stigmergy model from paper)
///
/// Uses the broker pub/sub pattern for actor registration:
/// - Sensors self-register via `SensorReady` broadcast
/// - Patch actors self-register via `PatchActorReady` broadcast
pub struct AsyncKernelBuilder {
    coordinator: KernelCoordinator,
    /// Sensors to spawn (they self-register via SensorReady broadcast)
    sensors: Vec<Arc<dyn Sensor>>,
    /// Validation sensor for RegionActors (first sensor added)
    validation_sensor: Option<Arc<dyn Sensor>>,
}

impl AsyncKernelBuilder {
    /// Create a new async kernel builder.
    pub fn new(config: KernelConfig, artifact: Box<dyn Artifact>) -> Self {
        Self {
            coordinator: KernelCoordinator::new(config, artifact),
            sensors: Vec::new(),
            validation_sensor: None,
        }
    }

    /// Register a sensor for measurement and validation.
    ///
    /// The first sensor added is also used for post-patch validation
    /// in RegionActors to ensure patches reduce pressure.
    ///
    /// Sensors are spawned during `spawn()` and self-register with the
    /// coordinator via `SensorReady` broker broadcast.
    pub fn add_sensor(mut self, sensor: Box<dyn Sensor>) -> Self {
        let sensor_arc: Arc<dyn Sensor> = Arc::from(sensor);
        // First sensor is used for validation
        if self.validation_sensor.is_none() {
            self.validation_sensor = Some(sensor_arc.clone());
        }
        self.sensors.push(sensor_arc);
        self
    }

    /// Spawn the kernel, sensor actors, and region actors.
    ///
    /// Returns the coordinator's actor handle. To run ticks:
    /// 1. Send `Tick { now_ms }` messages to the coordinator
    /// 2. Receive `TickComplete { result }` replies
    ///
    /// Sensors are spawned and self-register via `SensorReady` broker broadcast.
    pub async fn spawn(self, runtime: &mut ActorRuntime) -> ActorHandle {
        use crate::actors::SensorActor;

        // Get artifact info before moving to coordinator
        let region_ids: Vec<RegionId> = self.coordinator.artifact.region_ids();
        let region_views: Vec<_> = region_ids
            .iter()
            .filter_map(|rid| self.coordinator.artifact.read_region(*rid).ok().map(|v| (*rid, v)))
            .collect();
        let pressure_axes = self.coordinator.config.pressure_axes.clone();
        let validation_sensor = self.validation_sensor.clone();

        // Spawn the coordinator first (it subscribes to SensorReady)
        let coordinator_handle = self.coordinator.spawn(runtime).await;

        // Spawn sensor actors - they self-register via SensorReady broadcast
        for sensor in self.sensors {
            let sensor_actor = SensorActor::new(sensor);
            sensor_actor.spawn(runtime).await;
        }

        // Spawn RegionActors if we have a validation sensor
        if let Some(sensor) = validation_sensor {
            let mut region_actors: HashMap<RegionId, ActorHandle> = HashMap::new();

            for (rid, view) in region_views {
                let region_actor = RegionActor::new(
                    rid,
                    view.kind,
                    view.content,
                    view.metadata,
                    coordinator_handle.clone(),
                    sensor.clone(),
                    pressure_axes.clone(),
                );
                let handle = region_actor.spawn(runtime, 0).await;
                region_actors.insert(rid, handle);
            }

            // Register region actors with coordinator
            coordinator_handle
                .send(RegisterRegionActors {
                    actors: region_actors,
                })
                .await;
        }

        coordinator_handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_life_decay() {
        let mut value = 1.0;
        half_life_decay(&mut value, 600_000, 600_000); // one half-life
        assert!((value - 0.5).abs() < 0.01);
    }
}
