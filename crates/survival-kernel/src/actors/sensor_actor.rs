//! SensorActor: wraps a Sensor for concurrent measurement via acton-reactive.
//!
//! Uses the broker pub/sub pattern:
//! - Subscribes to `MeasureRegion` broadcasts
//! - Broadcasts `SensorReady` on start for coordinator tracking
//! - Broadcasts `MeasurementResult` responses

use std::sync::Arc;

use acton_reactive::prelude::*;

use crate::messages::{MeasureRegion, MeasurementResult, SensorReady};
use crate::pressure::Sensor;

/// Actor state for SensorActor.
#[derive(Default, Clone)]
pub struct SensorActorState {
    /// The wrapped sensor implementation
    sensor: Option<Arc<dyn Sensor>>,
}

impl std::fmt::Debug for SensorActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SensorActorState")
            .field("sensor", &self.sensor.as_ref().map(|s| s.name()))
            .finish()
    }
}

/// Actor wrapper for a [`Sensor`] implementation.
///
/// Each SensorActor subscribes to `MeasureRegion` broadcasts and handles them
/// concurrently (via `act_on`), broadcasting `MeasurementResult` responses.
///
/// On start, broadcasts `SensorReady` so the coordinator can track sensor count
/// without needing direct handles. The coordinator identifies the sensor via
/// the message envelope's sender ERN.
pub struct SensorActor {
    /// The wrapped sensor implementation
    pub sensor: Arc<dyn Sensor>,
}

impl SensorActor {
    /// Create a new SensorActor.
    pub fn new(sensor: Arc<dyn Sensor>) -> Self {
        Self { sensor }
    }

    /// Spawn this sensor actor in the given runtime.
    ///
    /// The actor will:
    /// 1. Subscribe to `MeasureRegion` broadcasts
    /// 2. Broadcast `SensorReady` on start
    /// 3. Handle measurements and broadcast results
    pub async fn spawn(self, runtime: &mut ActorRuntime) -> ActorHandle {
        let sensor_name = self.sensor.name().to_string();

        let mut actor =
            runtime.new_actor_with_name::<SensorActorState>(format!("Sensor:{}", sensor_name));

        // Set initial state
        actor.model.sensor = Some(self.sensor);

        // Subscribe to MeasureRegion broadcasts BEFORE starting
        actor.handle().subscribe::<MeasureRegion>().await;

        // Broadcast SensorReady on start so coordinator knows about us
        // Include our ERN since broker broadcasts don't preserve sender identity
        actor.after_start(|actor| {
            let broker = actor.broker().clone();
            let sensor_ern = actor.handle().name().to_string();

            Reply::pending(async move {
                broker.broadcast(SensorReady { sensor_ern }).await;
            })
        });

        // act_on = concurrent (multiple measurements in parallel)
        actor.act_on::<MeasureRegion>(|actor, context| {
            let msg = context.message().clone();
            let sensor = actor.model.sensor.clone();
            let broker = actor.broker().clone();

            let Some(sensor) = sensor else {
                tracing::error!("SensorActor: sensor not initialized");
                return Reply::ready();
            };

            let sensor_name = sensor.name().to_string();

            // Measure synchronously
            let result = sensor.measure(&msg.region_view);

            match result {
                Ok(signals) => {
                    let measurement = MeasurementResult {
                        correlation_id: msg.correlation_id,
                        region_id: msg.region_id,
                        sensor_name,
                        signals,
                    };

                    // Broadcast result (coordinator subscribes to MeasurementResult)
                    Reply::pending(async move {
                        broker.broadcast(measurement).await;
                    })
                }
                Err(e) => {
                    tracing::warn!(
                        sensor = sensor_name,
                        region = %msg.region_id,
                        error = %e,
                        "Sensor measurement failed"
                    );
                    Reply::ready()
                }
            }
        });

        actor.start().await
    }
}
