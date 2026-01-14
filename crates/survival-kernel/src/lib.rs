//! Survival Kernel: Emergent Coordination through Gradient Fields and Decay
//!
//! This crate implements a decentralized multi-agent coordination algorithm
//! based on local gradient descent over pressure fields with temporal decay.
//!
//! ## Quick Start
//!
//! ```ignore
//! use survival_kernel::{AsyncKernelBuilder, KernelConfig, Tick};
//! use acton_reactive::prelude::*;
//!
//! // Create runtime
//! let mut runtime = ActonApp::launch_async().await;
//!
//! // Spawn your patch actor first (it needs coordinator handle later)
//! // ... spawn LlmActor or other patch actors ...
//!
//! // Build kernel with sensors and pre-spawned actor handles
//! let coordinator = AsyncKernelBuilder::new(config, Box::new(artifact))
//!     .add_sensor(Box::new(MySensor))
//!     .add_patch_actor_handle(llm_actor_handle)
//!     .spawn(&mut runtime)
//!     .await;
//!
//! // Drive the kernel by sending Tick messages
//! coordinator.send(Tick { now_ms: 0 }).await;
//! ```

pub mod actors;
pub mod artifact;
pub mod config;
pub mod kernel;
pub mod messages;
pub mod pressure;
pub mod region;

pub use actors::{
    KernelCoordinator, KernelCoordinatorState, RegionActor, RegionActorState, SensorActor,
    SensorActorState,
};
pub use artifact::Artifact;
pub use config::{KernelConfig, PressureAxisConfig};
pub use kernel::{half_life_decay, AsyncKernelBuilder, TickResult};
pub use messages::{
    ApplyDecay, MeasureRegion, MeasurementResult, PatchActorReady, PatchActorsReady, PatchProposal,
    PressureResponse, ProposeForRegion, QueryPressure, RefreshContent, RegionApplyPatch,
    RegionPatchResult, RegisterRegionActors, RegisterTickDriver, SaveArtifact, SensorReady,
    SetOutputDir, Tick, TickComplete, ValidatePatch, ValidatePatchResponse, WaitForPatchActors,
};
pub use pressure::{measure_pressure_inline, Pressure, PressureVector, Sensor, Signals};
pub use region::{Patch, PatchOp, RegionId, RegionState, RegionView};
