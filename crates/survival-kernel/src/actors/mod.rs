//! Acton-reactive actors for the pressure-field kernel.
//!
//! Uses message correlation (via mti) to track async responses:
//!
//! ```text
//! Tick → Coordinator
//!   ├─ MeasureRegion (correlation_id) → SensorActors (concurrent)
//!   │   └─ MeasurementResult → RegionActors (each updates own state)
//!   ├─ QueryPressure → RegionActors → PressureResponse → Coordinator
//!   ├─ ProposeForRegion (correlation_id) → PatchActors (concurrent)
//!   │   └─ PatchProposal (correlation_id) → Coordinator
//!   ├─ RegionApplyPatch → RegionActor (validates, applies, responds)
//!   │   └─ RegionPatchResult → Coordinator
//!   └─ TickComplete ← Reply when done
//! ```
//!
//! RegionActors own their state, providing natural conflict resolution via mailbox
//! serialization and ensuring patches actually reduce pressure (δ_min > 0).

mod coordinator;
mod region_actor;
mod sensor_actor;

pub use coordinator::{KernelCoordinator, KernelCoordinatorState};
pub use region_actor::{RegionActor, RegionActorState};
pub use sensor_actor::{SensorActor, SensorActorState};
