//! Survival Kernel: Emergent Coordination through Gradient Fields and Decay
//!
//! This crate implements a decentralized multi-agent coordination algorithm
//! based on local gradient descent over pressure fields with temporal decay.

pub mod artifact;
pub mod config;
pub mod kernel;
pub mod pressure;
pub mod region;

pub use artifact::Artifact;
pub use config::KernelConfig;
pub use kernel::Kernel;
pub use pressure::{Pressure, PressureVector, Signals};
pub use region::{Patch, PatchOp, RegionId, RegionState, RegionView};
