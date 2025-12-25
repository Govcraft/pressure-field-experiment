//! Pressure and signal types: the gradient fields that drive coordination.

use std::collections::HashMap;

use crate::region::{RegionState, RegionView};

/// Signals are measurable features computed from a region.
/// These form the "state surface" that agents observe.
///
/// Examples:
/// - `parse_confidence`: 0.0 to 1.0
/// - `style_distance`: 0.0 to unbounded
/// - `test_failures`: count
/// - `lint_density`: warnings per line
pub type Signals = HashMap<String, f64>;

/// Pressure values per axis: higher means worse (more corrective pressure needed).
pub type PressureVector = HashMap<String, f64>;

/// A sensor measures signals from regions.
///
/// Sensors are the bridge between artifacts and the pressure system.
/// They must be:
/// - Local: only examine the region and its immediate neighborhood
/// - Deterministic: same input produces same output
/// - Fast: called frequently during ticks
pub trait Sensor: Send + Sync {
    /// Unique name for this sensor.
    fn name(&self) -> &str;

    /// Measure signals for a region.
    fn measure(&self, region: &RegionView) -> anyhow::Result<Signals>;
}

/// A pressure function computes "badness" from signals.
///
/// Pressures define the gradient field that agents descend.
/// Higher pressure means the region needs more attention.
pub trait Pressure: Send + Sync {
    /// Unique name for this pressure axis.
    fn name(&self) -> &str;

    /// Compute pressure from signals and prior state.
    fn compute(&self, region: &RegionView, signals: &Signals, prior: Option<&RegionState>) -> f64;
}

/// An actor proposes patches to reduce pressure.
///
/// Actors are the "agents" in this system, but they:
/// - Have no persistent state
/// - Make only local decisions
/// - Compete via patch selection, not coordination
pub trait Actor: Send + Sync {
    /// Unique name for this actor.
    fn name(&self) -> &str;

    /// Propose candidate patches for a high-pressure region.
    fn propose(
        &self,
        region: &RegionView,
        signals: &Signals,
        pressures: &PressureVector,
        state: &RegionState,
    ) -> anyhow::Result<Vec<crate::region::Patch>>;
}
