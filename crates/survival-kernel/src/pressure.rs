//! Pressure and signal types: the gradient fields that drive coordination.
//!
//! The `Sensor` trait defines the interface for measuring quality signals from regions.
//! Sensors are synchronous local computations.
//!
//! Actors (LLM proposers, etc.) are implemented as native acton-reactive actors that
//! handle `ProposeForRegion` messages directly. There is no `Actor` trait - actors
//! communicate via message passing.

use std::collections::HashMap;

use crate::region::RegionView;

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
///
/// This trait is **synchronous** - sensors should compute locally without I/O.
pub trait Sensor: Send + Sync {
    /// Unique name for this sensor.
    fn name(&self) -> &str;

    /// Measure signals for a region.
    ///
    /// This is synchronous - sensors should compute signals from the region data
    /// without making external calls.
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
    fn compute(
        &self,
        region: &RegionView,
        signals: &Signals,
        prior: Option<&crate::region::RegionState>,
    ) -> f64;
}

/// Measure total weighted pressure for content synchronously.
///
/// Used by RegionActors to validate that patches actually reduce pressure
/// before accepting them (ensures δ_min > 0 per the convergence theorem).
///
/// # Arguments
/// * `content` - The code content to measure
/// * `kind` - The region kind (e.g., "function")
/// * `sensor` - The sensor to measure signals
/// * `pressure_axes` - The pressure axis weights from config
///
/// # Returns
/// Total weighted pressure: Σ w_j * φ_j(σ(content))
pub fn measure_pressure_inline(
    content: &str,
    kind: &str,
    sensor: &dyn Sensor,
    pressure_axes: &[crate::config::PressureAxisConfig],
) -> anyhow::Result<f64> {
    use mti::prelude::*;
    use uuid::Uuid;

    // Create a placeholder MagicTypeId for temporary measurement
    let prefix = TypeIdPrefix::try_from("temp").expect("temp is valid prefix");
    let suffix = TypeIdSuffix::from(Uuid::nil());
    let placeholder_id = MagicTypeId::new(prefix, suffix);

    let view = RegionView {
        id: placeholder_id,
        kind: kind.to_string(),
        content: content.to_string(),
        metadata: HashMap::new(),
    };

    let signals = sensor.measure(&view)?;

    let total: f64 = pressure_axes
        .iter()
        .map(|axis| {
            // Get the signal value for this axis (using expr as signal name)
            let signal_value = signals.get(&axis.expr).copied().unwrap_or(0.0);
            // Apply kind-specific weight if present, otherwise use base weight
            let weight = axis.kind_weights.get(kind).copied().unwrap_or(axis.weight);
            signal_value * weight
        })
        .sum();

    Ok(total)
}
