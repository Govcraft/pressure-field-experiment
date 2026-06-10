//! Configuration types for the kernel.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level kernel configuration.
///
/// This defines the pressure landscape, decay dynamics, and selection criteria.
/// Loaded from TOML/JSON at runtime.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelConfig {
    /// Tick interval in milliseconds (for decay calculations and internal tick rate)
    pub tick_interval_ms: u64,

    /// Maximum ticks before stopping (0 = unlimited)
    #[serde(default)]
    pub max_ticks: usize,

    /// Consecutive stable ticks (no patches) required for convergence (0 = disable)
    #[serde(default)]
    pub stable_threshold: usize,

    /// Pressure axis definitions
    pub pressure_axes: Vec<PressureAxisConfig>,

    /// Smoothing factor for the per-region pressure EMA (0.0 to 1.0).
    /// Higher values track new measurements faster; lower values forget
    /// old measurements more slowly.
    pub pressure_ema_alpha: f64,

    /// Activation thresholds
    pub activation: ActivationConfig,

    /// Patch selection configuration
    pub selection: SelectionConfig,
}

/// Configuration for a single pressure axis.
#[derive(Debug, Clone, Deserialize)]
pub struct PressureAxisConfig {
    /// Unique name for this axis
    pub name: String,

    /// Base weight for this pressure
    pub weight: f64,

    /// Signal name or expression to evaluate
    /// Simple case: just the signal name (e.g., "lint_density")
    /// Complex case: expression like "max(0, 1.0 - parse_confidence)"
    pub expr: String,

    /// Per-region-kind weight overrides
    #[serde(default)]
    pub kind_weights: HashMap<String, f64>,
}

/// Activation configuration: when to trigger action proposals.
#[derive(Debug, Clone, Deserialize)]
pub struct ActivationConfig {
    /// Minimum total weighted pressure to trigger proposals
    pub min_total_pressure: f64,

    /// Inhibition window after patch application (logical ticks)
    pub inhibit_ticks: u64,
}

/// Selection configuration: how to choose among candidate patches.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectionConfig {
    /// Minimum expected improvement to accept a patch
    pub min_expected_improvement: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            tick_interval_ms: 250,
            max_ticks: 0,        // unlimited
            stable_threshold: 3, // stop after 3 ticks with no patches
            pressure_axes: Vec::new(),
            pressure_ema_alpha: 0.2,
            activation: ActivationConfig {
                min_total_pressure: 0.8,
                inhibit_ticks: 120,
            },
            selection: SelectionConfig {
                min_expected_improvement: 0.15,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_valid_smoothing_and_inhibition() {
        let config = KernelConfig::default();
        // EMA smoothing must be a valid mixing factor.
        assert!(config.pressure_ema_alpha > 0.0 && config.pressure_ema_alpha <= 1.0);
        // Inhibition must be a positive number of ticks by default.
        assert!(config.activation.inhibit_ticks > 0);
    }
}
