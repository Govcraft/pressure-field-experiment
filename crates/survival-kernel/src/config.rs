//! Configuration types for the kernel.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level kernel configuration.
///
/// This defines the pressure landscape, decay dynamics, and selection criteria.
/// Loaded from TOML/JSON at runtime.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelConfig {
    /// Tick interval in milliseconds (for decay calculations)
    pub tick_interval_ms: u64,

    /// Pressure axis definitions
    pub pressure_axes: Vec<PressureAxisConfig>,

    /// Decay configuration
    pub decay: DecayConfig,

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

/// Decay configuration: how quickly state erodes without reinforcement.
#[derive(Debug, Clone, Deserialize)]
pub struct DecayConfig {
    /// Half-life for fitness decay (milliseconds)
    pub fitness_half_life_ms: u64,

    /// Half-life for confidence decay (milliseconds)
    pub confidence_half_life_ms: u64,

    /// Smoothing factor for pressure EMA (0.0 to 1.0)
    pub ema_alpha: f64,
}

/// Activation configuration: when to trigger action proposals.
#[derive(Debug, Clone, Deserialize)]
pub struct ActivationConfig {
    /// Minimum total weighted pressure to trigger proposals
    pub min_total_pressure: f64,

    /// Inhibition window after patch application (milliseconds)
    pub inhibit_ms: u64,
}

/// Selection configuration: how to choose among candidate patches.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectionConfig {
    /// Maximum patches to apply per tick
    pub max_patches_per_tick: usize,

    /// Minimum expected improvement to accept a patch
    pub min_expected_improvement: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            tick_interval_ms: 250,
            pressure_axes: Vec::new(),
            decay: DecayConfig {
                fitness_half_life_ms: 600_000,    // 10 minutes
                confidence_half_life_ms: 1_800_000, // 30 minutes
                ema_alpha: 0.2,
            },
            activation: ActivationConfig {
                min_total_pressure: 0.8,
                inhibit_ms: 30_000,
            },
            selection: SelectionConfig {
                max_patches_per_tick: 3,
                min_expected_improvement: 0.15,
            },
        }
    }
}
