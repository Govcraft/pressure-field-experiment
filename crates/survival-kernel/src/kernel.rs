//! The coordination kernel: tick-based pressure descent with decay.

use std::collections::HashMap;

use crate::{
    artifact::Artifact,
    config::KernelConfig,
    pressure::{Actor, PressureVector, Sensor, Signals},
    region::{Patch, RegionId, RegionState},
};

/// The survival kernel: coordinates agents through gradient fields and decay.
///
/// The kernel runs a tick loop:
/// 1. Decay all region states
/// 2. Measure signals for each region
/// 3. Compute pressures from signals
/// 4. For high-pressure regions, collect patch proposals from actors
/// 5. Select and apply top patches
/// 6. Reinforce patched regions, inhibit for cooldown
pub struct Kernel {
    config: KernelConfig,
    sensors: Vec<Box<dyn Sensor>>,
    actors: Vec<Box<dyn Actor>>,
    state: HashMap<RegionId, RegionState>,
}

/// Result of a single tick.
#[derive(Debug, Default)]
pub struct TickResult {
    /// Patches that were applied
    pub applied: Vec<Patch>,
    /// Regions that were evaluated but not patched
    pub evaluated: usize,
    /// Regions that were skipped (inhibited)
    pub skipped: usize,
    /// Total pressure across all regions
    pub total_pressure: f64,
}

impl Kernel {
    /// Create a new kernel with the given configuration.
    pub fn new(config: KernelConfig) -> Self {
        Self {
            config,
            sensors: Vec::new(),
            actors: Vec::new(),
            state: HashMap::new(),
        }
    }

    /// Register a sensor.
    pub fn add_sensor(&mut self, sensor: Box<dyn Sensor>) {
        self.sensors.push(sensor);
    }

    /// Register an actor.
    pub fn add_actor(&mut self, actor: Box<dyn Actor>) {
        self.actors.push(actor);
    }

    /// Run a single tick of the coordination loop.
    pub fn tick(&mut self, artifact: &mut dyn Artifact, now_ms: u64) -> anyhow::Result<TickResult> {
        let mut result = TickResult::default();

        // 1. Apply decay to all region states
        self.apply_decay(now_ms);

        // 2. Ensure all regions have state entries and collect active regions
        let region_ids = artifact.region_ids();
        for &rid in &region_ids {
            self.state
                .entry(rid)
                .or_insert_with(|| RegionState::new(now_ms));
        }

        // 3. Measure, compute pressures, collect candidates
        // We collect intermediate data to avoid borrow conflicts
        struct RegionEval {
            rid: RegionId,
            kind: String,
            signals: Signals,
            pressures: PressureVector,
            total_pressure: f64,
            state_snapshot: RegionState,
        }

        let mut evaluations: Vec<RegionEval> = Vec::new();

        for rid in region_ids {
            let state = self.state.get(&rid).unwrap();

            // Skip if inhibited
            if state.is_inhibited(now_ms) {
                result.skipped += 1;
                continue;
            }

            let region = artifact.read_region(rid)?;
            result.evaluated += 1;

            // Measure signals
            let signals = self.measure_signals(&region)?;

            // Compute pressures
            let pressures = self.compute_pressures(&signals);

            // Calculate total weighted pressure
            let total = self.total_weighted_pressure(&region.kind, &pressures);
            result.total_pressure += total;

            evaluations.push(RegionEval {
                rid,
                kind: region.kind,
                signals,
                pressures,
                total_pressure: total,
                state_snapshot: state.clone(),
            });
        }

        // 4. Update pressure EMAs
        let ema_alpha = self.config.decay.ema_alpha;
        for eval in &evaluations {
            if let Some(state) = self.state.get_mut(&eval.rid) {
                for (name, value) in &eval.pressures {
                    let ema = state.pressure_ema.entry(name.clone()).or_insert(*value);
                    *ema = ema_alpha * value + (1.0 - ema_alpha) * *ema;
                }
            }
        }

        // 5. Collect patch proposals for high-pressure regions
        let mut candidates: Vec<(f64, Patch)> = Vec::new();
        let activation_threshold = self.config.activation.min_total_pressure;
        let min_improvement = self.config.selection.min_expected_improvement;

        for eval in &evaluations {
            if eval.total_pressure < activation_threshold {
                continue;
            }

            let region = artifact.read_region(eval.rid)?;

            for actor in &self.actors {
                let patches =
                    actor.propose(&region, &eval.signals, &eval.pressures, &eval.state_snapshot)?;
                for patch in patches {
                    let score = self.score_patch(&patch, &eval.kind);
                    if score >= min_improvement {
                        candidates.push((score, patch));
                    }
                }
            }
        }

        // 6. Select and apply top patches
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let inhibit_ms = self.config.activation.inhibit_ms;
        for (_, patch) in candidates
            .into_iter()
            .take(self.config.selection.max_patches_per_tick)
        {
            let rid = patch.region;
            artifact.apply_patch(patch.clone())?;

            // 7. Reinforce and inhibit
            if let Some(state) = self.state.get_mut(&rid) {
                state.last_updated_ms = now_ms;
                state.confidence = (state.confidence + 0.05).min(1.0);
                state.fitness = (state.fitness + 0.03).min(1.0);
                state.suppress_until_ms = Some(now_ms + inhibit_ms);
                state.provenance.push(patch.rationale.clone());
            }

            result.applied.push(patch);
        }

        Ok(result)
    }

    /// Check if the system has stabilized (no high-pressure regions).
    pub fn is_stable(&self) -> bool {
        self.state.values().all(|s| {
            let total: f64 = s.pressure_ema.values().sum();
            total < self.config.activation.min_total_pressure
        })
    }

    /// Get the current state for a region.
    pub fn region_state(&self, rid: RegionId) -> Option<&RegionState> {
        self.state.get(&rid)
    }

    fn measure_signals(
        &self,
        region: &crate::region::RegionView,
    ) -> anyhow::Result<Signals> {
        let mut signals = Signals::new();
        for sensor in &self.sensors {
            let measured = sensor.measure(region)?;
            for (k, v) in measured {
                *signals.entry(k).or_insert(0.0) += v;
            }
        }
        Ok(signals)
    }

    fn compute_pressures(&self, signals: &Signals) -> PressureVector {
        let mut pressures = PressureVector::new();
        for axis in &self.config.pressure_axes {
            // Simple case: expr is just a signal name
            let value = signals.get(&axis.expr).copied().unwrap_or(0.0).max(0.0);
            pressures.insert(axis.name.clone(), value);
        }
        pressures
    }

    fn total_weighted_pressure(&self, kind: &str, pressures: &PressureVector) -> f64 {
        let mut total = 0.0;
        for axis in &self.config.pressure_axes {
            let weight = axis
                .kind_weights
                .get(kind)
                .copied()
                .unwrap_or(axis.weight);
            if let Some(&value) = pressures.get(&axis.name) {
                total += weight * value;
            }
        }
        total
    }

    fn score_patch(&self, patch: &Patch, kind: &str) -> f64 {
        let mut score = 0.0;
        for axis in &self.config.pressure_axes {
            let weight = axis
                .kind_weights
                .get(kind)
                .copied()
                .unwrap_or(axis.weight);
            if let Some(&delta) = patch.expected_delta.get(&axis.name) {
                score += weight * delta;
            }
        }
        score
    }

    fn apply_decay(&mut self, now_ms: u64) {
        let fitness_hl = self.config.decay.fitness_half_life_ms;
        let confidence_hl = self.config.decay.confidence_half_life_ms;

        for state in self.state.values_mut() {
            let dt_ms = now_ms.saturating_sub(state.last_updated_ms);
            if dt_ms == 0 {
                continue;
            }

            half_life_decay(&mut state.fitness, dt_ms, fitness_hl);
            half_life_decay(&mut state.confidence, dt_ms, confidence_hl);
            state.last_updated_ms = now_ms;
        }
    }
}

/// Apply exponential decay with the given half-life.
fn half_life_decay(value: &mut f64, dt_ms: u64, half_life_ms: u64) {
    if half_life_ms == 0 {
        return;
    }
    let lambda = std::f64::consts::LN_2 / half_life_ms as f64;
    *value *= (-lambda * dt_ms as f64).exp();
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
