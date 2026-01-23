//! Region types: the atomic units of artifacts that can be independently scored and patched.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use mti::prelude::MagicTypeId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Unique identifier for a region within an artifact.
///
/// Uses MTI (Magic Type ID) for human-readable, type-safe identifiers like
/// `region_01h455vb4pex5vsknk084sn02q`. The type prefix provides debuggability
/// while the suffix ensures uniqueness (v5 for deterministic, v7 for time-ordered).
pub type RegionId = MagicTypeId;

/// A view into a region for measurement and action proposal.
///
/// Regions are the smallest independently scorable units:
/// - For text: paragraphs, sentences, or spans
/// - For code: functions, modules, or AST nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionView {
    pub id: RegionId,
    /// Region kind allows different pressure profiles (e.g., "heading", "code_fn", "test")
    pub kind: String,
    /// The content of the region
    pub content: String,
    /// Arbitrary metadata for sensors/actors
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A mutation that can be applied to a region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    pub region: RegionId,
    pub op: PatchOp,
    /// Human-readable explanation for audit trail
    pub rationale: String,
    /// Predicted improvement per pressure axis (used for selection)
    pub expected_delta: HashMap<String, f64>,
}

/// The operation to apply to a region.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum PatchOp {
    /// Replace the region's content entirely
    Replace(String),
    /// Delete the region
    Delete,
    /// Insert content after this region
    InsertAfter(String),
}

/// Persistent state for a region: the "pheromone" store.
///
/// This tracks fitness, confidence, and pressure history over time,
/// enabling decay and reinforcement dynamics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegionState {
    /// Last update timestamp (milliseconds since epoch or arbitrary tick)
    pub last_updated_ms: u64,
    /// Fitness score: higher means more likely to survive (0.0 to 1.0)
    pub fitness: f64,
    /// Confidence in current state (0.0 to 1.0)
    pub confidence: f64,
    /// Exponential moving average of pressure values per axis
    pub pressure_ema: HashMap<String, f64>,
    /// Inhibition window: suppress actions until this timestamp
    pub suppress_until_ms: Option<u64>,
    /// Audit trail of applied patches
    pub provenance: Vec<String>,
}

impl RegionState {
    /// Create a new region state with default initial values.
    pub fn new(now_ms: u64) -> Self {
        Self {
            last_updated_ms: now_ms,
            fitness: 0.5,
            confidence: 0.5,
            pressure_ema: HashMap::new(),
            suppress_until_ms: None,
            provenance: Vec::new(),
        }
    }

    /// Check if this region is currently inhibited.
    pub fn is_inhibited(&self, now_ms: u64) -> bool {
        self.suppress_until_ms.is_some_and(|until| now_ms < until)
    }
}

/// Thread-safe map of region states for concurrent access.
pub type RegionStateMap = Arc<DashMap<RegionId, AtomicRegionState>>;

/// Thread-safe region state using atomics for numeric fields.
///
/// This enables lock-free concurrent updates during the measurement
/// and proposal phases of the tick loop.
#[derive(Debug)]
pub struct AtomicRegionState {
    /// Last update timestamp (milliseconds)
    pub last_updated_ms: AtomicU64,
    /// Fitness score stored as f64 bits (0.0 to 1.0)
    fitness_bits: AtomicU64,
    /// Confidence stored as f64 bits (0.0 to 1.0)
    confidence_bits: AtomicU64,
    /// Exponential moving average of pressure values per axis
    pub pressure_ema: Arc<DashMap<String, AtomicU64>>,
    /// Inhibition timestamp (0 = not inhibited)
    suppress_until_ms: AtomicU64,
    /// Audit trail of applied patches (requires lock for modification)
    pub provenance: RwLock<Vec<String>>,
}

impl AtomicRegionState {
    /// Create a new atomic region state with default initial values.
    pub fn new(now_ms: u64) -> Self {
        Self {
            last_updated_ms: AtomicU64::new(now_ms),
            fitness_bits: AtomicU64::new(0.5_f64.to_bits()),
            confidence_bits: AtomicU64::new(0.5_f64.to_bits()),
            pressure_ema: Arc::new(DashMap::new()),
            suppress_until_ms: AtomicU64::new(0),
            provenance: RwLock::new(Vec::new()),
        }
    }

    /// Get the current fitness value.
    pub fn fitness(&self) -> f64 {
        f64::from_bits(self.fitness_bits.load(Ordering::Acquire))
    }

    /// Set the fitness value atomically.
    pub fn set_fitness(&self, value: f64) {
        self.fitness_bits.store(value.to_bits(), Ordering::Release);
    }

    /// Get the current confidence value.
    pub fn confidence(&self) -> f64 {
        f64::from_bits(self.confidence_bits.load(Ordering::Acquire))
    }

    /// Set the confidence value atomically.
    pub fn set_confidence(&self, value: f64) {
        self.confidence_bits
            .store(value.to_bits(), Ordering::Release);
    }

    /// Check if this region is currently inhibited.
    pub fn is_inhibited(&self, now_ms: u64) -> bool {
        let until = self.suppress_until_ms.load(Ordering::Acquire);
        until > 0 && now_ms < until
    }

    /// Set the inhibition window.
    pub fn set_inhibition(&self, until_ms: u64) {
        self.suppress_until_ms.store(until_ms, Ordering::Release);
    }

    /// Clear the inhibition window.
    pub fn clear_inhibition(&self) {
        self.suppress_until_ms.store(0, Ordering::Release);
    }

    /// Update a pressure EMA value atomically using compare-and-swap.
    pub fn update_pressure_ema(&self, axis: &str, new_value: f64, alpha: f64) {
        self.pressure_ema
            .entry(axis.to_string())
            .and_modify(|current| {
                loop {
                    let current_bits = current.load(Ordering::Acquire);
                    let current_val = f64::from_bits(current_bits);
                    let ema = alpha * new_value + (1.0 - alpha) * current_val;
                    let new_bits = ema.to_bits();

                    if current
                        .compare_exchange(
                            current_bits,
                            new_bits,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        break;
                    }
                }
            })
            .or_insert_with(|| AtomicU64::new(new_value.to_bits()));
    }

    /// Get the total pressure across all axes.
    pub fn total_pressure(&self) -> f64 {
        self.pressure_ema
            .iter()
            .map(|entry| f64::from_bits(entry.value().load(Ordering::Acquire)))
            .sum()
    }

    /// Get pressure value for a specific axis.
    pub fn pressure(&self, axis: &str) -> Option<f64> {
        self.pressure_ema
            .get(axis)
            .map(|entry| f64::from_bits(entry.value().load(Ordering::Acquire)))
    }

    /// Convert to a snapshot RegionState for passing to actors.
    pub fn to_snapshot(&self) -> RegionState {
        let pressure_ema: HashMap<String, f64> = self
            .pressure_ema
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    f64::from_bits(entry.value().load(Ordering::Acquire)),
                )
            })
            .collect();

        let suppress_until = self.suppress_until_ms.load(Ordering::Acquire);

        RegionState {
            last_updated_ms: self.last_updated_ms.load(Ordering::Acquire),
            fitness: self.fitness(),
            confidence: self.confidence(),
            pressure_ema,
            suppress_until_ms: if suppress_until > 0 {
                Some(suppress_until)
            } else {
                None
            },
            provenance: self.provenance.read().clone(),
        }
    }

    /// Apply decay to fitness and confidence values.
    pub fn apply_decay(
        &self,
        now_ms: u64,
        fitness_half_life_ms: u64,
        confidence_half_life_ms: u64,
    ) {
        let last_updated = self.last_updated_ms.load(Ordering::Acquire);
        let dt_ms = now_ms.saturating_sub(last_updated);

        if dt_ms == 0 {
            return;
        }

        // Apply decay to fitness
        if fitness_half_life_ms > 0 {
            let lambda = std::f64::consts::LN_2 / fitness_half_life_ms as f64;
            let current_fitness = self.fitness();
            let decayed = current_fitness * (-lambda * dt_ms as f64).exp();
            self.set_fitness(decayed);
        }

        // Apply decay to confidence
        if confidence_half_life_ms > 0 {
            let lambda = std::f64::consts::LN_2 / confidence_half_life_ms as f64;
            let current_confidence = self.confidence();
            let decayed = current_confidence * (-lambda * dt_ms as f64).exp();
            self.set_confidence(decayed);
        }

        self.last_updated_ms.store(now_ms, Ordering::Release);
    }
}

impl Default for AtomicRegionState {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Clone for AtomicRegionState {
    fn clone(&self) -> Self {
        let new_pressure_ema = Arc::new(DashMap::new());
        for entry in self.pressure_ema.iter() {
            new_pressure_ema.insert(
                entry.key().clone(),
                AtomicU64::new(entry.value().load(Ordering::Acquire)),
            );
        }

        Self {
            last_updated_ms: AtomicU64::new(self.last_updated_ms.load(Ordering::Acquire)),
            fitness_bits: AtomicU64::new(self.fitness_bits.load(Ordering::Acquire)),
            confidence_bits: AtomicU64::new(self.confidence_bits.load(Ordering::Acquire)),
            pressure_ema: new_pressure_ema,
            suppress_until_ms: AtomicU64::new(self.suppress_until_ms.load(Ordering::Acquire)),
            provenance: RwLock::new(self.provenance.read().clone()),
        }
    }
}
