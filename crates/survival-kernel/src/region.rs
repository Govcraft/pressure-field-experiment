//! Region types: the atomic units of artifacts that can be independently scored and patched.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a region within an artifact.
pub type RegionId = Uuid;

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
