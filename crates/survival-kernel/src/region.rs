//! Region types: the atomic units of artifacts that can be independently scored and patched.

use std::collections::HashMap;

use mti::prelude::MagicTypeId;
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

/// Persistent state for a region.
///
/// Tracks the smoothed pressure field per axis, the inhibition window, and an
/// audit trail of applied patches.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegionState {
    /// Exponential moving average of pressure values per axis
    pub pressure_ema: HashMap<String, f64>,
    /// Inhibition window: suppress actions until this logical tick
    pub suppress_until_tick: Option<u64>,
    /// Audit trail of applied patches
    pub provenance: Vec<String>,
}

impl RegionState {
    /// Check if this region is currently inhibited at the given logical tick.
    pub fn is_inhibited(&self, now_tick: u64) -> bool {
        self.suppress_until_tick
            .is_some_and(|until| now_tick < until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_state_inhibition_uses_logical_ticks() {
        let mut state = RegionState::default();
        assert!(!state.is_inhibited(0));

        // Inhibit until tick 10.
        state.suppress_until_tick = Some(10);
        assert!(state.is_inhibited(5));
        assert!(state.is_inhibited(9));
        // Inhibition ends exactly at the suppression tick.
        assert!(!state.is_inhibited(10));
        assert!(!state.is_inhibited(11));
    }
}
