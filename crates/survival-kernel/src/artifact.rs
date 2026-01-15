//! Artifact trait: the interface for mutable objects under pressure.

use crate::region::{Patch, RegionId, RegionView};

/// An artifact is any mutable object that can be refined through pressure-driven coordination.
///
/// Examples:
/// - Text documents (regions = paragraphs/spans)
/// - Source code (regions = functions/modules)
/// - Configuration files (regions = sections/keys)
/// - Proofs (regions = lemmas/steps)
///
/// The kernel is artifact-agnostic; all domain-specific logic lives in
/// the `Artifact` implementation and the sensors/actors.
pub trait Artifact: Send + Sync {
    /// List all region IDs in the current artifact state.
    fn region_ids(&self) -> Vec<RegionId>;

    /// Read a region by ID.
    fn read_region(&self, id: RegionId) -> anyhow::Result<RegionView>;

    /// Apply a patch to the artifact.
    ///
    /// This should be atomic: either the patch fully applies or it fails
    /// without modifying the artifact.
    fn apply_patch(&mut self, patch: Patch) -> anyhow::Result<()>;

    /// Optional: snapshot the artifact for rollback.
    fn snapshot(&self) -> Option<Box<dyn Artifact>> {
        None
    }

    /// Optional: get the full source content of the artifact.
    ///
    /// Used for saving the final state after optimization.
    fn source(&self) -> Option<String> {
        None
    }

    /// Optional: callback invoked after a patch is successfully applied.
    ///
    /// Use this for domain-specific learning, such as adding successful
    /// patches to an example bank for few-shot prompting (pheromone deposit).
    ///
    /// Default implementation does nothing.
    fn on_patch_applied(&mut self, _patch: &Patch) {}

    /// Optional: check if the artifact has reached a complete/solved state.
    ///
    /// The kernel will stop ticking when this returns true. This allows
    /// domain-specific completion checks (e.g., puzzle solved, all tests pass).
    ///
    /// Default implementation returns false (rely on pressure convergence).
    fn is_complete(&self) -> bool {
        false
    }

    /// Optional: evaluate a patch before applying it.
    ///
    /// This performs clone-based validation: applies the patch to a cloned
    /// copy of the artifact, measures the actual pressure change, and returns
    /// whether the patch should be accepted.
    ///
    /// Returns `(should_accept, pressure_delta)` where:
    /// - `should_accept`: true if the patch improves pressure (delta > 0)
    /// - `pressure_delta`: old_pressure - new_pressure (positive = improvement)
    ///
    /// This maintains Nash equilibrium by only accepting moves that improve state.
    ///
    /// Default implementation accepts all patches (backward compatible).
    fn evaluate_patch(&self, _patch: &Patch) -> (bool, f64) {
        (true, 0.0)
    }

    /// Optional: get the total pressure of the artifact.
    ///
    /// Returns the actual pressure computed directly from the artifact state.
    /// This is used for tick-level pressure reporting and should match the
    /// pressure calculation used in `evaluate_patch()`.
    ///
    /// Default implementation returns None (use EMA-based calculation).
    fn total_pressure(&self) -> Option<f64> {
        None
    }
}
