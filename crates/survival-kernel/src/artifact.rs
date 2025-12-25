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
}
