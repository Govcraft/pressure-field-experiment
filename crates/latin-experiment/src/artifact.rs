//! LatinSquareArtifact: Artifact trait implementation for Latin Square puzzles.
//!
//! Each row is a region that can be patched independently.
//! Uses deterministic UUIDs for stable region IDs across re-parses.

use std::collections::HashMap;
use std::fmt;

use anyhow::{Result, bail};
use mti::prelude::*;
use survival_kernel::artifact::Artifact;
use survival_kernel::region::{Patch, PatchOp, RegionId, RegionView};
use uuid::Uuid;

use crate::sensors::SharedGrid;

/// Namespace UUID for generating deterministic region IDs.
const REGION_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc9,
]);

/// Create a MagicTypeId for a region from puzzle_id and row index.
fn create_region_mti(puzzle_id: &str, row_idx: usize) -> RegionId {
    let name = format!("{}:row:{}", puzzle_id, row_idx);
    let v5_uuid = Uuid::new_v5(&REGION_NAMESPACE, name.as_bytes());
    let prefix = TypeIdPrefix::try_from("region").expect("region is a valid prefix");
    let suffix = TypeIdSuffix::from(v5_uuid);
    MagicTypeId::new(prefix, suffix)
}

/// A Latin Square puzzle artifact.
///
/// Each row is treated as an independent region that can be patched.
/// The grid uses Option<u8> where None represents an empty cell.
#[derive(Debug, Clone)]
pub struct LatinSquareArtifact {
    /// Grid size (n x n)
    n: usize,
    /// The puzzle grid: grid[row][col] = Some(value) or None for empty
    grid: Vec<Vec<Option<u8>>>,
    /// Fixed cells that were given initially (cannot be changed)
    fixed: Vec<Vec<bool>>,
    /// Map from region ID to row index
    region_map: HashMap<RegionId, usize>,
    /// Ordered list of region IDs (one per row)
    region_order: Vec<RegionId>,
    /// Puzzle identifier for deterministic region IDs
    puzzle_id: String,
    /// Optional shared grid for sensor synchronization (updated on patch)
    shared_grid: Option<SharedGrid>,
}

impl LatinSquareArtifact {
    /// Create a new Latin Square artifact from a grid.
    ///
    /// # Arguments
    /// * `n` - Grid size (n x n)
    /// * `grid` - Initial grid state (None for empty cells)
    /// * `puzzle_id` - Unique identifier for this puzzle instance
    pub fn new(n: usize, grid: Vec<Vec<Option<u8>>>, puzzle_id: impl Into<String>) -> Result<Self> {
        let puzzle_id = puzzle_id.into();

        if grid.len() != n {
            bail!("Grid has {} rows, expected {}", grid.len(), n);
        }

        for (i, row) in grid.iter().enumerate() {
            if row.len() != n {
                bail!("Row {} has {} columns, expected {}", i, row.len(), n);
            }
        }

        // Track which cells are fixed (initially given)
        let fixed: Vec<Vec<bool>> = grid
            .iter()
            .map(|row| row.iter().map(|cell| cell.is_some()).collect())
            .collect();

        // Generate deterministic region IDs for each row
        let mut region_map = HashMap::new();
        let mut region_order = Vec::with_capacity(n);

        for row_idx in 0..n {
            let region_id = create_region_mti(&puzzle_id, row_idx);
            region_map.insert(region_id.clone(), row_idx);
            region_order.push(region_id);
        }

        Ok(Self {
            n,
            grid,
            fixed,
            region_map,
            region_order,
            puzzle_id,
            shared_grid: None,
        })
    }

    /// Configure a shared grid for sensor synchronization.
    ///
    /// When set, the shared grid is automatically updated whenever
    /// a patch is applied, keeping sensors in sync with the artifact state.
    pub fn with_shared_grid(mut self, grid: SharedGrid) -> Self {
        self.shared_grid = Some(grid);
        self
    }

    /// Get the grid size.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the current grid state.
    pub fn grid(&self) -> &Vec<Vec<Option<u8>>> {
        &self.grid
    }

    /// Check if the puzzle is completely solved (all cells filled, no violations).
    pub fn is_solved(&self) -> bool {
        // Check all cells are filled
        for row in &self.grid {
            for cell in row {
                if cell.is_none() {
                    return false;
                }
            }
        }

        // Check row constraints
        for row in &self.grid {
            let mut seen = vec![false; self.n + 1];
            for v in row.iter().flatten() {
                let v = *v as usize;
                if v == 0 || v > self.n || seen[v] {
                    return false;
                }
                seen[v] = true;
            }
        }

        // Check column constraints
        for col in 0..self.n {
            let mut seen = vec![false; self.n + 1];
            for row in 0..self.n {
                if let Some(v) = self.grid[row][col] {
                    let v = v as usize;
                    if v == 0 || v > self.n || seen[v] {
                        return false;
                    }
                    seen[v] = true;
                }
            }
        }

        true
    }

    /// Count total constraint violations.
    pub fn total_violations(&self) -> usize {
        let mut violations = 0;

        // Row violations
        for row in &self.grid {
            let mut counts = vec![0usize; self.n + 1];
            for v in row.iter().flatten() {
                let v = *v as usize;
                if v > 0 && v <= self.n {
                    counts[v] += 1;
                }
            }
            for count in counts {
                if count > 1 {
                    violations += count - 1;
                }
            }
        }

        // Column violations
        for col in 0..self.n {
            let mut counts = vec![0usize; self.n + 1];
            for row in 0..self.n {
                if let Some(v) = self.grid[row][col] {
                    let v = v as usize;
                    if v > 0 && v <= self.n {
                        counts[v] += 1;
                    }
                }
            }
            for count in counts {
                if count > 1 {
                    violations += count - 1;
                }
            }
        }

        violations
    }

    /// Count empty cells.
    pub fn empty_count(&self) -> usize {
        self.grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| cell.is_none())
            .count()
    }

    /// Parse a row from string format "1 _ 3 _ 5 _".
    pub fn parse_row(s: &str, n: usize) -> Result<Vec<Option<u8>>> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != n {
            bail!("Row has {} values, expected {}", parts.len(), n);
        }

        let mut row = Vec::with_capacity(n);
        for part in parts {
            if part == "_" {
                row.push(None);
            } else {
                let v: u8 = part
                    .parse()
                    .map_err(|_| anyhow::anyhow!("Invalid value: {}", part))?;
                if v == 0 || v as usize > n {
                    bail!("Value {} out of range 1..{}", v, n);
                }
                row.push(Some(v));
            }
        }

        Ok(row)
    }

    /// Format a row to string "1 _ 3 _ 5 _".
    pub fn format_row(row: &[Option<u8>]) -> String {
        row.iter()
            .map(|cell| match cell {
                Some(v) => v.to_string(),
                None => "_".to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get column availability for a given row index.
    ///
    /// Returns a map of column index -> set of available values (not yet used in that column).
    pub fn column_availability(&self, _row_idx: usize) -> HashMap<usize, Vec<u8>> {
        let mut availability = HashMap::new();

        for col in 0..self.n {
            let mut used = vec![false; self.n + 1];
            for row in 0..self.n {
                if let Some(v) = self.grid[row][col] {
                    used[v as usize] = true;
                }
            }

            let available: Vec<u8> = (1..=self.n as u8).filter(|&v| !used[v as usize]).collect();

            availability.insert(col, available);
        }

        availability
    }

    /// Get row index for a region ID.
    pub fn row_index(&self, region_id: RegionId) -> Option<usize> {
        self.region_map.get(&region_id).copied()
    }
}

impl Artifact for LatinSquareArtifact {
    fn region_ids(&self) -> Vec<RegionId> {
        self.region_order.clone()
    }

    fn read_region(&self, id: RegionId) -> Result<RegionView> {
        let row_idx = self
            .region_map
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Region not found: {}", id))?;

        let row = &self.grid[*row_idx];
        let content = Self::format_row(row);

        let mut metadata = HashMap::new();
        metadata.insert("row_index".to_string(), serde_json::json!(*row_idx));
        metadata.insert("n".to_string(), serde_json::json!(self.n));
        metadata.insert("puzzle_id".to_string(), serde_json::json!(&self.puzzle_id));

        // Include column availability in metadata for LLM prompting
        let availability = self.column_availability(*row_idx);
        metadata.insert(
            "column_availability".to_string(),
            serde_json::json!(availability),
        );

        Ok(RegionView {
            id,
            kind: "row".to_string(),
            content,
            metadata,
        })
    }

    fn apply_patch(&mut self, patch: Patch) -> Result<()> {
        let row_idx = self
            .region_map
            .get(&patch.region)
            .ok_or_else(|| anyhow::anyhow!("Region not found: {}", patch.region))?;
        let row_idx = *row_idx;

        match &patch.op {
            PatchOp::Replace(content) => {
                let new_row = Self::parse_row(content, self.n)?;

                // Validate that fixed cells are preserved
                for (col, (is_fixed, new_val)) in
                    self.fixed[row_idx].iter().zip(new_row.iter()).enumerate()
                {
                    if *is_fixed {
                        let original = self.grid[row_idx][col];
                        if *new_val != original {
                            bail!(
                                "Cannot change fixed cell at ({}, {}): was {:?}, got {:?}",
                                row_idx,
                                col,
                                original,
                                new_val
                            );
                        }
                    }
                }

                self.grid[row_idx] = new_row;

                // Sync shared grid for sensors
                if let Some(ref shared) = self.shared_grid
                    && let Ok(mut locked) = shared.write()
                {
                    locked.clear();
                    locked.extend(self.grid.iter().cloned());
                }
            }
            PatchOp::Delete => {
                bail!("Cannot delete a row in a Latin Square");
            }
            PatchOp::InsertAfter(_) => {
                bail!("Cannot insert after a row in a Latin Square");
            }
        }

        Ok(())
    }

    fn source(&self) -> Option<String> {
        let mut lines = Vec::new();
        for row in &self.grid {
            lines.push(Self::format_row(row));
        }
        Some(lines.join("\n"))
    }

    fn is_complete(&self) -> bool {
        self.is_solved()
    }

    fn on_patch_applied(&mut self, _patch: &Patch) {
        // Learning callback - could be used for example bank/pheromone deposits
        // Currently a no-op; the experiment can handle this externally if needed
    }
}

impl fmt::Display for LatinSquareArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Latin Square {}x{} (puzzle: {})",
            self.n, self.n, self.puzzle_id
        )?;
        for (i, row) in self.grid.iter().enumerate() {
            write!(f, "  Row {}: ", i)?;
            for cell in row {
                match cell {
                    Some(v) => write!(f, "{} ", v)?,
                    None => write!(f, "_ ")?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_puzzle() -> LatinSquareArtifact {
        // 4x4 puzzle with some cells filled
        let grid = vec![
            vec![Some(1), None, Some(3), None],
            vec![None, Some(2), None, Some(4)],
            vec![Some(3), None, Some(1), None],
            vec![None, Some(4), None, Some(2)],
        ];
        LatinSquareArtifact::new(4, grid, "test-puzzle").unwrap()
    }

    #[test]
    fn test_create_artifact() {
        let artifact = sample_puzzle();
        assert_eq!(artifact.size(), 4);
        assert_eq!(artifact.region_ids().len(), 4);
    }

    #[test]
    fn test_read_region() {
        let artifact = sample_puzzle();
        let regions = artifact.region_ids();
        let view = artifact.read_region(regions[0].clone()).unwrap();

        assert_eq!(view.content, "1 _ 3 _");
        assert_eq!(view.kind, "row");
    }

    #[test]
    fn test_apply_patch() {
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill empty cells".to_string(),
            expected_delta: HashMap::new(),
        };

        artifact.apply_patch(patch).unwrap();

        let view = artifact.read_region(regions[0].clone()).unwrap();
        assert_eq!(view.content, "1 2 3 4");
    }

    #[test]
    fn test_cannot_change_fixed_cells() {
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        // Try to change fixed cell (1 at position 0)
        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("2 2 3 4".to_string()),
            rationale: "Bad patch".to_string(),
            expected_delta: HashMap::new(),
        };

        assert!(artifact.apply_patch(patch).is_err());
    }

    #[test]
    fn test_column_availability() {
        let artifact = sample_puzzle();
        let availability = artifact.column_availability(0);

        // Column 0 has 1 and 3, so 2 and 4 are available
        assert_eq!(availability[&0], vec![2, 4]);
    }

    #[test]
    fn test_is_solved() {
        // Create a solved 4x4 Latin square
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![Some(2), Some(1), Some(4), Some(3)],
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "solved").unwrap();
        assert!(artifact.is_solved());
    }

    #[test]
    fn test_not_solved_with_empty() {
        let artifact = sample_puzzle();
        assert!(!artifact.is_solved());
    }

    #[test]
    fn test_parse_row() {
        let row = LatinSquareArtifact::parse_row("1 _ 3 _", 4).unwrap();
        assert_eq!(row, vec![Some(1), None, Some(3), None]);
    }

    #[test]
    fn test_format_row() {
        let row = vec![Some(1), None, Some(3), None];
        assert_eq!(LatinSquareArtifact::format_row(&row), "1 _ 3 _");
    }

    #[test]
    fn test_apply_multiple_patches_sequentially() {
        // This tests the scenario that was buggy: multiple patches applied in sequence
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        // Patch row 0: fill empty cells
        let patch0 = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill row 0".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch0).unwrap();
        assert_eq!(artifact.grid()[0], vec![Some(1), Some(2), Some(3), Some(4)]);

        // Patch row 1: fill empty cells
        let patch1 = Patch {
            region: regions[1].clone(),
            op: PatchOp::Replace("3 2 4 4".to_string()),
            rationale: "Fill row 1".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch1).unwrap();
        assert_eq!(artifact.grid()[1], vec![Some(3), Some(2), Some(4), Some(4)]);

        // Verify both rows are updated
        let view0 = artifact.read_region(regions[0].clone()).unwrap();
        let view1 = artifact.read_region(regions[1].clone()).unwrap();
        assert_eq!(view0.content, "1 2 3 4");
        assert_eq!(view1.content, "3 2 4 4");
    }

    #[test]
    fn test_grid_returns_current_state() {
        // Verify that grid() returns the actual current state after patches
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        // Initial state
        let initial_grid = artifact.grid().clone();
        assert_eq!(initial_grid[0][1], None); // Row 0, Col 1 is empty

        // Apply patch
        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill row".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch).unwrap();

        // grid() should return updated state
        let updated_grid = artifact.grid();
        assert_eq!(updated_grid[0][1], Some(2)); // Now filled
        assert_ne!(initial_grid[0][1], updated_grid[0][1]);
    }

    #[test]
    fn test_patch_preserves_other_rows() {
        // Ensure patching one row doesn't affect others
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        let row1_before = artifact.grid()[1].clone();
        let row2_before = artifact.grid()[2].clone();
        let row3_before = artifact.grid()[3].clone();

        // Patch only row 0
        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill row 0".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch).unwrap();

        // Other rows unchanged
        assert_eq!(artifact.grid()[1], row1_before);
        assert_eq!(artifact.grid()[2], row2_before);
        assert_eq!(artifact.grid()[3], row3_before);
    }

    #[test]
    fn test_column_availability_updates_after_patch() {
        // Column availability should reflect current grid state
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        // Initial: column 1 has only value 2 (in row 1) and 4 (in row 3)
        // So for row 0, available values in col 1 are: 1, 3
        let avail_before = artifact.column_availability(0);
        assert!(avail_before[&1].contains(&1));
        assert!(avail_before[&1].contains(&3));

        // Patch row 0 to put 2 in column 1
        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill row".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch).unwrap();

        // Now for row 2, column 1 should NOT have 2 available (row 0 and row 1 both have 2)
        let avail_after = artifact.column_availability(2);
        assert!(
            !avail_after[&1].contains(&2),
            "2 should not be available in col 1 for row 2"
        );
    }

    // =========================================================================
    // Solve Detection Tests - Critical for experiment validity
    // =========================================================================

    #[test]
    fn test_is_solved_detects_row_duplicate() {
        // A grid that looks complete but has a row duplicate
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![Some(2), Some(2), Some(4), Some(3)], // Duplicate 2 in row
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert!(!artifact.is_solved(), "Should detect row duplicate");
    }

    #[test]
    fn test_is_solved_detects_column_duplicate() {
        // A grid that looks complete but has a column duplicate
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![Some(1), Some(3), Some(4), Some(2)], // Column 0 has duplicate 1
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(1), Some(2), Some(3)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert!(!artifact.is_solved(), "Should detect column duplicate");
    }

    #[test]
    fn test_is_solved_detects_out_of_range_value() {
        // A grid with a value outside valid range
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(5)], // 5 is out of range for 4x4
            vec![Some(2), Some(1), Some(4), Some(3)],
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert!(!artifact.is_solved(), "Should detect out-of-range value");
    }

    #[test]
    fn test_is_solved_detects_zero_value() {
        // A grid with zero (invalid)
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(0)], // 0 is invalid
            vec![Some(2), Some(1), Some(4), Some(3)],
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert!(!artifact.is_solved(), "Should detect zero value");
    }

    #[test]
    fn test_is_solved_valid_5x5() {
        // Valid 5x5 Latin square
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
            vec![Some(2), Some(3), Some(4), Some(5), Some(1)],
            vec![Some(3), Some(4), Some(5), Some(1), Some(2)],
            vec![Some(4), Some(5), Some(1), Some(2), Some(3)],
            vec![Some(5), Some(1), Some(2), Some(3), Some(4)],
        ];
        let artifact = LatinSquareArtifact::new(5, grid, "test").unwrap();
        assert!(artifact.is_solved(), "Valid 5x5 should be solved");
    }

    #[test]
    fn test_is_solved_valid_7x7() {
        // Valid 7x7 Latin square (the size used in experiments)
        let grid = vec![
            vec![
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
            ],
            vec![
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(1),
            ],
            vec![
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(1),
                Some(2),
            ],
            vec![
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(1),
                Some(2),
                Some(3),
            ],
            vec![
                Some(5),
                Some(6),
                Some(7),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
            ],
            vec![
                Some(6),
                Some(7),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
            ],
            vec![
                Some(7),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
            ],
        ];
        let artifact = LatinSquareArtifact::new(7, grid, "test").unwrap();
        assert!(artifact.is_solved(), "Valid 7x7 should be solved");
    }

    // =========================================================================
    // Violation Counting Tests - Used for pressure calculation
    // =========================================================================

    #[test]
    fn test_total_violations_empty_grid() {
        let grid = vec![
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert_eq!(
            artifact.total_violations(),
            0,
            "Empty grid has no violations"
        );
    }

    #[test]
    fn test_total_violations_counts_row_duplicates() {
        let grid = vec![
            vec![Some(1), Some(1), None, None], // One duplicate
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert_eq!(
            artifact.total_violations(),
            1,
            "Should count one row violation"
        );
    }

    #[test]
    fn test_total_violations_counts_column_duplicates() {
        let grid = vec![
            vec![Some(1), None, None, None],
            vec![Some(1), None, None, None], // Column duplicate
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert_eq!(
            artifact.total_violations(),
            1,
            "Should count one column violation"
        );
    }

    #[test]
    fn test_total_violations_multiple() {
        let grid = vec![
            vec![Some(1), Some(1), None, None], // Row duplicate
            vec![Some(1), None, None, None],    // Column duplicate with row 0
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        // Row 0 has 1 duplicate, Column 0 has 1 duplicate
        assert_eq!(
            artifact.total_violations(),
            2,
            "Should count row and column violations"
        );
    }

    #[test]
    fn test_empty_count() {
        let artifact = sample_puzzle();
        // sample_puzzle has 8 empty cells out of 16
        assert_eq!(artifact.empty_count(), 8, "Should count 8 empty cells");
    }

    #[test]
    fn test_empty_count_full_grid() {
        let grid = vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![Some(2), Some(1), Some(4), Some(3)],
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ];
        let artifact = LatinSquareArtifact::new(4, grid, "test").unwrap();
        assert_eq!(artifact.empty_count(), 0, "Full grid has no empty cells");
    }

    // =========================================================================
    // Region ID Stability Tests - Critical for patch routing
    // =========================================================================

    #[test]
    fn test_region_ids_stable_across_patches() {
        let mut artifact = sample_puzzle();
        let regions_before = artifact.region_ids();

        // Apply a patch
        let patch = Patch {
            region: regions_before[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "test".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch).unwrap();

        let regions_after = artifact.region_ids();
        assert_eq!(
            regions_before, regions_after,
            "Region IDs should be stable after patch"
        );
    }

    #[test]
    fn test_region_ids_deterministic() {
        // Same puzzle should produce same region IDs
        let artifact1 = sample_puzzle();
        let artifact2 = sample_puzzle();
        assert_eq!(
            artifact1.region_ids(),
            artifact2.region_ids(),
            "Same puzzle should have same region IDs"
        );
    }

    #[test]
    fn test_region_count_matches_grid_size() {
        for n in 4..=7 {
            let grid = vec![vec![None; n]; n];
            let artifact = LatinSquareArtifact::new(n, grid, "test").unwrap();
            assert_eq!(
                artifact.region_ids().len(),
                n,
                "Should have {} regions for {}x{} grid",
                n,
                n,
                n
            );
        }
    }
}
