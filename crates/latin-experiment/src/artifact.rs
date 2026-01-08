//! LatinSquareArtifact: Artifact trait implementation for Latin Square puzzles.
//!
//! Each row is a region that can be patched independently.
//! Uses deterministic UUIDs for stable region IDs across re-parses.

use std::collections::HashMap;
use std::fmt;

use anyhow::{bail, Result};
use survival_kernel::artifact::Artifact;
use survival_kernel::region::{Patch, PatchOp, RegionId, RegionView};
use uuid::Uuid;

/// Namespace UUID for generating deterministic region IDs.
const REGION_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc9,
]);

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
            let id_string = format!("{}:row:{}", puzzle_id, row_idx);
            let region_id = Uuid::new_v5(&REGION_NAMESPACE, id_string.as_bytes());
            region_map.insert(region_id, row_idx);
            region_order.push(region_id);
        }

        Ok(Self {
            n,
            grid,
            fixed,
            region_map,
            region_order,
            puzzle_id,
        })
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
                for (col, (is_fixed, new_val)) in self.fixed[row_idx]
                    .iter()
                    .zip(new_row.iter())
                    .enumerate()
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
}

impl fmt::Display for LatinSquareArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Latin Square {}x{} (puzzle: {})", self.n, self.n, self.puzzle_id)?;
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
        let view = artifact.read_region(regions[0]).unwrap();

        assert_eq!(view.content, "1 _ 3 _");
        assert_eq!(view.kind, "row");
    }

    #[test]
    fn test_apply_patch() {
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        let patch = Patch {
            region: regions[0],
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "Fill empty cells".to_string(),
            expected_delta: HashMap::new(),
        };

        artifact.apply_patch(patch).unwrap();

        let view = artifact.read_region(regions[0]).unwrap();
        assert_eq!(view.content, "1 2 3 4");
    }

    #[test]
    fn test_cannot_change_fixed_cells() {
        let mut artifact = sample_puzzle();
        let regions = artifact.region_ids();

        // Try to change fixed cell (1 at position 0)
        let patch = Patch {
            region: regions[0],
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
}
