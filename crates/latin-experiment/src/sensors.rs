//! LatinSquareSensor: Pure Rust validation for Latin Square constraints.
//!
//! This sensor measures quality signals without any subprocess spawning:
//! - empty_count: Number of unfilled cells in the row
//! - row_duplicates: Number of duplicate values within the row
//! - col_conflicts: Number of conflicts with other rows in same columns
//!
//! All validation is O(n) per row, making it extremely fast (~microseconds).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use survival_kernel::pressure::{Sensor, Signals};
use survival_kernel::region::RegionView;

/// Shared grid state for column conflict detection.
///
/// The sensor needs access to the full grid to detect column conflicts,
/// but the Sensor trait only receives individual regions.
pub type SharedGrid = Arc<RwLock<Vec<Vec<Option<u8>>>>>;

/// Sensor for Latin Square constraint validation.
///
/// Measures three quality signals:
/// - `empty_count`: Cells that need to be filled (lower is better)
/// - `row_duplicates`: Duplicate values in the row (0 is valid)
/// - `col_conflicts`: Values that conflict with other rows (0 is valid)
pub struct LatinSquareSensor {
    /// Grid size (n x n)
    n: usize,
    /// Shared grid state for column conflict detection
    grid: SharedGrid,
}

impl LatinSquareSensor {
    /// Create a new sensor with access to the shared grid.
    pub fn new(n: usize, grid: SharedGrid) -> Self {
        Self { n, grid }
    }

    /// Parse a row from its content string.
    fn parse_row(content: &str, n: usize) -> Vec<Option<u8>> {
        content
            .split_whitespace()
            .take(n)
            .map(|s| {
                if s == "_" {
                    None
                } else {
                    s.parse().ok()
                }
            })
            .collect()
    }

    /// Count duplicate values within a row.
    fn count_row_duplicates(row: &[Option<u8>], n: usize) -> usize {
        let mut counts = vec![0usize; n + 1];
        for v in row.iter().flatten() {
            let v = *v as usize;
            if v > 0 && v <= n {
                counts[v] += 1;
            }
        }

        counts.iter().filter(|&&c| c > 1).map(|c| c - 1).sum()
    }

    /// Count column conflicts between this row and other rows.
    fn count_column_conflicts(
        row: &[Option<u8>],
        row_idx: usize,
        grid: &[Vec<Option<u8>>],
        _n: usize,
    ) -> usize {
        let mut conflicts = 0;

        for (col, cell) in row.iter().enumerate() {
            if let Some(v) = cell {
                // Check all other rows in this column
                for (other_row_idx, other_row) in grid.iter().enumerate() {
                    if other_row_idx != row_idx
                        && let Some(other_v) = other_row.get(col).and_then(|c| *c)
                        && other_v == *v
                    {
                        conflicts += 1;
                    }
                }
            }
        }

        conflicts
    }
}

impl Sensor for LatinSquareSensor {
    fn name(&self) -> &str {
        "latin_square"
    }

    fn measure(&self, region: &RegionView) -> Result<Signals> {
        let row = Self::parse_row(&region.content, self.n);

        // Get row index from metadata
        let row_idx = region
            .metadata
            .get("row_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Count empty cells
        let empty_count = row.iter().filter(|c| c.is_none()).count() as f64;

        // Count row duplicates
        let row_duplicates = Self::count_row_duplicates(&row, self.n) as f64;

        // Count column conflicts (requires grid access)
        let col_conflicts = {
            let grid = self.grid.read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
            Self::count_column_conflicts(&row, row_idx, &grid, self.n) as f64
        };

        // Compute derived metrics
        let total_issues = row_duplicates + col_conflicts;
        let completeness = 1.0 - (empty_count / self.n as f64);

        let mut signals = HashMap::new();
        signals.insert("empty_count".to_string(), empty_count);
        signals.insert("row_duplicates".to_string(), row_duplicates);
        signals.insert("col_conflicts".to_string(), col_conflicts);
        signals.insert("total_issues".to_string(), total_issues);
        signals.insert("completeness".to_string(), completeness);

        Ok(signals)
    }
}

/// Update the shared grid state when the artifact changes.
///
/// This should be called after each successful patch application.
pub fn update_shared_grid(grid: &SharedGrid, new_state: &[Vec<Option<u8>>]) -> Result<()> {
    let mut locked = grid.write().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
    locked.clear();
    locked.extend(new_state.iter().cloned());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_grid() -> SharedGrid {
        Arc::new(RwLock::new(vec![
            vec![Some(1), None, Some(3), None],
            vec![None, Some(2), None, Some(4)],
            vec![Some(3), None, Some(1), None],
            vec![None, Some(4), None, Some(2)],
        ]))
    }

    #[test]
    fn test_count_empty() {
        let grid = create_test_grid();
        let sensor = LatinSquareSensor::new(4, grid);

        let view = RegionView {
            id: Uuid::nil(),
            kind: "row".to_string(),
            content: "1 _ 3 _".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(0));
                m
            },
        };

        let signals = sensor.measure(&view).unwrap();
        assert_eq!(signals["empty_count"], 2.0);
    }

    #[test]
    fn test_row_duplicates() {
        let grid = create_test_grid();
        let sensor = LatinSquareSensor::new(4, grid);

        let view = RegionView {
            id: Uuid::nil(),
            kind: "row".to_string(),
            content: "1 1 3 3".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(0));
                m
            },
        };

        let signals = sensor.measure(&view).unwrap();
        assert_eq!(signals["row_duplicates"], 2.0); // Two pairs of duplicates
    }

    #[test]
    fn test_column_conflicts() {
        let grid = create_test_grid();
        let sensor = LatinSquareSensor::new(4, grid);

        // Row 0 has value 1 in column 0
        // Row 2 also has value 3 in column 2, matching row 0
        let view = RegionView {
            id: Uuid::nil(),
            kind: "row".to_string(),
            content: "1 _ 3 _".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(0));
                m
            },
        };

        let signals = sensor.measure(&view).unwrap();
        // Column 2: row 0 has 3, row 2 has 1 - no conflict with 3
        // But wait, row 2 has 1 in col 2, not 3. So 3 is unique in col 2 for row 0
        // Actually 1 in col 0: row 0 has 1, row 2 has 3 - no conflict
        // So col_conflicts should be 0 for this row
        assert_eq!(signals["col_conflicts"], 0.0);
    }

    #[test]
    fn test_valid_row_no_issues() {
        let grid = Arc::new(RwLock::new(vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![Some(2), Some(1), Some(4), Some(3)],
            vec![Some(3), Some(4), Some(1), Some(2)],
            vec![Some(4), Some(3), Some(2), Some(1)],
        ]));
        let sensor = LatinSquareSensor::new(4, grid);

        let view = RegionView {
            id: Uuid::nil(),
            kind: "row".to_string(),
            content: "1 2 3 4".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(0));
                m
            },
        };

        let signals = sensor.measure(&view).unwrap();
        assert_eq!(signals["empty_count"], 0.0);
        assert_eq!(signals["row_duplicates"], 0.0);
        assert_eq!(signals["col_conflicts"], 0.0);
        assert_eq!(signals["completeness"], 1.0);
    }
}
