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
#[derive(Clone)]
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
    use mti::prelude::*;
    use survival_kernel::artifact::Artifact;
    use survival_kernel::region::RegionId;
    use uuid::Uuid;

    fn test_region_id() -> RegionId {
        let v5_uuid = Uuid::new_v5(&Uuid::NAMESPACE_DNS, b"test-region");
        let prefix = TypeIdPrefix::try_from("test").expect("test is valid prefix");
        let suffix = TypeIdSuffix::from(v5_uuid);
        MagicTypeId::new(prefix, suffix)
    }

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
            id: test_region_id(),
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
            id: test_region_id(),
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
            id: test_region_id(),
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
            id: test_region_id(),
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

    /// Test that update_shared_grid correctly synchronizes the sensor's view.
    ///
    /// This test validates the fix for the bug where shared_grid was never
    /// updated in the kernel path, causing sensors to validate patches against
    /// stale grid state and incorrectly detect column conflicts.
    #[test]
    fn test_shared_grid_update_affects_column_conflict_detection() {
        // Initial grid state: row 0 has value 2 in column 1
        let shared_grid = Arc::new(RwLock::new(vec![
            vec![Some(1), Some(2), None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ]));
        let sensor = LatinSquareSensor::new(4, shared_grid.clone());

        // If row 1 also puts 2 in column 1, we have a column conflict
        let view_row1_with_conflict = RegionView {
            id: test_region_id(),
            kind: "row".to_string(),
            content: "3 2 _ _".to_string(), // 2 in column 1 conflicts with row 0
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };

        let signals = sensor.measure(&view_row1_with_conflict).unwrap();
        assert_eq!(signals["col_conflicts"], 1.0, "Should detect conflict with row 0's column 1");

        // Now simulate a patch that changes row 0's column 1 from 2 to 4
        // After update_shared_grid, the conflict should no longer exist
        let new_grid_state = vec![
            vec![Some(1), Some(4), None, None], // Changed: column 1 is now 4
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        update_shared_grid(&shared_grid, &new_grid_state).unwrap();

        // Same row 1 content should now have no column conflict
        let signals = sensor.measure(&view_row1_with_conflict).unwrap();
        assert_eq!(
            signals["col_conflicts"], 0.0,
            "After shared_grid update, conflict should be resolved"
        );
    }

    /// Test that demonstrates the bug: if shared_grid is NOT updated,
    /// the sensor sees stale state and makes wrong decisions.
    #[test]
    fn test_stale_shared_grid_causes_incorrect_validation() {
        // Grid starts with row 0 having 1,2,3,4
        let shared_grid = Arc::new(RwLock::new(vec![
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ]));
        let sensor = LatinSquareSensor::new(4, shared_grid.clone());

        // Row 1 proposes 2,1,4,3 - this should be valid (no conflicts)
        let view_row1 = RegionView {
            id: test_region_id(),
            kind: "row".to_string(),
            content: "2 1 4 3".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };

        let signals = sensor.measure(&view_row1).unwrap();
        assert_eq!(signals["col_conflicts"], 0.0, "Row 1's 2,1,4,3 has no conflicts with row 0");

        // Suppose the kernel applied a patch to row 0, changing it to 2,1,3,4
        // But we DO NOT call update_shared_grid (simulating the bug)
        // The sensor still thinks row 0 is 1,2,3,4

        // If row 1 now proposes 1,2,4,3, sensor incorrectly sees no conflict
        // because it checks against stale 1,2,3,4 instead of actual 2,1,3,4
        let view_row1_bug = RegionView {
            id: test_region_id(),
            kind: "row".to_string(),
            content: "1 2 4 3".to_string(), // Should conflict with actual row 0: 2,1,3,4
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };

        // With stale grid (1,2,3,4 in row 0), this row (1,2,4,3) appears to have conflicts
        let signals = sensor.measure(&view_row1_bug).unwrap();
        assert_eq!(
            signals["col_conflicts"], 2.0,
            "Against stale row 0 (1,2,3,4), row 1 (1,2,4,3) has 2 conflicts at columns 0 and 1"
        );

        // Now properly update shared_grid to reflect actual row 0 state
        let actual_grid_state = vec![
            vec![Some(2), Some(1), Some(3), Some(4)], // Row 0 was actually changed to 2,1,3,4
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        update_shared_grid(&shared_grid, &actual_grid_state).unwrap();

        // Now sensor correctly detects different conflicts
        let signals = sensor.measure(&view_row1_bug).unwrap();
        // Row 1 (1,2,4,3) vs actual row 0 (2,1,3,4):
        // Col 0: row0=2, row1=1 - no conflict
        // Col 1: row0=1, row1=2 - no conflict
        // Col 2: row0=3, row1=4 - no conflict
        // Col 3: row0=4, row1=3 - no conflict
        assert_eq!(
            signals["col_conflicts"], 0.0,
            "Against actual row 0 (2,1,3,4), row 1 (1,2,4,3) has no conflicts"
        );
    }

    /// Integration test: artifact patch → update_shared_grid → sensor measurement
    /// This tests the exact flow that was broken in the kernel path.
    #[test]
    fn test_artifact_patch_to_sensor_integration() {
        use crate::artifact::LatinSquareArtifact;
        use survival_kernel::region::{Patch, PatchOp};

        // Create artifact with initial state
        let grid = vec![
            vec![Some(1), None, Some(3), None],
            vec![None, Some(2), None, Some(4)],
            vec![Some(3), None, Some(1), None],
            vec![None, Some(4), None, Some(2)],
        ];
        let mut artifact = LatinSquareArtifact::new(4, grid.clone(), "test").unwrap();
        let regions = artifact.region_ids();

        // Create shared_grid with initial state (like kernel path does)
        let shared_grid: SharedGrid = Arc::new(RwLock::new(artifact.grid().clone()));
        let sensor = LatinSquareSensor::new(4, shared_grid.clone());

        // Row 1 proposes "3 2 1 4" - check conflicts against INITIAL state
        let view_row1 = RegionView {
            id: regions[1].clone(),
            kind: "row".to_string(),
            content: "3 2 1 4".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };

        let signals_before = sensor.measure(&view_row1).unwrap();
        // Against initial grid: row 0 has (1, _, 3, _), row 2 has (3, _, 1, _)
        // Row 1 proposes (3, 2, 1, 4):
        //   Col 0: 3 conflicts with row 2's 3
        //   Col 2: 1 conflicts with row 2's 1
        assert_eq!(signals_before["col_conflicts"], 2.0, "Initial state should show 2 conflicts");

        // Now apply a patch to row 0 via the artifact
        let patch = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 4 3 2".to_string()), // Changed col 1 from _ to 4, col 3 from _ to 2
            rationale: "Fill row 0".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch).unwrap();

        // KEY STEP: Update shared_grid from artifact (this was missing in buggy code!)
        update_shared_grid(&shared_grid, artifact.grid()).unwrap();

        // Now measure again - sensor should see updated state
        let signals_after = sensor.measure(&view_row1).unwrap();
        // Row 0 is now (1, 4, 3, 2)
        // Row 1 proposes (3, 2, 1, 4)
        // Col 0: 3 still conflicts with row 2's 3
        // Col 2: 1 still conflicts with row 2's 1
        // But importantly, the sensor is seeing CURRENT state, not stale state
        assert_eq!(signals_after["col_conflicts"], 2.0, "After update, still 2 conflicts");
    }

    /// Test multiple sequential patches with shared_grid sync
    /// Simulates what happens during multiple kernel ticks
    #[test]
    fn test_multiple_patches_with_shared_grid_sync() {
        use crate::artifact::LatinSquareArtifact;
        use survival_kernel::region::{Patch, PatchOp};

        // Start with empty 4x4 grid except for a few fixed cells
        let grid = vec![
            vec![Some(1), None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let mut artifact = LatinSquareArtifact::new(4, grid.clone(), "test").unwrap();
        let regions = artifact.region_ids();

        let shared_grid: SharedGrid = Arc::new(RwLock::new(artifact.grid().clone()));
        let sensor = LatinSquareSensor::new(4, shared_grid.clone());

        // Tick 1: Patch row 0 to "1 2 3 4"
        let patch0 = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch0).unwrap();
        update_shared_grid(&shared_grid, artifact.grid()).unwrap();

        // Verify row 1 sees row 0's values when checking conflicts
        let view_row1_attempt1 = RegionView {
            id: regions[1].clone(),
            kind: "row".to_string(),
            content: "1 2 3 4".to_string(), // Same as row 0 - should have 4 conflicts!
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };
        let signals = sensor.measure(&view_row1_attempt1).unwrap();
        assert_eq!(signals["col_conflicts"], 4.0, "Row 1 duplicating row 0 should have 4 col conflicts");

        // Tick 2: Patch row 1 to "2 1 4 3" (valid, no conflicts)
        let patch1 = Patch {
            region: regions[1].clone(),
            op: PatchOp::Replace("2 1 4 3".to_string()),
            rationale: "".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch1).unwrap();
        update_shared_grid(&shared_grid, artifact.grid()).unwrap();

        // Verify row 2 sees both row 0 and row 1
        let view_row2_attempt = RegionView {
            id: regions[2].clone(),
            kind: "row".to_string(),
            content: "1 2 3 4".to_string(), // Duplicates row 0 entirely
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(2));
                m
            },
        };
        let signals = sensor.measure(&view_row2_attempt).unwrap();
        assert_eq!(signals["col_conflicts"], 4.0, "Row 2 duplicating row 0 should have 4 conflicts");

        // A valid row 2 proposal
        let view_row2_valid = RegionView {
            id: regions[2].clone(),
            kind: "row".to_string(),
            content: "3 4 1 2".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(2));
                m
            },
        };
        let signals = sensor.measure(&view_row2_valid).unwrap();
        assert_eq!(signals["col_conflicts"], 0.0, "Valid row 2 should have no conflicts");
    }

    /// Test that demonstrates the exact bug scenario:
    /// Without shared_grid updates, later patches are validated against stale state
    #[test]
    fn test_bug_scenario_stale_validation() {
        use crate::artifact::LatinSquareArtifact;
        use survival_kernel::region::{Patch, PatchOp};

        let grid = vec![
            vec![Some(1), None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ];
        let mut artifact = LatinSquareArtifact::new(4, grid.clone(), "test").unwrap();
        let regions = artifact.region_ids();

        // Create TWO shared_grids: one that gets updated (fixed), one that doesn't (buggy)
        let shared_grid_fixed: SharedGrid = Arc::new(RwLock::new(artifact.grid().clone()));
        let shared_grid_buggy: SharedGrid = Arc::new(RwLock::new(artifact.grid().clone()));

        let sensor_fixed = LatinSquareSensor::new(4, shared_grid_fixed.clone());
        let sensor_buggy = LatinSquareSensor::new(4, shared_grid_buggy.clone());

        // Apply patch to row 0
        let patch0 = Patch {
            region: regions[0].clone(),
            op: PatchOp::Replace("1 2 3 4".to_string()),
            rationale: "".to_string(),
            expected_delta: HashMap::new(),
        };
        artifact.apply_patch(patch0).unwrap();

        // FIXED path: update shared_grid
        update_shared_grid(&shared_grid_fixed, artifact.grid()).unwrap();
        // BUGGY path: DON'T update shared_grid (simulating the bug)

        // Now propose row 1 = "1 2 3 4" (same as row 0 - should be rejected)
        let view_row1 = RegionView {
            id: regions[1].clone(),
            kind: "row".to_string(),
            content: "1 2 3 4".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(1));
                m
            },
        };

        // Fixed sensor correctly detects 4 column conflicts
        let signals_fixed = sensor_fixed.measure(&view_row1).unwrap();
        assert_eq!(
            signals_fixed["col_conflicts"], 4.0,
            "Fixed sensor should see 4 conflicts (row 0 has 1,2,3,4)"
        );

        // Buggy sensor sees NO conflicts because it still thinks row 0 is "1 _ _ _"
        let signals_buggy = sensor_buggy.measure(&view_row1).unwrap();
        assert_eq!(
            signals_buggy["col_conflicts"], 1.0,
            "Buggy sensor only sees 1 conflict (stale row 0 only has 1 in col 0)"
        );
    }

    /// Test that simulates the kernel tick cycle:
    /// 1. Get TickResult with applied patches
    /// 2. Apply patches to local artifact
    /// 3. Update shared_grid
    /// 4. Next tick's sensor measurements see updated state
    #[test]
    fn test_simulated_kernel_tick_cycle() {
        use crate::artifact::LatinSquareArtifact;
        use survival_kernel::region::{Patch, PatchOp};

        // Initial 4x4 grid
        let grid = vec![
            vec![Some(1), None, None, None],
            vec![None, Some(2), None, None],
            vec![None, None, Some(3), None],
            vec![None, None, None, Some(4)],
        ];
        let mut artifact = LatinSquareArtifact::new(4, grid.clone(), "test").unwrap();
        let regions = artifact.region_ids();

        let shared_grid: SharedGrid = Arc::new(RwLock::new(artifact.grid().clone()));
        let sensor = LatinSquareSensor::new(4, shared_grid.clone());

        // Simulate 3 ticks, each applying one patch
        let patches = vec![
            (regions[0].clone(), "1 2 3 4"),
            (regions[1].clone(), "3 2 4 1"),
            (regions[2].clone(), "4 1 3 2"),
        ];

        for (tick, (region, content)) in patches.iter().enumerate() {
            // Simulate TickResult.applied containing one patch
            let patch = Patch {
                region: region.clone(),
                op: PatchOp::Replace(content.to_string()),
                rationale: format!("Tick {}", tick),
                expected_delta: HashMap::new(),
            };

            // Apply patch (like kernel does internally)
            artifact.apply_patch(patch).unwrap();

            // THE FIX: Update shared_grid after applying patches
            update_shared_grid(&shared_grid, artifact.grid()).unwrap();

            // Verify sensor sees current state by checking row 3
            let view_row3 = RegionView {
                id: regions[3].clone(),
                kind: "row".to_string(),
                content: "2 4 1 4".to_string(), // Some test content
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("row_index".to_string(), serde_json::json!(3));
                    m
                },
            };
            let _signals = sensor.measure(&view_row3).unwrap();
            // Just verify it doesn't crash and sensor is working
        }

        // After all patches, verify final state
        assert_eq!(artifact.grid()[0], vec![Some(1), Some(2), Some(3), Some(4)]);
        assert_eq!(artifact.grid()[1], vec![Some(3), Some(2), Some(4), Some(1)]);
        assert_eq!(artifact.grid()[2], vec![Some(4), Some(1), Some(3), Some(2)]);

        // Final check: row 3 proposing "2 4 1 3" should have 0 conflicts
        let view_row3_valid = RegionView {
            id: regions[3].clone(),
            kind: "row".to_string(),
            content: "2 4 1 3".to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("row_index".to_string(), serde_json::json!(3));
                m
            },
        };
        let signals = sensor.measure(&view_row3_valid).unwrap();
        assert_eq!(signals["col_conflicts"], 0.0, "Valid completion should have no conflicts");
        assert_eq!(signals["row_duplicates"], 0.0, "Valid completion should have no duplicates");
    }
}
