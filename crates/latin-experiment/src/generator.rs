//! Latin Square puzzle generator with controlled difficulty.
//!
//! Generates valid Latin squares and removes cells to create puzzles.
//! Difficulty is controlled by the number of empty cells.

use anyhow::Result;
use rand::prelude::*;
use uuid::Uuid;

use crate::artifact::LatinSquareArtifact;

/// Configuration for puzzle generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Grid size (n x n)
    pub n: usize,
    /// Number of cells to leave empty
    pub empty_cells: usize,
    /// Random seed for reproducibility (None for random)
    pub seed: Option<u64>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            n: 6,
            empty_cells: 15,
            seed: None,
        }
    }
}

/// Puzzle generator for Latin squares.
pub struct LatinSquareGenerator {
    config: GeneratorConfig,
}

impl LatinSquareGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a random valid Latin square.
    fn generate_valid_square(&self, rng: &mut impl Rng) -> Vec<Vec<u8>> {
        let n = self.config.n;

        // Start with a simple valid Latin square (cyclic permutation)
        let mut grid: Vec<Vec<u8>> = (0..n)
            .map(|row| (0..n).map(|col| ((row + col) % n + 1) as u8).collect())
            .collect();

        // Shuffle rows within blocks to add randomness
        grid.shuffle(rng);

        // Shuffle columns by transposing, shuffling rows, transposing back
        let mut transposed: Vec<Vec<u8>> = (0..n)
            .map(|col| (0..n).map(|row| grid[row][col]).collect())
            .collect();
        transposed.shuffle(rng);
        grid = (0..n)
            .map(|row| (0..n).map(|col| transposed[col][row]).collect())
            .collect();

        // Permute symbols
        let mut symbols: Vec<u8> = (1..=n as u8).collect();
        symbols.shuffle(rng);
        for row in &mut grid {
            for cell in row {
                *cell = symbols[(*cell - 1) as usize];
            }
        }

        grid
    }

    /// Generate a puzzle by removing cells from a valid square.
    pub fn generate(&self) -> Result<LatinSquareArtifact> {
        let mut rng: Box<dyn RngCore> = match self.config.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::rng()),
        };

        let n = self.config.n;
        let valid_square = self.generate_valid_square(&mut rng);

        // Convert to Option<u8> grid
        let mut grid: Vec<Vec<Option<u8>>> = valid_square
            .into_iter()
            .map(|row| row.into_iter().map(Some).collect())
            .collect();

        // Collect all cell positions
        let mut positions: Vec<(usize, usize)> = (0..n)
            .flat_map(|row| (0..n).map(move |col| (row, col)))
            .collect();
        positions.shuffle(&mut rng);

        // Remove cells to create the puzzle
        let cells_to_remove = self.config.empty_cells.min(n * n);
        for (row, col) in positions.into_iter().take(cells_to_remove) {
            grid[row][col] = None;
        }

        // Generate a unique puzzle ID
        let puzzle_id = Uuid::new_v4().to_string();

        LatinSquareArtifact::new(n, grid, puzzle_id)
    }

    /// Generate multiple puzzles.
    pub fn generate_batch(&self, count: usize) -> Result<Vec<LatinSquareArtifact>> {
        (0..count).map(|_| self.generate()).collect()
    }
}

/// Difficulty presets for experiments.
#[derive(Debug, Clone, Copy)]
pub enum Difficulty {
    /// Easy: 4x4 with 4 empty cells
    Easy,
    /// Medium: 6x6 with 12 empty cells
    Medium,
    /// Hard: 6x6 with 18 empty cells
    Hard,
    /// VeryHard: 8x8 with 24 empty cells
    VeryHard,
    /// Custom difficulty
    Custom { n: usize, empty_cells: usize },
}

impl Difficulty {
    /// Get the generator config for this difficulty.
    pub fn config(self) -> GeneratorConfig {
        match self {
            Difficulty::Easy => GeneratorConfig {
                n: 4,
                empty_cells: 4,
                seed: None,
            },
            Difficulty::Medium => GeneratorConfig {
                n: 6,
                empty_cells: 12,
                seed: None,
            },
            Difficulty::Hard => GeneratorConfig {
                n: 6,
                empty_cells: 18,
                seed: None,
            },
            Difficulty::VeryHard => GeneratorConfig {
                n: 8,
                empty_cells: 24,
                seed: None,
            },
            Difficulty::Custom { n, empty_cells } => GeneratorConfig {
                n,
                empty_cells,
                seed: None,
            },
        }
    }

    /// Create a generator for this difficulty.
    pub fn generator(self) -> LatinSquareGenerator {
        LatinSquareGenerator::new(self.config())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_valid_square() {
        let config = GeneratorConfig {
            n: 4,
            empty_cells: 0,
            seed: Some(42),
        };
        let generator = LatinSquareGenerator::new(config);
        let artifact = generator.generate().unwrap();

        // Should be complete (no empty cells)
        assert_eq!(artifact.empty_count(), 0);

        // Should be valid (no violations)
        assert!(artifact.is_solved());
    }

    #[test]
    fn test_generate_with_holes() {
        let config = GeneratorConfig {
            n: 6,
            empty_cells: 12,
            seed: Some(42),
        };
        let generator = LatinSquareGenerator::new(config);
        let artifact = generator.generate().unwrap();

        // Should have exactly 12 empty cells
        assert_eq!(artifact.empty_count(), 12);

        // Should not be solved (has empty cells)
        assert!(!artifact.is_solved());

        // But should have no violations (the filled cells are valid)
        assert_eq!(artifact.total_violations(), 0);
    }

    #[test]
    fn test_reproducible_with_seed() {
        let config = GeneratorConfig {
            n: 4,
            empty_cells: 4,
            seed: Some(12345),
        };

        let gen1 = LatinSquareGenerator::new(config.clone());
        let gen2 = LatinSquareGenerator::new(config);

        let artifact1 = gen1.generate().unwrap();
        let artifact2 = gen2.generate().unwrap();

        // Grids should be identical (same seed)
        assert_eq!(artifact1.grid(), artifact2.grid());
    }

    #[test]
    fn test_difficulty_presets() {
        for difficulty in [
            Difficulty::Easy,
            Difficulty::Medium,
            Difficulty::Hard,
            Difficulty::VeryHard,
        ] {
            let generator = difficulty.generator();
            let artifact = generator.generate().unwrap();

            // Should generate valid puzzles
            assert!(artifact.empty_count() > 0);
            assert_eq!(artifact.total_violations(), 0);
        }
    }
}
