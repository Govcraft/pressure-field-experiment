//! ExampleBank: Pheromone-like few-shot example accumulation system.
//!
//! Implements the swarm intelligence mechanism where:
//! - Successful patches are added as examples (pheromone deposit)
//! - Example weights decay over time (evaporation)
//! - Weights are reinforced when examples lead to success (positive feedback)
//! - Examples are selected by weight for few-shot prompting (trail following)

use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A validated example that can be used for few-shot prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Unique identifier for this example
    pub id: Uuid,
    /// Tick when this example was created
    pub created_tick: usize,
    /// Tick when this example was last used
    pub last_used_tick: usize,
    /// Number of times this example has been used
    pub use_count: usize,
    /// Original row content before the patch
    pub before: String,
    /// Row content after the successful patch
    pub after: String,
    /// Pressure before the patch
    pub pressure_before: f64,
    /// Pressure after the patch
    pub pressure_after: f64,
    /// Pheromone weight (starts at 1.0, decays over time, reinforced on use)
    pub weight: f64,
    /// Row index context (for relevance matching)
    pub row_context: Option<usize>,
    /// Empty cell positions that were filled
    pub filled_positions: Vec<usize>,
}

impl Example {
    /// Create a new example from a successful patch.
    pub fn new(
        tick: usize,
        before: String,
        after: String,
        pressure_before: f64,
        pressure_after: f64,
    ) -> Self {
        // Identify which positions were filled
        let before_parts: Vec<&str> = before.split_whitespace().collect();
        let after_parts: Vec<&str> = after.split_whitespace().collect();
        let filled_positions: Vec<usize> = before_parts
            .iter()
            .zip(after_parts.iter())
            .enumerate()
            .filter(|(_, (b, a))| *b == &"_" && *a != &"_")
            .map(|(i, _)| i)
            .collect();

        Self {
            id: Uuid::new_v4(),
            created_tick: tick,
            last_used_tick: tick,
            use_count: 0,
            before,
            after,
            pressure_before,
            pressure_after,
            weight: 1.0,
            row_context: None,
            filled_positions,
        }
    }

    /// Get the pressure reduction achieved by this example.
    pub fn pressure_delta(&self) -> f64 {
        self.pressure_before - self.pressure_after
    }

    /// Format this example for inclusion in an LLM prompt.
    pub fn format_for_prompt(&self) -> String {
        format!(
            "Example: \"{}\" -> \"{}\" (reduced pressure by {:.1})",
            self.before,
            self.after,
            self.pressure_delta()
        )
    }
}

/// Configuration for the example bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleBankConfig {
    /// Maximum number of examples to keep
    pub max_examples: usize,
    /// Decay factor applied each tick (e.g., 0.95 = 5% decay per tick)
    pub decay_factor: f64,
    /// Reinforcement added when an example is used successfully
    pub reinforcement: f64,
    /// Minimum weight before example is evicted
    pub eviction_threshold: f64,
    /// Number of examples to include in prompts
    pub prompt_examples: usize,
    /// Whether to enable the example bank (for ablation studies)
    pub enabled: bool,
}

impl Default for ExampleBankConfig {
    fn default() -> Self {
        Self {
            max_examples: 50,
            decay_factor: 0.95,
            reinforcement: 0.3,
            eviction_threshold: 0.1,
            prompt_examples: 3,
            enabled: true,
        }
    }
}

/// The example bank: a pheromone-based few-shot learning system.
///
/// Thread-safe via RwLock for concurrent access during experiments.
#[derive(Debug)]
pub struct ExampleBank {
    /// Configuration
    config: ExampleBankConfig,
    /// All examples, sorted by weight (highest first)
    examples: Arc<RwLock<Vec<Example>>>,
    /// Current tick (for decay calculations)
    current_tick: Arc<RwLock<usize>>,
}

impl ExampleBank {
    /// Create a new empty example bank.
    pub fn new(config: ExampleBankConfig) -> Self {
        Self {
            config,
            examples: Arc::new(RwLock::new(Vec::new())),
            current_tick: Arc::new(RwLock::new(0)),
        }
    }

    /// Create a disabled example bank (for ablation studies).
    pub fn disabled() -> Self {
        Self::new(ExampleBankConfig {
            enabled: false,
            ..Default::default()
        })
    }

    /// Check if the bank is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Add a new example from a successful patch.
    pub fn add_example(
        &self,
        before: String,
        after: String,
        pressure_before: f64,
        pressure_after: f64,
    ) {
        if !self.config.enabled {
            return;
        }

        let tick = *self.current_tick.read().unwrap();
        let example = Example::new(tick, before, after, pressure_before, pressure_after);

        let mut examples = self.examples.write().unwrap();

        // Add new example
        examples.push(example);

        // Sort by weight (highest first)
        examples.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

        // Evict if over capacity
        if examples.len() > self.config.max_examples {
            examples.truncate(self.config.max_examples);
        }
    }

    /// Apply decay to all examples (pheromone evaporation).
    pub fn apply_decay(&self) {
        if !self.config.enabled {
            return;
        }

        let mut examples = self.examples.write().unwrap();

        for example in examples.iter_mut() {
            example.weight *= self.config.decay_factor;
        }

        // Evict examples below threshold
        examples.retain(|e| e.weight >= self.config.eviction_threshold);

        // Update tick counter
        let mut tick = self.current_tick.write().unwrap();
        *tick += 1;
    }

    /// Reinforce an example that led to a successful outcome.
    pub fn reinforce(&self, example_id: Uuid) {
        if !self.config.enabled {
            return;
        }

        let tick = *self.current_tick.read().unwrap();
        let mut examples = self.examples.write().unwrap();

        if let Some(example) = examples.iter_mut().find(|e| e.id == example_id) {
            example.weight += self.config.reinforcement;
            example.weight = example.weight.min(2.0); // Cap at 2x initial weight
            example.last_used_tick = tick;
            example.use_count += 1;
        }

        // Re-sort by weight
        examples.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get the top examples for few-shot prompting.
    ///
    /// Returns examples sorted by weight, limited to `prompt_examples`.
    pub fn get_examples_for_prompt(&self) -> Vec<Example> {
        if !self.config.enabled {
            return Vec::new();
        }

        let examples = self.examples.read().unwrap();
        examples
            .iter()
            .take(self.config.prompt_examples)
            .cloned()
            .collect()
    }

    /// Get examples relevant to a specific pattern.
    ///
    /// Prioritizes examples that filled similar positions or have similar structure.
    pub fn get_relevant_examples(&self, empty_positions: &[usize], n: usize) -> Vec<Example> {
        if !self.config.enabled {
            return Vec::new();
        }

        let examples = self.examples.read().unwrap();

        // Score each example by relevance
        let mut scored: Vec<(f64, &Example)> = examples
            .iter()
            .map(|e| {
                // Relevance = weight * position_overlap
                let overlap = e
                    .filled_positions
                    .iter()
                    .filter(|p| empty_positions.contains(p))
                    .count();
                let position_score = if !e.filled_positions.is_empty() {
                    overlap as f64 / e.filled_positions.len() as f64
                } else {
                    0.0
                };
                let relevance = e.weight * (0.5 + 0.5 * position_score);
                (relevance, e)
            })
            .collect();

        // Sort by relevance (highest first)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(n)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Get current statistics for logging.
    pub fn stats(&self) -> ExampleBankStats {
        let examples = self.examples.read().unwrap();
        let tick = *self.current_tick.read().unwrap();

        ExampleBankStats {
            total_examples: examples.len(),
            total_weight: examples.iter().map(|e| e.weight).sum(),
            avg_weight: if examples.is_empty() {
                0.0
            } else {
                examples.iter().map(|e| e.weight).sum::<f64>() / examples.len() as f64
            },
            max_weight: examples.iter().map(|e| e.weight).fold(0.0, f64::max),
            current_tick: tick,
            total_uses: examples.iter().map(|e| e.use_count).sum(),
        }
    }

    /// Reset the example bank (for new experiments).
    pub fn reset(&self) {
        let mut examples = self.examples.write().unwrap();
        let mut tick = self.current_tick.write().unwrap();
        examples.clear();
        *tick = 0;
    }
}

/// Statistics about the example bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleBankStats {
    pub total_examples: usize,
    pub total_weight: f64,
    pub avg_weight: f64,
    pub max_weight: f64,
    pub current_tick: usize,
    pub total_uses: usize,
}

impl Clone for ExampleBank {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            examples: Arc::new(RwLock::new(self.examples.read().unwrap().clone())),
            current_tick: Arc::new(RwLock::new(*self.current_tick.read().unwrap())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_example() {
        let bank = ExampleBank::new(ExampleBankConfig::default());

        bank.add_example(
            "1 _ 3 _".to_string(),
            "1 2 3 4".to_string(),
            4.0,
            0.0,
        );

        let examples = bank.get_examples_for_prompt();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].before, "1 _ 3 _");
        assert_eq!(examples[0].after, "1 2 3 4");
        assert_eq!(examples[0].pressure_delta(), 4.0);
    }

    #[test]
    fn test_decay() {
        let bank = ExampleBank::new(ExampleBankConfig {
            decay_factor: 0.5, // 50% decay per tick for testing
            ..Default::default()
        });

        bank.add_example("_ _".to_string(), "1 2".to_string(), 2.0, 0.0);

        // Initial weight is 1.0
        let examples = bank.get_examples_for_prompt();
        assert!((examples[0].weight - 1.0).abs() < 0.001);

        // After decay, weight should be 0.5
        bank.apply_decay();
        let examples = bank.get_examples_for_prompt();
        assert!((examples[0].weight - 0.5).abs() < 0.001);

        // After another decay, weight should be 0.25
        bank.apply_decay();
        let examples = bank.get_examples_for_prompt();
        assert!((examples[0].weight - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_eviction() {
        let bank = ExampleBank::new(ExampleBankConfig {
            decay_factor: 0.3,
            eviction_threshold: 0.2,
            ..Default::default()
        });

        bank.add_example("_ _".to_string(), "1 2".to_string(), 2.0, 0.0);

        // After first decay: 0.3
        bank.apply_decay();
        assert_eq!(bank.get_examples_for_prompt().len(), 1);

        // After second decay: 0.09 < 0.2 threshold
        bank.apply_decay();
        assert_eq!(bank.get_examples_for_prompt().len(), 0);
    }

    #[test]
    fn test_reinforcement() {
        let bank = ExampleBank::new(ExampleBankConfig {
            reinforcement: 0.5,
            decay_factor: 0.9,
            ..Default::default()
        });

        bank.add_example("_ _".to_string(), "1 2".to_string(), 2.0, 0.0);

        let examples = bank.get_examples_for_prompt();
        let id = examples[0].id;

        // Reinforce
        bank.reinforce(id);

        let examples = bank.get_examples_for_prompt();
        assert!((examples[0].weight - 1.5).abs() < 0.001);
        assert_eq!(examples[0].use_count, 1);
    }

    #[test]
    fn test_disabled_bank() {
        let bank = ExampleBank::disabled();

        bank.add_example("_ _".to_string(), "1 2".to_string(), 2.0, 0.0);

        assert!(bank.get_examples_for_prompt().is_empty());
    }

    #[test]
    fn test_max_examples() {
        let bank = ExampleBank::new(ExampleBankConfig {
            max_examples: 3,
            ..Default::default()
        });

        for i in 0..5 {
            bank.add_example(
                format!("_ {}", i),
                format!("1 {}", i),
                (5 - i) as f64, // Higher pressure reduction = higher weight
                0.0,
            );
        }

        // Should only keep top 3
        assert_eq!(bank.get_examples_for_prompt().len(), 3);
    }

    #[test]
    fn test_filled_positions() {
        let bank = ExampleBank::new(ExampleBankConfig::default());

        bank.add_example(
            "1 _ 3 _".to_string(),
            "1 2 3 4".to_string(),
            4.0,
            0.0,
        );

        let examples = bank.get_examples_for_prompt();
        assert_eq!(examples[0].filled_positions, vec![1, 3]);
    }
}
