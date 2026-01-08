//! Results collection and output for Latin Square experiments.
//!
//! Captures metrics like:
//! - Ticks to convergence
//! - Pressure reduction per tick
//! - LLM call efficiency
//! - Example bank statistics
//! - Model escalation events

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::example_bank::ExampleBankStats;

/// Record of a model escalation event.
///
/// When the pressure-field strategy detects zero progress for multiple ticks,
/// it escalates to a larger model in the chain. This struct captures when
/// and what model transition occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationEvent {
    /// Tick at which escalation occurred
    pub tick: usize,
    /// Model being escalated from
    pub from_model: String,
    /// Model being escalated to
    pub to_model: String,
}

/// Results from a single experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Experiment configuration
    pub config: ExperimentConfig,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// End time
    pub ended_at: DateTime<Utc>,
    /// Total ticks executed
    pub total_ticks: usize,
    /// Whether the puzzle was solved
    pub solved: bool,
    /// Final pressure (0 if solved)
    pub final_pressure: f64,
    /// Pressure at each tick
    pub pressure_history: Vec<f64>,
    /// Patches applied at each tick
    pub patches_per_tick: Vec<usize>,
    /// Empty cells remaining at each tick
    pub empty_cells_history: Vec<usize>,
    /// Example bank stats at end
    pub example_bank_stats: Option<ExampleBankStats>,
    /// Per-tick metrics
    pub tick_metrics: Vec<TickMetrics>,
    /// Model escalation events (when larger models were activated)
    pub escalation_events: Vec<EscalationEvent>,
    /// Which model tier was active when solved (or last model if unsolved)
    pub final_model: String,
    /// Conversation statistics (for Conversation strategy only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_stats: Option<ConversationStats>,
}

/// Configuration for an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Strategy name
    pub strategy: String,
    /// Number of agents
    pub agent_count: usize,
    /// Grid size
    pub n: usize,
    /// Initial empty cells
    pub empty_cells: usize,
    /// Whether decay is enabled
    pub decay_enabled: bool,
    /// Whether inhibition is enabled
    pub inhibition_enabled: bool,
    /// Whether few-shot examples are enabled
    pub examples_enabled: bool,
    /// Trial number (for repeated experiments)
    pub trial: usize,
    /// Random seed (if reproducible)
    pub seed: Option<u64>,
}

/// Metrics for a single tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickMetrics {
    pub tick: usize,
    pub pressure_before: f64,
    pub pressure_after: f64,
    pub patches_proposed: usize,
    pub patches_applied: usize,
    pub empty_cells: usize,
    pub violations: usize,
    pub llm_calls: usize,
    pub duration_ms: u64,
    /// Number of conversation messages exchanged (for Conversation strategy only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages_per_tick: Option<usize>,
}

/// Statistics about conversation-based coordination (AutoGen-style baseline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    /// Total messages across all ticks
    pub total_messages: usize,
    /// Average messages per tick
    pub avg_messages_per_tick: f64,
    /// Total LLM calls (each message = 1 call)
    pub total_llm_calls: usize,
    /// Consensus rate (percentage of ticks reaching explicit consensus)
    pub consensus_rate: f64,
    /// Average turns to consensus
    pub avg_turns_to_consensus: f64,
}

/// Aggregate results from a grid experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridResults {
    /// All individual results
    pub results: Vec<ExperimentResult>,
    /// Summary statistics by configuration
    pub summary: HashMap<String, ConfigSummary>,
}

/// Summary statistics for a configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub config_key: String,
    pub trials: usize,
    pub solve_rate: f64,
    /// Standard error of solve rate: sqrt(p(1-p)/n)
    pub solve_rate_se: f64,
    /// 95% confidence interval for solve rate: (lower, upper)
    pub solve_rate_ci: (f64, f64),
    pub avg_ticks: f64,
    /// Standard error of avg_ticks
    pub avg_ticks_se: f64,
    pub avg_final_pressure: f64,
    pub min_ticks: usize,
    pub max_ticks: usize,
}

impl GridResults {
    /// Create a new empty grid results.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            summary: HashMap::new(),
        }
    }

    /// Add a result.
    pub fn add(&mut self, result: ExperimentResult) {
        self.results.push(result);
    }

    /// Compute summary statistics.
    pub fn compute_summary(&mut self) {
        let mut by_config: HashMap<String, Vec<&ExperimentResult>> = HashMap::new();

        for result in &self.results {
            let key = format!(
                "{}:agents={}:decay={}:inhibit={}:examples={}",
                result.config.strategy,
                result.config.agent_count,
                result.config.decay_enabled,
                result.config.inhibition_enabled,
                result.config.examples_enabled
            );
            by_config.entry(key).or_default().push(result);
        }

        for (key, results) in by_config {
            let trials = results.len();
            let n = trials as f64;
            let solved_count = results.iter().filter(|r| r.solved).count();
            let solve_rate = solved_count as f64 / n;

            // Standard error for proportion: SE = sqrt(p(1-p)/n)
            let solve_rate_se = if trials > 1 {
                (solve_rate * (1.0 - solve_rate) / n).sqrt()
            } else {
                0.0
            };

            // 95% CI: p ± 1.96 * SE, clamped to [0, 1]
            let z = 1.96;
            let solve_rate_ci = (
                (solve_rate - z * solve_rate_se).max(0.0),
                (solve_rate + z * solve_rate_se).min(1.0),
            );

            let ticks: Vec<f64> = results.iter().map(|r| r.total_ticks as f64).collect();
            let avg_ticks = ticks.iter().sum::<f64>() / n;

            // Standard error for continuous: SE = std_dev / sqrt(n)
            let avg_ticks_se = if trials > 1 {
                let variance =
                    ticks.iter().map(|t| (t - avg_ticks).powi(2)).sum::<f64>() / (n - 1.0);
                variance.sqrt() / n.sqrt()
            } else {
                0.0
            };

            let min_ticks = ticks.iter().map(|t| *t as usize).min().unwrap_or(0);
            let max_ticks = ticks.iter().map(|t| *t as usize).max().unwrap_or(0);

            let avg_final_pressure =
                results.iter().map(|r| r.final_pressure).sum::<f64>() / n;

            self.summary.insert(
                key.clone(),
                ConfigSummary {
                    config_key: key,
                    trials,
                    solve_rate,
                    solve_rate_se,
                    solve_rate_ci,
                    avg_ticks,
                    avg_ticks_se,
                    avg_final_pressure,
                    min_ticks,
                    max_ticks,
                },
            );
        }
    }

    /// Save results to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load results from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let results = serde_json::from_str(&json)?;
        Ok(results)
    }
}

impl Default for GridResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a duration in milliseconds for display.
pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{:.1}m", ms as f64 / 60_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_results_summary() {
        let mut results = GridResults::new();

        // Add some test results
        for trial in 0..3 {
            results.add(ExperimentResult {
                config: ExperimentConfig {
                    strategy: "pressure_field".to_string(),
                    agent_count: 2,
                    n: 6,
                    empty_cells: 12,
                    decay_enabled: true,
                    inhibition_enabled: true,
                    examples_enabled: true,
                    trial,
                    seed: Some(trial as u64),
                },
                started_at: Utc::now(),
                ended_at: Utc::now(),
                total_ticks: 10 + trial,
                solved: trial < 2,
                final_pressure: if trial < 2 { 0.0 } else { 1.0 },
                pressure_history: vec![],
                patches_per_tick: vec![],
                empty_cells_history: vec![],
                example_bank_stats: None,
                tick_metrics: vec![],
                escalation_events: vec![],
                final_model: "test-model".to_string(),
                conversation_stats: None,
            });
        }

        results.compute_summary();

        let key = "pressure_field:agents=2:decay=true:inhibit=true:examples=true";
        let summary = results.summary.get(key).unwrap();

        assert_eq!(summary.trials, 3);
        assert!((summary.solve_rate - 0.666).abs() < 0.01);

        // Verify new SE/CI fields
        assert!(summary.solve_rate_se > 0.0);
        assert!(summary.solve_rate_ci.0 <= summary.solve_rate);
        assert!(summary.solve_rate_ci.1 >= summary.solve_rate);
    }
}
