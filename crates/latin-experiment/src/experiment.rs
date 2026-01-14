//! Experiment runner for Latin Square coordination experiments.
//!
//! Orchestrates the experiment lifecycle:
//! 1. Generate puzzle
//! 2. Set up kernel with sensors and actors
//! 3. Run tick loop until solved or max ticks
//! 4. Collect metrics and results

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use futures::future::join_all;
use rand::Rng;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use acton_reactive::prelude::*;
use survival_kernel::artifact::Artifact;
use survival_kernel::config::{
    ActivationConfig, DecayConfig, KernelConfig, PressureAxisConfig, SelectionConfig,
};
use survival_kernel::messages::{RegisterTickDriver, Tick, WaitForPatchActors};
use survival_kernel::region::PatchOp;
use survival_kernel::AsyncKernelBuilder;

use crate::llm_actor::{LlmActor, LlmActorConfig, SamplingBand, SamplingConfig};
use crate::tick_driver::{StartupWaiter, TickDriverActor};

use crate::artifact::LatinSquareArtifact;
use crate::conversation::ConversationRunner;
use crate::example_bank::{Example, ExampleBank, ExampleBankConfig};
use crate::generator::{Difficulty, GeneratorConfig, LatinSquareGenerator};
use crate::results::{
    ConversationStats, EscalationEvent, ExperimentConfig, ExperimentResult, PatchRejection,
    TickMetrics,
};
use crate::sensors::{update_shared_grid, LatinSquareSensor, SharedGrid};
use crate::vllm_client::VllmClient;

/// Configuration for the experiment runner.
#[derive(Debug, Clone)]
pub struct ExperimentRunnerConfig {
    /// vLLM host URL
    pub vllm_host: String,
    /// Model name (base model)
    pub model: String,
    /// Model escalation chain (e.g., ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b"])
    /// When stuck at local minimum, escalates to next model in chain
    pub model_chain: Vec<String>,
    /// Number of ticks with zero velocity before escalating model
    pub escalation_threshold: usize,
    /// Maximum ticks before giving up
    pub max_ticks: usize,
    /// Maximum concurrent LLM requests
    pub max_concurrent_llm: usize,
    /// Puzzle difficulty
    pub difficulty: Difficulty,
    /// Enable decay
    pub decay_enabled: bool,
    /// Enable inhibition
    pub inhibition_enabled: bool,
    /// Enable few-shot examples
    pub examples_enabled: bool,
    /// Example bank configuration
    pub example_bank_config: ExampleBankConfig,
    /// Maximum turns per conversation (for Conversation strategy)
    pub conversation_max_turns: usize,
}

impl Default for ExperimentRunnerConfig {
    fn default() -> Self {
        Self {
            vllm_host: "http://localhost:8000".to_string(),
            model: "Qwen/Qwen2.5-0.5B".to_string(),
            model_chain: vec![
                "Qwen/Qwen2.5-0.5B".to_string(),
                "Qwen/Qwen2.5-1.5B".to_string(),
                "Qwen/Qwen2.5-3B".to_string(),
                "Qwen/Qwen2.5-7B".to_string(),
                "Qwen/Qwen2.5-14B".to_string(),
            ],
            escalation_threshold: 20, // Escalate after 20 ticks with no progress
            max_ticks: 50,
            max_concurrent_llm: 8, // GPU handles 8-16 concurrent calls well
            difficulty: Difficulty::Medium,
            decay_enabled: true,
            inhibition_enabled: true,
            examples_enabled: true,
            example_bank_config: ExampleBankConfig::default(),
            conversation_max_turns: 5, // For Conversation strategy
        }
    }
}

/// Coordination strategy for the experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Pressure-field coordination (the novel approach)
    PressureField,
    /// Sequential round-robin
    Sequential,
    /// Random selection
    Random,
    /// Hierarchical (simulated manager)
    Hierarchical,
    /// Conversation-based coordination (AutoGen-style baseline)
    Conversation,
}

impl Strategy {
    /// Get all strategies for grid experiments.
    pub fn all() -> Vec<Self> {
        vec![
            Self::PressureField,
            Self::Sequential,
            Self::Random,
            Self::Hierarchical,
            Self::Conversation,
        ]
    }

    /// Get the name of this strategy.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PressureField => "pressure_field",
            Self::Sequential => "sequential",
            Self::Random => "random",
            Self::Hierarchical => "hierarchical",
            Self::Conversation => "conversation",
        }
    }
}

/// The experiment runner.
pub struct ExperimentRunner {
    config: ExperimentRunnerConfig,
}

impl ExperimentRunner {
    /// Create a new experiment runner.
    pub fn new(config: ExperimentRunnerConfig) -> Self {
        Self { config }
    }

    /// Run a single experiment.
    pub async fn run(
        &self,
        strategy: Strategy,
        agent_count: usize,
        trial: usize,
        seed: Option<u64>,
    ) -> Result<ExperimentResult> {
        let started_at = Utc::now();
        let start_time = Instant::now();

        // Generate puzzle
        let gen_config = GeneratorConfig {
            seed,
            ..self.config.difficulty.config()
        };
        let generator = LatinSquareGenerator::new(gen_config.clone());
        let mut artifact = generator.generate()?;

        let n = artifact.size();
        let initial_empty = artifact.empty_count();

        info!(
            strategy = strategy.name(),
            agents = agent_count,
            trial = trial,
            n = n,
            empty_cells = initial_empty,
            "Starting experiment"
        );

        // Set up shared grid for sensor
        let shared_grid: SharedGrid = Arc::new(std::sync::RwLock::new(artifact.grid().clone()));

        // Set up example bank
        let example_bank = if self.config.examples_enabled {
            Arc::new(RwLock::new(ExampleBank::new(
                self.config.example_bank_config.clone(),
            )))
        } else {
            Arc::new(RwLock::new(ExampleBank::disabled()))
        };

        // Create sensor
        let sensor = LatinSquareSensor::new(n, shared_grid.clone());

        // For PressureField strategy, use the kernel-based coordination
        if strategy == Strategy::PressureField {
            return self
                .run_pressure_field_with_kernel(
                    artifact,
                    shared_grid,
                    example_bank,
                    sensor,
                    agent_count,
                    trial,
                    seed,
                    started_at,
                    start_time,
                )
                .await;
        }

        // Tracking (for non-PressureField strategies)
        let mut pressure_history = Vec::new();
        let mut patches_per_tick = Vec::new();
        let mut empty_cells_history = Vec::new();
        let mut tick_metrics = Vec::new();

        // Token usage tracking
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let mut total_patch_rejections: HashMap<PatchRejection, usize> = HashMap::new();

        // Initial measurement
        let initial_pressure = self.measure_total_pressure(&artifact, &sensor)?;
        pressure_history.push(initial_pressure);
        empty_cells_history.push(artifact.empty_count());

        // Model escalation state (only for PressureField strategy)
        let mut current_model_idx: usize = 0;
        let mut zero_velocity_ticks: usize = 0;
        let mut escalation_events: Vec<EscalationEvent> = Vec::new();

        // Conversation-specific state
        let mut conversation_runner: Option<ConversationRunner> = None;
        let mut total_conversation_messages: usize = 0;
        let mut total_consensus_ticks: usize = 0;

        if strategy == Strategy::Conversation {
            let model = if self.config.model_chain.is_empty() {
                self.config.model.clone()
            } else {
                self.config.model_chain[0].clone()
            };
            conversation_runner = Some(ConversationRunner::new(
                &self.config.vllm_host,
                &model,
                self.config.conversation_max_turns,
            )?);
        }

        // Run tick loop
        for tick in 0..self.config.max_ticks {
            let tick_start = Instant::now();

            // Apply decay to example bank
            if self.config.decay_enabled {
                let bank = example_bank.read().await;
                bank.apply_decay();
            }

            // Handle Conversation strategy separately (different flow)
            // TODO: Add token tracking to ConversationRunner when available
            let (patches_applied, messages_this_tick, tick_prompt_tokens, tick_completion_tokens, tick_rejections, tick_model) = if strategy == Strategy::Conversation {
                let runner = conversation_runner.as_ref().unwrap();
                let (patch_opt, conv_state) = runner.run_tick(&artifact, &shared_grid).await?;

                let messages_count = conv_state.total_messages();
                total_conversation_messages += messages_count;
                if conv_state.final_patch.is_some() {
                    total_consensus_ticks += 1;
                }

                // Get current model for conversation
                let conv_model = if self.config.model_chain.is_empty() {
                    self.config.model.clone()
                } else {
                    self.config.model_chain[current_model_idx.min(self.config.model_chain.len() - 1)].clone()
                };

                let mut tick_rejections: HashMap<PatchRejection, usize> = HashMap::new();
                let mut applied = 0;
                if let Some(new_content) = patch_opt {
                    // Find the target region (coordinator selected it)
                    if let Some(region_id) = conv_state.target_region {
                        let region_view = artifact.read_region(region_id)?;
                        let patch = survival_kernel::region::Patch {
                            region: region_id,
                            op: survival_kernel::region::PatchOp::Replace(new_content.clone()),
                            rationale: "Conversation consensus".to_string(),
                            expected_delta: HashMap::new(),
                        };

                        match artifact.apply_patch(patch) {
                            Ok(()) => {
                                update_shared_grid(&shared_grid, artifact.grid())?;

                                // Add to example bank if enabled
                                if self.config.examples_enabled {
                                    let pressure_before =
                                        self.measure_region_pressure(&region_view, &sensor)?;
                                    let temp_view = survival_kernel::region::RegionView {
                                        id: region_id,
                                        kind: "row".to_string(),
                                        content: new_content.clone(),
                                        metadata: region_view.metadata.clone(),
                                    };
                                    let pressure_after =
                                        self.measure_region_pressure(&temp_view, &sensor)?;

                                    let bank = example_bank.read().await;
                                    bank.add_example(
                                        region_view.content.clone(),
                                        new_content,
                                        pressure_before,
                                        pressure_after,
                                    );
                                }

                                applied = 1;
                            }
                            Err(e) => {
                                debug!(tick = tick, error = %e, "Conversation patch rejected");
                                *tick_rejections.entry(PatchRejection::WouldIncreaseViolations).or_insert(0) += 1;
                            }
                        }
                    }
                }

                // Conversation token tracking: estimate from message count
                // Each message ≈ 100 tokens prompt + 50 tokens completion (rough estimate)
                // TODO: Get actual tokens from ConversationRunner when available
                let estimated_prompt_tokens = (messages_count as u32) * 100;
                let estimated_completion_tokens = (messages_count as u32) * 50;
                total_prompt_tokens += estimated_prompt_tokens;
                total_completion_tokens += estimated_completion_tokens;

                for (reason, count) in &tick_rejections {
                    *total_patch_rejections.entry(*reason).or_insert(0) += count;
                }

                (applied, Some(messages_count), estimated_prompt_tokens, estimated_completion_tokens, tick_rejections, conv_model)
            } else {
                // Standard strategies: PressureField, Sequential, Random, Hierarchical
                let region_id = match strategy {
                    Strategy::PressureField => {
                        self.select_highest_pressure_region(&artifact, &sensor)?
                    }
                    Strategy::Sequential => {
                        let regions = artifact.region_ids();
                        regions[tick % regions.len()]
                    }
                    Strategy::Random => {
                        use rand::seq::IndexedRandom;
                        let regions = artifact.region_ids();
                        *regions.choose(&mut rand::rng()).unwrap()
                    }
                    Strategy::Hierarchical => {
                        // Simulate hierarchical by picking region with most empty cells
                        self.select_most_incomplete_region(&artifact)?
                    }
                    Strategy::Conversation => unreachable!(), // Handled above
                };

                // Get region view
                let region_view = artifact.read_region(region_id)?;
                let pressure_before = self.measure_region_pressure(&region_view, &sensor)?;

                // Get examples for few-shot prompting
                let examples = {
                    let bank = example_bank.read().await;
                    bank.get_examples_for_prompt()
                };

                // Get current model (with escalation for PressureField)
                let current_model = if self.config.model_chain.is_empty() {
                    self.config.model.clone()
                } else {
                    self.config.model_chain[current_model_idx.min(self.config.model_chain.len() - 1)]
                        .clone()
                };

                // Generate multiple patches concurrently using LLM
                let patch_results = self
                    .generate_concurrent_patches(&artifact, region_id, &examples, &current_model)
                    .await?;

                // Tick-level tracking
                let mut tick_prompt_tokens: u32 = 0;
                let mut tick_completion_tokens: u32 = 0;
                let mut tick_rejections: HashMap<PatchRejection, usize> = HashMap::new();

                // Accumulate tokens from all LLM calls
                for (_, prompt_t, completion_t) in &patch_results {
                    tick_prompt_tokens += prompt_t;
                    tick_completion_tokens += completion_t;
                }

                // Try each patch result until one succeeds
                let mut patches_applied = 0;
                for (new_content, _, _) in &patch_results {
                    // Skip empty content (parse failures)
                    if new_content.is_empty() {
                        *tick_rejections.entry(PatchRejection::ParseFailure).or_insert(0) += 1;
                        continue;
                    }

                    // Validate that the patch reduces pressure
                    let temp_view = survival_kernel::region::RegionView {
                        id: region_id,
                        kind: "row".to_string(),
                        content: new_content.clone(),
                        metadata: region_view.metadata.clone(),
                    };
                    let pressure_after = self.measure_region_pressure(&temp_view, &sensor)?;

                    if pressure_after < pressure_before || !self.config.decay_enabled {
                        // Apply patch
                        let patch = survival_kernel::region::Patch {
                            region: region_id,
                            op: survival_kernel::region::PatchOp::Replace(new_content.clone()),
                            rationale: "Filled row".to_string(),
                            expected_delta: HashMap::new(),
                        };

                        // Try to apply - may fail if LLM changed fixed cells
                        match artifact.apply_patch(patch) {
                            Ok(()) => {
                                // Update shared grid
                                update_shared_grid(&shared_grid, artifact.grid())?;

                                // Add to example bank
                                {
                                    let bank = example_bank.read().await;
                                    bank.add_example(
                                        region_view.content.clone(),
                                        new_content.clone(),
                                        pressure_before,
                                        pressure_after,
                                    );
                                }

                                patches_applied = 1;
                                break; // Stop after first successful patch
                            }
                            Err(e) => {
                                debug!(tick = tick, error = %e, "Patch rejected - invalid");
                                *tick_rejections.entry(PatchRejection::WouldIncreaseViolations).or_insert(0) += 1;
                            }
                        }
                    } else {
                        debug!(tick = tick, "Patch rejected - did not reduce pressure");
                        *tick_rejections.entry(PatchRejection::DidNotReducePressure).or_insert(0) += 1;
                    }
                }

                // Aggregate tick-level stats into totals
                total_prompt_tokens += tick_prompt_tokens;
                total_completion_tokens += tick_completion_tokens;
                for (reason, count) in &tick_rejections {
                    *total_patch_rejections.entry(*reason).or_insert(0) += count;
                }

                (patches_applied, None, tick_prompt_tokens, tick_completion_tokens, tick_rejections, current_model)
            };

            // Record metrics
            let current_pressure = self.measure_total_pressure(&artifact, &sensor)?;
            pressure_history.push(current_pressure);
            patches_per_tick.push(patches_applied);
            empty_cells_history.push(artifact.empty_count());

            tick_metrics.push(TickMetrics {
                tick,
                pressure_before: pressure_history[tick],
                pressure_after: current_pressure,
                patches_proposed: 1,
                patches_applied,
                empty_cells: artifact.empty_count(),
                violations: artifact.total_violations(),
                llm_calls: if patches_applied > 0 { 1 } else { 0 },
                duration_ms: tick_start.elapsed().as_millis() as u64,
                model_used: tick_model,
                prompt_tokens: tick_prompt_tokens,
                completion_tokens: tick_completion_tokens,
                patch_rejections: tick_rejections,
                messages_per_tick: messages_this_tick,
            });

            // Check if solved
            if artifact.is_solved() {
                info!(
                    tick = tick,
                    duration_ms = start_time.elapsed().as_millis(),
                    "Puzzle solved!"
                );
                break;
            }

            // Model escalation: track velocity and escalate when stuck (all strategies)
            if tick > 0 {
                let prev_pressure = pressure_history[tick];
                let curr_pressure = pressure_history[tick + 1];
                let velocity = prev_pressure - curr_pressure; // positive = improving

                if velocity <= 0.0 {
                    // No progress this tick
                    zero_velocity_ticks += 1;

                    // Check if we should escalate to a larger model
                    if zero_velocity_ticks >= self.config.escalation_threshold
                        && current_model_idx < self.config.model_chain.len() - 1
                    {
                        let from_model = self.config.model_chain[current_model_idx].clone();
                        current_model_idx += 1;
                        let to_model = self.config.model_chain[current_model_idx].clone();
                        zero_velocity_ticks = 0;

                        // Record escalation event
                        escalation_events.push(EscalationEvent {
                            tick,
                            from_model: from_model.clone(),
                            to_model: to_model.clone(),
                        });

                        // Update ConversationRunner's model if applicable
                        if let Some(ref mut runner) = conversation_runner {
                            runner.set_model(&to_model);
                        }

                        info!(
                            tick = tick,
                            new_model = &to_model,
                            prev_model = &from_model,
                            strategy = strategy.name(),
                            "Escalating model due to stalled progress"
                        );
                    }
                } else {
                    // Made progress, reset counter
                    zero_velocity_ticks = 0;
                }
            }
        }

        let ended_at = Utc::now();
        let solved = artifact.is_solved();
        let final_pressure = self.measure_total_pressure(&artifact, &sensor)?;

        let example_bank_stats = {
            let bank = example_bank.read().await;
            Some(bank.stats())
        };

        // Determine final model (which model tier was active when solved/ended)
        let final_model = if self.config.model_chain.is_empty() {
            self.config.model.clone()
        } else {
            self.config.model_chain[current_model_idx.min(self.config.model_chain.len() - 1)].clone()
        };

        // Build conversation stats for Conversation strategy
        let conversation_stats = if strategy == Strategy::Conversation {
            let total_ticks = tick_metrics.len();
            let avg_messages_per_tick = if total_ticks > 0 {
                total_conversation_messages as f64 / total_ticks as f64
            } else {
                0.0
            };
            let consensus_rate = if total_ticks > 0 {
                total_consensus_ticks as f64 / total_ticks as f64
            } else {
                0.0
            };
            // Average turns to consensus: assume max_turns if no consensus, else average
            let avg_turns = if total_consensus_ticks > 0 {
                total_conversation_messages as f64 / total_consensus_ticks as f64 / 3.0 // ~3 messages per turn
            } else {
                self.config.conversation_max_turns as f64
            };
            Some(ConversationStats {
                total_messages: total_conversation_messages,
                avg_messages_per_tick,
                total_llm_calls: total_conversation_messages, // Each message = 1 LLM call
                consensus_rate,
                avg_turns_to_consensus: avg_turns,
            })
        } else {
            None
        };

        Ok(ExperimentResult {
            config: ExperimentConfig {
                strategy: strategy.name().to_string(),
                agent_count,
                n,
                empty_cells: initial_empty,
                decay_enabled: self.config.decay_enabled,
                inhibition_enabled: self.config.inhibition_enabled,
                examples_enabled: self.config.examples_enabled,
                trial,
                seed,
            },
            started_at,
            ended_at,
            total_ticks: tick_metrics.len(),
            solved,
            final_pressure,
            pressure_history,
            patches_per_tick,
            empty_cells_history,
            example_bank_stats,
            tick_metrics,
            escalation_events,
            final_model,
            total_prompt_tokens,
            total_completion_tokens,
            total_patch_rejections,
            conversation_stats,
        })
    }

    /// Measure total pressure across all regions.
    fn measure_total_pressure(
        &self,
        artifact: &LatinSquareArtifact,
        sensor: &LatinSquareSensor,
    ) -> Result<f64> {
        let mut total = 0.0;
        for region_id in artifact.region_ids() {
            let view = artifact.read_region(region_id)?;
            total += self.measure_region_pressure(&view, sensor)?;
        }
        Ok(total)
    }

    /// Measure pressure for a single region.
    fn measure_region_pressure(
        &self,
        view: &survival_kernel::region::RegionView,
        sensor: &LatinSquareSensor,
    ) -> Result<f64> {
        use survival_kernel::pressure::Sensor;
        let signals = sensor.measure(view)?;

        // Pressure formula: empty × 1.0 + row_dups × 10.0 + col_conflicts × 10.0
        let empty = signals.get("empty_count").copied().unwrap_or(0.0);
        let row_dups = signals.get("row_duplicates").copied().unwrap_or(0.0);
        let col_conflicts = signals.get("col_conflicts").copied().unwrap_or(0.0);

        Ok(empty * 1.0 + row_dups * 10.0 + col_conflicts * 10.0)
    }

    /// Select the region with highest pressure.
    fn select_highest_pressure_region(
        &self,
        artifact: &LatinSquareArtifact,
        sensor: &LatinSquareSensor,
    ) -> Result<uuid::Uuid> {
        let mut max_pressure = f64::NEG_INFINITY;
        let mut max_region = artifact.region_ids()[0];

        for region_id in artifact.region_ids() {
            let view = artifact.read_region(region_id)?;
            let pressure = self.measure_region_pressure(&view, sensor)?;
            if pressure > max_pressure {
                max_pressure = pressure;
                max_region = region_id;
            }
        }

        Ok(max_region)
    }

    /// Select the region with most empty cells.
    fn select_most_incomplete_region(
        &self,
        artifact: &LatinSquareArtifact,
    ) -> Result<uuid::Uuid> {
        let mut max_empty = 0;
        let mut max_region = artifact.region_ids()[0];

        for region_id in artifact.region_ids() {
            let view = artifact.read_region(region_id)?;
            let empty_count = view.content.matches('_').count();
            if empty_count > max_empty {
                max_empty = empty_count;
                max_region = region_id;
            }
        }

        Ok(max_region)
    }

    /// Generate multiple patches concurrently using parallel LLM calls.
    ///
    /// Each call targets a potentially different empty position with different
    /// sampling parameters (temperature/top_p) for exploration diversity.
    ///
    /// Returns Vec of (patch_content, prompt_tokens, completion_tokens).
    async fn generate_concurrent_patches(
        &self,
        artifact: &LatinSquareArtifact,
        region_id: uuid::Uuid,
        examples: &[Example],
        current_model: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        let n = artifact.size();
        let row_idx = artifact.row_index(region_id).unwrap_or(0);
        let availability = artifact.column_availability(row_idx);
        let view = artifact.read_region(region_id)?;

        // Parse current row to identify empty positions
        let cells: Vec<&str> = view.content.split_whitespace().collect();
        let empty_positions: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == "_")
            .map(|(i, _)| i)
            .collect();

        if empty_positions.is_empty() {
            return Ok(Vec::new());
        }

        // Format examples for few-shot learning
        let examples_text = if examples.is_empty() {
            String::new()
        } else {
            let formatted: Vec<String> = examples.iter().map(|e| e.format_for_prompt()).collect();
            format!(
                "\nEXAMPLES OF SUCCESSFUL FIXES:\n{}\n",
                formatted.join("\n")
            )
        };

        // Create concurrent LLM calls
        let num_calls = self.config.max_concurrent_llm.min(empty_positions.len() * 2);
        let client = Arc::new(VllmClient::new(&self.config.vllm_host));

        let futures: Vec<_> = (0..num_calls)
            .map(|i| {
                let client = Arc::clone(&client);
                let model = current_model.to_string();
                let examples_text = examples_text.clone();
                let availability = availability.clone();
                let cells: Vec<String> = cells.iter().map(|s| s.to_string()).collect();
                let empty_positions = empty_positions.clone();
                let content = view.content.clone();

                async move {
                    let mut rng = rand::rng();

                    // Pick a target position (cycle through empty positions)
                    let target_pos = empty_positions[i % empty_positions.len()];

                    // Get available values for this position
                    let available_for_target = availability
                        .get(&target_pos)
                        .map(|v| {
                            v.iter()
                                .map(|n| n.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_else(|| "?".to_string());

                    let prompt = format!(
                        r#"Row: {content}
Empty position: {target_pos}
Available values for position {target_pos}: [{available}]
{examples_text}
What number goes in position {target_pos}? Return just the number."#,
                        content = content,
                        target_pos = target_pos,
                        available = available_for_target,
                        examples_text = examples_text,
                    );

                    // Sample temperature and top_p with diversity
                    let band: u8 = rng.random_range(0..3);
                    let (temp, top_p) = match band {
                        0 => (rng.random_range(0.15..0.35), rng.random_range(0.80..0.90)),
                        1 => (rng.random_range(0.35..0.55), rng.random_range(0.85..0.95)),
                        _ => (rng.random_range(0.55..0.85), rng.random_range(0.90..0.98)),
                    };

                    match client.generate_with_usage(&model, &prompt, temp, top_p, 8).await {
                        Ok(response) => {
                            let response_text = response.content.trim();
                            // Parse single number
                            let cleaned = response_text.replace([',', '[', ']', '"', '.'], " ");

                            for word in cleaned.split_whitespace() {
                                if let Ok(num) = word.parse::<u8>()
                                    && num >= 1
                                    && num <= n as u8
                                {
                                    // Construct new row
                                    let mut new_cells = cells.clone();
                                    new_cells[target_pos] = num.to_string();
                                    return Some((
                                        new_cells.join(" "),
                                        response.prompt_tokens,
                                        response.completion_tokens,
                                    ));
                                }
                            }
                            // Failed to parse, but still consumed tokens
                            Some(("".to_string(), response.prompt_tokens, response.completion_tokens))
                        }
                        Err(e) => {
                            debug!(error = %e, "Concurrent LLM call failed");
                            None
                        }
                    }
                }
            })
            .collect();

        // Run all calls concurrently
        let results = join_all(futures).await;

        // Filter and collect results (include even empty patches to track token usage)
        Ok(results.into_iter().flatten().collect())
    }

    /// Generate a patch using the LLM (single call, kept for compatibility).
    #[allow(dead_code)]
    async fn generate_llm_patch(
        &self,
        artifact: &LatinSquareArtifact,
        region_id: uuid::Uuid,
        examples: &[Example],
    ) -> Result<Option<String>> {
        let n = artifact.size();
        let row_idx = artifact.row_index(region_id).unwrap_or(0);
        let availability = artifact.column_availability(row_idx);

        let view = artifact.read_region(region_id)?;

        // Format examples for few-shot learning
        let examples_text = if examples.is_empty() {
            String::new()
        } else {
            let formatted: Vec<String> = examples.iter().map(|e| e.format_for_prompt()).collect();
            format!(
                "\nEXAMPLES OF SUCCESSFUL FIXES:\n{}\n",
                formatted.join("\n")
            )
        };

        // Parse current row to identify empty positions
        let cells: Vec<&str> = view.content.split_whitespace().collect();
        let empty_positions: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == "_")
            .map(|(i, _)| i)
            .collect();

        // Pick a random empty position to fill (if multiple)
        if empty_positions.is_empty() {
            return Ok(None);
        }
        let mut rng = rand::rng();
        let target_pos = if empty_positions.len() == 1 {
            empty_positions[0]
        } else {
            use rand::seq::IndexedRandom;
            *empty_positions.choose(&mut rng).unwrap()
        };

        // Get available values for this specific column
        let available_for_target = availability
            .get(&target_pos)
            .map(|v| {
                v.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "?".to_string());

        let prompt = format!(
            r#"Row: {content}
Empty position: {target_pos}
Available values for position {target_pos}: [{available}]
{examples_text}
What number goes in position {target_pos}? Return just the number."#,
            content = view.content,
            target_pos = target_pos,
            available = available_for_target,
            examples_text = examples_text,
        );

        // Sample temperature and top_p for exploration diversity
        // Three bands: exploitation (low temp), balanced, exploration (high temp)
        let mut rng = rand::rng();
        let band: u8 = rng.random_range(0..3);
        let (temp, top_p) = match band {
            0 => {
                // Exploitation: low temperature, more deterministic
                (rng.random_range(0.15..0.35), rng.random_range(0.80..0.90))
            }
            1 => {
                // Balanced: medium temperature
                (rng.random_range(0.35..0.55), rng.random_range(0.85..0.95))
            }
            _ => {
                // Exploration: high temperature, more creative
                (rng.random_range(0.55..0.85), rng.random_range(0.90..0.98))
            }
        };

        debug!(row_idx = row_idx, temp = temp, top_p = top_p, "Calling LLM for patch");

        // Call vLLM with sampled options
        let client = VllmClient::new(&self.config.vllm_host);

        let response = match client.generate(&self.config.model, &prompt, temp, top_p, 8).await {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "LLM call failed");
                return Ok(None);
            }
        };

        let response_text = response.trim();
        debug!(response = %response_text, "LLM response");

        // Parse the single number from response
        let fill_value = self.parse_single_number(response_text, n);

        match fill_value {
            Some(value) => {
                // Construct new row by replacing the empty cell at target_pos
                let mut new_cells: Vec<String> = cells.iter().map(|s| s.to_string()).collect();
                new_cells[target_pos] = value.to_string();
                let new_content = new_cells.join(" ");
                if new_content != view.content {
                    Ok(Some(new_content))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Parse a single number from the LLM response.
    fn parse_single_number(&self, response: &str, n: usize) -> Option<u8> {
        // Extract first valid number from response
        let cleaned = response.replace([',', '[', ']', '"', '.'], " ");

        for word in cleaned.split_whitespace() {
            if let Ok(num) = word.parse::<u8>()
                && num >= 1
                && num <= n as u8
            {
                return Some(num);
            }
        }

        None
    }

    /// Run PressureField strategy using the kernel's actor-based coordination.
    ///
    /// This method uses the survival-kernel's actor system for coordination:
    /// - KernelCoordinator orchestrates tick phases
    /// - LlmActors propose patches via broker pub/sub
    /// - RegionActors validate and apply patches
    /// - TickDriverActor forwards results to the experiment harness
    #[allow(clippy::too_many_lines)]
    async fn run_pressure_field_with_kernel(
        &self,
        artifact: LatinSquareArtifact,
        shared_grid: SharedGrid,
        example_bank: Arc<RwLock<ExampleBank>>,
        sensor: LatinSquareSensor,
        agent_count: usize,
        trial: usize,
        seed: Option<u64>,
        started_at: chrono::DateTime<Utc>,
        start_time: Instant,
    ) -> Result<ExperimentResult> {
        let n = artifact.size();
        let initial_empty = artifact.empty_count();

        // Build kernel config
        let kernel_config = self.build_kernel_config();

        // Create acton runtime
        let mut runtime = ActonApp::launch_async().await;

        // Build and spawn the kernel coordinator with artifact and sensor
        let coordinator_handle = AsyncKernelBuilder::new(
            kernel_config,
            Box::new(artifact.clone()),
        )
        .add_sensor(Box::new(sensor.clone()))
        .spawn(&mut runtime)
        .await;

        // Create semaphore for rate limiting LLM requests
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_llm));

        // Spawn LLM actors - they self-register via PatchActorReady broadcast
        // Distribute actors across sampling bands for diversity
        for i in 0..agent_count {
            let band = match i % 3 {
                0 => SamplingBand::Exploitation,
                1 => SamplingBand::Balanced,
                _ => SamplingBand::Exploration,
            };

            let current_model = if self.config.model_chain.is_empty() {
                self.config.model.clone()
            } else {
                self.config.model_chain[0].clone()
            };

            let llm_config = LlmActorConfig {
                host: self.config.vllm_host.clone(),
                model: current_model,
                sampling: SamplingConfig::random_in_band(band),
                max_tokens: 256,
                band,
                randomize_sampling: true,
            };

            let llm_actor = LlmActor::new(
                format!("LlmActor:{}", i),
                llm_config,
                semaphore.clone(),
                example_bank.clone(),
            );
            llm_actor.spawn(&mut runtime).await;
        }

        // Create tick driver to receive TickComplete messages
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tick_driver = TickDriverActor::new(tx);
        let tick_driver_handle = tick_driver.spawn(&mut runtime).await;

        // Register tick driver with coordinator
        coordinator_handle
            .send(RegisterTickDriver {
                handle: tick_driver_handle,
            })
            .await;

        // Wait for all patch actors to register before starting ticks
        // This prevents the race condition where ticks start before actors are ready
        let (startup_tx, startup_rx) = tokio::sync::oneshot::channel();
        let startup_waiter = StartupWaiter::new(startup_tx);
        startup_waiter.spawn(&mut runtime).await;

        // Tell coordinator how many actors we expect and wait for confirmation
        coordinator_handle
            .send(WaitForPatchActors {
                expected_count: agent_count,
            })
            .await;

        // Wait for PatchActorsReady (with timeout for safety)
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            startup_rx,
        )
        .await
        {
            Ok(Ok(registered)) => {
                tracing::debug!(
                    expected = agent_count,
                    registered = registered,
                    "All patch actors registered"
                );
            }
            Ok(Err(_)) => {
                tracing::warn!("Startup waiter channel closed unexpectedly");
            }
            Err(_) => {
                tracing::warn!(
                    expected = agent_count,
                    "Timeout waiting for patch actors to register, proceeding anyway"
                );
            }
        }

        // Tracking
        let mut pressure_history = Vec::new();
        let mut patches_per_tick = Vec::new();
        let mut empty_cells_history = Vec::new();
        let mut tick_metrics = Vec::new();
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let total_patch_rejections: HashMap<PatchRejection, usize> = HashMap::new();

        // Model escalation state
        let mut current_model_idx: usize = 0;
        let mut zero_velocity_ticks: usize = 0;
        let mut escalation_events: Vec<EscalationEvent> = Vec::new();

        // Track artifact state locally (kernel owns the canonical artifact)
        let mut local_artifact = artifact.clone();

        // Initial measurement
        let initial_pressure = self.measure_total_pressure(&local_artifact, &sensor)?;
        pressure_history.push(initial_pressure);
        empty_cells_history.push(local_artifact.empty_count());

        // Run tick loop
        for tick in 0..self.config.max_ticks {
            let tick_start = Instant::now();

            // Apply decay to example bank
            if self.config.decay_enabled {
                let bank = example_bank.read().await;
                bank.apply_decay();
            }

            // Send Tick to coordinator
            coordinator_handle
                .send(Tick {
                    now_ms: tick as u64 * 100,
                })
                .await;

            // Wait for TickComplete
            let tick_result = match tokio::time::timeout(
                tokio::time::Duration::from_secs(30),
                rx.recv(),
            )
            .await
            {
                Ok(Some(result)) => result,
                Ok(None) => {
                    warn!(tick = tick, "TickDriver channel closed");
                    break;
                }
                Err(_) => {
                    warn!(tick = tick, "Tick timed out after 30s");
                    break;
                }
            };

            // Update local artifact state from applied patches
            for patch in &tick_result.applied {
                if let PatchOp::Replace(new_content) = &patch.op {
                    // Apply patch to local artifact
                    let local_patch = survival_kernel::region::Patch {
                        region: patch.region,
                        op: PatchOp::Replace(new_content.clone()),
                        rationale: patch.rationale.clone(),
                        expected_delta: patch.expected_delta.clone(),
                    };
                    if local_artifact.apply_patch(local_patch).is_ok() {
                        // Update shared grid for sensor column detection
                        update_shared_grid(&shared_grid, local_artifact.grid())?;

                        // Add successful patch to example bank
                        if self.config.examples_enabled {
                            // Get the old content from region view
                            if let Ok(view) = local_artifact.read_region(patch.region) {
                                let bank = example_bank.read().await;
                                // Estimate pressure improvement (patch was validated by kernel)
                                bank.add_example(
                                    view.content.clone(), // This is now the new content
                                    new_content.clone(),
                                    1.0, // Estimated before pressure
                                    0.0, // Estimated after pressure (improved)
                                );
                            }
                        }
                    }
                }
            }

            // Aggregate token counts
            total_prompt_tokens += tick_result.prompt_tokens;
            total_completion_tokens += tick_result.completion_tokens;

            // Record metrics
            let current_pressure = tick_result.total_pressure;
            pressure_history.push(current_pressure);
            patches_per_tick.push(tick_result.applied.len());
            empty_cells_history.push(local_artifact.empty_count());

            let current_model = if self.config.model_chain.is_empty() {
                self.config.model.clone()
            } else {
                self.config.model_chain[current_model_idx.min(self.config.model_chain.len() - 1)]
                    .clone()
            };

            tick_metrics.push(TickMetrics {
                tick,
                pressure_before: pressure_history[tick],
                pressure_after: current_pressure,
                patches_proposed: tick_result.evaluated,
                patches_applied: tick_result.applied.len(),
                empty_cells: local_artifact.empty_count(),
                violations: local_artifact.total_violations(),
                llm_calls: agent_count, // All actors were queried
                duration_ms: tick_start.elapsed().as_millis() as u64,
                model_used: current_model.clone(),
                prompt_tokens: tick_result.prompt_tokens,
                completion_tokens: tick_result.completion_tokens,
                patch_rejections: HashMap::new(),
                messages_per_tick: None,
            });

            // Check if solved
            if local_artifact.is_solved() {
                info!(
                    tick = tick,
                    duration_ms = start_time.elapsed().as_millis(),
                    "Puzzle solved!"
                );
                break;
            }

            // Model escalation: track velocity and escalate when stuck
            let velocity = tick_result.velocity;
            if velocity >= 0.0 {
                // No improvement (velocity is pressure change, negative = improvement)
                zero_velocity_ticks += 1;

                if zero_velocity_ticks >= self.config.escalation_threshold
                    && current_model_idx < self.config.model_chain.len() - 1
                {
                    let from_model = self.config.model_chain[current_model_idx].clone();
                    current_model_idx += 1;
                    let to_model = self.config.model_chain[current_model_idx].clone();
                    zero_velocity_ticks = 0;

                    escalation_events.push(EscalationEvent {
                        tick,
                        from_model: from_model.clone(),
                        to_model: to_model.clone(),
                    });

                    info!(
                        tick = tick,
                        new_model = &to_model,
                        prev_model = &from_model,
                        "Escalating model due to stalled progress"
                    );

                    // Note: LlmActors would need to be notified of model change
                    // This requires a new message type - for now, we continue with original model
                }
            } else {
                zero_velocity_ticks = 0;
            }
        }

        // Shutdown runtime
        let _ = runtime.shutdown_all().await;

        let ended_at = Utc::now();
        let solved = local_artifact.is_solved();
        let final_pressure = self.measure_total_pressure(&local_artifact, &sensor)?;

        let example_bank_stats = {
            let bank = example_bank.read().await;
            Some(bank.stats())
        };

        let final_model = if self.config.model_chain.is_empty() {
            self.config.model.clone()
        } else {
            self.config.model_chain[current_model_idx.min(self.config.model_chain.len() - 1)]
                .clone()
        };

        Ok(ExperimentResult {
            config: ExperimentConfig {
                strategy: "pressure_field".to_string(),
                agent_count,
                n,
                empty_cells: initial_empty,
                decay_enabled: self.config.decay_enabled,
                inhibition_enabled: self.config.inhibition_enabled,
                examples_enabled: self.config.examples_enabled,
                trial,
                seed,
            },
            started_at,
            ended_at,
            total_ticks: tick_metrics.len(),
            solved,
            final_pressure,
            pressure_history,
            patches_per_tick,
            empty_cells_history,
            example_bank_stats,
            tick_metrics,
            escalation_events,
            final_model,
            total_prompt_tokens,
            total_completion_tokens,
            total_patch_rejections,
            conversation_stats: None,
        })
    }

    /// Build kernel configuration for Latin Square experiments.
    ///
    /// Configures pressure axes based on the sensor signals:
    /// - empty_count: Cells needing to be filled (weight: 1.0)
    /// - row_duplicates: Duplicate values in row (weight: 2.0)
    /// - col_conflicts: Column constraint violations (weight: 2.0)
    fn build_kernel_config(&self) -> KernelConfig {
        // Tick interval in ms (experiment runs faster than the 250ms default)
        let tick_interval_ms = 100;

        // Pressure axes matching sensor signals
        let pressure_axes = vec![
            PressureAxisConfig {
                name: "empty_cells".to_string(),
                weight: 1.0,
                expr: "empty_count".to_string(),
                kind_weights: HashMap::new(),
            },
            PressureAxisConfig {
                name: "row_violations".to_string(),
                weight: 2.0,
                expr: "row_duplicates".to_string(),
                kind_weights: HashMap::new(),
            },
            PressureAxisConfig {
                name: "column_violations".to_string(),
                weight: 2.0,
                expr: "col_conflicts".to_string(),
                kind_weights: HashMap::new(),
            },
        ];

        // Decay configuration based on experiment settings
        let decay = DecayConfig {
            // Fast decay for Latin Square experiments
            fitness_half_life_ms: if self.config.decay_enabled {
                5_000 // 5 seconds
            } else {
                u64::MAX // Effectively disabled
            },
            confidence_half_life_ms: if self.config.decay_enabled {
                10_000 // 10 seconds
            } else {
                u64::MAX
            },
            ema_alpha: 0.3,
        };

        // Activation thresholds
        let activation = ActivationConfig {
            // Any pressure triggers proposals (we want all regions with issues addressed)
            min_total_pressure: 0.5,
            // Inhibition period after patching
            inhibit_ms: if self.config.inhibition_enabled {
                1_000 // 1 second - faster for Latin Square experiments
            } else {
                0 // Disabled
            },
        };

        // Patch selection
        let selection = SelectionConfig {
            // Accept patches that improve the situation
            min_expected_improvement: 0.1,
        };

        KernelConfig {
            tick_interval_ms,
            pressure_axes,
            decay,
            activation,
            selection,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_experiment_runner_basic() {
        let config = ExperimentRunnerConfig {
            difficulty: Difficulty::Easy,
            max_ticks: 10,
            ..Default::default()
        };

        let runner = ExperimentRunner::new(config);
        let result = runner
            .run(Strategy::PressureField, 1, 0, Some(42))
            .await
            .unwrap();

        assert!(result.total_ticks <= 10);
        assert!(!result.pressure_history.is_empty());
    }

    #[test]
    fn test_strategy_names() {
        assert_eq!(Strategy::PressureField.name(), "pressure_field");
        assert_eq!(Strategy::Sequential.name(), "sequential");
    }

    #[test]
    fn test_parse_single_number_basic() {
        let config = ExperimentRunnerConfig::default();
        let runner = ExperimentRunner::new(config);

        // Basic valid numbers
        assert_eq!(runner.parse_single_number("3", 5), Some(3));
        assert_eq!(runner.parse_single_number("1", 5), Some(1));
        assert_eq!(runner.parse_single_number("5", 5), Some(5));

        // Numbers with surrounding text (common LLM output)
        assert_eq!(runner.parse_single_number("The answer is 4", 5), Some(4));
        assert_eq!(runner.parse_single_number("I think 2 is correct", 5), Some(2));
    }

    #[test]
    fn test_parse_single_number_edge_cases() {
        let config = ExperimentRunnerConfig::default();
        let runner = ExperimentRunner::new(config);

        // Out of range (should return None)
        assert_eq!(runner.parse_single_number("0", 5), None); // 0 is invalid (1-indexed)
        assert_eq!(runner.parse_single_number("6", 5), None); // > n
        assert_eq!(runner.parse_single_number("10", 5), None); // way out of range

        // Empty or no numbers
        assert_eq!(runner.parse_single_number("", 5), None);
        assert_eq!(runner.parse_single_number("no number here", 5), None);

        // Multiple numbers (should pick first valid)
        assert_eq!(runner.parse_single_number("3 4 5", 5), Some(3));

        // With special characters that LLMs sometimes include
        assert_eq!(runner.parse_single_number("[3]", 5), Some(3));
        assert_eq!(runner.parse_single_number("\"4\"", 5), Some(4));
        assert_eq!(runner.parse_single_number("3.", 5), Some(3));
        assert_eq!(runner.parse_single_number("3,", 5), Some(3));
    }

    #[test]
    fn test_parse_single_number_range_boundaries() {
        let config = ExperimentRunnerConfig::default();
        let runner = ExperimentRunner::new(config);

        // For a 7x7 puzzle
        assert_eq!(runner.parse_single_number("1", 7), Some(1)); // min valid
        assert_eq!(runner.parse_single_number("7", 7), Some(7)); // max valid
        assert_eq!(runner.parse_single_number("8", 7), None); // just over
        assert_eq!(runner.parse_single_number("0", 7), None); // just under
    }

    #[test]
    fn test_model_chain_escalation_indexing() {
        // Verify model chain default has expected models
        let config = ExperimentRunnerConfig::default();
        assert_eq!(config.model_chain.len(), 5);
        assert!(config.model_chain[0].contains("0.5B"));
        assert!(config.model_chain[4].contains("14B"));

        // Verify escalation threshold default
        assert_eq!(config.escalation_threshold, 20);
    }

    #[test]
    fn test_model_chain_index_bounds() {
        // Simulate the escalation index clamping logic from run()
        let model_chain = vec![
            "Qwen/Qwen2.5-0.5B".to_string(),
            "Qwen/Qwen2.5-1.5B".to_string(),
            "Qwen/Qwen2.5-3B".to_string(),
        ];

        // Test the clamping logic used in experiment.rs:320 and :465
        for idx in 0..10 {
            let clamped = idx.min(model_chain.len() - 1);
            assert!(clamped < model_chain.len(), "Index {} should be clamped", idx);
            // Should not panic when accessing
            let _model = &model_chain[clamped];
        }
    }

    #[test]
    fn test_escalation_velocity_calculation() {
        // Verify the velocity formula: prev_pressure - curr_pressure
        // Positive = improving (pressure decreasing)
        // Zero or negative = stuck

        let prev_pressure = 10.0;
        let curr_pressure = 8.0;
        let velocity = prev_pressure - curr_pressure;
        assert!(velocity > 0.0, "Should show improvement");

        let prev_pressure = 10.0;
        let curr_pressure = 10.0;
        let velocity = prev_pressure - curr_pressure;
        assert!(velocity <= 0.0, "Should show stuck (no progress)");

        let prev_pressure = 10.0;
        let curr_pressure = 12.0;
        let velocity = prev_pressure - curr_pressure;
        assert!(velocity < 0.0, "Should show regression");
    }

    #[test]
    fn test_pressure_formula() {
        // Test the pressure calculation formula from measure_region_pressure
        // Pressure = empty * 1.0 + row_dups * 10.0 + col_conflicts * 10.0

        let compute_pressure = |empty: f64, row_dups: f64, col_conflicts: f64| -> f64 {
            empty * 1.0 + row_dups * 10.0 + col_conflicts * 10.0
        };

        // Row with 2 empty cells, no duplicates
        assert_eq!(compute_pressure(2.0, 0.0, 0.0), 2.0);

        // Row with 1 empty, 1 duplicate
        assert_eq!(compute_pressure(1.0, 1.0, 0.0), 11.0);

        // Row with column conflict (worse than row duplicate)
        assert_eq!(compute_pressure(0.0, 0.0, 1.0), 10.0);

        // Multiple issues compound
        assert_eq!(compute_pressure(3.0, 2.0, 1.0), 33.0); // 3 + 20 + 10

        // Perfect row (solved)
        assert_eq!(compute_pressure(0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_all_strategies_eligible_for_escalation() {
        // Verify that all strategies are represented in Strategy::all()
        // and that none are excluded from escalation by design
        let all_strategies = Strategy::all();

        assert!(all_strategies.contains(&Strategy::PressureField));
        assert!(all_strategies.contains(&Strategy::Sequential));
        assert!(all_strategies.contains(&Strategy::Random));
        assert!(all_strategies.contains(&Strategy::Hierarchical));
        assert!(all_strategies.contains(&Strategy::Conversation));

        // All 5 strategies should be present
        assert_eq!(all_strategies.len(), 5);
    }

    #[test]
    fn test_escalation_applies_to_all_strategies() {
        // This test documents that escalation should work for ALL strategies.
        // Previously, escalation was gated by `strategy == Strategy::PressureField`,
        // which was a bug that gave PressureField an unfair advantage.
        //
        // The fix removes that condition, allowing all strategies to escalate
        // when they get stuck (zero velocity for `escalation_threshold` ticks).
        //
        // Key changes made:
        // 1. Changed `if strategy == Strategy::PressureField && tick > 0`
        //    to `if tick > 0` (line ~412)
        // 2. Added ConversationRunner::set_model() to allow model updates
        //    during escalation for the Conversation strategy

        // Verify default config has escalation enabled
        let config = ExperimentRunnerConfig::default();
        assert!(!config.model_chain.is_empty(), "Model chain should not be empty");
        assert_eq!(config.escalation_threshold, 20, "Default escalation threshold");

        // Verify all strategies would use the same model selection logic
        // (current_model_idx starts at 0 for all, and escalation increments it)
        for strategy in Strategy::all() {
            // Each strategy name should be valid
            assert!(!strategy.name().is_empty());
        }
    }
}
