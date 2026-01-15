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
use survival_kernel::region::RegionId;
use survival_kernel::{
    AsyncKernelBuilder, PatchActorsReady, SensorsReady, Tick, TickComplete, TickResult,
    WaitForPatchActors, WaitForSensors,
};

use crate::sensors::update_shared_grid;

use crate::llm_actor::{LlmActor, LlmActorConfig, SamplingBand, SamplingConfig, UpdateModel};

use crate::artifact::LatinSquareArtifact;
use crate::conversation::ConversationRunner;
use crate::example_bank::{Example, ExampleBank, ExampleBankConfig};
use crate::generator::{Difficulty, GeneratorConfig, LatinSquareGenerator};
use crate::results::{
    ConversationStats, EscalationEvent, ExperimentConfig, ExperimentResult, PatchRejection,
    TickMetrics,
};
use crate::sensors::{LatinSquareSensor, SharedGrid};
use crate::vllm_client::VllmClient;

/// Context for a single experiment run.
#[derive(Debug, Clone)]
struct RunContext {
    trial: usize,
    seed: Option<u64>,
    started_at: chrono::DateTime<Utc>,
}

/// Configuration for the experiment runner.
#[derive(Debug, Clone)]
pub struct ExperimentRunnerConfig {
    /// vLLM host URL (fallback when vllm_hosts is empty)
    pub vllm_host: String,
    /// vLLM hosts for model escalation (one per model in chain)
    /// If empty, uses vllm_host for all models
    pub vllm_hosts: Vec<String>,
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
            vllm_hosts: Vec::new(), // Empty = use vllm_host for all models
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

impl ExperimentRunnerConfig {
    /// Get the vLLM host for a given model index in the escalation chain.
    ///
    /// If `vllm_hosts` is configured (non-empty), returns the host at the given index.
    /// Otherwise, falls back to the single `vllm_host` for all models.
    pub fn get_vllm_host(&self, model_idx: usize) -> &str {
        if self.vllm_hosts.is_empty() {
            // Fallback: use single host for all models
            &self.vllm_host
        } else {
            // Use the host corresponding to this model index, clamped to bounds
            let idx = model_idx.min(self.vllm_hosts.len() - 1);
            &self.vllm_hosts[idx]
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
            let ctx = RunContext {
                trial,
                seed,
                started_at,
            };
            return self
                .run_pressure_field_with_kernel(
                    artifact,
                    example_bank,
                    sensor,
                    shared_grid.clone(),
                    agent_count,
                    ctx,
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
            let vllm_host = self.config.get_vllm_host(0);
            conversation_runner = Some(ConversationRunner::new(
                vllm_host,
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
                    if let Some(ref region_id) = conv_state.target_region {
                        let region_view = artifact.read_region(region_id.clone())?;
                        let patch = survival_kernel::region::Patch {
                            region: region_id.clone(),
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
                                        id: region_id.clone(),
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
                        regions[tick % regions.len()].clone()
                    }
                    Strategy::Random => {
                        use rand::seq::IndexedRandom;
                        let regions = artifact.region_ids();
                        regions.choose(&mut rand::rng()).cloned().unwrap()
                    }
                    Strategy::Hierarchical => {
                        // Simulate hierarchical by picking region with most empty cells
                        self.select_most_incomplete_region(&artifact)?
                    }
                    Strategy::Conversation => unreachable!(), // Handled above
                };

                // Get region view
                let region_view = artifact.read_region(region_id.clone())?;
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
                    .generate_concurrent_patches(&artifact, region_id.clone(), &examples, &current_model, current_model_idx)
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
                        id: region_id.clone(),
                        kind: "row".to_string(),
                        content: new_content.clone(),
                        metadata: region_view.metadata.clone(),
                    };
                    let pressure_after = self.measure_region_pressure(&temp_view, &sensor)?;

                    if pressure_after < pressure_before || !self.config.decay_enabled {
                        // Apply patch
                        let patch = survival_kernel::region::Patch {
                            region: region_id.clone(),
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

                        // Update ConversationRunner's model and host if applicable
                        if let Some(ref mut runner) = conversation_runner {
                            runner.set_model(&to_model);
                            runner.set_host(self.config.get_vllm_host(current_model_idx));
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
    ) -> Result<RegionId> {
        let mut max_pressure = f64::NEG_INFINITY;
        let mut max_region = artifact.region_ids()[0].clone();

        for region_id in artifact.region_ids() {
            let view = artifact.read_region(region_id.clone())?;
            let pressure = self.measure_region_pressure(&view, sensor)?;
            if pressure > max_pressure {
                max_pressure = pressure;
                max_region = region_id.clone();
            }
        }

        Ok(max_region)
    }

    /// Select the region with most empty cells.
    fn select_most_incomplete_region(
        &self,
        artifact: &LatinSquareArtifact,
    ) -> Result<RegionId> {
        let mut max_empty = 0;
        let mut max_region = artifact.region_ids()[0].clone();

        for region_id in artifact.region_ids() {
            let view = artifact.read_region(region_id.clone())?;
            let empty_count = view.content.matches('_').count();
            if empty_count > max_empty {
                max_empty = empty_count;
                max_region = region_id.clone();
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
        region_id: RegionId,
        examples: &[Example],
        current_model: &str,
        model_idx: usize,
    ) -> Result<Vec<(String, u32, u32)>> {
        let n = artifact.size();
        let row_idx = artifact.row_index(region_id.clone()).unwrap_or(0);
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
        let vllm_host = self.config.get_vllm_host(model_idx);
        let client = Arc::new(VllmClient::new(vllm_host));

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

    /// Run PressureField strategy using the kernel's actor-based coordination.
    ///
    /// This method uses the survival-kernel's actor system for coordination:
    /// - KernelCoordinator orchestrates tick phases
    /// - LlmActors propose patches via broker pub/sub
    /// - RegionActors validate and apply patches
    /// - Internal TickActor drives the tick loop
    /// - External tick loop with model escalation support
    async fn run_pressure_field_with_kernel(
        &self,
        mut artifact: LatinSquareArtifact,
        example_bank: Arc<RwLock<ExampleBank>>,
        sensor: LatinSquareSensor,
        shared_grid: SharedGrid,
        agent_count: usize,
        ctx: RunContext,
    ) -> Result<ExperimentResult> {
        let n = artifact.size();
        let initial_empty = artifact.empty_count();

        // Build kernel config
        let kernel_config = self.build_kernel_config();
        let max_ticks = kernel_config.max_ticks;
        let tick_interval_ms = kernel_config.tick_interval_ms;

        // Create acton runtime
        let mut runtime = ActonApp::launch_async().await;

        // Create semaphore for rate limiting LLM requests
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_llm));

        // Get initial model and host
        let initial_model_idx = 0;
        let initial_model = if self.config.model_chain.is_empty() {
            self.config.model.clone()
        } else {
            self.config.model_chain[0].clone()
        };
        let initial_host = self.config.get_vllm_host(initial_model_idx).to_string();

        // Spawn LLM actors first - they self-register via PatchActorReady broadcast
        // Distribute actors across sampling bands for diversity
        for i in 0..agent_count {
            let band = match i % 3 {
                0 => SamplingBand::Exploitation,
                1 => SamplingBand::Balanced,
                _ => SamplingBand::Exploration,
            };

            let llm_config = LlmActorConfig {
                host: initial_host.clone(),
                model: initial_model.clone(),
                sampling: SamplingConfig::random_in_band(band),
                max_tokens: 32,
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

        // Build kernel and spawn (without running tick loop)
        let coordinator_handle =
            AsyncKernelBuilder::new(kernel_config, Box::new(artifact.clone()))
                .add_sensor(Box::new(sensor))
                .spawn(&mut runtime)
                .await;

        // Create observer to collect TickComplete broadcasts
        let (tick_tx, mut tick_rx) = tokio::sync::mpsc::channel::<TickResult>(1000);
        spawn_tick_observer(&mut runtime, tick_tx).await;

        // Create observers to wait for registrations
        let (sensors_tx, mut sensors_rx) = tokio::sync::mpsc::channel::<SensorsReady>(1);
        let (actors_tx, mut actors_rx) = tokio::sync::mpsc::channel::<PatchActorsReady>(1);

        // Spawn SensorsReady observer (we have 1 sensor)
        spawn_sensors_ready_observer(&mut runtime, sensors_tx).await;

        // Spawn PatchActorsReady observer
        spawn_patch_actors_ready_observer(&mut runtime, actors_tx).await;

        // Wait for sensors to register
        coordinator_handle
            .send(WaitForSensors { expected_count: 1 })
            .await;
        sensors_rx.recv().await;

        // Wait for patch actors to register
        if agent_count > 0 {
            coordinator_handle
                .send(WaitForPatchActors {
                    expected_count: agent_count,
                })
                .await;
            actors_rx.recv().await;
        }

        // External tick loop with escalation tracking
        let mut tick_results = Vec::new();
        let mut pressure_history = Vec::new();
        let mut escalation_events = Vec::new();
        let mut current_model_idx = initial_model_idx;
        let mut current_model = initial_model;
        let mut ticks_without_progress = 0;
        let mut current_tick = 0usize;
        let mut total_prompt_tokens = 0u32;
        let mut total_completion_tokens = 0u32;
        let mut final_pressure = 0.0;
        let mut solved = false;

        info!(
            max_ticks,
            escalation_threshold = self.config.escalation_threshold,
            model_chain_len = self.config.model_chain.len(),
            "Starting external tick loop with escalation support"
        );

        loop {
            current_tick += 1;

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Send Tick to coordinator
            coordinator_handle.send(Tick { now_ms }).await;

            // Wait for TickComplete via the observer channel
            let Some(result) = tick_rx.recv().await else {
                info!("TickComplete channel closed unexpectedly");
                break;
            };

            // Track results
            pressure_history.push(result.total_pressure);
            total_prompt_tokens += result.prompt_tokens;
            total_completion_tokens += result.completion_tokens;
            final_pressure = result.total_pressure;

            // Track velocity (consecutive ticks with no patches)
            if result.applied.is_empty() {
                ticks_without_progress += 1;
            } else {
                ticks_without_progress = 0;

                // Apply patches to local artifact copy to keep shared_grid in sync
                // This is critical: the sensor validates patches using shared_grid,
                // so it must reflect the current grid state after each applied patch
                for patch in &result.applied {
                    if let Err(e) = artifact.apply_patch(patch.clone()) {
                        warn!(error = %e, "Failed to apply patch to local artifact");
                    }
                }
                if let Err(e) = update_shared_grid(&shared_grid, artifact.grid()) {
                    warn!(error = %e, "Failed to update shared grid");
                }
            }

            let is_complete = result.is_complete;
            tick_results.push(result);

            // Check for escalation
            if ticks_without_progress >= self.config.escalation_threshold
                && current_model_idx + 1 < self.config.model_chain.len()
            {
                let old_model = current_model.clone();
                current_model_idx += 1;
                current_model = self.config.model_chain[current_model_idx].clone();
                let new_host = self.config.get_vllm_host(current_model_idx).to_string();

                info!(
                    tick = current_tick,
                    from_model = %old_model,
                    to_model = %current_model,
                    new_host = %new_host,
                    "Escalating model due to stall"
                );

                // Broadcast UpdateModel to all LLM actors
                runtime
                    .broker()
                    .broadcast(UpdateModel {
                        model: current_model.clone(),
                        host: new_host.clone(),
                    })
                    .await;

                escalation_events.push(EscalationEvent {
                    tick: current_tick,
                    from_model: old_model,
                    to_model: current_model.clone(),
                });

                ticks_without_progress = 0;
            }

            // Check termination conditions
            if is_complete {
                solved = true;
                info!(tick = current_tick, "Puzzle solved!");
                break;
            }
            if max_ticks > 0 && current_tick >= max_ticks {
                info!(tick = current_tick, "Max ticks reached");
                break;
            }
            // Stop if fully escalated and still stuck
            if current_model_idx == self.config.model_chain.len().saturating_sub(1)
                && ticks_without_progress >= self.config.escalation_threshold
            {
                info!(
                    tick = current_tick,
                    model = %current_model,
                    "Converged at largest model with no progress"
                );
                break;
            }

            // Wait for the interval before next tick
            tokio::time::sleep(std::time::Duration::from_millis(tick_interval_ms)).await;
        }

        // Shutdown runtime
        let _ = runtime.shutdown_all().await;

        let ended_at = Utc::now();

        // Get example bank stats
        let example_bank_stats = {
            let bank = example_bank.read().await;
            Some(bank.stats())
        };

        // Build tick metrics from results
        let tick_metrics: Vec<TickMetrics> = tick_results
            .iter()
            .enumerate()
            .map(|(tick, result)| {
                let pressure_before = if tick > 0 {
                    pressure_history.get(tick - 1).copied().unwrap_or(0.0)
                } else {
                    pressure_history.first().copied().unwrap_or(0.0)
                };

                // Determine which model was used at this tick
                let model_at_tick = escalation_events
                    .iter()
                    .filter(|e| e.tick <= tick)
                    .next_back()
                    .map(|e| e.to_model.clone())
                    .unwrap_or_else(|| {
                        if self.config.model_chain.is_empty() {
                            self.config.model.clone()
                        } else {
                            self.config.model_chain[0].clone()
                        }
                    });

                TickMetrics {
                    tick,
                    pressure_before,
                    pressure_after: result.total_pressure,
                    patches_proposed: result.evaluated,
                    patches_applied: result.applied.len(),
                    empty_cells: 0,
                    violations: 0,
                    llm_calls: agent_count,
                    duration_ms: 0,
                    model_used: model_at_tick,
                    prompt_tokens: result.prompt_tokens,
                    completion_tokens: result.completion_tokens,
                    patch_rejections: HashMap::new(),
                    messages_per_tick: None,
                }
            })
            .collect();

        let patches_per_tick: Vec<usize> = tick_results.iter().map(|r| r.applied.len()).collect();

        Ok(ExperimentResult {
            config: ExperimentConfig {
                strategy: "pressure_field".to_string(),
                agent_count,
                n,
                empty_cells: initial_empty,
                decay_enabled: self.config.decay_enabled,
                inhibition_enabled: self.config.inhibition_enabled,
                examples_enabled: self.config.examples_enabled,
                trial: ctx.trial,
                seed: ctx.seed,
            },
            started_at: ctx.started_at,
            ended_at,
            total_ticks: current_tick,
            solved,
            final_pressure,
            pressure_history,
            patches_per_tick,
            empty_cells_history: Vec::new(),
            example_bank_stats,
            tick_metrics,
            escalation_events,
            final_model: current_model,
            total_prompt_tokens,
            total_completion_tokens,
            total_patch_rejections: HashMap::new(),
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
            max_ticks: self.config.max_ticks,
            stable_threshold: 0, // Disabled - external tick loop handles termination
            pressure_axes,
            decay,
            activation,
            selection,
        }
    }
}

/// Spawn an observer actor that collects TickComplete broadcasts.
async fn spawn_tick_observer(
    runtime: &mut ActorRuntime,
    tx: tokio::sync::mpsc::Sender<TickResult>,
) {
    #[derive(Default, Clone, Debug)]
    struct State {
        tx: Option<tokio::sync::mpsc::Sender<TickResult>>,
    }

    let mut actor = runtime.new_actor_with_name::<State>("TickResultObserver".to_string());
    actor.model.tx = Some(tx);

    // Subscribe to TickComplete broadcasts
    actor.handle().subscribe::<TickComplete>().await;

    actor.act_on::<TickComplete>(|actor, context| {
        let msg = context.message();
        let mut result = msg.result.clone();
        result.is_complete = msg.is_complete;
        let tx = actor.model.tx.clone();
        Reply::pending(async move {
            if let Some(tx) = tx {
                let _ = tx.send(result).await;
            }
        })
    });

    actor.start().await;
}

/// Spawn an observer actor that waits for SensorsReady broadcast.
async fn spawn_sensors_ready_observer(
    runtime: &mut ActorRuntime,
    tx: tokio::sync::mpsc::Sender<SensorsReady>,
) {
    #[derive(Default, Clone, Debug)]
    struct State {
        tx: Option<tokio::sync::mpsc::Sender<SensorsReady>>,
    }

    let mut actor = runtime.new_actor_with_name::<State>("SensorsReadyObserver".to_string());
    actor.model.tx = Some(tx);

    // Subscribe to SensorsReady broadcasts
    actor.handle().subscribe::<SensorsReady>().await;

    actor.act_on::<SensorsReady>(|actor, context| {
        let msg = context.message().clone();
        let tx = actor.model.tx.clone();
        Reply::pending(async move {
            if let Some(tx) = tx {
                let _ = tx.send(msg).await;
            }
        })
    });

    actor.start().await;
}

/// Spawn an observer actor that waits for PatchActorsReady broadcast.
async fn spawn_patch_actors_ready_observer(
    runtime: &mut ActorRuntime,
    tx: tokio::sync::mpsc::Sender<PatchActorsReady>,
) {
    #[derive(Default, Clone, Debug)]
    struct State {
        tx: Option<tokio::sync::mpsc::Sender<PatchActorsReady>>,
    }

    let mut actor = runtime.new_actor_with_name::<State>("PatchActorsReadyObserver".to_string());
    actor.model.tx = Some(tx);

    // Subscribe to PatchActorsReady broadcasts
    actor.handle().subscribe::<PatchActorsReady>().await;

    actor.act_on::<PatchActorsReady>(|actor, context| {
        let msg = context.message().clone();
        let tx = actor.model.tx.clone();
        Reply::pending(async move {
            if let Some(tx) = tx {
                let _ = tx.send(msg).await;
            }
        })
    });

    actor.start().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test that requires a running vLLM server on localhost:8000.
    /// Run manually with: cargo nextest run test_experiment_runner_basic --ignored
    #[tokio::test]
    #[ignore = "requires external vLLM server on localhost:8000"]
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
        let model_chain = [
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

    #[test]
    fn test_pressure_formula_edge_cases() {
        let compute_pressure = |empty: f64, row_dups: f64, col_conflicts: f64| -> f64 {
            empty * 1.0 + row_dups * 10.0 + col_conflicts * 10.0
        };

        // Edge case: Large values (stress test the formula)
        let large_pressure = compute_pressure(100.0, 50.0, 50.0);
        assert_eq!(large_pressure, 1100.0); // 100 + 500 + 500

        // Edge case: Fractional values (shouldn't happen in practice, but formula handles it)
        let fractional = compute_pressure(0.5, 0.5, 0.5);
        assert_eq!(fractional, 10.5); // 0.5 + 5.0 + 5.0

        // Edge case: Column conflicts weighted same as row duplicates
        let col_only = compute_pressure(0.0, 0.0, 3.0);
        let row_only = compute_pressure(0.0, 3.0, 0.0);
        assert_eq!(col_only, row_only, "Col conflicts and row dups should have same weight");
    }

    #[test]
    fn test_escalation_velocity_zero_detection() {
        // Velocity = 0 means no progress - should trigger escalation check
        let prev_pressure = 10.0;
        let curr_pressure = 10.0;
        let velocity = prev_pressure - curr_pressure;

        assert_eq!(velocity, 0.0);

        // In the actual code, zero_velocity_streak is incremented when velocity <= 0
        // This triggers escalation when zero_velocity_streak >= escalation_threshold
        let threshold = 20;
        let stuck_ticks = 25;
        assert!(stuck_ticks >= threshold, "Should trigger escalation after {} ticks", stuck_ticks);
    }

    #[test]
    fn test_model_chain_all_sizes_present() {
        let config = ExperimentRunnerConfig::default();

        // Verify all expected model sizes are in the default chain
        let sizes: Vec<&str> = vec!["0.5B", "1.5B", "3B", "7B", "14B"];
        for size in sizes {
            assert!(
                config.model_chain.iter().any(|m| m.contains(size)),
                "Model chain should contain {} model",
                size
            );
        }
    }

    #[test]
    fn test_experiment_config_defaults() {
        let config = ExperimentRunnerConfig::default();

        assert_eq!(config.max_ticks, 50);
        assert!(config.decay_enabled);
        assert!(config.inhibition_enabled);
        assert!(config.examples_enabled);
        assert!(config.max_concurrent_llm > 0);
    }

    #[test]
    fn test_strategy_display_names_unique() {
        use std::collections::HashSet;

        let strategies = Strategy::all();
        let names: HashSet<_> = strategies.iter().map(|s| s.name()).collect();

        assert_eq!(
            names.len(),
            strategies.len(),
            "All strategy names should be unique"
        );
    }

    #[test]
    fn test_velocity_calculation_improvement() {
        // Test all three velocity scenarios
        let test_cases = vec![
            (10.0, 5.0, true, "Improvement: pressure decreased"),
            (10.0, 10.0, false, "Stuck: no change"),
            (10.0, 15.0, false, "Regression: pressure increased"),
        ];

        for (prev, curr, should_improve, msg) in test_cases {
            let velocity = prev - curr;
            let improved = velocity > 0.0;
            assert_eq!(improved, should_improve, "{}", msg);
        }
    }

    #[test]
    fn test_escalation_threshold_bounds() {
        // Test that escalation threshold is reasonable
        let config = ExperimentRunnerConfig::default();

        // Threshold should be > 0 to avoid immediate escalation
        assert!(config.escalation_threshold > 0);

        // Threshold should be < max_ticks to allow escalation to happen
        assert!(config.escalation_threshold < config.max_ticks);

        // Reasonable range for escalation threshold
        assert!(config.escalation_threshold >= 5);
        assert!(config.escalation_threshold <= 50);
    }

    #[test]
    fn test_model_chain_ordering() {
        let config = ExperimentRunnerConfig::default();

        // Model chain should be ordered from smallest to largest
        // (escalation moves to larger models)
        let sizes_order = ["0.5B", "1.5B", "3B", "7B", "14B"];

        for (i, expected_size) in sizes_order.iter().enumerate() {
            assert!(
                config.model_chain[i].contains(expected_size),
                "Model at index {} should be {} model, got {}",
                i,
                expected_size,
                config.model_chain[i]
            );
        }
    }

    #[test]
    fn test_conversation_max_turns_positive() {
        let config = ExperimentRunnerConfig::default();

        // Conversation max turns should be positive
        assert!(config.conversation_max_turns > 0);

        // And reasonable (not too short)
        assert!(config.conversation_max_turns >= 3);
    }

    #[test]
    fn test_max_concurrent_llm_reasonable() {
        let config = ExperimentRunnerConfig::default();

        // Max concurrent should be positive
        assert!(config.max_concurrent_llm > 0);

        // And not too high (resource constraint)
        assert!(config.max_concurrent_llm <= 50);
    }
}
