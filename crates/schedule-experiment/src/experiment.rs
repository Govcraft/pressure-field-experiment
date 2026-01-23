//! Experiment runner for schedule coordination experiments.
//!
//! Orchestrates the experiment lifecycle:
//! 1. Generate scheduling problem
//! 2. Set up kernel with sensors and actors
//! 3. Run tick loop until solved or max ticks
//! 4. Collect metrics and results

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use rand::Rng;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use acton_reactive::prelude::*;
use survival_kernel::artifact::Artifact;
use survival_kernel::config::{
    ActivationConfig, DecayConfig, KernelConfig, PressureAxisConfig, SelectionConfig,
};
use survival_kernel::pressure::Sensor;
use survival_kernel::{
    AsyncKernelBuilder, PatchActorsReady, SensorsReady, Tick, TickComplete, TickResult,
    WaitForPatchActors, WaitForSensors,
};

use crate::artifact::{ScheduleArtifact, SharedSchedule};
use crate::conversation::ConversationRunner;
use crate::example_bank::{ExampleBank, ExampleBankConfig};
use crate::generator::{ScheduleGenerator, ScheduleGeneratorConfig};
use crate::llm_actor::{
    LlmActor, LlmActorConfig, SamplingBand, SamplingConfig, UpdateBand, UpdateModel,
};
use crate::results::{
    BandEscalationEvent, ConversationStats, EscalationEvent, ExperimentConfig, ExperimentResult,
    PatchRejection, TickMetrics,
};
use crate::sensors::CombinedScheduleSensor;
use crate::vllm_client::VllmClient;

/// Context for a single experiment run.
#[derive(Debug, Clone)]
struct RunContext {
    agent_count: usize,
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
    pub vllm_hosts: Vec<String>,
    /// Base model name
    pub model: String,
    /// Model escalation chain
    pub model_chain: Vec<String>,
    /// Number of ticks with zero velocity before escalating model
    pub escalation_threshold: usize,
    /// Number of ticks before escalating sampling band (before model escalation)
    /// Band escalates at: band_escalation_interval, 2*band_escalation_interval
    /// Model escalates at: 3*band_escalation_interval
    pub band_escalation_interval: usize,
    /// Maximum ticks before giving up
    pub max_ticks: usize,
    /// Maximum concurrent LLM requests
    pub max_concurrent_llm: usize,
    /// Generator configuration
    pub generator_config: ScheduleGeneratorConfig,
    /// Enable decay
    pub decay_enabled: bool,
    /// Enable inhibition
    pub inhibition_enabled: bool,
    /// Enable few-shot examples
    pub examples_enabled: bool,
    /// Example bank configuration
    pub example_bank_config: ExampleBankConfig,
}

impl Default for ExperimentRunnerConfig {
    fn default() -> Self {
        Self {
            vllm_host: "http://localhost:11434".to_string(),
            vllm_hosts: Vec::new(),
            model: "qwen2.5:1.5b".to_string(),
            model_chain: vec![
                "qwen2.5:1.5b".to_string(),
                "qwen2.5:7b".to_string(),
                "qwen2.5:14b".to_string(),
            ],
            escalation_threshold: 21, // Divisible by 3 for clean band intervals
            band_escalation_interval: 7, // 21 / 3 = 7 ticks per band level
            max_ticks: 50,
            max_concurrent_llm: 8,
            generator_config: ScheduleGeneratorConfig::easy(),
            decay_enabled: true,
            inhibition_enabled: true,
            examples_enabled: true,
            example_bank_config: ExampleBankConfig::default(),
        }
    }
}

impl ExperimentRunnerConfig {
    /// Get the vLLM host for a given model index in the escalation chain.
    pub fn get_vllm_host(&self, model_idx: usize) -> &str {
        if self.vllm_hosts.is_empty() {
            &self.vllm_host
        } else {
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
    /// Conversation-based coordination (AutoGen-style baseline)
    Conversation,
    /// Sequential round-robin
    Sequential,
    /// Random selection
    Random,
    /// Hierarchical (simulated manager)
    Hierarchical,
}

impl Strategy {
    /// Get all strategies for grid experiments.
    pub fn all() -> Vec<Self> {
        vec![
            Self::PressureField,
            Self::Conversation,
            Self::Sequential,
            Self::Random,
            Self::Hierarchical,
        ]
    }

    /// Get the name of this strategy.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PressureField => "pressure_field",
            Self::Conversation => "conversation",
            Self::Sequential => "sequential",
            Self::Random => "random",
            Self::Hierarchical => "hierarchical",
        }
    }
}

/// Context for running baseline strategies.
struct BaselineRunContext {
    artifact: ScheduleArtifact,
    example_bank: Arc<RwLock<ExampleBank>>,
    sensor: CombinedScheduleSensor,
    shared_schedule: SharedSchedule,
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
        let _start_time = Instant::now();

        // Generate schedule problem
        let actual_seed = seed.unwrap_or_else(rand::random);
        let mut generator =
            ScheduleGenerator::new(self.config.generator_config.clone(), actual_seed);
        let artifact = generator.generate();

        let num_meetings = artifact.meetings().len();
        let num_rooms = artifact.rooms().len();
        let initial_unscheduled = artifact.schedule().count_unscheduled();

        info!(
            strategy = strategy.name(),
            agents = agent_count,
            trial = trial,
            meetings = num_meetings,
            rooms = num_rooms,
            unscheduled = initial_unscheduled,
            "Starting schedule experiment"
        );

        // Set up shared schedule for sensors
        let shared_schedule: SharedSchedule = artifact
            .shared_schedule()
            .unwrap_or_else(|| Arc::new(std::sync::RwLock::new(artifact.schedule().clone())));

        // Set up example bank
        let example_bank = if self.config.examples_enabled {
            Arc::new(RwLock::new(ExampleBank::new(
                self.config.example_bank_config.clone(),
            )))
        } else {
            Arc::new(RwLock::new(ExampleBank::disabled()))
        };

        // Create sensor
        let sensor = CombinedScheduleSensor::new(shared_schedule.clone());

        // For PressureField strategy, use the kernel-based coordination
        if strategy == Strategy::PressureField {
            let ctx = RunContext {
                agent_count,
                trial,
                seed,
                started_at,
            };
            return self
                .run_pressure_field_with_kernel(
                    artifact,
                    example_bank,
                    sensor,
                    shared_schedule,
                    agent_count,
                    ctx,
                )
                .await;
        }

        // For Conversation strategy, use multi-agent dialogue
        if strategy == Strategy::Conversation {
            let baseline_ctx = BaselineRunContext {
                artifact,
                example_bank,
                sensor,
                shared_schedule,
            };
            let ctx = RunContext {
                agent_count,
                trial,
                seed,
                started_at,
            };
            return self.run_conversation_strategy(ctx, baseline_ctx).await;
        }

        // For baseline strategies, run simple tick loop
        let baseline_ctx = BaselineRunContext {
            artifact,
            example_bank,
            sensor,
            shared_schedule,
        };
        let run_ctx = RunContext {
            agent_count,
            trial,
            seed,
            started_at,
        };
        self.run_baseline_strategy(strategy, baseline_ctx, run_ctx)
            .await
    }

    /// Run the pressure-field strategy using the kernel.
    async fn run_pressure_field_with_kernel(
        &self,
        mut artifact: ScheduleArtifact,
        example_bank: Arc<RwLock<ExampleBank>>,
        sensor: CombinedScheduleSensor,
        shared_schedule: SharedSchedule,
        agent_count: usize,
        ctx: RunContext,
    ) -> Result<ExperimentResult> {
        let num_meetings = artifact.meetings().len();
        let initial_unscheduled = artifact.schedule().count_unscheduled();

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

        // Create shared artifact for LLM actors (shares rejected_patches via Arc)
        let shared_artifact = Arc::new(artifact.clone());

        // Spawn LLM actors - they self-register via PatchActorReady broadcast
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
                max_tokens: 256, // Schedules need more tokens than Latin squares
                band,
                randomize_sampling: true,
            };

            LlmActor::spawn(
                &mut runtime,
                format!("LlmActor:{}", i),
                llm_config,
                semaphore.clone(),
                example_bank.clone(),
                shared_artifact.clone(),
            )
            .await;
        }

        // Build kernel and spawn
        let coordinator_handle = AsyncKernelBuilder::new(kernel_config, Box::new(artifact.clone()))
            .add_sensor(Box::new(sensor))
            .spawn(&mut runtime)
            .await;

        // Create observer to collect TickComplete broadcasts
        let (tick_tx, mut tick_rx) = tokio::sync::mpsc::channel::<TickResult>(1000);
        spawn_tick_observer(&mut runtime, tick_tx).await;

        // Create observers to wait for registrations
        let (sensors_tx, mut sensors_rx) = tokio::sync::mpsc::channel::<SensorsReady>(1);
        let (actors_tx, mut actors_rx) = tokio::sync::mpsc::channel::<PatchActorsReady>(1);

        spawn_sensors_ready_observer(&mut runtime, sensors_tx).await;
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
        let mut band_escalation_events = Vec::new();
        let mut current_model_idx = initial_model_idx;
        let mut current_model = initial_model;
        let mut current_band_level: usize = 0; // 0 = Exploitation, 1 = Balanced, 2 = Exploration
        let mut ticks_without_progress = 0;
        let mut previous_pressure = f64::MAX; // First tick always shows "progress"
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

            // Track velocity based on actual pressure improvement (not just patch application)
            // This ensures escalation triggers when stuck even if patches are being applied
            let pressure_improved = result.total_pressure < previous_pressure - 0.01; // Small epsilon
            if pressure_improved {
                debug!(
                    prev = previous_pressure,
                    curr = result.total_pressure,
                    "Pressure improved, resetting stall counter"
                );
                ticks_without_progress = 0;
            } else {
                ticks_without_progress += 1;
                debug!(
                    prev = previous_pressure,
                    curr = result.total_pressure,
                    ticks_without_progress,
                    "No pressure improvement"
                );
            }
            previous_pressure = result.total_pressure;

            // Apply patches regardless (they may have local effects)
            if !result.applied.is_empty() {
                for patch in &result.applied {
                    if let Err(e) = artifact.apply_patch(patch.clone()) {
                        warn!(error = %e, "Failed to apply patch to local artifact");
                    }
                }
                // Update shared schedule
                if let Ok(mut schedule) = shared_schedule.write() {
                    *schedule = artifact.schedule().clone();
                }
            }

            // Apply rejection decay each tick (negative pheromone evaporation)
            shared_artifact.apply_rejection_decay(0.95, 0.1);

            // Log rejection stats periodically
            let (rejection_count, total_weight) = shared_artifact.rejection_stats();
            if rejection_count > 0 {
                debug!(
                    tick = current_tick,
                    rejection_count,
                    total_weight = format!("{:.2}", total_weight),
                    "Negative pheromone stats"
                );
            }

            let is_complete = result.is_complete;
            tick_results.push(result);

            // Check for escalation (progressive: band first, then model)
            if ticks_without_progress >= self.config.band_escalation_interval {
                if current_band_level < 2 {
                    // Escalate sampling band first
                    let old_band = match current_band_level {
                        0 => SamplingBand::Exploitation,
                        1 => SamplingBand::Balanced,
                        _ => SamplingBand::Exploration,
                    };
                    current_band_level += 1;
                    let new_band = match current_band_level {
                        1 => SamplingBand::Balanced,
                        _ => SamplingBand::Exploration,
                    };

                    info!(
                        tick = current_tick,
                        from_band = ?old_band,
                        to_band = ?new_band,
                        "Escalating sampling band before model escalation"
                    );

                    // Broadcast UpdateBand to all LLM actors
                    runtime
                        .broker()
                        .broadcast(UpdateBand { band: new_band })
                        .await;

                    band_escalation_events.push(BandEscalationEvent {
                        tick: current_tick,
                        from_band: format!("{:?}", old_band),
                        to_band: format!("{:?}", new_band),
                    });

                    ticks_without_progress = 0;
                } else if current_model_idx + 1 < self.config.model_chain.len() {
                    // Already at Exploration band, escalate model and reset band
                    let old_model = current_model.clone();
                    current_model_idx += 1;
                    current_model = self.config.model_chain[current_model_idx].clone();
                    let new_host = self.config.get_vllm_host(current_model_idx).to_string();

                    info!(
                        tick = current_tick,
                        from_model = %old_model,
                        to_model = %current_model,
                        new_host = %new_host,
                        "Escalating model and resetting band to Exploitation"
                    );

                    // Broadcast UpdateModel to all LLM actors
                    runtime
                        .broker()
                        .broadcast(UpdateModel {
                            model: current_model.clone(),
                            host: new_host,
                        })
                        .await;

                    // Reset to Exploitation band for new model
                    current_band_level = 0;
                    runtime
                        .broker()
                        .broadcast(UpdateBand {
                            band: SamplingBand::Exploitation,
                        })
                        .await;

                    escalation_events.push(EscalationEvent {
                        tick: current_tick,
                        from_model: old_model,
                        to_model: current_model.clone(),
                    });

                    ticks_without_progress = 0;
                }
            }

            // Check termination conditions
            if is_complete {
                solved = true;
                info!(trial = ctx.trial, tick = current_tick, "Schedule solved!");
                break;
            }
            if max_ticks > 0 && current_tick >= max_ticks {
                info!(
                    trial = ctx.trial,
                    tick = current_tick,
                    "Schedule unsolved! Max ticks reached"
                );
                break;
            }
            // Stop if fully escalated (max band + max model) and still stuck
            if current_model_idx == self.config.model_chain.len().saturating_sub(1)
                && current_band_level >= 2
                && ticks_without_progress >= self.config.band_escalation_interval
            {
                info!(
                    trial = ctx.trial,
                    tick = current_tick,
                    model = %current_model,
                    band_level = current_band_level,
                    "Schedule unsolved! Converged at largest model and highest band"
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
                n: num_meetings,
                empty_cells: initial_unscheduled,
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
            band_escalation_events,
            final_model: current_model,
            total_prompt_tokens,
            total_completion_tokens,
            total_patch_rejections: HashMap::new(),
            conversation_stats: None,
        })
    }

    /// Run baseline strategies (Sequential, Random, Hierarchical).
    async fn run_baseline_strategy(
        &self,
        strategy: Strategy,
        mut ctx: BaselineRunContext,
        run_ctx: RunContext,
    ) -> Result<ExperimentResult> {
        let num_meetings = ctx.artifact.meetings().len();
        let initial_unscheduled = ctx.artifact.schedule().count_unscheduled();

        // Tracking
        let mut pressure_history = Vec::new();
        let mut patches_per_tick = Vec::new();
        let mut tick_metrics = Vec::new();

        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let total_patch_rejections: HashMap<PatchRejection, usize> = HashMap::new();

        // Initial measurement
        let initial_pressure = self.measure_total_pressure(&ctx.artifact, &ctx.sensor)?;
        pressure_history.push(initial_pressure);

        // Model escalation state
        let mut current_model_idx: usize = 0;
        let mut zero_velocity_ticks: usize = 0;
        let mut escalation_events: Vec<EscalationEvent> = Vec::new();

        let vllm_client = VllmClient::new(self.config.get_vllm_host(0));
        let mut current_model = if self.config.model_chain.is_empty() {
            self.config.model.clone()
        } else {
            self.config.model_chain[0].clone()
        };

        // Run tick loop
        for tick in 0..self.config.max_ticks {
            let tick_start = Instant::now();

            // Apply decay to example bank
            if self.config.decay_enabled {
                let bank = ctx.example_bank.read().await;
                bank.apply_decay();
            }

            // Select region based on strategy
            let region_id = match strategy {
                Strategy::PressureField => unreachable!("PressureField handled separately"),
                Strategy::Conversation => unreachable!("Conversation handled separately"),
                Strategy::Sequential => {
                    let regions = ctx.artifact.region_ids();
                    regions[tick % regions.len()].clone()
                }
                Strategy::Random => {
                    let regions = ctx.artifact.region_ids();
                    let idx = rand::rng().random_range(0..regions.len());
                    regions[idx].clone()
                }
                Strategy::Hierarchical => {
                    self.select_highest_pressure_region(&ctx.artifact, &ctx.sensor)?
                }
            };

            let region_view = ctx.artifact.read_region(region_id.clone())?;

            // Get examples for few-shot learning
            let examples = if self.config.examples_enabled {
                let bank = ctx.example_bank.read().await;
                bank.get_examples_for_prompt()
            } else {
                vec![]
            };

            // Generate prompt and call LLM
            let prompt = self.build_schedule_prompt(&region_view, &examples);
            let response = vllm_client
                .generate_with_usage(&current_model, &prompt, 0.4, 0.9, 256)
                .await?;

            total_prompt_tokens += response.prompt_tokens;
            total_completion_tokens += response.completion_tokens;

            // Parse response and apply patch
            let mut patches_applied = 0;
            if let Some(new_content) = self.parse_schedule_response(&response.content)
                && new_content != region_view.content
            {
                let patch = survival_kernel::region::Patch {
                    region: region_id.clone(),
                    op: survival_kernel::region::PatchOp::Replace(new_content.clone()),
                    rationale: format!("{} strategy patch", strategy.name()),
                    expected_delta: HashMap::new(),
                };

                // Hierarchical validates patches (centralized control with quality check)
                // Sequential/Random do not validate (uncoordinated controls)
                let should_apply = match strategy {
                    Strategy::Hierarchical => {
                        let (should_accept, _delta) = ctx.artifact.evaluate_patch(&patch);
                        should_accept
                    }
                    _ => true, // Sequential and Random apply unconditionally
                };

                if should_apply && ctx.artifact.apply_patch(patch).is_ok() {
                    // Update shared schedule
                    if let Ok(mut schedule) = ctx.shared_schedule.write() {
                        *schedule = ctx.artifact.schedule().clone();
                    }

                    // Add to example bank
                    if self.config.examples_enabled {
                        let pressure_before =
                            self.measure_region_pressure(&region_view, &ctx.sensor)?;
                        let temp_view = survival_kernel::region::RegionView {
                            id: region_id.clone(),
                            kind: "time_block".to_string(),
                            content: new_content.clone(),
                            metadata: region_view.metadata.clone(),
                        };
                        let pressure_after =
                            self.measure_region_pressure(&temp_view, &ctx.sensor)?;

                        let bank = ctx.example_bank.read().await;
                        bank.add_example(
                            region_view.content.clone(),
                            new_content,
                            pressure_before,
                            pressure_after,
                        );
                    }

                    patches_applied = 1;
                }
            }

            // Track tick results
            let current_pressure = self.measure_total_pressure(&ctx.artifact, &ctx.sensor)?;
            pressure_history.push(current_pressure);
            patches_per_tick.push(patches_applied);

            // Track velocity for escalation
            if patches_applied == 0 {
                zero_velocity_ticks += 1;
            } else {
                zero_velocity_ticks = 0;
            }

            // Check for model escalation
            if zero_velocity_ticks >= self.config.escalation_threshold
                && current_model_idx + 1 < self.config.model_chain.len()
            {
                let old_model = current_model.clone();
                current_model_idx += 1;
                current_model = self.config.model_chain[current_model_idx].clone();

                info!(
                    tick = tick,
                    from_model = %old_model,
                    to_model = %current_model,
                    "Escalating model due to stall"
                );

                escalation_events.push(EscalationEvent {
                    tick,
                    from_model: old_model,
                    to_model: current_model.clone(),
                });

                zero_velocity_ticks = 0;
            }

            let tick_duration = tick_start.elapsed();

            tick_metrics.push(TickMetrics {
                tick,
                pressure_before: pressure_history.get(tick).copied().unwrap_or(0.0),
                pressure_after: current_pressure,
                patches_proposed: 1,
                patches_applied,
                empty_cells: 0,
                violations: 0,
                llm_calls: 1,
                duration_ms: tick_duration.as_millis() as u64,
                model_used: current_model.clone(),
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
                patch_rejections: HashMap::new(),
                messages_per_tick: None,
            });

            // Check completion
            if ctx.artifact.is_solved() {
                info!(trial = run_ctx.trial, tick = tick, "Schedule solved!");
                break;
            }

            // Check fully escalated and stuck
            if current_model_idx == self.config.model_chain.len().saturating_sub(1)
                && zero_velocity_ticks >= self.config.escalation_threshold
            {
                info!(
                    trial = run_ctx.trial,
                    tick = tick,
                    "Schedule unsolved! Converged at largest model"
                );
                break;
            }
        }

        let ended_at = Utc::now();
        let final_pressure = pressure_history.last().copied().unwrap_or(0.0);
        let solved = ctx.artifact.is_solved();

        let example_bank_stats = {
            let bank = ctx.example_bank.read().await;
            Some(bank.stats())
        };

        Ok(ExperimentResult {
            config: ExperimentConfig {
                strategy: strategy.name().to_string(),
                agent_count: run_ctx.agent_count,
                n: num_meetings,
                empty_cells: initial_unscheduled,
                decay_enabled: self.config.decay_enabled,
                inhibition_enabled: self.config.inhibition_enabled,
                examples_enabled: self.config.examples_enabled,
                trial: run_ctx.trial,
                seed: run_ctx.seed,
            },
            started_at: run_ctx.started_at,
            ended_at,
            total_ticks: tick_metrics.len(),
            solved,
            final_pressure,
            pressure_history,
            patches_per_tick,
            empty_cells_history: Vec::new(),
            example_bank_stats,
            tick_metrics,
            escalation_events,
            band_escalation_events: Vec::new(), // Baselines don't use band escalation
            final_model: current_model,
            total_prompt_tokens,
            total_completion_tokens,
            total_patch_rejections,
            conversation_stats: None,
        })
    }

    /// Run the Conversation strategy (AutoGen-style multi-agent dialogue).
    ///
    /// This is the key comparison baseline for pressure-field coordination:
    /// - Uses explicit message-passing between roles (Coordinator, Proposer, Validator)
    /// - Sequential by design - cannot parallelize
    /// - 3-5 LLM calls per tick (vs N for pressure_field, 1 for other baselines)
    /// - Validates proposals before applying (like pressure_field)
    async fn run_conversation_strategy(
        &self,
        ctx: RunContext,
        mut baseline_ctx: BaselineRunContext,
    ) -> Result<ExperimentResult> {
        let num_meetings = baseline_ctx.artifact.meetings().len();
        let initial_unscheduled = baseline_ctx.artifact.schedule().count_unscheduled();

        // Create conversation runner
        let initial_model = if self.config.model_chain.is_empty() {
            self.config.model.clone()
        } else {
            self.config.model_chain[0].clone()
        };
        let mut runner = ConversationRunner::new(
            self.config.get_vllm_host(0),
            &initial_model,
            5, // max_turns per tick
        )?;

        // Tracking
        let mut pressure_history = Vec::new();
        let mut patches_per_tick = Vec::new();
        let mut tick_metrics = Vec::new();
        let mut escalation_events: Vec<EscalationEvent> = Vec::new();
        let mut total_messages: usize = 0;
        let mut consensus_ticks: usize = 0;
        let mut total_turns_to_consensus: usize = 0;

        // Initial measurement
        let initial_pressure =
            self.measure_total_pressure(&baseline_ctx.artifact, &baseline_ctx.sensor)?;
        pressure_history.push(initial_pressure);

        // Model escalation state
        let mut current_model_idx: usize = 0;
        let mut current_model = initial_model;
        let mut zero_velocity_ticks: usize = 0;

        // Run tick loop
        let mut tick_count = 0usize;
        for tick in 0..self.config.max_ticks {
            tick_count = tick + 1;
            let tick_start = Instant::now();

            // Run conversation for this tick
            let (final_patch, state) = runner
                .run_tick(&baseline_ctx.artifact, &baseline_ctx.shared_schedule)
                .await?;

            let messages_this_tick = state.total_messages();
            total_messages += messages_this_tick;

            if state.final_patch.is_some() {
                consensus_ticks += 1;
                total_turns_to_consensus += state.current_turn;
            }

            // Apply validated patch
            let mut patches_applied = 0;
            if let Some(patch_content) = final_patch {
                // Get target region
                if let Some(region_id) = state.target_region {
                    let patch = survival_kernel::region::Patch {
                        region: region_id.clone(),
                        op: survival_kernel::region::PatchOp::Replace(patch_content.clone()),
                        rationale: "conversation consensus patch".to_string(),
                        expected_delta: HashMap::new(),
                    };

                    // Evaluate patch before applying (like pressure_field)
                    let (should_accept, _delta) = baseline_ctx.artifact.evaluate_patch(&patch);

                    if should_accept && baseline_ctx.artifact.apply_patch(patch).is_ok() {
                        // Update shared schedule
                        if let Ok(mut schedule) = baseline_ctx.shared_schedule.write() {
                            *schedule = baseline_ctx.artifact.schedule().clone();
                        }
                        patches_applied = 1;
                    }
                }
            }

            // Track tick results
            let current_pressure =
                self.measure_total_pressure(&baseline_ctx.artifact, &baseline_ctx.sensor)?;
            pressure_history.push(current_pressure);
            patches_per_tick.push(patches_applied);

            // Track velocity for escalation
            if patches_applied == 0 {
                zero_velocity_ticks += 1;
            } else {
                zero_velocity_ticks = 0;
            }

            // Check for model escalation
            if zero_velocity_ticks >= self.config.escalation_threshold
                && current_model_idx + 1 < self.config.model_chain.len()
            {
                let old_model = current_model.clone();
                current_model_idx += 1;
                current_model = self.config.model_chain[current_model_idx].clone();
                let new_host = self.config.get_vllm_host(current_model_idx);

                info!(
                    tick = tick,
                    from_model = %old_model,
                    to_model = %current_model,
                    "Escalating model due to stall"
                );

                // Update runner with new model
                runner.set_model(&current_model);
                runner.set_host(new_host);

                escalation_events.push(EscalationEvent {
                    tick,
                    from_model: old_model,
                    to_model: current_model.clone(),
                });

                zero_velocity_ticks = 0;
            }

            let tick_duration = tick_start.elapsed();

            tick_metrics.push(TickMetrics {
                tick,
                pressure_before: pressure_history.get(tick).copied().unwrap_or(0.0),
                pressure_after: current_pressure,
                patches_proposed: 1, // Conversation proposes one patch per tick
                patches_applied,
                empty_cells: 0,
                violations: 0,
                llm_calls: messages_this_tick, // Each message = 1 LLM call
                duration_ms: tick_duration.as_millis() as u64,
                model_used: current_model.clone(),
                prompt_tokens: 0, // Could track via runner if needed
                completion_tokens: 0,
                patch_rejections: HashMap::new(),
                messages_per_tick: Some(messages_this_tick),
            });

            // Check completion
            if baseline_ctx.artifact.is_solved() {
                info!(trial = ctx.trial, tick = tick, "Schedule solved!");
                break;
            }

            // Check fully escalated and stuck
            if current_model_idx == self.config.model_chain.len().saturating_sub(1)
                && zero_velocity_ticks >= self.config.escalation_threshold
            {
                info!(
                    trial = ctx.trial,
                    tick = tick,
                    "Schedule unsolved! Converged at largest model"
                );
                break;
            }
        }

        let ended_at = Utc::now();
        let final_pressure = pressure_history.last().copied().unwrap_or(0.0);
        let solved = baseline_ctx.artifact.is_solved();

        // Log if max ticks reached without solving
        if !solved && tick_count >= self.config.max_ticks {
            info!(
                trial = ctx.trial,
                tick = tick_count,
                "Schedule unsolved! Max ticks reached"
            );
        }

        let example_bank_stats = {
            let bank = baseline_ctx.example_bank.read().await;
            Some(bank.stats())
        };

        // Build conversation statistics
        let avg_messages_per_tick = if tick_count > 0 {
            total_messages as f64 / tick_count as f64
        } else {
            0.0
        };
        let consensus_rate = if tick_count > 0 {
            consensus_ticks as f64 / tick_count as f64
        } else {
            0.0
        };
        let avg_turns_to_consensus = if consensus_ticks > 0 {
            total_turns_to_consensus as f64 / consensus_ticks as f64
        } else {
            0.0
        };

        Ok(ExperimentResult {
            config: ExperimentConfig {
                strategy: "conversation".to_string(),
                agent_count: ctx.agent_count,
                n: num_meetings,
                empty_cells: initial_unscheduled,
                decay_enabled: self.config.decay_enabled,
                inhibition_enabled: self.config.inhibition_enabled,
                examples_enabled: self.config.examples_enabled,
                trial: ctx.trial,
                seed: ctx.seed,
            },
            started_at: ctx.started_at,
            ended_at,
            total_ticks: tick_metrics.len(),
            solved,
            final_pressure,
            pressure_history,
            patches_per_tick,
            empty_cells_history: Vec::new(),
            example_bank_stats,
            tick_metrics,
            escalation_events,
            band_escalation_events: Vec::new(), // Conversation doesn't use band escalation
            final_model: current_model,
            total_prompt_tokens: 0, // Could track if needed
            total_completion_tokens: 0,
            total_patch_rejections: HashMap::new(),
            conversation_stats: Some(ConversationStats {
                total_messages,
                avg_messages_per_tick,
                total_llm_calls: total_messages,
                consensus_rate,
                avg_turns_to_consensus,
            }),
        })
    }

    /// Build kernel configuration for schedule experiments.
    fn build_kernel_config(&self) -> KernelConfig {
        let tick_interval_ms = 100;

        // Pressure axes matching sensor signals
        let pressure_axes = vec![
            PressureAxisConfig {
                name: "gaps".to_string(),
                weight: 1.0,
                expr: "gap_ratio".to_string(),
                kind_weights: HashMap::new(),
            },
            PressureAxisConfig {
                name: "overlaps".to_string(),
                weight: 2.0,
                expr: "overlap_count".to_string(),
                kind_weights: HashMap::new(),
            },
            PressureAxisConfig {
                name: "utilization".to_string(),
                weight: 0.5,
                expr: "utilization_variance".to_string(),
                kind_weights: HashMap::new(),
            },
            PressureAxisConfig {
                name: "unscheduled".to_string(),
                weight: 1.5,
                expr: "unscheduled_count".to_string(),
                kind_weights: HashMap::new(),
            },
        ];

        let decay = DecayConfig {
            fitness_half_life_ms: if self.config.decay_enabled {
                5_000
            } else {
                u64::MAX
            },
            confidence_half_life_ms: if self.config.decay_enabled {
                10_000
            } else {
                u64::MAX
            },
            ema_alpha: 0.2,
        };

        let activation = ActivationConfig {
            min_total_pressure: 0.1,
            inhibit_ms: if self.config.inhibition_enabled {
                2_000
            } else {
                0
            },
        };

        let selection = SelectionConfig {
            min_expected_improvement: 0.0,
        };

        KernelConfig {
            pressure_axes,
            decay,
            activation,
            selection,
            max_ticks: self.config.max_ticks,
            tick_interval_ms,
            stable_threshold: 10,
        }
    }

    /// Measure total pressure across all regions.
    fn measure_total_pressure(
        &self,
        artifact: &ScheduleArtifact,
        sensor: &CombinedScheduleSensor,
    ) -> Result<f64> {
        let mut total = 0.0;
        for region_id in artifact.region_ids() {
            let region_view = artifact.read_region(region_id)?;
            let signals = sensor.measure(&region_view)?;

            // Weighted sum of pressure signals
            total += signals.get("gap_ratio").unwrap_or(&0.0) * 1.0;
            total += signals.get("overlap_count").unwrap_or(&0.0) * 2.0;
            total += signals.get("utilization_variance").unwrap_or(&0.0) * 0.5;
        }
        // Add global unscheduled count
        if let Some(first_region) = artifact.region_ids().first() {
            let region_view = artifact.read_region(first_region.clone())?;
            let signals = sensor.measure(&region_view)?;
            total += signals.get("unscheduled_count").unwrap_or(&0.0) * 1.5;
        }
        Ok(total)
    }

    /// Measure pressure for a single region.
    fn measure_region_pressure(
        &self,
        region_view: &survival_kernel::region::RegionView,
        sensor: &CombinedScheduleSensor,
    ) -> Result<f64> {
        let signals = sensor.measure(region_view)?;
        let pressure = signals.get("gap_ratio").unwrap_or(&0.0) * 1.0
            + signals.get("overlap_count").unwrap_or(&0.0) * 2.0
            + signals.get("utilization_variance").unwrap_or(&0.0) * 0.5;
        Ok(pressure)
    }

    /// Select the highest-pressure region (for Hierarchical strategy).
    fn select_highest_pressure_region(
        &self,
        artifact: &ScheduleArtifact,
        sensor: &CombinedScheduleSensor,
    ) -> Result<survival_kernel::region::RegionId> {
        let mut max_pressure = f64::NEG_INFINITY;
        let mut selected = artifact.region_ids()[0].clone();

        for region_id in artifact.region_ids() {
            let region_view = artifact.read_region(region_id.clone())?;
            let pressure = self.measure_region_pressure(&region_view, sensor)?;
            if pressure > max_pressure {
                max_pressure = pressure;
                selected = region_id.clone();
            }
        }

        Ok(selected)
    }

    /// Build a schedule prompt for the LLM.
    fn build_schedule_prompt(
        &self,
        region_view: &survival_kernel::region::RegionView,
        examples: &[crate::example_bank::Example],
    ) -> String {
        let time_range = region_view
            .metadata
            .get("time_range")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown time");

        let rooms_info = region_view
            .metadata
            .get("rooms")
            .cloned()
            .unwrap_or(serde_json::json!([]));

        let rooms_text = if let Some(rooms) = rooms_info.as_array() {
            rooms
                .iter()
                .map(|r| {
                    format!(
                        "  Room {}: capacity {}",
                        r.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
                        r.get("capacity").and_then(|v| v.as_u64()).unwrap_or(0)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            String::new()
        };

        // Extract unscheduled meetings from metadata (matching pressure_field prompt)
        let unscheduled = region_view
            .metadata
            .get("unscheduled_meetings")
            .cloned()
            .unwrap_or(serde_json::json!([]));

        let unscheduled_text = if let Some(meetings) = unscheduled.as_array() {
            if meetings.is_empty() {
                "  None".to_string()
            } else {
                meetings
                    .iter()
                    .map(|m| {
                        let id = m.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                        let duration = m
                            .get("duration_slots")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let attendees = m.get("attendees").and_then(|v| v.as_u64()).unwrap_or(0);
                        let duration_min = duration * 30;
                        format!(
                            "  Meeting {}: {}min, {} attendees",
                            id, duration_min, attendees
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        } else {
            "  None".to_string()
        };

        let examples_text = if examples.is_empty() {
            String::new()
        } else {
            let formatted: Vec<String> = examples.iter().map(|e| e.format_for_prompt()).collect();
            format!(
                "\nEXAMPLES OF SUCCESSFUL SCHEDULES:\n{}\n",
                formatted.join("\n")
            )
        };

        format!(
            r#"You are a meeting room scheduler. Output schedules in the exact format requested. No explanations, just the schedule.

Meeting Room Schedule Optimization.
Goal: Schedule meetings to minimize gaps and avoid conflicts.

Time Block: {time_range}

Rooms:
{rooms_text}

Current assignments:
{current_schedule}

Unscheduled meetings that could fit in this block:
{unscheduled_text}

Constraints:
- No attendee can be in multiple meetings at the same time
- Room capacity must fit attendees
{examples_text}
Output the schedule for this time block. For each room, list the meeting IDs and times:
Room A: meeting_id (start-end), ...
Room B: meeting_id (start-end), ...

Answer:"#,
            time_range = time_range,
            rooms_text = rooms_text,
            current_schedule = region_view.content,
            unscheduled_text = unscheduled_text,
            examples_text = examples_text,
        )
    }

    /// Parse a schedule response from the LLM.
    fn parse_schedule_response(&self, response: &str) -> Option<String> {
        let mut lines = Vec::new();
        let mut found_room = false;

        for line in response.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.to_lowercase().starts_with("room") {
                found_room = true;
                lines.push(trimmed.to_string());
            } else if found_room && trimmed.contains(':') {
                lines.push(trimmed.to_string());
            }
        }

        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    }
}

/// Spawn a tick observer actor that forwards TickComplete to a channel.
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

/// Spawn an observer for SensorsReady.
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

/// Spawn an observer for PatchActorsReady.
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

    #[test]
    fn test_strategy_names() {
        assert_eq!(Strategy::PressureField.name(), "pressure_field");
        assert_eq!(Strategy::Conversation.name(), "conversation");
        assert_eq!(Strategy::Sequential.name(), "sequential");
        assert_eq!(Strategy::Random.name(), "random");
        assert_eq!(Strategy::Hierarchical.name(), "hierarchical");
    }

    #[test]
    fn test_strategy_all() {
        let all = Strategy::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_config_get_vllm_host_fallback() {
        let config = ExperimentRunnerConfig::default();
        // With empty vllm_hosts, should fallback to vllm_host
        assert_eq!(config.get_vllm_host(0), config.vllm_host);
        assert_eq!(config.get_vllm_host(5), config.vllm_host);
    }

    #[test]
    fn test_config_get_vllm_host_multi() {
        let config = ExperimentRunnerConfig {
            vllm_hosts: vec![
                "http://host1:8000".to_string(),
                "http://host2:8000".to_string(),
            ],
            ..Default::default()
        };
        assert_eq!(config.get_vllm_host(0), "http://host1:8000");
        assert_eq!(config.get_vllm_host(1), "http://host2:8000");
        assert_eq!(config.get_vllm_host(5), "http://host2:8000"); // Clamped
    }
}
