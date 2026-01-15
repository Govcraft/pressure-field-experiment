//! LLM actor for meeting room scheduling using vLLM.
//!
//! Proposes schedule patches to fill empty slots and resolve conflicts.
//! Supports sampling diversity (varying temperature/top-p) and few-shot learning.

use std::collections::HashMap;
use std::sync::Arc;

use acton_reactive::prelude::*;
use anyhow::Result;
use rand::Rng;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};

use survival_kernel::messages::{
    CoordinatorReady, PatchActorReady, PatchProposal, ProposeForRegion,
};
use survival_kernel::region::{Patch, PatchOp};

/// Message to update the model/host for escalation.
///
/// Broadcast to all LLM actors when the experiment detects a stall
/// and wants to escalate to a larger model.
#[derive(Clone, Debug)]
pub struct UpdateModel {
    /// New model name (e.g., "Qwen/Qwen2.5-7B")
    pub model: String,
    /// New vLLM host URL (e.g., "http://localhost:8004")
    pub host: String,
}

use crate::example_bank::ExampleBank;
use crate::vllm_client::VllmClient;

/// System prompt for meeting room scheduling.
const SCHEDULE_SYSTEM_PROMPT: &str = "You are a meeting room scheduler. Output schedules in the exact format requested. No explanations, just the schedule.";

/// Sampling configuration for diversity.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for generation (0.0-1.0)
    pub temperature: f32,
    /// Top-p nucleus sampling (0.0-1.0)
    pub top_p: f32,
    /// Top-k sampling (number of tokens to consider)
    pub top_k: u32,
}

impl SamplingConfig {
    /// Conservative sampling (exploitation)
    pub fn exploitation() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.85,
            top_k: 25,
        }
    }

    /// Balanced sampling
    pub fn balanced() -> Self {
        Self {
            temperature: 0.4,
            top_p: 0.9,
            top_k: 40,
        }
    }

    /// Exploratory sampling
    pub fn exploration() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 60,
        }
    }

    /// Random sampling within a band.
    pub fn random_in_band(band: SamplingBand) -> Self {
        let mut rng = rand::rng();
        match band {
            SamplingBand::Exploitation => Self {
                temperature: rng.random_range(0.15..0.3),
                top_p: rng.random_range(0.8..0.9),
                top_k: rng.random_range(15..30),
            },
            SamplingBand::Balanced => Self {
                temperature: rng.random_range(0.3..0.55),
                top_p: rng.random_range(0.85..0.95),
                top_k: rng.random_range(30..50),
            },
            SamplingBand::Exploration => Self {
                temperature: rng.random_range(0.5..0.85),
                top_p: rng.random_range(0.9..0.98),
                top_k: rng.random_range(45..80),
            },
        }
    }
}

/// Sampling bands for diversity.
#[derive(Debug, Clone, Copy)]
pub enum SamplingBand {
    Exploitation,
    Balanced,
    Exploration,
}

/// Configuration for the LLM actor.
#[derive(Debug, Clone)]
pub struct LlmActorConfig {
    /// vLLM host URL
    pub host: String,
    /// Model name
    pub model: String,
    /// Base sampling configuration
    pub sampling: SamplingConfig,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Sampling band for this actor
    pub band: SamplingBand,
    /// Whether to use random sampling within the band
    pub randomize_sampling: bool,
}

impl Default for LlmActorConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            model: "qwen2.5:1.5b".to_string(),
            sampling: SamplingConfig::balanced(),
            max_tokens: 256, // Schedules need more tokens than Latin squares
            band: SamplingBand::Balanced,
            randomize_sampling: true,
        }
    }
}

/// Actor state for LLM-based patch proposal.
#[derive(Default, Clone)]
pub struct LlmActorState {
    /// Actor name
    pub name: String,
    /// LLM configuration (wrapped for interior mutability during escalation)
    pub config: Option<Arc<RwLock<LlmActorConfig>>>,
    /// Semaphore for rate limiting concurrent LLM requests
    pub semaphore: Option<Arc<Semaphore>>,
    /// Example bank for few-shot learning
    pub example_bank: Option<Arc<RwLock<ExampleBank>>>,
}

impl std::fmt::Debug for LlmActorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmActorState")
            .field("name", &self.name)
            .field("has_example_bank", &self.example_bank.is_some())
            .finish()
    }
}

/// LLM-based actor for schedule patch proposals.
pub struct LlmActor;

impl LlmActor {
    /// Spawn a new LLM actor.
    pub async fn spawn(
        runtime: &mut ActorRuntime,
        name: String,
        config: LlmActorConfig,
        semaphore: Arc<Semaphore>,
        example_bank: Arc<RwLock<ExampleBank>>,
    ) -> ActorHandle {
        let mut actor = runtime.new_actor_with_name::<LlmActorState>(name.clone());

        actor.model.name = name;
        actor.model.config = Some(Arc::new(RwLock::new(config)));
        actor.model.semaphore = Some(semaphore);
        actor.model.example_bank = Some(example_bank);

        // Subscribe to messages before starting
        actor.handle().subscribe::<ProposeForRegion>().await;
        actor.handle().subscribe::<CoordinatorReady>().await;
        actor.handle().subscribe::<UpdateModel>().await;

        // On CoordinatorReady, announce ourselves
        actor.act_on::<CoordinatorReady>(|actor, _context| {
            let broker = actor.broker().clone();
            let handle = actor.handle().clone();
            let actor_ern = actor.handle().name().to_string();

            Reply::pending(async move {
                info!(actor = %actor_ern, "Schedule LLM actor ready");
                broker
                    .broadcast(PatchActorReady { actor_ern, handle })
                    .await;
            })
        });

        // Handle model updates during escalation
        actor.act_on::<UpdateModel>(|actor, context| {
            let msg = context.message().clone();
            let config = actor.model.config.clone();
            let actor_name = actor.model.name.clone();

            Reply::pending(async move {
                if let Some(config) = config {
                    let mut cfg = config.write().await;
                    info!(
                        actor = %actor_name,
                        from = %cfg.model,
                        to = %msg.model,
                        "Escalating model"
                    );
                    cfg.model = msg.model;
                    cfg.host = msg.host;
                }
            })
        });

        // Handle ProposeForRegion - generate a schedule patch
        actor.act_on::<ProposeForRegion>(|actor, context| {
            let msg = context.message().clone();
            let broker = actor.broker().clone();
            let config = actor.model.config.clone();
            let semaphore = actor.model.semaphore.clone();
            let example_bank = actor.model.example_bank.clone();
            let actor_name = actor.model.name.clone();

            Reply::pending(async move {
                // Acquire semaphore permit for rate limiting
                let _permit = if let Some(sem) = &semaphore {
                    Some(sem.acquire().await.expect("semaphore closed"))
                } else {
                    None
                };

                // Get examples for few-shot learning
                let examples = if let Some(bank) = &example_bank {
                    bank.read().await.get_examples_for_prompt()
                } else {
                    vec![]
                };

                // Generate patch
                let config_guard = config.as_ref().unwrap().read().await;
                let result = generate_schedule_patch(&config_guard, &msg, &examples).await;

                drop(config_guard);

                match result {
                    Ok((patch, prompt_tokens, completion_tokens)) => {
                        let patches = patch
                            .map(|p| vec![(1.0, p)])
                            .unwrap_or_default();

                        broker
                            .broadcast(PatchProposal {
                                correlation_id: msg.correlation_id.clone(),
                                actor_name,
                                patches,
                                prompt_tokens,
                                completion_tokens,
                            })
                            .await;
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to generate schedule patch");
                        broker
                            .broadcast(PatchProposal {
                                correlation_id: msg.correlation_id.clone(),
                                actor_name,
                                patches: vec![],
                                prompt_tokens: 0,
                                completion_tokens: 0,
                            })
                            .await;
                    }
                }
            })
        });

        actor.start().await
    }
}

/// Generate a schedule patch using the LLM.
async fn generate_schedule_patch(
    config: &LlmActorConfig,
    msg: &ProposeForRegion,
    examples: &[crate::example_bank::Example],
) -> Result<(Option<Patch>, u32, u32)> {
    // Extract time block info from region metadata
    let time_range = msg
        .region_view
        .metadata
        .get("time_range")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown time");

    let rooms_info = msg
        .region_view
        .metadata
        .get("rooms")
        .cloned()
        .unwrap_or(serde_json::json!([]));

    let unscheduled = msg
        .region_view
        .metadata
        .get("unscheduled_meetings")
        .cloned()
        .unwrap_or(serde_json::json!([]));

    // Format room info for prompt
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

    // Format unscheduled meetings for prompt
    let unscheduled_text = if let Some(meetings) = unscheduled.as_array() {
        if meetings.is_empty() {
            "  None".to_string()
        } else {
            meetings
                .iter()
                .map(|m| {
                    let id = m.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                    let duration = m.get("duration_slots").and_then(|v| v.as_u64()).unwrap_or(0);
                    let attendees = m.get("attendees").and_then(|v| v.as_u64()).unwrap_or(0);
                    let duration_min = duration * 30;
                    format!("  Meeting {}: {}min, {} attendees", id, duration_min, attendees)
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    } else {
        "  None".to_string()
    };

    // Format examples for few-shot learning
    let examples_text = if examples.is_empty() {
        String::new()
    } else {
        let formatted: Vec<String> = examples.iter().map(|e| e.format_for_prompt()).collect();
        format!(
            "\nEXAMPLES OF SUCCESSFUL SCHEDULES:\n{}\n",
            formatted.join("\n")
        )
    };

    let prompt = format!(
        r#"Meeting Room Schedule Optimization.
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
        current_schedule = msg.region_view.content,
        unscheduled_text = unscheduled_text,
        examples_text = examples_text,
    );

    info!(
        region = %msg.region_id,
        model = %config.model,
        time_range = %time_range,
        "Generate patch"
    );

    // Determine sampling parameters
    let sampling = if config.randomize_sampling {
        SamplingConfig::random_in_band(config.band)
    } else {
        config.sampling.clone()
    };

    // Call vLLM with usage tracking
    let client = VllmClient::new(&config.host);

    let response = client
        .generate_with_system_and_usage(
            &config.model,
            SCHEDULE_SYSTEM_PROMPT,
            &prompt,
            sampling.temperature,
            sampling.top_p,
            config.max_tokens,
        )
        .await?;

    let prompt_tokens = response.prompt_tokens;
    let completion_tokens = response.completion_tokens;
    let response_text = response.content.trim();

    debug!(response = %response_text, prompt_tokens, completion_tokens, "LLM response");

    // Parse the response
    let new_content = parse_schedule_response(response_text);

    let Some(new_content) = new_content else {
        debug!("Could not parse valid schedule from LLM response");
        return Ok((None, prompt_tokens, completion_tokens));
    };

    // Don't return if content is unchanged
    if new_content == msg.region_view.content {
        debug!("LLM returned unchanged content");
        return Ok((None, prompt_tokens, completion_tokens));
    }

    Ok((
        Some(Patch {
            region: msg.region_id.clone(),
            op: PatchOp::Replace(new_content),
            rationale: "Optimized schedule for time block".to_string(),
            expected_delta: HashMap::new(),
        }),
        prompt_tokens,
        completion_tokens,
    ))
}

/// Parse a schedule response from the LLM.
///
/// Accepts formats like:
/// - "Room A: 5 (10:00-11:00), 7 (11:00-12:00)"
/// - "Room B: [empty]"
fn parse_schedule_response(response: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut found_room = false;

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Look for lines starting with "Room"
        if trimmed.to_lowercase().starts_with("room") {
            found_room = true;
            lines.push(trimmed.to_string());
        } else if found_room && trimmed.contains(':') {
            // Also accept other room-like formats
            lines.push(trimmed.to_string());
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schedule_response_basic() {
        let response = r#"
Room A: 5 (10:00-11:00), 7 (11:00-12:00)
Room B: [empty]
Room C: 12 (10:30-11:30)
"#;
        let parsed = parse_schedule_response(response);
        assert!(parsed.is_some());
        let content = parsed.unwrap();
        assert!(content.contains("Room A:"));
        assert!(content.contains("Room B:"));
        assert!(content.contains("Room C:"));
    }

    #[test]
    fn test_parse_schedule_response_empty() {
        let response = "I don't understand the question.";
        let parsed = parse_schedule_response(response);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_sampling_config_bands() {
        let exploitation = SamplingConfig::exploitation();
        assert!(exploitation.temperature < 0.3);

        let balanced = SamplingConfig::balanced();
        assert!(balanced.temperature >= 0.3 && balanced.temperature <= 0.5);

        let exploration = SamplingConfig::exploration();
        assert!(exploration.temperature >= 0.6);
    }
}
