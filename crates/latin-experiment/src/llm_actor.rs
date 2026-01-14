//! LLM actor for Latin Square puzzle solving using vLLM.
//!
//! Proposes row patches to fill empty cells and resolve conflicts.
//! Supports sampling diversity (varying temperature/top-p) and few-shot learning.

use std::collections::HashMap;
use std::sync::Arc;

use acton_reactive::prelude::*;
use anyhow::Result;
use rand::Rng;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};

use survival_kernel::messages::{CoordinatorReady, PatchActorReady, PatchProposal, ProposeForRegion};
use survival_kernel::region::{Patch, PatchOp};

use crate::example_bank::ExampleBank;
use crate::vllm_client::VllmClient;

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
            host: "http://localhost:8000".to_string(),
            model: "Qwen/Qwen2.5-1.5B".to_string(),
            sampling: SamplingConfig::balanced(),
            max_tokens: 256,
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
    /// LLM configuration
    pub config: Option<LlmActorConfig>,
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

/// LLM actor for proposing Latin Square improvements.
///
/// Uses the broker pub/sub pattern:
/// - Subscribes to `ProposeForRegion` broadcasts
/// - Broadcasts `PatchActorReady` on start for coordinator tracking
/// - Broadcasts `PatchProposal` responses
pub struct LlmActor {
    /// Actor name
    pub name: String,
    /// Configuration
    pub config: LlmActorConfig,
    /// Semaphore for rate limiting (shared across all actors)
    pub semaphore: Arc<Semaphore>,
    /// Example bank for few-shot learning
    pub example_bank: Arc<RwLock<ExampleBank>>,
}

impl LlmActor {
    /// Create a new LLM actor.
    pub fn new(
        name: impl Into<String>,
        config: LlmActorConfig,
        semaphore: Arc<Semaphore>,
        example_bank: Arc<RwLock<ExampleBank>>,
    ) -> Self {
        Self {
            name: name.into(),
            config,
            semaphore,
            example_bank,
        }
    }

    /// Spawn the actor in the runtime.
    ///
    /// The actor will:
    /// 1. Subscribe to `ProposeForRegion` broadcasts
    /// 2. Broadcast `PatchActorReady` on start
    /// 3. Handle proposals and broadcast results
    pub async fn spawn(self, runtime: &mut ActorRuntime) -> ActorHandle {
        let mut actor = runtime.new_actor_with_name::<LlmActorState>(self.name.clone());

        actor.model.name = self.name;
        actor.model.config = Some(self.config);
        actor.model.semaphore = Some(self.semaphore);
        actor.model.example_bank = Some(self.example_bank);

        // Subscribe to ProposeForRegion broadcasts BEFORE starting
        actor.handle().subscribe::<ProposeForRegion>().await;

        // Subscribe to CoordinatorReady - we'll respond with PatchActorReady
        // This ensures the coordinator exists before we try to register
        actor.handle().subscribe::<CoordinatorReady>().await;

        // Handle CoordinatorReady by broadcasting PatchActorReady
        actor.act_on::<CoordinatorReady>(|actor, _context| {
            let broker = actor.broker().clone();
            let actor_ern = actor.handle().name().to_string();
            let handle = actor.handle().clone();
            Reply::pending(async move {
                broker.broadcast(PatchActorReady { actor_ern, handle }).await;
            })
        });

        configure_llm_actor(&mut actor);

        actor.start().await
    }
}

fn configure_llm_actor(actor: &mut ManagedActor<Idle, LlmActorState>) {
    actor.act_on::<ProposeForRegion>(|actor, context| {
        let msg = context.message().clone();
        let broker = actor.broker().clone();
        let name = actor.model.name.clone();
        let config = actor.model.config.clone();
        let semaphore = actor.model.semaphore.clone();
        let example_bank = actor.model.example_bank.clone();

        let Some(config) = config else {
            warn!("ProposeForRegion: no config");
            return Reply::ready();
        };

        Reply::pending(async move {
            // Acquire semaphore permit for rate limiting
            let _permit = if let Some(ref sem) = semaphore {
                match sem.acquire().await {
                    Ok(permit) => Some(permit),
                    Err(_) => {
                        warn!("Semaphore closed");
                        None
                    }
                }
            } else {
                None
            };

            // Get examples from the bank
            let examples = if let Some(ref bank) = example_bank {
                let bank = bank.read().await;
                bank.get_examples_for_prompt()
            } else {
                vec![]
            };

            let result = generate_patch(&config, &msg, &examples).await;
            // Permit is dropped here, releasing the slot

            let (patches, prompt_tokens, completion_tokens) = match result {
                Ok((Some(patch), prompt_t, completion_t)) => {
                    (vec![(1.0, patch)], prompt_t, completion_t)
                }
                Ok((None, prompt_t, completion_t)) => {
                    debug!(region_id = %msg.region_id, "No patch generated");
                    (vec![], prompt_t, completion_t)
                }
                Err(e) => {
                    warn!(region_id = %msg.region_id, error = %e, "Failed to generate patch");
                    (vec![], 0, 0)
                }
            };

            let proposal = PatchProposal {
                correlation_id: msg.correlation_id,
                actor_name: name,
                patches,
                prompt_tokens,
                completion_tokens,
            };

            // Broadcast result (coordinator subscribes to PatchProposal)
            broker.broadcast(proposal).await;
        })
    });
}

/// Generate a patch for the given region using the LLM.
///
/// Returns (Option<Patch>, prompt_tokens, completion_tokens).
async fn generate_patch(
    config: &LlmActorConfig,
    msg: &ProposeForRegion,
    examples: &[crate::example_bank::Example],
) -> Result<(Option<Patch>, u32, u32)> {
    let n = msg
        .region_view
        .metadata
        .get("n")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;

    let row_idx = msg
        .region_view
        .metadata
        .get("row_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Parse column availability from metadata
    let column_availability: HashMap<String, Vec<u8>> = msg
        .region_view
        .metadata
        .get("column_availability")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Format column availability for prompt
    let availability_text: String = (0..n)
        .map(|col| {
            let available = column_availability
                .get(&col.to_string())
                .map(|v| {
                    v.iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_else(|| "?".to_string());
            format!("  Column {}: [{}]", col, available)
        })
        .collect::<Vec<_>>()
        .join("\n");

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

    let prompt = format!(
        r#"You are solving a Latin Square puzzle. Each row and column must contain each number from 1 to {n} exactly once.

Current row {row_idx}: {content}

Available values for each column (numbers NOT yet used in that column):
{availability_text}
{examples_text}
Fill all empty cells (_) with valid numbers from the available values.
Return ONLY the complete row as space-separated numbers. Example: "1 2 3 4 5 6""#,
        n = n,
        row_idx = row_idx,
        content = msg.region_view.content,
        availability_text = availability_text,
        examples_text = examples_text,
    );

    info!(
        region_id = %msg.region_id,
        row_idx = row_idx,
        example_count = examples.len(),
        "Generating patch with LLM"
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
        .generate_with_usage(
            &config.model,
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

    // Parse the response - should be space-separated numbers
    let new_content = parse_row_response(response_text, n);

    let Some(new_content) = new_content else {
        debug!("Could not parse valid row from LLM response");
        return Ok((None, prompt_tokens, completion_tokens));
    };

    // Don't return if content is unchanged
    if new_content == msg.region_view.content {
        debug!("LLM returned unchanged content");
        return Ok((None, prompt_tokens, completion_tokens));
    }

    Ok((
        Some(Patch {
            region: msg.region_id,
            op: PatchOp::Replace(new_content),
            rationale: "Filled empty cells in row".to_string(),
            expected_delta: HashMap::new(),
        }),
        prompt_tokens,
        completion_tokens,
    ))
}

/// Parse a row response from the LLM.
///
/// Accepts formats like:
/// - "1 2 3 4 5 6"
/// - "1, 2, 3, 4, 5, 6"
/// - Just the numbers on a line
fn parse_row_response(response: &str, n: usize) -> Option<String> {
    // Try to find a line with n numbers
    for line in response.lines() {
        let cleaned = line.replace([',', '[', ']', '"'], " ");

        let numbers: Vec<u8> = cleaned
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .filter(|&v| v >= 1 && v <= n as u8)
            .collect();

        if numbers.len() == n {
            return Some(
                numbers
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_row_response_simple() {
        assert_eq!(
            parse_row_response("1 2 3 4", 4),
            Some("1 2 3 4".to_string())
        );
    }

    #[test]
    fn test_parse_row_response_with_commas() {
        assert_eq!(
            parse_row_response("1, 2, 3, 4", 4),
            Some("1 2 3 4".to_string())
        );
    }

    #[test]
    fn test_parse_row_response_with_explanation() {
        let response = r#"Based on the constraints, the answer is:
1 2 3 4
This satisfies all requirements."#;
        assert_eq!(parse_row_response(response, 4), Some("1 2 3 4".to_string()));
    }

    #[test]
    fn test_parse_row_response_invalid() {
        assert_eq!(parse_row_response("invalid response", 4), None);
        assert_eq!(parse_row_response("1 2 3", 4), None); // Wrong length
    }

    #[test]
    fn test_sampling_config_bands() {
        let exploit = SamplingConfig::exploitation();
        assert!(exploit.temperature < 0.3);

        let explore = SamplingConfig::exploration();
        assert!(explore.temperature > 0.5);
    }
}
