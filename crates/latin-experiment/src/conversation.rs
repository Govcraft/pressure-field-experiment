//! Multi-turn conversation coordination (AutoGen-style baseline).
//!
//! Implements conversation-based agent coordination where agents communicate
//! via messages to reach consensus on patches. This serves as a realistic
//! baseline representing explicit message-passing coordination (AutoGen-style)
//! vs. the stigmergic pressure-field approach.
//!
//! Key patterns captured:
//! - Role-based agents (Coordinator, Proposer, Validator)
//! - Multi-turn dialogue until consensus
//! - Sequential message passing (vs. parallel pressure activation)
//! - Explicit coordination overhead (tracked as messages_per_tick)

use std::sync::Arc;

use anyhow::{Context, Result};
use regex::Regex;
use survival_kernel::artifact::Artifact;
use tokio::sync::Semaphore;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::artifact::LatinSquareArtifact;
use crate::sensors::SharedGrid;
use crate::vllm_client::VllmClient;

/// Agent roles in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Selects target region, synthesizes final decision
    Coordinator,
    /// Generates candidate patches
    Proposer,
    /// Critiques proposals against constraints
    Validator,
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coordinator => write!(f, "Coordinator"),
            Self::Proposer => write!(f, "Proposer"),
            Self::Validator => write!(f, "Validator"),
        }
    }
}

/// A single message in the conversation.
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub turn: usize,
    pub role: AgentRole,
    pub content: String,
    pub region_id: Option<Uuid>,
}

/// Conversation state for a single tick.
#[derive(Debug, Clone)]
pub struct ConversationState {
    pub messages: Vec<ConversationMessage>,
    pub current_turn: usize,
    pub max_turns: usize,
    pub target_region: Option<Uuid>,
    pub final_patch: Option<String>,
    /// Number of ticks where consensus was reached explicitly
    pub consensus_ticks: usize,
    /// Number of ticks completed
    pub total_ticks: usize,
}

impl ConversationState {
    pub fn new(max_turns: usize) -> Self {
        Self {
            messages: Vec::new(),
            current_turn: 0,
            max_turns,
            target_region: None,
            final_patch: None,
            consensus_ticks: 0,
            total_ticks: 0,
        }
    }

    pub fn add_message(&mut self, role: AgentRole, content: String, region_id: Option<Uuid>) {
        self.messages.push(ConversationMessage {
            turn: self.current_turn,
            role,
            content,
            region_id,
        });
    }

    pub fn total_messages(&self) -> usize {
        self.messages.len()
    }

    pub fn is_complete(&self) -> bool {
        self.final_patch.is_some() || self.current_turn >= self.max_turns
    }

    /// Format conversation history for LLM context.
    pub fn format_history(&self) -> String {
        self.messages
            .iter()
            .map(|m| format!("[{}] {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Reset for a new tick while preserving stats.
    pub fn reset_for_tick(&mut self) {
        self.messages.clear();
        self.current_turn = 0;
        self.target_region = None;
        self.final_patch = None;
    }
}

/// Prompt templates for each agent role.
pub struct PromptTemplates;

impl PromptTemplates {
    pub fn coordinator_select_region(puzzle_state: &str, n: usize) -> String {
        format!(
            r#"You are a Coordinator agent solving a {n}x{n} Latin Square puzzle.

Current puzzle state (each row is numbered, _ means empty):
{puzzle_state}

Task: Identify which row needs the most attention. Consider:
1. Rows with empty cells
2. Rows with constraint violations

Respond with ONLY: TARGET row=<N>
Where N is the 0-indexed row number (0 to {max_row}).

Example: TARGET row=2"#,
            n = n,
            puzzle_state = puzzle_state,
            max_row = n - 1
        )
    }

    pub fn coordinator_decide(conversation_history: &str, region_content: &str, n: usize) -> String {
        format!(
            r#"You are a Coordinator agent. Based on the conversation, make the final decision.

Conversation so far:
{conversation_history}

Target row content: {region_content}

If a valid proposal was APPROVED by the Validator, output:
APPLY <complete row with the change, {n} space-separated numbers, use the same format as the row>

If the proposal was REJECTED or no valid proposal exists, output:
REJECT

Only output ONE of these two options, nothing else."#,
            conversation_history = conversation_history,
            region_content = region_content,
            n = n
        )
    }

    pub fn proposer(
        region_content: &str,
        availability: &str,
        conversation_history: &str,
        row_idx: usize,
    ) -> String {
        format!(
            r#"You are a Proposer agent solving a Latin Square puzzle.

Target row {row_idx}: {region_content}
Available values per column position (values NOT yet used in that column):
{availability}

Previous messages:
{conversation_history}

Propose ONE value for ONE empty cell (_).
Format: PROPOSE position=<col> value=<num>

Choose a position that has _ and pick a value from that position's available list.
Example: PROPOSE position=2 value=4"#,
            row_idx = row_idx,
            region_content = region_content,
            availability = availability,
            conversation_history = if conversation_history.is_empty() {
                "(none yet)".to_string()
            } else {
                conversation_history.to_string()
            }
        )
    }

    pub fn validator(
        region_content: &str,
        proposal: &str,
        column_values: &str,
        row_values: &str,
    ) -> String {
        format!(
            r#"You are a Validator agent checking Latin Square constraints.

Row: {region_content}
Proposal: {proposal}

Values already in the target column: {column_values}
Values already in the row (excluding empty cells): {row_values}

Check if the proposed value:
1. Already exists in the column (VIOLATION)
2. Already exists in the row (VIOLATION)
3. Is within valid range for this puzzle (VIOLATION if out of range)

Respond with ONLY ONE of:
APPROVE
REJECT <brief reason>"#,
            region_content = region_content,
            proposal = proposal,
            column_values = column_values,
            row_values = row_values
        )
    }
}

/// Orchestrates a multi-turn conversation for one tick.
pub struct ConversationRunner {
    client: VllmClient,
    model: String,
    max_turns: usize,
    llm_semaphore: Arc<Semaphore>,
}

impl ConversationRunner {
    pub fn new(vllm_host: &str, model: &str, max_turns: usize) -> Result<Self> {
        let client = VllmClient::new(vllm_host);
        Ok(Self {
            client,
            model: model.to_string(),
            max_turns,
            llm_semaphore: Arc::new(Semaphore::new(1)), // Sequential for conversation
        })
    }

    /// Run a complete conversation for one tick.
    /// Returns (final_patch, conversation_state).
    pub async fn run_tick(
        &self,
        artifact: &LatinSquareArtifact,
        shared_grid: &SharedGrid,
    ) -> Result<(Option<String>, ConversationState)> {
        let mut state = ConversationState::new(self.max_turns);
        let n = artifact.size();

        // Turn 1: Coordinator selects region
        let puzzle_state = self.format_puzzle_state(artifact);
        let region_idx = self.coordinator_select_region(&mut state, &puzzle_state, n).await?;

        if region_idx >= n {
            warn!(
                region_idx = region_idx,
                n = n,
                "Coordinator selected invalid region, falling back to 0"
            );
            state.target_region = artifact.region_ids().first().copied();
        } else {
            state.target_region = artifact.region_ids().get(region_idx).copied();
        }

        let Some(region_id) = state.target_region else {
            return Ok((None, state));
        };

        let region_view = artifact.read_region(region_id)?;
        let row_idx = region_idx;
        let availability = self.format_availability(artifact, row_idx);

        // Turns 2-N: Proposer/Validator dialogue
        let mut last_proposal: Option<(usize, u8)> = None;
        let mut last_approved = false;

        for turn in 1..self.max_turns {
            state.current_turn = turn;

            // Proposer turn
            let proposal = self
                .proposer_propose(&mut state, &region_view.content, &availability, row_idx)
                .await?;

            if let Some((pos, val)) = proposal {
                last_proposal = Some((pos, val));

                // Validator turn
                let column_values = self.get_column_values(shared_grid, pos, row_idx);
                let row_values = self.get_row_values(&region_view.content);

                let approved = self
                    .validator_check(
                        &mut state,
                        &region_view.content,
                        pos,
                        val,
                        &column_values,
                        &row_values,
                    )
                    .await?;

                if approved {
                    last_approved = true;
                    state.consensus_ticks += 1;
                    // Build the patch
                    let new_content = self.apply_value_to_row(&region_view.content, pos, val, n);
                    state.final_patch = Some(new_content);
                    break;
                }
            } else {
                // Proposer failed to make a valid proposal
                debug!(turn = turn, "Proposer failed to make valid proposal");
            }
        }

        // Final Coordinator decision if no early consensus
        if state.final_patch.is_none() && last_proposal.is_some() && last_approved {
            state.final_patch = self
                .coordinator_decide(&mut state, &region_view.content, n)
                .await?;
        }

        state.total_ticks += 1;
        Ok((state.final_patch.clone(), state))
    }

    fn format_puzzle_state(&self, artifact: &LatinSquareArtifact) -> String {
        artifact
            .region_ids()
            .iter()
            .enumerate()
            .filter_map(|(i, id)| {
                artifact
                    .read_region(*id)
                    .ok()
                    .map(|r| format!("Row {}: {}", i, r.content))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_availability(
        &self,
        artifact: &LatinSquareArtifact,
        row_idx: usize,
    ) -> String {
        let availability = artifact.column_availability(row_idx);
        let mut entries: Vec<_> = availability.iter().collect();
        entries.sort_by_key(|(col, _)| *col);
        entries
            .iter()
            .map(|(col, vals)| {
                let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                format!("Position {}: [{}]", col, vals_str.join(", "))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn get_column_values(&self, shared_grid: &SharedGrid, col: usize, exclude_row: usize) -> String {
        let grid = shared_grid.read().unwrap();
        let values: Vec<String> = grid
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != exclude_row)
            .filter_map(|(_, row)| row.get(col).and_then(|v| *v).map(|v| v.to_string()))
            .collect();
        if values.is_empty() {
            "(none)".to_string()
        } else {
            values.join(", ")
        }
    }

    fn get_row_values(&self, row_content: &str) -> String {
        let values: Vec<String> = row_content
            .split_whitespace()
            .filter(|s| *s != "_")
            .map(|s| s.to_string())
            .collect();
        if values.is_empty() {
            "(none)".to_string()
        } else {
            values.join(", ")
        }
    }

    async fn coordinator_select_region(
        &self,
        state: &mut ConversationState,
        puzzle_state: &str,
        n: usize,
    ) -> Result<usize> {
        let prompt = PromptTemplates::coordinator_select_region(puzzle_state, n);
        let response = self.call_llm(&prompt).await?;

        state.add_message(AgentRole::Coordinator, response.clone(), None);

        parse_coordinator_target(&response).unwrap_or(0)
            .pipe(Ok)
    }

    async fn coordinator_decide(
        &self,
        state: &mut ConversationState,
        region_content: &str,
        n: usize,
    ) -> Result<Option<String>> {
        let history = state.format_history();
        let prompt = PromptTemplates::coordinator_decide(&history, region_content, n);
        let response = self.call_llm(&prompt).await?;

        state.add_message(AgentRole::Coordinator, response.clone(), state.target_region);

        Ok(parse_coordinator_decision(&response, n))
    }

    async fn proposer_propose(
        &self,
        state: &mut ConversationState,
        region_content: &str,
        availability: &str,
        row_idx: usize,
    ) -> Result<Option<(usize, u8)>> {
        let history = state.format_history();
        let prompt = PromptTemplates::proposer(region_content, availability, &history, row_idx);
        let response = self.call_llm(&prompt).await?;

        state.add_message(AgentRole::Proposer, response.clone(), state.target_region);

        Ok(parse_proposer_proposal(&response))
    }

    async fn validator_check(
        &self,
        state: &mut ConversationState,
        region_content: &str,
        pos: usize,
        val: u8,
        column_values: &str,
        row_values: &str,
    ) -> Result<bool> {
        let proposal = format!("position={} value={}", pos, val);
        let prompt =
            PromptTemplates::validator(region_content, &proposal, column_values, row_values);
        let response = self.call_llm(&prompt).await?;

        state.add_message(AgentRole::Validator, response.clone(), state.target_region);

        Ok(parse_validator_response(&response))
    }

    fn apply_value_to_row(&self, row_content: &str, pos: usize, val: u8, _n: usize) -> String {
        let parts: Vec<&str> = row_content.split_whitespace().collect();
        parts
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if i == pos {
                    val.to_string()
                } else {
                    s.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    async fn call_llm(&self, prompt: &str) -> Result<String> {
        let _permit = self.llm_semaphore.acquire().await?;

        // Use generate_with_system for conversation agents - they have their own prompts
        // Default temperature/top_p for conversation-style responses
        let response = self
            .client
            .generate_with_system(
                &self.model,
                "You are a collaborative agent solving Latin Square puzzles. Follow the instructions precisely and respond in the exact format requested.",
                prompt,
                0.3,  // Lower temperature for more consistent structured output
                0.9,
                256,
            )
            .await
            .context("LLM generation failed")?;

        Ok(response.trim().to_string())
    }
}

// Response parsing functions

fn parse_coordinator_target(response: &str) -> Option<usize> {
    // Look for "TARGET row=N" or "row N" or just a number
    let re = Regex::new(r"(?i)(?:TARGET\s+)?row\s*=?\s*(\d+)").ok()?;
    re.captures(response)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn parse_proposer_proposal(response: &str) -> Option<(usize, u8)> {
    // Look for "PROPOSE position=N value=V"
    let re = Regex::new(r"(?i)PROPOSE\s+position\s*=\s*(\d+)\s+value\s*=\s*(\d+)").ok()?;
    re.captures(response).and_then(|c| {
        let pos = c.get(1)?.as_str().parse().ok()?;
        let val = c.get(2)?.as_str().parse().ok()?;
        Some((pos, val))
    })
}

fn parse_validator_response(response: &str) -> bool {
    response.to_uppercase().contains("APPROVE")
}

fn parse_coordinator_decision(response: &str, n: usize) -> Option<String> {
    if response.to_uppercase().contains("APPLY") {
        // Extract the row content after APPLY
        let re = Regex::new(r"(?i)APPLY\s+(.+)").ok()?;
        re.captures(response)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| {
                // Validate it looks like a valid row
                let parts: Vec<&str> = s.split_whitespace().collect();
                parts.len() == n && parts.iter().all(|p| p.parse::<u8>().is_ok() || *p == "_")
            })
    } else {
        None
    }
}

/// Extension trait for pipe operator.
trait Pipe: Sized {
    fn pipe<R, F: FnOnce(Self) -> R>(self, f: F) -> R {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_coordinator_target() {
        assert_eq!(parse_coordinator_target("TARGET row=3"), Some(3));
        assert_eq!(parse_coordinator_target("TARGET row=0"), Some(0));
        assert_eq!(parse_coordinator_target("I think row 2 needs work"), Some(2));
        assert_eq!(parse_coordinator_target("invalid"), None);
    }

    #[test]
    fn test_parse_proposer_proposal() {
        assert_eq!(
            parse_proposer_proposal("PROPOSE position=2 value=4"),
            Some((2, 4))
        );
        assert_eq!(
            parse_proposer_proposal("PROPOSE position=0 value=1"),
            Some((0, 1))
        );
        assert_eq!(
            parse_proposer_proposal("I propose position=1 value=3 because..."),
            Some((1, 3))
        );
        assert_eq!(parse_proposer_proposal("invalid"), None);
    }

    #[test]
    fn test_parse_validator_response() {
        assert!(parse_validator_response("APPROVE"));
        assert!(parse_validator_response("APPROVE - no violations"));
        assert!(parse_validator_response("approve"));
        assert!(!parse_validator_response("REJECT - column conflict"));
        assert!(!parse_validator_response("invalid"));
    }

    #[test]
    fn test_parse_coordinator_decision() {
        assert_eq!(
            parse_coordinator_decision("APPLY 1 2 3 4", 4),
            Some("1 2 3 4".to_string())
        );
        assert_eq!(
            parse_coordinator_decision("APPLY 1 2 3 4 5 6 7", 7),
            Some("1 2 3 4 5 6 7".to_string())
        );
        assert_eq!(parse_coordinator_decision("REJECT", 4), None);
        assert_eq!(parse_coordinator_decision("APPLY 1 2 3", 4), None); // Wrong length
    }

    #[test]
    fn test_conversation_state() {
        let mut state = ConversationState::new(5);
        state.add_message(AgentRole::Coordinator, "TARGET row=0".to_string(), None);
        assert_eq!(state.total_messages(), 1);
        assert!(!state.is_complete());

        state.final_patch = Some("1 2 3 4".to_string());
        assert!(state.is_complete());
    }

    #[test]
    fn test_format_history() {
        let mut state = ConversationState::new(5);
        state.add_message(AgentRole::Coordinator, "TARGET row=0".to_string(), None);
        state.add_message(
            AgentRole::Proposer,
            "PROPOSE position=1 value=3".to_string(),
            None,
        );

        let history = state.format_history();
        assert!(history.contains("[Coordinator]"));
        assert!(history.contains("[Proposer]"));
        assert!(history.contains("TARGET row=0"));
    }
}
