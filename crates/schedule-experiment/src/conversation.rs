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
use survival_kernel::region::RegionId;
use tokio::sync::Semaphore;
use tracing::{debug, warn};

use crate::artifact::{ScheduleArtifact, SharedSchedule};
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
    pub region_id: Option<RegionId>,
}

/// Conversation state for a single tick.
#[derive(Debug, Clone)]
pub struct ConversationState {
    pub messages: Vec<ConversationMessage>,
    pub current_turn: usize,
    pub max_turns: usize,
    pub target_region: Option<RegionId>,
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

    pub fn add_message(&mut self, role: AgentRole, content: String, region_id: Option<RegionId>) {
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
    /// Coordinator selects which time block needs most attention.
    pub fn coordinator_select_region(schedule_state: &str, num_blocks: usize) -> String {
        format!(
            r#"You are a Coordinator agent for meeting room scheduling.

Current schedule state by time block:
{schedule_state}

Task: Identify which time block needs the most attention. Consider:
1. Time blocks with unscheduled meetings
2. Time blocks with attendee conflicts (overlaps)
3. Time blocks with gaps or underutilization

Respond with ONLY: TARGET block=<N>
Where N is the 0-indexed block number (0 to {max_block}).

Example: TARGET block=2"#,
            schedule_state = schedule_state,
            max_block = num_blocks.saturating_sub(1)
        )
    }

    /// Proposer suggests moving/scheduling a meeting.
    pub fn proposer(
        block_content: &str,
        unscheduled_info: &str,
        rooms_info: &str,
        time_range: &str,
        conversation_history: &str,
    ) -> String {
        format!(
            r#"You are a Proposer agent for meeting room scheduling.

Time Block: {time_range}

Current assignments:
{block_content}

Unscheduled meetings that could fit this block:
{unscheduled_info}

Available rooms:
{rooms_info}

Previous messages:
{conversation_history}

Propose ONE scheduling action. Choose from:
1. Schedule an unscheduled meeting: PROPOSE schedule meeting=<id> room=<room_name> start=<HH:MM>
2. Move a scheduled meeting: PROPOSE move meeting=<id> to room=<room_name> start=<HH:MM>

Example: PROPOSE schedule meeting=5 room=A start=10:00"#,
            time_range = time_range,
            block_content = block_content,
            unscheduled_info = if unscheduled_info.is_empty() {
                "(no unscheduled meetings fit this block)".to_string()
            } else {
                unscheduled_info.to_string()
            },
            rooms_info = rooms_info,
            conversation_history = if conversation_history.is_empty() {
                "(none yet)".to_string()
            } else {
                conversation_history.to_string()
            }
        )
    }

    /// Validator checks scheduling constraints.
    pub fn validator(proposal: &str, room_schedule: &str, block_content: &str) -> String {
        format!(
            r#"You are a Validator agent checking scheduling constraints.

Current block assignments:
{block_content}

Room schedule for proposed room:
{room_schedule}

Proposal: {proposal}

Check if the proposal:
1. Causes room conflicts (room already booked at that time)
2. Causes attendee conflicts (attendee already in another meeting)
3. Exceeds room capacity
4. Falls outside the time block

Respond with ONLY ONE of:
APPROVE
REJECT <brief reason>"#,
            block_content = block_content,
            room_schedule = room_schedule,
            proposal = proposal
        )
    }

    /// Coordinator makes final decision after dialogue.
    pub fn coordinator_decide(
        conversation_history: &str,
        block_content: &str,
        rooms_info: &str,
    ) -> String {
        format!(
            r#"You are a Coordinator agent. Based on the conversation, make the final decision.

Conversation so far:
{conversation_history}

Current block content:
{block_content}

Available rooms:
{rooms_info}

If a valid proposal was APPROVED by the Validator, output the updated schedule:
APPLY
Room A: <meeting assignments for room A>
Room B: <meeting assignments for room B>
...

Format meeting assignments as: <meeting_id> (HH:MM-HH:MM), ...
Use [empty] if room has no meetings in this block.

If the proposal was REJECTED or no valid proposal exists, output:
REJECT"#,
            conversation_history = conversation_history,
            block_content = block_content,
            rooms_info = rooms_info
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

    /// Update the model used for LLM calls (for model escalation).
    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    /// Update the vLLM host (for model escalation with multi-host setup).
    pub fn set_host(&mut self, vllm_host: &str) {
        self.client = VllmClient::new(vllm_host);
    }

    /// Get the current model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Run a complete conversation for one tick.
    /// Returns (final_patch, conversation_state).
    pub async fn run_tick(
        &self,
        artifact: &ScheduleArtifact,
        _shared_schedule: &SharedSchedule,
    ) -> Result<(Option<String>, ConversationState)> {
        let mut state = ConversationState::new(self.max_turns);
        let region_ids = artifact.region_ids();
        let num_blocks = region_ids.len();

        // Turn 1: Coordinator selects region
        let schedule_state = self.format_schedule_state(artifact);
        let region_idx = self
            .coordinator_select_region(&mut state, &schedule_state, num_blocks)
            .await?;

        if region_idx >= num_blocks {
            warn!(
                region_idx = region_idx,
                num_blocks = num_blocks,
                "Coordinator selected invalid region, falling back to 0"
            );
            state.target_region = region_ids.first().cloned();
        } else {
            state.target_region = region_ids.get(region_idx).cloned();
        }

        let Some(ref region_id) = state.target_region else {
            return Ok((None, state));
        };

        let region_view = artifact.read_region(region_id.clone())?;
        let block_content = &region_view.content;
        let time_range = region_view
            .metadata
            .get("time_range")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown time");
        let rooms_info = self.format_rooms_info(artifact);
        let unscheduled_info = self.format_unscheduled_info(&region_view.metadata);

        // Turns 2-N: Proposer/Validator dialogue
        let mut last_proposal: Option<String> = None;
        let mut last_approved = false;

        for turn in 1..self.max_turns {
            state.current_turn = turn;

            // Proposer turn
            let proposal = self
                .proposer_propose(
                    &mut state,
                    block_content,
                    &unscheduled_info,
                    &rooms_info,
                    time_range,
                )
                .await?;

            if let Some(ref prop) = proposal {
                last_proposal = Some(prop.clone());

                // Validator turn
                let room_schedule = self.get_room_schedule_for_proposal(artifact, prop);

                let approved = self
                    .validator_check(&mut state, prop, &room_schedule, block_content)
                    .await?;

                if approved {
                    last_approved = true;
                    state.consensus_ticks += 1;

                    // Get final decision from Coordinator
                    let final_content = self
                        .coordinator_decide(&mut state, block_content, &rooms_info)
                        .await?;

                    if let Some(content) = final_content {
                        state.final_patch = Some(content);
                    }
                    break;
                }
            } else {
                // Proposer failed to make a valid proposal
                debug!(turn = turn, "Proposer failed to make valid proposal");
            }
        }

        // Final Coordinator decision if no early consensus but we had an approved proposal
        if state.final_patch.is_none() && last_proposal.is_some() && last_approved {
            state.final_patch = self
                .coordinator_decide(&mut state, block_content, &rooms_info)
                .await?;
        }

        state.total_ticks += 1;
        Ok((state.final_patch.clone(), state))
    }

    fn format_schedule_state(&self, artifact: &ScheduleArtifact) -> String {
        artifact
            .region_ids()
            .iter()
            .enumerate()
            .filter_map(|(i, id)| {
                artifact.read_region(id.clone()).ok().map(|r| {
                    let time_range = r
                        .metadata
                        .get("time_range")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let unscheduled: usize = r
                        .metadata
                        .get("unscheduled_meetings")
                        .and_then(|v| v.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    format!(
                        "Block {}: {} ({} unscheduled)\n{}",
                        i, time_range, unscheduled, r.content
                    )
                })
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn format_rooms_info(&self, artifact: &ScheduleArtifact) -> String {
        artifact
            .rooms()
            .iter()
            .map(|r| format!("Room {}: capacity {}", r.name, r.capacity))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_unscheduled_info(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> String {
        metadata
            .get("unscheduled_meetings")
            .and_then(|v| v.as_array())
            .map(|meetings| {
                meetings
                    .iter()
                    .filter_map(|m| {
                        let id = m.get("id")?.as_u64()?;
                        let duration = m.get("duration_slots")?.as_u64()?;
                        let attendees = m.get("attendees")?.as_u64().unwrap_or(0);
                        Some(format!(
                            "Meeting {}: {} slots, {} attendees",
                            id, duration, attendees
                        ))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    fn get_room_schedule_for_proposal(
        &self,
        artifact: &ScheduleArtifact,
        proposal: &str,
    ) -> String {
        // Extract room name from proposal
        let room_re = Regex::new(r"room\s*=\s*([A-Za-z]+)").ok();
        let room_name = room_re
            .and_then(|re| re.captures(proposal))
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_uppercase());

        if let Some(room_name) = room_name {
            // Get current room schedule from artifact
            if let Some(source) = artifact.source() {
                source
                    .lines()
                    .filter(|line| line.contains(&format!("Room {}:", room_name)))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                format!("Room {}: (unknown)", room_name)
            }
        } else {
            "(could not determine room)".to_string()
        }
    }

    async fn coordinator_select_region(
        &self,
        state: &mut ConversationState,
        schedule_state: &str,
        num_blocks: usize,
    ) -> Result<usize> {
        let prompt = PromptTemplates::coordinator_select_region(schedule_state, num_blocks);
        let response = self.call_llm(&prompt).await?;

        state.add_message(AgentRole::Coordinator, response.clone(), None);

        parse_coordinator_target(&response).unwrap_or(0).pipe(Ok)
    }

    async fn coordinator_decide(
        &self,
        state: &mut ConversationState,
        block_content: &str,
        rooms_info: &str,
    ) -> Result<Option<String>> {
        let history = state.format_history();
        let prompt = PromptTemplates::coordinator_decide(&history, block_content, rooms_info);
        let response = self.call_llm(&prompt).await?;

        state.add_message(
            AgentRole::Coordinator,
            response.clone(),
            state.target_region.clone(),
        );

        Ok(parse_coordinator_decision(&response))
    }

    async fn proposer_propose(
        &self,
        state: &mut ConversationState,
        block_content: &str,
        unscheduled_info: &str,
        rooms_info: &str,
        time_range: &str,
    ) -> Result<Option<String>> {
        let history = state.format_history();
        let prompt = PromptTemplates::proposer(
            block_content,
            unscheduled_info,
            rooms_info,
            time_range,
            &history,
        );
        let response = self.call_llm(&prompt).await?;

        state.add_message(
            AgentRole::Proposer,
            response.clone(),
            state.target_region.clone(),
        );

        Ok(parse_proposer_proposal(&response))
    }

    async fn validator_check(
        &self,
        state: &mut ConversationState,
        proposal: &str,
        room_schedule: &str,
        block_content: &str,
    ) -> Result<bool> {
        let prompt = PromptTemplates::validator(proposal, room_schedule, block_content);
        let response = self.call_llm(&prompt).await?;

        state.add_message(
            AgentRole::Validator,
            response.clone(),
            state.target_region.clone(),
        );

        Ok(parse_validator_response(&response))
    }

    async fn call_llm(&self, prompt: &str) -> Result<String> {
        let _permit = self.llm_semaphore.acquire().await?;

        // Use generate_with_system for conversation agents
        let response = self
            .client
            .generate_with_system(
                &self.model,
                "You are a collaborative agent for meeting room scheduling. Follow the instructions precisely and respond in the exact format requested.",
                prompt,
                0.3, // Lower temperature for more consistent structured output
                0.9,
                512, // Schedules need more tokens than Latin squares
            )
            .await
            .context("LLM generation failed")?;

        Ok(response.trim().to_string())
    }
}

// Response parsing functions

fn parse_coordinator_target(response: &str) -> Option<usize> {
    // Look for "TARGET block=N" or "block N" or just a number
    let re = Regex::new(r"(?i)(?:TARGET\s+)?block\s*=?\s*(\d+)").ok()?;
    re.captures(response)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn parse_proposer_proposal(response: &str) -> Option<String> {
    // Look for "PROPOSE schedule/move meeting=..."
    let re = Regex::new(r"(?i)PROPOSE\s+((?:schedule|move)\s+meeting\s*=\s*\d+.*)").ok()?;
    re.captures(response)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
}

fn parse_validator_response(response: &str) -> bool {
    response.to_uppercase().contains("APPROVE")
}

fn parse_coordinator_decision(response: &str) -> Option<String> {
    if response.to_uppercase().contains("APPLY") {
        // Extract everything after APPLY that looks like room assignments
        let lines: Vec<&str> = response.lines().collect();
        let apply_idx = lines
            .iter()
            .position(|l| l.to_uppercase().contains("APPLY"));

        if let Some(idx) = apply_idx {
            let content_lines: Vec<&str> = lines[idx + 1..]
                .iter()
                .copied()
                .filter(|l| !l.trim().is_empty())
                .filter(|l| {
                    let trimmed = l.trim().to_lowercase();
                    trimmed.starts_with("room") || trimmed.contains(':')
                })
                .collect();

            if !content_lines.is_empty() {
                return Some(content_lines.join("\n"));
            }
        }
    }
    None
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
        assert_eq!(parse_coordinator_target("TARGET block=3"), Some(3));
        assert_eq!(parse_coordinator_target("TARGET block=0"), Some(0));
        assert_eq!(
            parse_coordinator_target("I think block 2 needs work"),
            Some(2)
        );
        assert_eq!(parse_coordinator_target("invalid"), None);
    }

    #[test]
    fn test_parse_proposer_proposal() {
        assert_eq!(
            parse_proposer_proposal("PROPOSE schedule meeting=5 room=A start=10:00"),
            Some("schedule meeting=5 room=A start=10:00".to_string())
        );
        assert_eq!(
            parse_proposer_proposal("PROPOSE move meeting=3 to room=B start=14:00"),
            Some("move meeting=3 to room=B start=14:00".to_string())
        );
        assert_eq!(parse_proposer_proposal("invalid"), None);
    }

    #[test]
    fn test_parse_validator_response() {
        assert!(parse_validator_response("APPROVE"));
        assert!(parse_validator_response("APPROVE - no conflicts"));
        assert!(parse_validator_response("approve"));
        assert!(!parse_validator_response("REJECT - room conflict"));
        assert!(!parse_validator_response("invalid"));
    }

    #[test]
    fn test_parse_coordinator_decision() {
        let response = "APPLY\nRoom A: 5 (10:00-11:00)\nRoom B: [empty]";
        let result = parse_coordinator_decision(response);
        assert!(result.is_some());
        assert!(result.as_ref().unwrap().contains("Room A"));

        assert_eq!(parse_coordinator_decision("REJECT"), None);
    }

    #[test]
    fn test_conversation_state() {
        let mut state = ConversationState::new(5);
        state.add_message(AgentRole::Coordinator, "TARGET block=0".to_string(), None);
        assert_eq!(state.total_messages(), 1);
        assert!(!state.is_complete());

        state.final_patch = Some("Room A: 5 (10:00-11:00)".to_string());
        assert!(state.is_complete());
    }

    #[test]
    fn test_format_history() {
        let mut state = ConversationState::new(5);
        state.add_message(AgentRole::Coordinator, "TARGET block=0".to_string(), None);
        state.add_message(
            AgentRole::Proposer,
            "PROPOSE schedule meeting=5 room=A start=10:00".to_string(),
            None,
        );

        let history = state.format_history();
        assert!(history.contains("[Coordinator]"));
        assert!(history.contains("[Proposer]"));
        assert!(history.contains("TARGET block=0"));
    }
}
