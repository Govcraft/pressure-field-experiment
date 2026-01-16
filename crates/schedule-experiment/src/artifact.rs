//! ScheduleArtifact: Artifact trait implementation for meeting room scheduling.
//!
//! Each time block is a region that can be patched independently.
//! Uses deterministic UUIDs for stable region IDs across re-parses.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Result};
use mti::prelude::*;
use serde::{Deserialize, Serialize};
use survival_kernel::artifact::Artifact;
use survival_kernel::region::{Patch, PatchOp, RegionId, RegionView};
use uuid::Uuid;

/// Namespace UUID for generating deterministic region IDs (MTI v5).
const REGION_NAMESPACE: Uuid = Uuid::from_bytes([
    0x7b, 0xa8, 0xc9, 0x20, 0xae, 0xbe, 0x22, 0xe2, 0x91, 0xc5, 0x01, 0xd1, 0x5e, 0xe5, 0x41, 0xda,
]);

/// Create a deterministic MTI-based region ID.
///
/// Uses MTI v5 pattern: namespace UUID + name -> deterministic, stable ID.
/// Returns `MagicTypeId` like `region_01h455vb4pex5vsknk084sn02q`.
fn create_region_mti(schedule_id: &str, day: u8, block: u8) -> RegionId {
    let name = format!("{}:day:{}:block:{}", schedule_id, day, block);
    // 1. Generate UUIDv5 from namespace + name (deterministic)
    let v5_uuid = Uuid::new_v5(&REGION_NAMESPACE, name.as_bytes());
    // 2. Create TypeIdPrefix (region is a valid prefix - lowercase, no special chars)
    let prefix = TypeIdPrefix::try_from("region").expect("region is a valid prefix");
    // 3. Create TypeIdSuffix from the UUID
    let suffix = TypeIdSuffix::from(v5_uuid);
    // 4. Combine into MagicTypeId
    MagicTypeId::new(prefix, suffix)
}

/// Unique identifier for a meeting.
pub type MeetingId = u32;

/// Unique identifier for an attendee.
pub type AttendeeId = u32;

/// Unique identifier for a room.
pub type RoomId = u32;

/// A time slot in the schedule (30-minute granularity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeSlot {
    /// Day index (0 = Monday, 4 = Friday for a 5-day week)
    pub day: u8,
    /// Slot index within the day (0 = 8:00, 1 = 8:30, etc.)
    pub slot: u8,
}

impl TimeSlot {
    pub fn new(day: u8, slot: u8) -> Self {
        Self { day, slot }
    }

    /// Get the hour and minute for this slot (assuming 8:00 start).
    pub fn to_time(&self) -> (u8, u8) {
        let hour = 8 + (self.slot / 2);
        let minute = (self.slot % 2) * 30;
        (hour, minute)
    }

    /// Format as "Day HH:MM".
    pub fn format(&self) -> String {
        let (hour, minute) = self.to_time();
        let day_name = match self.day {
            0 => "Mon",
            1 => "Tue",
            2 => "Wed",
            3 => "Thu",
            4 => "Fri",
            _ => "???",
        };
        format!("{} {:02}:{:02}", day_name, hour, minute)
    }
}

/// A meeting to be scheduled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meeting {
    pub id: MeetingId,
    /// Duration in 30-minute slots.
    pub duration_slots: u8,
    /// List of attendee IDs who must attend.
    pub attendees: Vec<AttendeeId>,
    /// Preferred rooms (in order of preference).
    pub preferred_rooms: Vec<RoomId>,
    /// Is this meeting currently scheduled?
    pub scheduled: Option<ScheduledMeeting>,
}

/// A scheduled meeting instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledMeeting {
    pub room: RoomId,
    pub start: TimeSlot,
}

impl ScheduledMeeting {
    /// Get all time slots occupied by this meeting.
    pub fn slots(&self, duration_slots: u8) -> Vec<TimeSlot> {
        let mut slots = Vec::with_capacity(duration_slots as usize);
        for i in 0..duration_slots {
            slots.push(TimeSlot {
                day: self.start.day,
                slot: self.start.slot + i,
            });
        }
        slots
    }
}

/// Room definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub id: RoomId,
    pub name: String,
    pub capacity: u8,
}

/// Shared grid for sensor synchronization.
pub type SharedSchedule = Arc<RwLock<ScheduleGrid>>;

/// A rejected patch that should be avoided in future attempts (negative pheromone).
///
/// These are tracked when patches fail evaluation and can be included in prompts
/// to help LLMs avoid repeating ineffective patterns.
#[derive(Debug, Clone)]
pub struct RejectedPatch {
    /// The region this patch targeted.
    pub region_id: RegionId,
    /// The proposed content that was rejected.
    pub proposed_content: String,
    /// The pressure delta (negative = harmful).
    pub pressure_delta: f64,
    /// The tick when this rejection occurred.
    pub tick: usize,
    /// Pheromone weight (starts at 1.0, decays over time).
    pub weight: f64,
}

impl RejectedPatch {
    /// Format for inclusion in an LLM prompt.
    pub fn format_for_prompt(&self) -> String {
        // Take first line only to keep prompts concise
        let first_line = self.proposed_content.lines().next().unwrap_or("");
        format!(
            "AVOID: \"{}\" (worsened by {:.1})",
            first_line,
            -self.pressure_delta
        )
    }
}

/// Grid representation of the schedule for sensor access.
#[derive(Debug, Clone, Default)]
pub struct ScheduleGrid {
    /// Grid: rooms × days × slots → Option<MeetingId>
    pub grid: Vec<Vec<Vec<Option<MeetingId>>>>,
    /// Meeting definitions.
    pub meetings: HashMap<MeetingId, Meeting>,
    /// Room definitions.
    pub rooms: Vec<Room>,
    /// Number of days in the schedule.
    pub num_days: u8,
    /// Number of slots per day.
    pub slots_per_day: u8,
}

impl ScheduleGrid {
    /// Create a new schedule grid.
    pub fn new(rooms: Vec<Room>, num_days: u8, slots_per_day: u8) -> Self {
        let num_rooms = rooms.len();
        let grid = vec![vec![vec![None; slots_per_day as usize]; num_days as usize]; num_rooms];
        Self {
            grid,
            meetings: HashMap::new(),
            rooms,
            num_days,
            slots_per_day,
        }
    }

    /// Get meeting at a specific slot.
    pub fn get(&self, room: RoomId, day: u8, slot: u8) -> Option<MeetingId> {
        self.grid
            .get(room as usize)
            .and_then(|r| r.get(day as usize))
            .and_then(|d| d.get(slot as usize))
            .copied()
            .flatten()
    }

    /// Set meeting at a specific slot.
    pub fn set(&mut self, room: RoomId, day: u8, slot: u8, meeting: Option<MeetingId>) {
        if let Some(r) = self.grid.get_mut(room as usize)
            && let Some(d) = r.get_mut(day as usize)
            && let Some(s) = d.get_mut(slot as usize)
        {
            *s = meeting;
        }
    }

    /// Count total overlaps (attendee double-bookings).
    pub fn count_overlaps(&self) -> usize {
        let mut overlaps = 0;

        for day in 0..self.num_days {
            for slot in 0..self.slots_per_day {
                let mut attendees_in_slot: HashMap<AttendeeId, Vec<MeetingId>> = HashMap::new();

                for room in 0..self.rooms.len() as u32 {
                    if let Some(meeting_id) = self.get(room, day, slot)
                        && let Some(meeting) = self.meetings.get(&meeting_id)
                    {
                        for &attendee in &meeting.attendees {
                            attendees_in_slot
                                .entry(attendee)
                                .or_default()
                                .push(meeting_id);
                        }
                    }
                }

                // Count overlaps: attendees in multiple meetings
                for meetings in attendees_in_slot.values() {
                    if meetings.len() > 1 {
                        overlaps += meetings.len() - 1;
                    }
                }
            }
        }

        overlaps
    }

    /// Count unscheduled meetings.
    pub fn count_unscheduled(&self) -> usize {
        self.meetings
            .values()
            .filter(|m| m.scheduled.is_none())
            .count()
    }

    /// Count total scheduled meeting-slots (for utilization).
    pub fn count_scheduled_slots(&self) -> usize {
        self.grid
            .iter()
            .flat_map(|r| r.iter())
            .flat_map(|d| d.iter())
            .filter(|s| s.is_some())
            .count()
    }

    /// Get total capacity (room-slots available).
    pub fn total_capacity(&self) -> usize {
        self.rooms.len() * self.num_days as usize * self.slots_per_day as usize
    }
}

/// A meeting room scheduling artifact.
///
/// Regions are time blocks (e.g., 2-hour windows) that can be optimized independently.
/// The goal is to minimize gaps, overlaps, and preference violations.
#[derive(Debug, Clone)]
pub struct ScheduleArtifact {
    /// Schedule grid.
    schedule: ScheduleGrid,
    /// Map from region ID to (day, start_slot, end_slot).
    region_map: HashMap<RegionId, (u8, u8, u8)>,
    /// Ordered list of region IDs.
    region_order: Vec<RegionId>,
    /// Schedule identifier for deterministic region IDs.
    schedule_id: String,
    /// Slots per time block (region granularity).
    _slots_per_block: u8,
    /// Optional shared schedule for sensor synchronization.
    shared_schedule: Option<SharedSchedule>,
    /// Rejected patches with decay weights (negative pheromones).
    rejected_patches: Arc<RwLock<Vec<RejectedPatch>>>,
    /// Current tick (for decay calculations).
    current_tick: Arc<RwLock<usize>>,
}

impl ScheduleArtifact {
    /// Create a new schedule artifact.
    ///
    /// # Arguments
    /// * `rooms` - Room definitions
    /// * `meetings` - Meeting definitions
    /// * `num_days` - Number of days in the schedule
    /// * `slots_per_day` - Number of 30-minute slots per day (e.g., 16 for 8am-4pm)
    /// * `slots_per_block` - Slots per time block (region granularity, e.g., 4 for 2-hour blocks)
    /// * `schedule_id` - Unique identifier for this schedule instance
    pub fn new(
        rooms: Vec<Room>,
        meetings: Vec<Meeting>,
        num_days: u8,
        slots_per_day: u8,
        slots_per_block: u8,
        schedule_id: impl Into<String>,
    ) -> Result<Self> {
        let schedule_id = schedule_id.into();

        if !slots_per_day.is_multiple_of(slots_per_block) {
            bail!(
                "slots_per_day ({}) must be divisible by slots_per_block ({})",
                slots_per_day,
                slots_per_block
            );
        }

        let mut schedule = ScheduleGrid::new(rooms, num_days, slots_per_day);

        // Add meetings to the grid
        for meeting in meetings {
            // If meeting is already scheduled, place it on the grid
            if let Some(ref scheduled) = meeting.scheduled {
                for slot in scheduled.slots(meeting.duration_slots) {
                    schedule.set(scheduled.room, slot.day, slot.slot, Some(meeting.id));
                }
            }
            schedule.meetings.insert(meeting.id, meeting);
        }

        // Generate deterministic region IDs for each time block
        let blocks_per_day = slots_per_day / slots_per_block;
        let mut region_map = HashMap::new();
        let mut region_order = Vec::new();

        for day in 0..num_days {
            for block in 0..blocks_per_day {
                let start_slot = block * slots_per_block;
                let end_slot = start_slot + slots_per_block;

                let region_id = create_region_mti(&schedule_id, day, block);

                region_map.insert(region_id.clone(), (day, start_slot, end_slot));
                region_order.push(region_id);
            }
        }

        Ok(Self {
            schedule,
            region_map,
            region_order,
            schedule_id,
            _slots_per_block: slots_per_block,
            shared_schedule: None,
            rejected_patches: Arc::new(RwLock::new(Vec::new())),
            current_tick: Arc::new(RwLock::new(0)),
        })
    }

    /// Configure a shared schedule for sensor synchronization.
    pub fn with_shared_schedule(mut self, schedule: SharedSchedule) -> Self {
        self.shared_schedule = Some(schedule);
        self
    }

    /// Get the shared schedule if configured.
    pub fn shared_schedule(&self) -> Option<SharedSchedule> {
        self.shared_schedule.clone()
    }

    /// Get the schedule grid.
    pub fn schedule(&self) -> &ScheduleGrid {
        &self.schedule
    }

    /// Get rooms.
    pub fn rooms(&self) -> &[Room] {
        &self.schedule.rooms
    }

    /// Get meetings.
    pub fn meetings(&self) -> &HashMap<MeetingId, Meeting> {
        &self.schedule.meetings
    }

    /// Get a meeting by ID.
    pub fn meeting(&self, id: MeetingId) -> Option<&Meeting> {
        self.schedule.meetings.get(&id)
    }

    /// Check if the schedule is complete (all meetings scheduled, no overlaps).
    pub fn is_solved(&self) -> bool {
        self.schedule.count_unscheduled() == 0 && self.schedule.count_overlaps() == 0
    }

    /// Get total pressure (higher = worse).
    pub fn total_pressure(&self) -> f64 {
        compute_pressure_from_grid(&self.schedule)
    }

    /// Get region metadata (day, start_slot, end_slot) for a region ID.
    pub fn region_metadata(&self, region_id: &RegionId) -> Option<(u8, u8, u8)> {
        self.region_map.get(region_id).copied()
    }

    /// Evaluate a patch by applying to a cloned grid and measuring actual pressure.
    ///
    /// Returns (should_accept, pressure_delta) where:
    /// - should_accept: true if patch improves pressure (delta > 0)
    /// - pressure_delta: old_pressure - new_pressure (positive = improvement)
    ///
    /// This maintains Nash equilibrium by only accepting moves that improve state.
    /// Rejected patches are tracked as negative pheromones for future prompt guidance.
    pub fn evaluate_patch(&self, patch: &Patch) -> (bool, f64) {
        // Get region metadata
        let Some((day, start_slot, end_slot)) = self.region_metadata(&patch.region) else {
            return (false, 0.0);
        };

        // Clone the schedule grid
        let mut test_schedule = self.schedule.clone();

        // Extract content for potential rejection tracking
        let content = if let PatchOp::Replace(c) = &patch.op {
            Some(c.clone())
        } else {
            None
        };

        // Apply patch to the clone
        if let Some(ref new_content) = content {
            if let Ok(assignments) = self.parse_block_schedule(new_content, day, start_slot, end_slot)
            {
                apply_block_schedule_to_grid(
                    &mut test_schedule,
                    day,
                    start_slot,
                    end_slot,
                    &assignments,
                );
            } else {
                // Parse failed - reject patch
                return (false, 0.0);
            }
        } else {
            // Only Replace operations are supported for schedule patches
            return (false, 0.0);
        }

        // Measure pressure from both grids (actual state)
        let old_pressure = compute_pressure_from_grid(&self.schedule);
        let new_pressure = compute_pressure_from_grid(&test_schedule);

        let delta = old_pressure - new_pressure;
        let should_accept = delta > 0.0;

        // Track rejected patches as negative pheromones (only if harmful)
        if !should_accept
            && delta < 0.0
            && let Some(proposed_content) = content
            && let Ok(tick) = self.current_tick.read()
            && let Ok(mut rejected) = self.rejected_patches.write()
        {
            rejected.push(RejectedPatch {
                region_id: patch.region.clone(),
                proposed_content,
                pressure_delta: delta,
                tick: *tick,
                weight: 1.0,
            });
        }

        (should_accept, delta)
    }

    /// Get rejected patches for a specific region (for prompt injection).
    ///
    /// Returns up to `max` patches, sorted by weight (highest first).
    pub fn get_rejected_for_region(&self, region_id: &RegionId, max: usize) -> Vec<RejectedPatch> {
        let Ok(rejected) = self.rejected_patches.read() else {
            return Vec::new();
        };

        let mut matches: Vec<_> = rejected
            .iter()
            .filter(|r| &r.region_id == region_id)
            .cloned()
            .collect();

        // Sort by weight (highest first)
        matches.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(max);
        matches
    }

    /// Get all rejected patches (for debugging/stats).
    pub fn get_all_rejected(&self) -> Vec<RejectedPatch> {
        self.rejected_patches
            .read()
            .map(|r| r.clone())
            .unwrap_or_default()
    }

    /// Apply decay to all rejected patches (call each tick).
    ///
    /// Default decay_factor: 0.95 (5% decay per tick)
    /// Default eviction_threshold: 0.1
    pub fn apply_rejection_decay(&self, decay_factor: f64, eviction_threshold: f64) {
        if let Ok(mut rejected) = self.rejected_patches.write() {
            for r in rejected.iter_mut() {
                r.weight *= decay_factor;
            }
            rejected.retain(|r| r.weight >= eviction_threshold);
        }

        if let Ok(mut tick) = self.current_tick.write() {
            *tick += 1;
        }
    }

    /// Get current rejection stats.
    pub fn rejection_stats(&self) -> (usize, f64) {
        let rejected = self.rejected_patches.read().ok();
        let count = rejected.as_ref().map(|r| r.len()).unwrap_or(0);
        let total_weight = rejected
            .as_ref()
            .map(|r| r.iter().map(|p| p.weight).sum())
            .unwrap_or(0.0);
        (count, total_weight)
    }

    /// Sync the shared schedule after a patch.
    fn sync_shared_schedule(&self) {
        if let Some(ref shared) = self.shared_schedule
            && let Ok(mut locked) = shared.write()
        {
            locked.grid = self.schedule.grid.clone();
            locked.meetings = self.schedule.meetings.clone();
        }
    }

    /// Get meetings that could potentially be scheduled in a time block.
    fn unscheduled_meetings_for_block(&self, _day: u8, start_slot: u8, end_slot: u8) -> Vec<&Meeting> {
        self.schedule
            .meetings
            .values()
            .filter(|m| {
                // Not yet scheduled
                m.scheduled.is_none()
                    // And fits within the block
                    && m.duration_slots <= (end_slot - start_slot)
            })
            .collect()
    }

    /// Get current assignments in a time block.
    fn block_assignments(&self, day: u8, start_slot: u8, end_slot: u8) -> HashMap<RoomId, Vec<(MeetingId, u8, u8)>> {
        let mut assignments: HashMap<RoomId, Vec<(MeetingId, u8, u8)>> = HashMap::new();

        for room in &self.schedule.rooms {
            let mut room_meetings = Vec::new();
            let mut slot = start_slot;

            while slot < end_slot {
                if let Some(meeting_id) = self.schedule.get(room.id, day, slot) {
                    // Find the meeting and its duration
                    if let Some(meeting) = self.schedule.meetings.get(&meeting_id) {
                        let meeting_start = slot;
                        let meeting_end = (slot + meeting.duration_slots).min(end_slot);

                        // Avoid duplicates (meeting spans multiple slots)
                        if room_meetings.last().map(|(id, _, _)| *id) != Some(meeting_id) {
                            room_meetings.push((meeting_id, meeting_start, meeting_end));
                        }

                        slot = meeting_end;
                    } else {
                        slot += 1;
                    }
                } else {
                    slot += 1;
                }
            }

            assignments.insert(room.id, room_meetings);
        }

        assignments
    }

    /// Parse a block schedule from LLM response.
    ///
    /// Expected format:
    /// ```text
    /// Room A: 5 (10:00-11:00), 7 (11:00-12:00)
    /// Room B: 12 (10:30-11:30)
    /// Room C: [empty]
    /// ```
    pub fn parse_block_schedule(
        &self,
        response: &str,
        _day: u8,
        start_slot: u8,
        _end_slot: u8,
    ) -> Result<Vec<(MeetingId, RoomId, u8)>> {
        let mut assignments = Vec::new();

        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() || line.to_lowercase().contains("empty") {
                continue;
            }

            // Parse "Room X: meeting_id (time), meeting_id (time), ..."
            if let Some(colon_pos) = line.find(':') {
                let room_part = line[..colon_pos].trim();
                let meetings_part = line[colon_pos + 1..].trim();

                // Extract room ID from room name
                let room_id = self
                    .schedule
                    .rooms
                    .iter()
                    .find(|r| room_part.contains(&r.name) || room_part.ends_with(&r.id.to_string()))
                    .map(|r| r.id);

                if let Some(room_id) = room_id {
                    // Parse meeting assignments
                    for meeting_str in meetings_part.split(',') {
                        let meeting_str = meeting_str.trim();
                        if meeting_str.is_empty() || meeting_str.to_lowercase().contains("empty") {
                            continue;
                        }

                        // Extract meeting ID (first number in the string)
                        if let Some(id) = extract_meeting_id(meeting_str) {
                            // Extract start slot from time if present, else use block start
                            let slot = extract_start_slot(meeting_str).unwrap_or(start_slot);
                            assignments.push((id, room_id, slot));
                        }
                    }
                }
            }
        }

        Ok(assignments)
    }

    /// Apply a parsed block schedule.
    fn apply_block_schedule(
        &mut self,
        day: u8,
        start_slot: u8,
        end_slot: u8,
        assignments: &[(MeetingId, RoomId, u8)],
    ) -> Result<()> {
        // First, clear all meetings from this block
        for room in 0..self.schedule.rooms.len() as u32 {
            for slot in start_slot..end_slot {
                if let Some(meeting_id) = self.schedule.get(room, day, slot) {
                    // Mark meeting as unscheduled
                    if let Some(meeting) = self.schedule.meetings.get_mut(&meeting_id) {
                        meeting.scheduled = None;
                    }
                }
                self.schedule.set(room, day, slot, None);
            }
        }

        // Then, place new assignments
        for &(meeting_id, room_id, slot) in assignments {
            // Get duration first (immutable borrow)
            let duration = self
                .schedule
                .meetings
                .get(&meeting_id)
                .map(|m| m.duration_slots)
                .unwrap_or(0);

            if duration == 0 {
                continue;
            }

            // Place meeting on grid
            for s in slot..slot + duration {
                if s < end_slot {
                    self.schedule.set(room_id, day, s, Some(meeting_id));
                }
            }

            // Update meeting's scheduled state (separate mutable borrow)
            if let Some(meeting) = self.schedule.meetings.get_mut(&meeting_id) {
                meeting.scheduled = Some(ScheduledMeeting {
                    room: room_id,
                    start: TimeSlot::new(day, slot),
                });
            }
        }

        Ok(())
    }
}

/// Compute total pressure from a schedule grid.
///
/// This measures actual state, not estimated state.
fn compute_pressure_from_grid(schedule: &ScheduleGrid) -> f64 {
    let unscheduled = schedule.count_unscheduled() as f64;
    let overlaps = schedule.count_overlaps() as f64;

    // Weight overlaps heavily (constraint violations)
    unscheduled * 1.0 + overlaps * 10.0
}

/// Apply a block schedule to any ScheduleGrid (not just self.schedule).
///
/// This enables clone-based patch evaluation.
fn apply_block_schedule_to_grid(
    schedule: &mut ScheduleGrid,
    day: u8,
    start_slot: u8,
    end_slot: u8,
    assignments: &[(MeetingId, RoomId, u8)],
) {
    // First, clear all meetings from this block
    for room in 0..schedule.rooms.len() as u32 {
        for slot in start_slot..end_slot {
            if let Some(meeting_id) = schedule.get(room, day, slot) {
                // Mark meeting as unscheduled
                if let Some(meeting) = schedule.meetings.get_mut(&meeting_id) {
                    meeting.scheduled = None;
                }
            }
            schedule.set(room, day, slot, None);
        }
    }

    // Then, place new assignments
    for &(meeting_id, room_id, slot) in assignments {
        // Get duration first (immutable borrow)
        let duration = schedule
            .meetings
            .get(&meeting_id)
            .map(|m| m.duration_slots)
            .unwrap_or(0);

        if duration == 0 {
            continue;
        }

        // Place meeting on grid
        for s in slot..slot + duration {
            if s < end_slot {
                schedule.set(room_id, day, s, Some(meeting_id));
            }
        }

        // Update meeting's scheduled state (separate mutable borrow)
        if let Some(meeting) = schedule.meetings.get_mut(&meeting_id) {
            meeting.scheduled = Some(ScheduledMeeting {
                room: room_id,
                start: TimeSlot::new(day, slot),
            });
        }
    }
}

/// Extract meeting ID from a string like "5 (10:00-11:00)" or "Meeting 5".
fn extract_meeting_id(s: &str) -> Option<MeetingId> {
    // Try to find first number in the string
    let mut num_start = None;
    let mut num_end = 0;

    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() {
            if num_start.is_none() {
                num_start = Some(i);
            }
            num_end = i + 1;
        } else if num_start.is_some() {
            break;
        }
    }

    if let Some(start) = num_start {
        s[start..num_end].parse().ok()
    } else {
        None
    }
}

/// Extract start slot from time string like "(10:00-11:00)".
fn extract_start_slot(s: &str) -> Option<u8> {
    // Look for HH:MM pattern
    let re = regex::Regex::new(r"(\d{1,2}):(\d{2})").ok()?;
    let caps = re.captures(s)?;

    let hour: u8 = caps.get(1)?.as_str().parse().ok()?;
    let minute: u8 = caps.get(2)?.as_str().parse().ok()?;

    // Convert to slot (assuming 8:00 start, 30-minute slots)
    if hour >= 8 {
        let slot = (hour - 8) * 2 + minute / 30;
        Some(slot)
    } else {
        None
    }
}

impl Artifact for ScheduleArtifact {
    fn region_ids(&self) -> Vec<RegionId> {
        self.region_order.clone()
    }

    fn read_region(&self, id: RegionId) -> Result<RegionView> {
        let (day, start_slot, end_slot) = self
            .region_map
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Region not found: {}", id))?;

        let day = *day;
        let start_slot = *start_slot;
        let end_slot = *end_slot;

        // Build content: current assignments
        let assignments = self.block_assignments(day, start_slot, end_slot);
        let mut content_lines = Vec::new();

        for room in &self.schedule.rooms {
            let room_assignments = assignments.get(&room.id).cloned().unwrap_or_default();
            if room_assignments.is_empty() {
                content_lines.push(format!("Room {}: [empty]", room.name));
            } else {
                let meeting_strs: Vec<String> = room_assignments
                    .iter()
                    .map(|(id, start, end)| {
                        let start_time = TimeSlot::new(day, *start);
                        let end_time = TimeSlot::new(day, *end);
                        format!(
                            "{} ({}-{})",
                            id,
                            start_time.format().split(' ').nth(1).unwrap_or("??:??"),
                            end_time.format().split(' ').nth(1).unwrap_or("??:??")
                        )
                    })
                    .collect();
                content_lines.push(format!("Room {}: {}", room.name, meeting_strs.join(", ")));
            }
        }

        let content = content_lines.join("\n");

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("day".to_string(), serde_json::json!(day));
        metadata.insert("start_slot".to_string(), serde_json::json!(start_slot));
        metadata.insert("end_slot".to_string(), serde_json::json!(end_slot));
        metadata.insert("schedule_id".to_string(), serde_json::json!(&self.schedule_id));

        // Include unscheduled meetings that could fit
        let unscheduled: Vec<serde_json::Value> = self
            .unscheduled_meetings_for_block(day, start_slot, end_slot)
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "duration_slots": m.duration_slots,
                    "attendees": m.attendees.len(),
                    "preferred_rooms": m.preferred_rooms,
                })
            })
            .collect();
        metadata.insert("unscheduled_meetings".to_string(), serde_json::json!(unscheduled));

        // Include room info
        let rooms_info: Vec<serde_json::Value> = self
            .schedule
            .rooms
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "name": r.name,
                    "capacity": r.capacity,
                })
            })
            .collect();
        metadata.insert("rooms".to_string(), serde_json::json!(rooms_info));

        // Block time range
        let start_time = TimeSlot::new(day, start_slot);
        let end_time = TimeSlot::new(day, end_slot);
        metadata.insert("time_range".to_string(), serde_json::json!(format!(
            "{} - {}",
            start_time.format(),
            end_time.format().split(' ').nth(1).unwrap_or("??:??")
        )));

        Ok(RegionView {
            id,
            kind: "time_block".to_string(),
            content,
            metadata,
        })
    }

    fn apply_patch(&mut self, patch: Patch) -> Result<()> {
        let (day, start_slot, end_slot) = self
            .region_map
            .get(&patch.region)
            .ok_or_else(|| anyhow::anyhow!("Region not found: {}", patch.region))?;

        let day = *day;
        let start_slot = *start_slot;
        let end_slot = *end_slot;

        match &patch.op {
            PatchOp::Replace(content) => {
                // Parse the new block schedule
                let assignments = self.parse_block_schedule(content, day, start_slot, end_slot)?;

                // Apply the new schedule
                self.apply_block_schedule(day, start_slot, end_slot, &assignments)?;

                // Sync shared schedule for sensors
                self.sync_shared_schedule();
            }
            PatchOp::Delete => {
                bail!("Cannot delete a time block");
            }
            PatchOp::InsertAfter(_) => {
                bail!("Cannot insert after a time block");
            }
        }

        Ok(())
    }

    fn source(&self) -> Option<String> {
        let mut lines = Vec::new();

        for day in 0..self.schedule.num_days {
            let day_name = match day {
                0 => "Monday",
                1 => "Tuesday",
                2 => "Wednesday",
                3 => "Thursday",
                4 => "Friday",
                _ => "Unknown",
            };
            lines.push(format!("=== {} ===", day_name));

            for room in &self.schedule.rooms {
                let mut room_line = format!("Room {}: ", room.name);
                let mut slot = 0;
                let mut meetings = Vec::new();

                while slot < self.schedule.slots_per_day {
                    if let Some(meeting_id) = self.schedule.get(room.id, day, slot) {
                        if let Some(meeting) = self.schedule.meetings.get(&meeting_id) {
                            let start_time = TimeSlot::new(day, slot);
                            let end_slot = slot + meeting.duration_slots;
                            let end_time = TimeSlot::new(day, end_slot);
                            meetings.push(format!(
                                "M{} ({}-{})",
                                meeting_id,
                                start_time.format().split(' ').nth(1).unwrap_or("??:??"),
                                end_time.format().split(' ').nth(1).unwrap_or("??:??")
                            ));
                            slot = end_slot;
                        } else {
                            slot += 1;
                        }
                    } else {
                        slot += 1;
                    }
                }

                if meetings.is_empty() {
                    room_line.push_str("[empty]");
                } else {
                    room_line.push_str(&meetings.join(", "));
                }
                lines.push(room_line);
            }
            lines.push(String::new());
        }

        Some(lines.join("\n"))
    }

    fn is_complete(&self) -> bool {
        self.is_solved()
    }

    fn on_patch_applied(&mut self, _patch: &Patch) {
        // Learning callback - could be used for example bank/pheromone deposits
    }

    fn evaluate_patch(&self, patch: &Patch) -> (bool, f64) {
        // Clone-based validation - apply to clone, measure actual pressure
        ScheduleArtifact::evaluate_patch(self, patch)
    }

    fn total_pressure(&self) -> Option<f64> {
        // Return actual pressure from grid state
        Some(ScheduleArtifact::total_pressure(self))
    }
}

impl fmt::Display for ScheduleArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Schedule ({}): {} rooms, {} meetings, {} days",
            self.schedule_id,
            self.schedule.rooms.len(),
            self.schedule.meetings.len(),
            self.schedule.num_days
        )?;
        writeln!(
            f,
            "  Unscheduled: {}, Overlaps: {}, Pressure: {:.1}",
            self.schedule.count_unscheduled(),
            self.schedule.count_overlaps(),
            self.total_pressure()
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rooms() -> Vec<Room> {
        vec![
            Room { id: 0, name: "A".to_string(), capacity: 10 },
            Room { id: 1, name: "B".to_string(), capacity: 6 },
            Room { id: 2, name: "C".to_string(), capacity: 4 },
        ]
    }

    fn sample_meetings() -> Vec<Meeting> {
        vec![
            Meeting {
                id: 1,
                duration_slots: 2, // 1 hour
                attendees: vec![1, 2, 3],
                preferred_rooms: vec![0],
                scheduled: None,
            },
            Meeting {
                id: 2,
                duration_slots: 4, // 2 hours
                attendees: vec![4, 5],
                preferred_rooms: vec![1, 0],
                scheduled: None,
            },
            Meeting {
                id: 3,
                duration_slots: 1, // 30 min
                attendees: vec![1, 4],
                preferred_rooms: vec![2],
                scheduled: None,
            },
        ]
    }

    fn sample_artifact() -> ScheduleArtifact {
        ScheduleArtifact::new(
            sample_rooms(),
            sample_meetings(),
            5,  // 5 days
            16, // 8 hours (8am-4pm)
            4,  // 2-hour blocks
            "test-schedule",
        )
        .unwrap()
    }

    #[test]
    fn test_create_artifact() {
        let artifact = sample_artifact();
        assert_eq!(artifact.schedule.rooms.len(), 3);
        assert_eq!(artifact.schedule.meetings.len(), 3);
        // 5 days × 4 blocks per day = 20 regions
        assert_eq!(artifact.region_ids().len(), 20);
    }

    #[test]
    fn test_read_region() {
        let artifact = sample_artifact();
        let regions = artifact.region_ids();
        let view = artifact.read_region(regions[0].clone()).unwrap();

        assert_eq!(view.kind, "time_block");
        assert!(view.content.contains("Room A"));
        assert!(view.content.contains("Room B"));
        assert!(view.content.contains("Room C"));
    }

    #[test]
    fn test_region_ids_stable() {
        let artifact1 = sample_artifact();
        let artifact2 = sample_artifact();
        assert_eq!(artifact1.region_ids(), artifact2.region_ids());
    }

    #[test]
    fn test_is_solved_initially_false() {
        let artifact = sample_artifact();
        assert!(!artifact.is_solved());
    }

    #[test]
    fn test_pressure_calculation() {
        let artifact = sample_artifact();
        let pressure = artifact.total_pressure();
        // 3 unscheduled meetings × 1.0 = 3.0
        assert_eq!(pressure, 3.0);
    }

    #[test]
    fn test_time_slot_format() {
        let slot = TimeSlot::new(0, 4); // Monday, 10:00
        assert_eq!(slot.format(), "Mon 10:00");

        let slot2 = TimeSlot::new(2, 9); // Wednesday, 12:30
        assert_eq!(slot2.format(), "Wed 12:30");
    }

    #[test]
    fn test_extract_meeting_id() {
        assert_eq!(extract_meeting_id("5 (10:00-11:00)"), Some(5));
        assert_eq!(extract_meeting_id("Meeting 12"), Some(12));
        assert_eq!(extract_meeting_id("M3 (09:00-10:00)"), Some(3));
        assert_eq!(extract_meeting_id("[empty]"), None);
    }

    #[test]
    fn test_extract_start_slot() {
        assert_eq!(extract_start_slot("(10:00-11:00)"), Some(4)); // 10:00 = slot 4
        assert_eq!(extract_start_slot("(08:30-09:00)"), Some(1)); // 8:30 = slot 1
        assert_eq!(extract_start_slot("(12:00-13:00)"), Some(8)); // 12:00 = slot 8
    }

    #[test]
    fn test_rejected_patch_format_for_prompt() {
        // Use a real region ID from the test artifact
        let region_id = create_region_mti("test_schedule", 0, 0);

        let rejected = RejectedPatch {
            region_id,
            proposed_content: "Room A: 5 (10:00-11:00)\nRoom B: [empty]".to_string(),
            pressure_delta: -3.0, // Made things worse
            tick: 5,
            weight: 1.0,
        };

        let formatted = rejected.format_for_prompt();
        // Should only show first line
        assert!(formatted.contains("Room A: 5"));
        assert!(!formatted.contains("Room B"));
        // Should show the worsening amount
        assert!(formatted.contains("worsened by 3.0"));
    }

    #[test]
    fn test_negative_pheromone_tracking_and_decay() {
        let artifact = sample_artifact();

        // Create test region IDs
        let region1_id = create_region_mti("test_schedule", 0, 0);
        let region2_id = create_region_mti("test_schedule", 0, 1);

        // Initially no rejections
        let (count, weight) = artifact.rejection_stats();
        assert_eq!(count, 0);
        assert_eq!(weight, 0.0);

        // Add some rejections manually (simulating evaluate_patch behavior)
        {
            let mut rejected = artifact.rejected_patches.write().unwrap();
            rejected.push(RejectedPatch {
                region_id: region1_id.clone(),
                proposed_content: "Bad schedule 1".to_string(),
                pressure_delta: -2.0,
                tick: 1,
                weight: 1.0,
            });
            rejected.push(RejectedPatch {
                region_id: region1_id.clone(),
                proposed_content: "Bad schedule 2".to_string(),
                pressure_delta: -1.0,
                tick: 2,
                weight: 1.0,
            });
            rejected.push(RejectedPatch {
                region_id: region2_id.clone(),
                proposed_content: "Bad schedule 3".to_string(),
                pressure_delta: -5.0,
                tick: 3,
                weight: 1.0,
            });
        }

        // Should have 3 rejections
        let (count, weight) = artifact.rejection_stats();
        assert_eq!(count, 3);
        assert!((weight - 3.0).abs() < 0.01);

        // Query for region1 should return 2
        let region1_rejections = artifact.get_rejected_for_region(&region1_id, 10);
        assert_eq!(region1_rejections.len(), 2);

        // Query for region2 should return 1
        let region2_rejections = artifact.get_rejected_for_region(&region2_id, 10);
        assert_eq!(region2_rejections.len(), 1);

        // Apply decay
        artifact.apply_rejection_decay(0.5, 0.1);

        // Weights should be halved
        let (count, weight) = artifact.rejection_stats();
        assert_eq!(count, 3);
        assert!((weight - 1.5).abs() < 0.01);

        // Apply more decay to trigger eviction
        artifact.apply_rejection_decay(0.1, 0.1);

        // All rejections should be evicted (weights below 0.1)
        let (count, _) = artifact.rejection_stats();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_rejected_patches_shared_across_clones() {
        let artifact1 = sample_artifact();

        // Create test region IDs
        let region1_id = create_region_mti("test_schedule", 0, 0);
        let region2_id = create_region_mti("test_schedule", 0, 1);

        // Add a rejection through artifact1
        {
            let mut rejected = artifact1.rejected_patches.write().unwrap();
            rejected.push(RejectedPatch {
                region_id: region1_id,
                proposed_content: "Bad schedule".to_string(),
                pressure_delta: -2.0,
                tick: 1,
                weight: 1.0,
            });
        }

        // Clone artifact1
        let artifact2 = artifact1.clone();

        // Both should see the same rejection (Arc sharing)
        assert_eq!(artifact1.rejection_stats().0, 1);
        assert_eq!(artifact2.rejection_stats().0, 1);

        // Adding through artifact2 should be visible in artifact1
        {
            let mut rejected = artifact2.rejected_patches.write().unwrap();
            rejected.push(RejectedPatch {
                region_id: region2_id,
                proposed_content: "Another bad schedule".to_string(),
                pressure_delta: -3.0,
                tick: 2,
                weight: 1.0,
            });
        }

        // Both should see 2 rejections now
        assert_eq!(artifact1.rejection_stats().0, 2);
        assert_eq!(artifact2.rejection_stats().0, 2);
    }
}
