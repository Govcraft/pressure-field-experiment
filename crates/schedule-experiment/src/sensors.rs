//! Schedule sensors for measuring pressure signals.
//!
//! These sensors measure various quality signals that contribute to
//! the overall pressure of a schedule region (time block).
//!
//! IMPORTANT: Sensors parse RegionView.content directly to support
//! proposed patch evaluation. The kernel creates hypothetical RegionViews
//! with new content to measure what pressure would be after a patch.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use survival_kernel::pressure::{Sensor, Signals};
use survival_kernel::region::RegionView;

use crate::artifact::ScheduleGrid;

/// Shared schedule grid for sensor access.
pub type SharedSchedule = Arc<RwLock<ScheduleGrid>>;

/// A sensor for schedule problems.
///
/// All schedule sensors share access to the schedule grid to compute
/// signals based on the current state.
pub trait ScheduleSensor: Sensor {
    /// Get the shared schedule grid.
    fn schedule(&self) -> &SharedSchedule;
}

/// Parse the content string to count filled slots per room.
///
/// Content format:
/// ```text
/// Room A: 5 (08:00-09:00), 7 (09:00-10:00)
/// Room B: [empty]
/// Room C: 12 (08:30-09:30)
/// ```
///
/// Returns (empty_room_count, total_room_count, filled_slot_estimate)
fn parse_content_for_gaps(content: &str, block_slots: usize) -> (usize, usize, usize) {
    let mut empty_rooms = 0;
    let mut total_rooms = 0;
    let mut filled_slots = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Look for "Room X:" lines
        if line.to_lowercase().starts_with("room") && line.contains(':') {
            total_rooms += 1;

            if line.to_lowercase().contains("[empty]") || line.ends_with(':') {
                empty_rooms += 1;
            } else {
                // Count meetings in this room by counting commas + 1 (or just meeting IDs)
                // Each meeting takes approximately duration_slots slots
                // For simplicity, count meeting entries and assume average 2 slots each
                let after_colon = line.split(':').nth(1).unwrap_or("");
                let meeting_count = after_colon
                    .split(',')
                    .filter(|s| !s.trim().is_empty() && !s.to_lowercase().contains("empty"))
                    .count();

                // Estimate filled slots (assume 2 slots per meeting on average)
                filled_slots += meeting_count * 2;
            }
        }
    }

    (empty_rooms, total_rooms, filled_slots.min(total_rooms * block_slots))
}

/// Parse content to extract meeting IDs scheduled in each room for overlap detection.
fn parse_content_for_meetings(content: &str) -> HashMap<String, Vec<u32>> {
    let mut room_meetings: HashMap<String, Vec<u32>> = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.to_lowercase().starts_with("room") && line.contains(':') {
            // Extract room name (e.g., "Room A" -> "A")
            let room_name = line
                .split(':')
                .next()
                .unwrap_or("")
                .trim()
                .replace("Room ", "")
                .replace("room ", "");

            let after_colon = line.split(':').nth(1).unwrap_or("");
            let mut meetings = Vec::new();

            // Extract meeting IDs (first number in each comma-separated segment)
            for segment in after_colon.split(',') {
                let segment = segment.trim();
                if segment.is_empty() || segment.to_lowercase().contains("empty") {
                    continue;
                }

                // Find first number in segment
                let mut num_str = String::new();
                for c in segment.chars() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                    } else if !num_str.is_empty() {
                        break;
                    }
                }

                if let Ok(meeting_id) = num_str.parse::<u32>() {
                    meetings.push(meeting_id);
                }
            }

            room_meetings.insert(room_name, meetings);
        }
    }

    room_meetings
}

/// Sensor that measures gap (unscheduled) time in a time block.
///
/// Higher values indicate more empty slots that could be filled.
#[derive(Clone)]
pub struct GapSensor {
    schedule: SharedSchedule,
}

impl GapSensor {
    pub fn new(schedule: SharedSchedule) -> Self {
        Self { schedule }
    }
}

impl Sensor for GapSensor {
    fn name(&self) -> &str {
        "gap"
    }

    fn measure(&self, region: &RegionView) -> Result<Signals> {
        let mut signals = HashMap::new();

        // Extract time block info from metadata
        let start_slot = region
            .metadata
            .get("start_slot")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;
        let end_slot = region
            .metadata
            .get("end_slot")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;

        let block_size = (end_slot - start_slot) as usize;

        // Get room count from shared schedule
        let num_rooms = {
            let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;
            schedule.rooms.len()
        };

        let total_slots = block_size * num_rooms;

        // Parse content to count filled slots (supports proposed patch evaluation)
        let (_empty_rooms, _total_rooms, filled_slots) =
            parse_content_for_gaps(&region.content, block_size);

        let empty_slots = total_slots.saturating_sub(filled_slots);

        // Gap ratio: 0.0 = fully scheduled, 1.0 = completely empty
        let gap_ratio = if total_slots > 0 {
            empty_slots as f64 / total_slots as f64
        } else {
            0.0
        };

        signals.insert("empty_slots".to_string(), empty_slots as f64);
        signals.insert("total_slots".to_string(), total_slots as f64);
        signals.insert("gap_ratio".to_string(), gap_ratio);

        Ok(signals)
    }
}

impl ScheduleSensor for GapSensor {
    fn schedule(&self) -> &SharedSchedule {
        &self.schedule
    }
}

/// Sensor that measures attendee double-booking conflicts.
///
/// Higher values indicate more scheduling conflicts.
#[derive(Clone)]
pub struct OverlapSensor {
    schedule: SharedSchedule,
}

impl OverlapSensor {
    pub fn new(schedule: SharedSchedule) -> Self {
        Self { schedule }
    }
}

impl Sensor for OverlapSensor {
    fn name(&self) -> &str {
        "overlap"
    }

    fn measure(&self, region: &RegionView) -> Result<Signals> {
        let mut signals = HashMap::new();

        // Parse meeting IDs from content (supports proposed patch evaluation)
        let room_meetings = parse_content_for_meetings(&region.content);

        // Get all unique meeting IDs from content
        let meeting_ids: Vec<u32> = room_meetings.values().flatten().copied().collect();

        // Look up attendees from shared schedule
        let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;

        // Build attendee -> meetings map
        let mut attendee_meetings: HashMap<u32, Vec<u32>> = HashMap::new();
        for &meeting_id in &meeting_ids {
            if let Some(meeting) = schedule.meetings.get(&meeting_id) {
                for &attendee in &meeting.attendees {
                    attendee_meetings
                        .entry(attendee)
                        .or_default()
                        .push(meeting_id);
                }
            }
        }

        // Count overlaps: attendees in multiple meetings in this block
        let mut overlaps = 0;
        for meetings in attendee_meetings.values() {
            if meetings.len() > 1 {
                // Each additional meeting after the first is an overlap
                overlaps += meetings.len() - 1;
            }
        }

        signals.insert("overlap_count".to_string(), overlaps as f64);

        Ok(signals)
    }
}

impl ScheduleSensor for OverlapSensor {
    fn schedule(&self) -> &SharedSchedule {
        &self.schedule
    }
}

/// Sensor that measures room utilization balance.
///
/// Higher values indicate uneven utilization across rooms.
#[derive(Clone)]
pub struct UtilizationSensor {
    schedule: SharedSchedule,
}

impl UtilizationSensor {
    pub fn new(schedule: SharedSchedule) -> Self {
        Self { schedule }
    }
}

impl Sensor for UtilizationSensor {
    fn name(&self) -> &str {
        "utilization"
    }

    fn measure(&self, region: &RegionView) -> Result<Signals> {
        let mut signals = HashMap::new();

        // Extract time block info from metadata
        let start_slot = region
            .metadata
            .get("start_slot")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;
        let end_slot = region
            .metadata
            .get("end_slot")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;

        let block_size = (end_slot - start_slot) as usize;

        // Get room count from shared schedule
        let num_rooms = {
            let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;
            schedule.rooms.len()
        };

        if num_rooms == 0 || block_size == 0 {
            signals.insert("utilization_variance".to_string(), 0.0);
            signals.insert("avg_utilization".to_string(), 0.0);
            return Ok(signals);
        }

        // Parse content to get meetings per room (supports proposed patch evaluation)
        let room_meetings = parse_content_for_meetings(&region.content);

        // Calculate utilization per room (estimate 2 slots per meeting)
        let mut room_utilizations = Vec::new();
        for (_room_name, meetings) in &room_meetings {
            let estimated_slots = (meetings.len() * 2).min(block_size);
            let utilization = estimated_slots as f64 / block_size as f64;
            room_utilizations.push(utilization);
        }

        // Pad with zeros for rooms not mentioned in content (assumed empty)
        while room_utilizations.len() < num_rooms {
            room_utilizations.push(0.0);
        }

        // Calculate average and variance
        let avg_utilization: f64 = if room_utilizations.is_empty() {
            0.0
        } else {
            room_utilizations.iter().sum::<f64>() / room_utilizations.len() as f64
        };

        let variance: f64 = if room_utilizations.is_empty() {
            0.0
        } else {
            room_utilizations
                .iter()
                .map(|u| (u - avg_utilization).powi(2))
                .sum::<f64>()
                / room_utilizations.len() as f64
        };

        signals.insert("avg_utilization".to_string(), avg_utilization);
        signals.insert("utilization_variance".to_string(), variance);

        Ok(signals)
    }
}

impl ScheduleSensor for UtilizationSensor {
    fn schedule(&self) -> &SharedSchedule {
        &self.schedule
    }
}

/// Sensor that measures unscheduled meeting count.
///
/// This is a global sensor that doesn't depend on the region.
#[derive(Clone)]
pub struct UnscheduledSensor {
    schedule: SharedSchedule,
}

impl UnscheduledSensor {
    pub fn new(schedule: SharedSchedule) -> Self {
        Self { schedule }
    }
}

impl Sensor for UnscheduledSensor {
    fn name(&self) -> &str {
        "unscheduled"
    }

    fn measure(&self, _region: &RegionView) -> Result<Signals> {
        let mut signals = HashMap::new();

        let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;
        let unscheduled = schedule.count_unscheduled();
        let total = schedule.meetings.len();

        signals.insert("unscheduled_count".to_string(), unscheduled as f64);
        signals.insert("total_meetings".to_string(), total as f64);
        signals.insert(
            "scheduled_ratio".to_string(),
            if total > 0 {
                (total - unscheduled) as f64 / total as f64
            } else {
                1.0
            },
        );

        Ok(signals)
    }
}

impl ScheduleSensor for UnscheduledSensor {
    fn schedule(&self) -> &SharedSchedule {
        &self.schedule
    }
}

/// Combined schedule sensor that produces all signals.
#[derive(Clone)]
pub struct CombinedScheduleSensor {
    gap: GapSensor,
    overlap: OverlapSensor,
    utilization: UtilizationSensor,
    unscheduled: UnscheduledSensor,
}

impl CombinedScheduleSensor {
    pub fn new(schedule: SharedSchedule) -> Self {
        Self {
            gap: GapSensor::new(schedule.clone()),
            overlap: OverlapSensor::new(schedule.clone()),
            utilization: UtilizationSensor::new(schedule.clone()),
            unscheduled: UnscheduledSensor::new(schedule),
        }
    }
}

impl Sensor for CombinedScheduleSensor {
    fn name(&self) -> &str {
        "schedule"
    }

    fn measure(&self, region: &RegionView) -> Result<Signals> {
        let mut signals = HashMap::new();

        // Combine signals from all sensors
        signals.extend(self.gap.measure(region)?);
        signals.extend(self.overlap.measure(region)?);
        signals.extend(self.utilization.measure(region)?);
        signals.extend(self.unscheduled.measure(region)?);

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{Meeting, Room, ScheduleArtifact, ScheduledMeeting, TimeSlot};
    use survival_kernel::artifact::Artifact;

    fn create_test_schedule() -> (ScheduleArtifact, SharedSchedule) {
        let rooms = vec![
            Room {
                id: 0,
                name: "A".to_string(),
                capacity: 10,
            },
            Room {
                id: 1,
                name: "B".to_string(),
                capacity: 6,
            },
        ];

        let meetings = vec![
            Meeting {
                id: 0,
                duration_slots: 2,
                attendees: vec![1, 2],
                preferred_rooms: vec![0],
                scheduled: Some(ScheduledMeeting {
                    room: 0,
                    start: TimeSlot::new(0, 0),
                }),
            },
            Meeting {
                id: 1,
                duration_slots: 2,
                attendees: vec![3, 4],
                preferred_rooms: vec![1],
                scheduled: None, // Not scheduled
            },
        ];

        let schedule = Arc::new(RwLock::new(crate::artifact::ScheduleGrid::new(
            rooms.clone(),
            5,
            16,
        )));

        // Place meeting 0 on grid
        {
            let mut s = schedule.write().unwrap();
            s.set(0, 0, 0, Some(0));
            s.set(0, 0, 1, Some(0));
            for m in &meetings {
                s.meetings.insert(m.id, m.clone());
            }
        }

        let artifact = ScheduleArtifact::new(rooms, meetings, 5, 16, 4, "test")
            .unwrap()
            .with_shared_schedule(schedule.clone());

        (artifact, schedule)
    }

    #[test]
    fn test_gap_sensor() {
        let (artifact, schedule) = create_test_schedule();
        let sensor = GapSensor::new(schedule);

        let regions = artifact.region_ids();
        let region = artifact.read_region(regions[0].clone()).unwrap();
        let signals = sensor.measure(&region).unwrap();

        // Block has 2 rooms × 4 slots = 8 total slots
        // 2 slots used by meeting 0
        assert_eq!(signals["total_slots"], 8.0);
        assert_eq!(signals["empty_slots"], 6.0);
    }

    #[test]
    fn test_overlap_sensor() {
        let (artifact, schedule) = create_test_schedule();
        let sensor = OverlapSensor::new(schedule);

        let regions = artifact.region_ids();
        let region = artifact.read_region(regions[0].clone()).unwrap();
        let signals = sensor.measure(&region).unwrap();

        // No overlaps in the test schedule
        assert_eq!(signals["overlap_count"], 0.0);
    }

    #[test]
    fn test_unscheduled_sensor() {
        let (artifact, schedule) = create_test_schedule();
        let sensor = UnscheduledSensor::new(schedule);

        let regions = artifact.region_ids();
        let region = artifact.read_region(regions[0].clone()).unwrap();
        let signals = sensor.measure(&region).unwrap();

        // 1 meeting unscheduled out of 2
        assert_eq!(signals["unscheduled_count"], 1.0);
        assert_eq!(signals["total_meetings"], 2.0);
    }

    #[test]
    fn test_combined_sensor() {
        let (artifact, schedule) = create_test_schedule();
        let sensor = CombinedScheduleSensor::new(schedule);

        let regions = artifact.region_ids();
        let region = artifact.read_region(regions[0].clone()).unwrap();
        let signals = sensor.measure(&region).unwrap();

        // Should have signals from all sub-sensors
        assert!(signals.contains_key("empty_slots"));
        assert!(signals.contains_key("overlap_count"));
        assert!(signals.contains_key("avg_utilization"));
        assert!(signals.contains_key("unscheduled_count"));
    }
}
