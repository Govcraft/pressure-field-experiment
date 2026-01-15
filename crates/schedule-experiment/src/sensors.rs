//! Schedule sensors for measuring pressure signals.
//!
//! These sensors measure various quality signals that contribute to
//! the overall pressure of a schedule region (time block).
//!
//! All sensors read directly from the shared ScheduleGrid (actual state).
//! Clone-based patch validation is handled at the artifact level, so
//! sensors only need to measure current state for EMA updates.

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
        let day = region
            .metadata
            .get("day")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;
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

        // Read actual state from shared grid
        let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;
        let num_rooms = schedule.rooms.len();
        let total_slots = block_size * num_rooms;

        // Count filled slots by scanning the actual grid
        let mut filled_slots = 0;
        for room in &schedule.rooms {
            for slot in start_slot..end_slot {
                if schedule.get(room.id, day, slot).is_some() {
                    filled_slots += 1;
                }
            }
        }

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

        // Extract time block info from metadata
        let day = region
            .metadata
            .get("day")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;
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

        // Read actual state from shared grid
        let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;

        // Collect unique meeting IDs in this block from the grid
        let mut meeting_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for room in &schedule.rooms {
            for slot in start_slot..end_slot {
                if let Some(meeting_id) = schedule.get(room.id, day, slot) {
                    meeting_ids.insert(meeting_id);
                }
            }
        }

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
        let day = region
            .metadata
            .get("day")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;
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

        // Read actual state from shared grid
        let schedule = self.schedule.read().map_err(|e| anyhow::anyhow!("{}", e))?;
        let num_rooms = schedule.rooms.len();

        if num_rooms == 0 || block_size == 0 {
            signals.insert("utilization_variance".to_string(), 0.0);
            signals.insert("avg_utilization".to_string(), 0.0);
            return Ok(signals);
        }

        // Calculate utilization per room from actual grid state
        let mut room_utilizations = Vec::new();
        for room in &schedule.rooms {
            let mut filled_slots = 0;
            for slot in start_slot..end_slot {
                if schedule.get(room.id, day, slot).is_some() {
                    filled_slots += 1;
                }
            }
            let utilization = filled_slots as f64 / block_size as f64;
            room_utilizations.push(utilization);
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
/// Reports global unscheduled meeting count from the grid.
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

        // Read actual unscheduled count from grid
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
