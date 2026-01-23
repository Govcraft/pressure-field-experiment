//! Schedule problem generator.
//!
//! Generates random but solvable meeting room scheduling problems.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::artifact::{Meeting, Room, ScheduleArtifact, ScheduledMeeting, TimeSlot};

/// Configuration for generating schedule problems.
#[derive(Debug, Clone)]
pub struct ScheduleGeneratorConfig {
    /// Number of rooms.
    pub num_rooms: usize,
    /// Number of days in the schedule.
    pub num_days: u8,
    /// Number of 30-minute slots per day (e.g., 16 for 8am-4pm).
    pub slots_per_day: u8,
    /// Number of meetings to schedule.
    pub num_meetings: usize,
    /// Number of unique attendees.
    pub num_attendees: usize,
    /// Fraction of meetings to pre-schedule (0.0 to 1.0).
    pub pre_scheduled_fraction: f64,
    /// Room capacity range (min, max).
    pub room_capacity_range: (u8, u8),
    /// Meeting duration range in slots (min, max).
    pub meeting_duration_range: (u8, u8),
    /// Attendees per meeting range (min, max).
    pub attendees_per_meeting_range: (usize, usize),
}

impl Default for ScheduleGeneratorConfig {
    fn default() -> Self {
        Self {
            num_rooms: 3,
            num_days: 5,
            slots_per_day: 16, // 8am-4pm
            num_meetings: 20,
            num_attendees: 10,
            pre_scheduled_fraction: 0.6, // 60% pre-scheduled
            room_capacity_range: (4, 12),
            meeting_duration_range: (1, 4), // 30min to 2hr
            attendees_per_meeting_range: (2, 6),
        }
    }
}

impl ScheduleGeneratorConfig {
    /// Easy difficulty: few meetings, more rooms, high pre-scheduled fraction.
    pub fn easy() -> Self {
        Self {
            num_rooms: 3,
            num_days: 5,
            slots_per_day: 16,
            num_meetings: 20,
            num_attendees: 10,
            pre_scheduled_fraction: 0.7,
            room_capacity_range: (4, 12),
            meeting_duration_range: (1, 3),
            attendees_per_meeting_range: (2, 4),
        }
    }

    /// Medium difficulty: moderate meetings and rooms.
    pub fn medium() -> Self {
        Self {
            num_rooms: 5,
            num_days: 5,
            slots_per_day: 16,
            num_meetings: 40,
            num_attendees: 15,
            pre_scheduled_fraction: 0.5,
            room_capacity_range: (4, 12),
            meeting_duration_range: (1, 4),
            attendees_per_meeting_range: (2, 5),
        }
    }

    /// Hard difficulty: many meetings, fewer rooms, low pre-scheduled fraction.
    pub fn hard() -> Self {
        Self {
            num_rooms: 5,
            num_days: 5,
            slots_per_day: 16,
            num_meetings: 60,
            num_attendees: 20,
            pre_scheduled_fraction: 0.3,
            room_capacity_range: (4, 10),
            meeting_duration_range: (2, 4),
            attendees_per_meeting_range: (3, 6),
        }
    }
}

/// Generator for schedule problems.
pub struct ScheduleGenerator {
    config: ScheduleGeneratorConfig,
    rng: ChaCha8Rng,
}

impl ScheduleGenerator {
    /// Create a new generator with the given config and seed.
    pub fn new(config: ScheduleGeneratorConfig, seed: u64) -> Self {
        Self {
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate a schedule problem.
    pub fn generate(&mut self) -> ScheduleArtifact {
        let rooms = self.generate_rooms();
        let meetings = self.generate_meetings(&rooms);
        let schedule_id = format!("schedule-{}", self.rng.random::<u32>());

        ScheduleArtifact::new(
            rooms,
            meetings,
            self.config.num_days,
            self.config.slots_per_day,
            4, // 2-hour blocks
            schedule_id,
        )
        .expect("Failed to create schedule artifact")
    }

    /// Generate room definitions.
    fn generate_rooms(&mut self) -> Vec<Room> {
        let room_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
        let (min_cap, max_cap) = self.config.room_capacity_range;

        (0..self.config.num_rooms)
            .map(|i| Room {
                id: i as u32,
                name: room_names
                    .get(i)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| format!("R{}", i)),
                capacity: self.rng.random_range(min_cap..=max_cap),
            })
            .collect()
    }

    /// Generate meeting definitions.
    fn generate_meetings(&mut self, rooms: &[Room]) -> Vec<Meeting> {
        let (min_dur, max_dur) = self.config.meeting_duration_range;
        let (min_att, max_att) = self.config.attendees_per_meeting_range;
        let num_pre_scheduled =
            (self.config.num_meetings as f64 * self.config.pre_scheduled_fraction) as usize;

        // Track scheduled slots to avoid initial overlaps
        let mut scheduled_slots: Vec<Vec<Vec<bool>>> =
            vec![
                vec![
                    vec![false; self.config.slots_per_day as usize];
                    self.config.num_days as usize
                ];
                rooms.len()
            ];

        // Track attendee schedules to avoid initial double-bookings
        let mut attendee_schedules: Vec<Vec<Vec<bool>>> =
            vec![
                vec![
                    vec![false; self.config.slots_per_day as usize];
                    self.config.num_days as usize
                ];
                self.config.num_attendees
            ];

        let mut meetings = Vec::with_capacity(self.config.num_meetings);

        for i in 0..self.config.num_meetings {
            let duration_slots = self.rng.random_range(min_dur..=max_dur);
            let num_attendees = self
                .rng
                .random_range(min_att..=max_att)
                .min(self.config.num_attendees);

            // Select random attendees
            let mut attendees: Vec<u32> = (0..self.config.num_attendees as u32).collect();
            attendees.shuffle(&mut self.rng);
            let attendees: Vec<u32> = attendees.into_iter().take(num_attendees).collect();

            // Select preferred rooms (up to 2)
            let mut room_ids: Vec<u32> = (0..rooms.len() as u32).collect();
            room_ids.shuffle(&mut self.rng);
            let preferred_rooms: Vec<u32> = room_ids.into_iter().take(2).collect();

            // Should this meeting be pre-scheduled?
            let scheduled = if i < num_pre_scheduled {
                self.try_schedule_meeting(
                    duration_slots,
                    &attendees,
                    rooms,
                    &mut scheduled_slots,
                    &mut attendee_schedules,
                )
            } else {
                None
            };

            meetings.push(Meeting {
                id: i as u32,
                duration_slots,
                attendees,
                preferred_rooms,
                scheduled,
            });
        }

        meetings
    }

    /// Try to find a valid slot for a meeting without conflicts.
    fn try_schedule_meeting(
        &mut self,
        duration_slots: u8,
        attendees: &[u32],
        rooms: &[Room],
        scheduled_slots: &mut [Vec<Vec<bool>>],
        attendee_schedules: &mut [Vec<Vec<bool>>],
    ) -> Option<ScheduledMeeting> {
        // Try up to 50 random positions
        for _ in 0..50 {
            let room_id = self.rng.random_range(0..rooms.len() as u32);
            let day = self.rng.random_range(0..self.config.num_days);
            let max_start = self.config.slots_per_day.saturating_sub(duration_slots);
            if max_start == 0 {
                continue;
            }
            let start_slot = self.rng.random_range(0..max_start);

            // Check if room is available
            let room_available = (start_slot..start_slot + duration_slots)
                .all(|slot| !scheduled_slots[room_id as usize][day as usize][slot as usize]);

            if !room_available {
                continue;
            }

            // Check if all attendees are available
            let attendees_available = attendees.iter().all(|&att| {
                (start_slot..start_slot + duration_slots)
                    .all(|slot| !attendee_schedules[att as usize][day as usize][slot as usize])
            });

            if !attendees_available {
                continue;
            }

            // Mark slots as occupied
            for slot in start_slot..start_slot + duration_slots {
                scheduled_slots[room_id as usize][day as usize][slot as usize] = true;
                for &att in attendees {
                    attendee_schedules[att as usize][day as usize][slot as usize] = true;
                }
            }

            return Some(ScheduledMeeting {
                room: room_id,
                start: TimeSlot::new(day, start_slot),
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use survival_kernel::artifact::Artifact;

    #[test]
    fn test_generate_easy() {
        let config = ScheduleGeneratorConfig::easy();
        let mut sched_gen = ScheduleGenerator::new(config, 42);
        let artifact = sched_gen.generate();

        assert_eq!(artifact.rooms().len(), 3);
        assert!(!artifact.meetings().is_empty());
    }

    #[test]
    fn test_generate_medium() {
        let config = ScheduleGeneratorConfig::medium();
        let mut sched_gen = ScheduleGenerator::new(config, 42);
        let artifact = sched_gen.generate();

        assert_eq!(artifact.rooms().len(), 5);
        assert_eq!(artifact.meetings().len(), 40);
    }

    #[test]
    fn test_generate_hard() {
        let config = ScheduleGeneratorConfig::hard();
        let mut sched_gen = ScheduleGenerator::new(config, 42);
        let artifact = sched_gen.generate();

        assert_eq!(artifact.rooms().len(), 5);
        assert_eq!(artifact.meetings().len(), 60);
    }

    #[test]
    fn test_deterministic() {
        let config = ScheduleGeneratorConfig::easy();
        let mut sched_gen1 = ScheduleGenerator::new(config.clone(), 123);
        let mut sched_gen2 = ScheduleGenerator::new(config, 123);

        let art1 = sched_gen1.generate();
        let art2 = sched_gen2.generate();

        // Same seed should produce same regions
        assert_eq!(art1.region_ids(), art2.region_ids());
    }

    #[test]
    fn test_no_initial_overlaps() {
        let config = ScheduleGeneratorConfig::medium();
        let mut sched_gen = ScheduleGenerator::new(config, 42);
        let artifact = sched_gen.generate();

        // Check no overlaps in pre-scheduled meetings
        assert_eq!(artifact.schedule().count_overlaps(), 0);
    }
}
