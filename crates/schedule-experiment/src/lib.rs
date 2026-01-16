//! Schedule Experiment - Coverage scheduling domain for pressure-field coordination.
//!
//! This experiment domain demonstrates emergent coordination on a problem with
//! **continuous quality gradients** (unlike Latin squares' binary pass/fail).
//!
//! ## Domain: Meeting Room Scheduling
//!
//! - N meetings to schedule across R rooms over D days
//! - Each meeting has: duration, attendees, room preferences
//! - Goal: Minimize gaps, overlaps, and preference violations
//!
//! ## Why This Favors Pressure-Field Coordination
//!
//! 1. **Continuous gradients**: Every gap reduced is progress
//! 2. **Local effects**: Moving one meeting doesn't break distant constraints
//! 3. **Parallel benefit**: Agents can optimize different time blocks simultaneously

pub mod artifact;
pub mod conversation;
pub mod example_bank;
pub mod experiment;
pub mod generator;
pub mod llm_actor;
pub mod results;
pub mod sensors;
pub mod vllm_client;

pub use artifact::{Meeting, MeetingId, ScheduleArtifact, ScheduleGrid, TimeSlot};
pub use generator::{ScheduleGenerator, ScheduleGeneratorConfig};
pub use sensors::{
    CombinedScheduleSensor, GapSensor, OverlapSensor, ScheduleSensor, UnscheduledSensor,
    UtilizationSensor,
};
