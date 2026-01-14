//! Latin Square Experiment: A simple domain for validating pressure-field coordination.
//!
//! This crate implements a Latin Square completion experiment that:
//! - Uses pure Rust validation (no subprocess spawning or forking)
//! - Demonstrates emergent coordination through column constraints
//! - Implements swarm intelligence via few-shot accumulation (pheromone-like)

pub mod artifact;
pub mod conversation;
pub mod example_bank;
pub mod experiment;
pub mod generator;
pub mod llm_actor;
pub mod results;
pub mod sensors;
pub mod vllm_client;
