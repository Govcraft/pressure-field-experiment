//! Schedule Experiment CLI.
//!
//! Run coverage scheduling experiments to demonstrate pressure-field coordination.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use schedule_experiment::experiment::{ExperimentRunner, ExperimentRunnerConfig, Strategy};
use schedule_experiment::generator::ScheduleGeneratorConfig;
use schedule_experiment::results::{ExperimentResult, GridResults};
use schedule_experiment::ScheduleGenerator;
use survival_kernel::artifact::Artifact;

#[derive(Parser)]
#[command(name = "schedule-experiment")]
#[command(about = "Coverage scheduling experiments for pressure-field coordination")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama/vLLM host URL
    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    host: String,

    /// Enable verbose output
    #[arg(long, short)]
    verbose: bool,

    /// Model escalation chain (comma-separated, e.g., "qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b")
    #[arg(long, default_value = "qwen2.5:1.5b,qwen2.5:7b,qwen2.5:14b")]
    model_chain: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate and display a schedule problem.
    Generate {
        /// Difficulty: easy, medium, hard
        #[arg(short, long, default_value = "easy")]
        difficulty: String,
        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },

    /// Run a single experiment.
    Single {
        /// Strategy: pressure_field, sequential, random, hierarchical
        #[arg(short, long, default_value = "pressure_field")]
        strategy: String,
        /// Number of agents
        #[arg(short, long, default_value = "2")]
        agents: usize,
        /// Difficulty: easy, medium, hard
        #[arg(short, long, default_value = "easy")]
        difficulty: String,
        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,
        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run a grid of experiments.
    Grid {
        /// Number of trials per configuration
        #[arg(short, long, default_value = "30")]
        trials: usize,
        /// Strategies (comma-separated)
        #[arg(short, long, default_value = "pressure_field,sequential,random,hierarchical")]
        strategies: String,
        /// Agent counts (comma-separated)
        #[arg(short, long, default_value = "1,2,4")]
        agents: String,
        /// Difficulties (comma-separated): easy, medium, hard
        #[arg(short, long, default_value = "easy,medium,hard")]
        difficulties: String,
        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,
        /// Output file for results (JSON)
        #[arg(short, long, default_value = "results/schedule-experiment.json")]
        output: PathBuf,
    },

    /// Run ablation study (varying decay/inhibition/examples).
    Ablation {
        /// Number of trials per configuration
        #[arg(short, long, default_value = "10")]
        trials: usize,
        /// Number of agents
        #[arg(short, long, default_value = "2")]
        agents: usize,
        /// Difficulty: easy, medium, hard
        #[arg(short, long, default_value = "easy")]
        difficulty: String,
        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,
        /// Output file for results (JSON)
        #[arg(short, long, default_value = "results/schedule-ablation.json")]
        output: PathBuf,
    },
}

fn parse_strategy(s: &str) -> Option<Strategy> {
    match s.to_lowercase().as_str() {
        "pressure_field" | "pressurefield" => Some(Strategy::PressureField),
        "sequential" => Some(Strategy::Sequential),
        "random" => Some(Strategy::Random),
        "hierarchical" => Some(Strategy::Hierarchical),
        _ => None,
    }
}

fn parse_difficulty(s: &str) -> ScheduleGeneratorConfig {
    match s.to_lowercase().as_str() {
        "easy" => ScheduleGeneratorConfig::easy(),
        "medium" => ScheduleGeneratorConfig::medium(),
        "hard" => ScheduleGeneratorConfig::hard(),
        _ => {
            eprintln!("Unknown difficulty: {}. Using 'easy'.", s);
            ScheduleGeneratorConfig::easy()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match cli.command {
        Commands::Generate { difficulty, seed } => {
            let config = parse_difficulty(&difficulty);
            let mut generator = ScheduleGenerator::new(config, seed);
            let artifact = generator.generate();

            println!("{}", artifact);
            if let Some(source) = Artifact::source(&artifact) {
                println!("\nSchedule:\n{}", source);
            }
        }

        Commands::Single {
            strategy,
            agents,
            difficulty,
            seed,
            max_ticks,
            output,
        } => {
            let strategy = parse_strategy(&strategy)
                .unwrap_or_else(|| {
                    eprintln!("Unknown strategy: {}. Using 'pressure_field'.", strategy);
                    Strategy::PressureField
                });

            let generator_config = parse_difficulty(&difficulty);

            let model_chain: Vec<String> = cli.model_chain
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();

            let config = ExperimentRunnerConfig {
                vllm_host: cli.host.clone(),
                model: model_chain.first().cloned().unwrap_or_else(|| "qwen2.5:1.5b".to_string()),
                model_chain,
                generator_config,
                max_ticks,
                ..Default::default()
            };

            let runner = ExperimentRunner::new(config);
            let result = runner.run(strategy, agents, 0, seed).await?;

            let initial_pressure = result.pressure_history.first().copied().unwrap_or(0.0);
            let pressure_delta = initial_pressure - result.final_pressure;

            println!("\n=== Experiment Complete ===");
            println!("Strategy: {}", strategy.name());
            println!("Agents: {}", agents);
            println!("Solved: {}", result.solved);
            println!("Ticks: {}", result.total_ticks);
            println!(
                "Pressure: {:.2} -> {:.2} (delta: {:.2})",
                initial_pressure, result.final_pressure, pressure_delta
            );
            println!("Final model: {}", result.final_model);
            println!("Prompt tokens: {}", result.total_prompt_tokens);
            println!("Completion tokens: {}", result.total_completion_tokens);

            if !result.band_escalation_events.is_empty() {
                println!("\nBand escalation events:");
                for event in &result.band_escalation_events {
                    println!("  Tick {}: {} -> {}", event.tick, event.from_band, event.to_band);
                }
            }

            if !result.escalation_events.is_empty() {
                println!("\nModel escalation events:");
                for event in &result.escalation_events {
                    println!("  Tick {}: {} -> {}", event.tick, event.from_model, event.to_model);
                }
            }

            if let Some(output) = output {
                let json = serde_json::to_string_pretty(&result)?;
                std::fs::create_dir_all(output.parent().unwrap_or(&PathBuf::from(".")))?;
                std::fs::write(&output, json)?;
                println!("\nResults written to: {}", output.display());
            }
        }

        Commands::Grid {
            trials,
            strategies,
            agents,
            difficulties,
            max_ticks,
            output,
        } => {
            let strategies: Vec<Strategy> = strategies
                .split(',')
                .filter_map(|s| parse_strategy(s.trim()))
                .collect();

            let agent_counts: Vec<usize> = agents
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            let difficulty_names: Vec<&str> = difficulties
                .split(',')
                .map(|s| s.trim())
                .collect();

            let model_chain: Vec<String> = cli.model_chain
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();

            info!(
                strategies = ?strategies.iter().map(|s| s.name()).collect::<Vec<_>>(),
                agent_counts = ?agent_counts,
                difficulties = ?difficulty_names,
                model_chain = ?model_chain,
                trials,
                "Starting grid experiment"
            );

            let mut results: Vec<ExperimentResult> = Vec::new();

            for difficulty in &difficulty_names {
                let generator_config = parse_difficulty(difficulty);

                let config = ExperimentRunnerConfig {
                    vllm_host: cli.host.clone(),
                    model: model_chain.first().cloned().unwrap_or_else(|| "qwen2.5:1.5b".to_string()),
                    model_chain: model_chain.clone(),
                    generator_config,
                    max_ticks,
                    ..Default::default()
                };

                let runner = ExperimentRunner::new(config);

                for strategy in &strategies {
                    for &agent_count in &agent_counts {
                        for trial in 0..trials {
                            info!(
                                difficulty = %difficulty,
                                strategy = strategy.name(),
                                agents = agent_count,
                                trial,
                                "Running experiment"
                            );

                            let seed = Some((trial as u64) * 1000 + (agent_count as u64));
                            let result = runner.run(*strategy, agent_count, trial, seed).await?;

                            println!(
                                "{} difficulty={} agents={} trial={}: solved={} ticks={} pressure={:.2}",
                                strategy.name(),
                                difficulty,
                                agent_count,
                                trial,
                                result.solved,
                                result.total_ticks,
                                result.final_pressure
                            );

                            results.push(result);
                        }
                    }
                }
            }

            // Create GridResults and save
            let mut grid_results = GridResults::new();
            for result in results {
                grid_results.add(result);
            }
            grid_results.compute_summary();

            println!("\n=== Grid Results Summary ===");
            for (key, summary) in &grid_results.summary {
                let solved_count = (summary.solve_rate * summary.trials as f64).round() as usize;
                println!(
                    "{}: {:.1}% solved ({}/{}), avg ticks={:.1}",
                    key,
                    summary.solve_rate * 100.0,
                    solved_count,
                    summary.trials,
                    summary.avg_ticks
                );
            }

            let json = serde_json::to_string_pretty(&grid_results)?;
            std::fs::create_dir_all(output.parent().unwrap_or(&PathBuf::from(".")))?;
            std::fs::write(&output, json)?;
            println!("\nResults written to: {}", output.display());
        }

        Commands::Ablation {
            trials,
            agents,
            difficulty,
            max_ticks,
            output,
        } => {
            let generator_config = parse_difficulty(&difficulty);

            let model_chain: Vec<String> = cli.model_chain
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();

            // Ablation configurations: (decay, inhibition, examples)
            let ablation_configs = vec![
                (true, true, true, "full"),
                (false, true, true, "no_decay"),
                (true, false, true, "no_inhibition"),
                (true, true, false, "no_examples"),
                (false, false, true, "no_decay_no_inhibition"),
                (false, true, false, "no_decay_no_examples"),
                (true, false, false, "no_inhibition_no_examples"),
                (false, false, false, "baseline"),
            ];

            info!(
                configs = ablation_configs.len(),
                trials,
                agents,
                model_chain = ?model_chain,
                "Starting ablation study"
            );

            let mut results: Vec<ExperimentResult> = Vec::new();

            for (decay, inhibition, examples, name) in &ablation_configs {
                info!(
                    config = name,
                    decay,
                    inhibition,
                    examples,
                    "Running ablation config"
                );

                let config = ExperimentRunnerConfig {
                    vllm_host: cli.host.clone(),
                    model: model_chain.first().cloned().unwrap_or_else(|| "qwen2.5:1.5b".to_string()),
                    model_chain: model_chain.clone(),
                    generator_config: generator_config.clone(),
                    max_ticks,
                    decay_enabled: *decay,
                    inhibition_enabled: *inhibition,
                    examples_enabled: *examples,
                    ..Default::default()
                };

                let runner = ExperimentRunner::new(config);

                for trial in 0..trials {
                    let seed = Some((trial as u64) * 1000);
                    let result = runner.run(Strategy::PressureField, agents, trial, seed).await?;

                    println!(
                        "{}: trial={} solved={} ticks={} pressure={:.2}",
                        name,
                        trial,
                        result.solved,
                        result.total_ticks,
                        result.final_pressure
                    );

                    results.push(result);
                }
            }

            // Create GridResults and save
            let mut grid_results = GridResults::new();
            for result in results {
                grid_results.add(result);
            }
            grid_results.compute_summary();

            println!("\n=== Ablation Results Summary ===");
            for (key, summary) in &grid_results.summary {
                println!(
                    "{}: {:.1}% solved, avg ticks={:.1}",
                    key,
                    summary.solve_rate * 100.0,
                    summary.avg_ticks
                );
            }

            let json = serde_json::to_string_pretty(&grid_results)?;
            std::fs::create_dir_all(output.parent().unwrap_or(&PathBuf::from(".")))?;
            std::fs::write(&output, json)?;
            println!("\nResults written to: {}", output.display());
        }
    }

    Ok(())
}
