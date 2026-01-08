//! Latin Square Experiment CLI.
//!
//! Commands:
//! - single: Run a single experiment
//! - grid: Run full grid experiment (strategies × agent counts × trials)
//! - ablation: Run ablation studies (decay/inhibition/examples on/off)

use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Local;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use latin_experiment::experiment::{ExperimentRunner, ExperimentRunnerConfig, Strategy};
use latin_experiment::generator::Difficulty;
use latin_experiment::results::GridResults;

/// Generate a timestamped output path from the given path.
/// e.g., "results.json" -> "results-20260108-010530.json"
fn timestamped_path(path: &Path) -> PathBuf {
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("results");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("json");
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    parent.join(format!("{}-{}.{}", stem, timestamp, ext))
}

#[derive(Parser)]
#[command(name = "latin-experiment")]
#[command(version)]
#[command(about = "Latin Square coordination experiments")]
struct Cli {
    /// Ollama host URL
    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    ollama_host: String,

    /// Model name (base model, first in escalation chain)
    #[arg(long, default_value = "qwen2.5:1.5b")]
    model: String,

    /// Model escalation chain (comma-separated, e.g., "qwen2.5:1.5b,qwen2.5:7b,qwen2.5:14b")
    /// When stuck at local minimum, escalates to larger models
    #[arg(long, default_value = "qwen2.5:1.5b,qwen2.5:7b,qwen2.5:14b", value_delimiter = ',')]
    model_chain: Vec<String>,

    /// Ticks with zero progress before escalating to larger model
    #[arg(long, default_value = "5")]
    escalation_threshold: usize,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single experiment
    Single {
        /// Strategy to use
        #[arg(long, default_value = "pressure_field")]
        strategy: String,

        /// Number of agents
        #[arg(long, default_value = "2")]
        agents: usize,

        /// Grid size
        #[arg(long, default_value = "6")]
        n: usize,

        /// Number of empty cells
        #[arg(long, default_value = "15")]
        empty: usize,

        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Run grid experiment
    Grid {
        /// Number of trials per configuration
        #[arg(long, default_value = "5")]
        trials: usize,

        /// Grid size
        #[arg(long, default_value = "6")]
        n: usize,

        /// Number of empty cells
        #[arg(long, default_value = "15")]
        empty: usize,

        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,

        /// Output file for results
        #[arg(long, default_value = "results.json")]
        output: PathBuf,

        /// Agent counts to test (comma-separated)
        #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
        agents: Vec<usize>,
    },

    /// Run ablation study
    Ablation {
        /// Number of trials per configuration
        #[arg(long, default_value = "5")]
        trials: usize,

        /// Grid size
        #[arg(long, default_value = "6")]
        n: usize,

        /// Number of empty cells
        #[arg(long, default_value = "15")]
        empty: usize,

        /// Maximum ticks
        #[arg(long, default_value = "50")]
        max_ticks: usize,

        /// Output file for results
        #[arg(long, default_value = "ablation.json")]
        output: PathBuf,
    },

    /// Generate and display a puzzle
    Generate {
        /// Grid size
        #[arg(long, default_value = "6")]
        n: usize,

        /// Number of empty cells
        #[arg(long, default_value = "15")]
        empty: usize,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .compact()
        .init();

    match cli.command {
        Commands::Single {
            strategy,
            agents,
            n,
            empty,
            max_ticks,
            seed,
        } => {
            let strategy = parse_strategy(&strategy)?;

            let config = ExperimentRunnerConfig {
                ollama_host: cli.ollama_host,
                model: cli.model,
                model_chain: cli.model_chain,
                escalation_threshold: cli.escalation_threshold,
                max_ticks,
                difficulty: Difficulty::Custom {
                    n,
                    empty_cells: empty,
                },
                ..Default::default()
            };

            let runner = ExperimentRunner::new(config);
            let result = runner.run(strategy, agents, 0, seed).await?;

            println!("\n=== Experiment Result ===");
            println!("Strategy: {}", result.config.strategy);
            println!("Agents: {}", result.config.agent_count);
            println!("Grid: {}x{} with {} empty cells", n, n, empty);
            println!("Solved: {}", result.solved);
            println!("Ticks: {}", result.total_ticks);
            println!("Final pressure: {:.2}", result.final_pressure);

            if let Some(stats) = &result.example_bank_stats {
                println!("\nExample Bank Stats:");
                println!("  Total examples: {}", stats.total_examples);
                println!("  Avg weight: {:.2}", stats.avg_weight);
                println!("  Total uses: {}", stats.total_uses);
            }

            println!("\nPressure history:");
            for (i, p) in result.pressure_history.iter().enumerate() {
                println!("  Tick {}: {:.2}", i, p);
            }
        }

        Commands::Grid {
            trials,
            n,
            empty,
            max_ticks,
            output,
            agents,
        } => {
            info!(
                trials = trials,
                n = n,
                empty = empty,
                "Starting grid experiment"
            );

            let mut results = GridResults::new();
            let strategies = Strategy::all();

            let total = strategies.len() * agents.len() * trials;
            let mut completed = 0;

            for strategy in &strategies {
                for &agent_count in &agents {
                    for trial in 0..trials {
                        let config = ExperimentRunnerConfig {
                            ollama_host: cli.ollama_host.clone(),
                            model: cli.model.clone(),
                            model_chain: cli.model_chain.clone(),
                            escalation_threshold: cli.escalation_threshold,
                            max_ticks,
                            difficulty: Difficulty::Custom {
                                n,
                                empty_cells: empty,
                            },
                            ..Default::default()
                        };

                        let runner = ExperimentRunner::new(config);
                        let result = runner.run(*strategy, agent_count, trial, None).await?;

                        results.add(result);
                        completed += 1;

                        info!(
                            progress = format!("{}/{}", completed, total),
                            strategy = strategy.name(),
                            agents = agent_count,
                            trial = trial,
                            "Completed run"
                        );
                    }
                }
            }

            results.compute_summary();
            let output_path = timestamped_path(&output);
            results.save(&output_path)?;

            println!("\n=== Grid Experiment Complete ===");
            println!("Results saved to: {}", output_path.display());
            println!("\nSummary:");
            for (key, summary) in &results.summary {
                println!(
                    "  {}: solve_rate={:.1}%, avg_ticks={:.1}",
                    key,
                    summary.solve_rate * 100.0,
                    summary.avg_ticks
                );
            }
        }

        Commands::Ablation {
            trials,
            n,
            empty,
            max_ticks,
            output,
        } => {
            info!(trials = trials, "Starting ablation study");

            let mut results = GridResults::new();

            // Ablation configurations: (decay, inhibition, examples)
            let configs = vec![
                (true, true, true),   // Full model
                (false, true, true),  // No decay
                (true, false, true),  // No inhibition
                (true, true, false),  // No examples
                (false, false, true), // No decay, no inhibition
                (true, false, false), // No inhibition, no examples
                (false, true, false), // No decay, no examples
                (false, false, false), // Baseline (nothing)
            ];

            let total = configs.len() * trials;
            let mut completed = 0;

            for (decay, inhibition, examples) in configs {
                for trial in 0..trials {
                    let config = ExperimentRunnerConfig {
                        ollama_host: cli.ollama_host.clone(),
                        model: cli.model.clone(),
                        model_chain: cli.model_chain.clone(),
                        escalation_threshold: cli.escalation_threshold,
                        max_ticks,
                        difficulty: Difficulty::Custom {
                            n,
                            empty_cells: empty,
                        },
                        decay_enabled: decay,
                        inhibition_enabled: inhibition,
                        examples_enabled: examples,
                        ..Default::default()
                    };

                    let runner = ExperimentRunner::new(config);
                    let result = runner.run(Strategy::PressureField, 2, trial, None).await?;

                    results.add(result);
                    completed += 1;

                    info!(
                        progress = format!("{}/{}", completed, total),
                        decay = decay,
                        inhibition = inhibition,
                        examples = examples,
                        trial = trial,
                        "Completed ablation run"
                    );
                }
            }

            results.compute_summary();
            let output_path = timestamped_path(&output);
            results.save(&output_path)?;

            println!("\n=== Ablation Study Complete ===");
            println!("Results saved to: {}", output_path.display());
        }

        Commands::Generate { n, empty, seed } => {
            use latin_experiment::generator::{GeneratorConfig, LatinSquareGenerator};

            let config = GeneratorConfig {
                n,
                empty_cells: empty,
                seed,
            };

            let generator = LatinSquareGenerator::new(config);
            let artifact = generator.generate()?;

            println!("{}", artifact);
            println!("Empty cells: {}", artifact.empty_count());
            println!("Violations: {}", artifact.total_violations());
        }
    }

    Ok(())
}

fn parse_strategy(s: &str) -> Result<Strategy> {
    match s.to_lowercase().as_str() {
        "pressure_field" | "pressure-field" | "pf" => Ok(Strategy::PressureField),
        "sequential" | "seq" => Ok(Strategy::Sequential),
        "random" | "rand" => Ok(Strategy::Random),
        "hierarchical" | "hier" => Ok(Strategy::Hierarchical),
        _ => anyhow::bail!("Unknown strategy: {}", s),
    }
}
