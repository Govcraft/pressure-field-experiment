#!/usr/bin/env Rscript
# Statistical Analysis of Latin Square Experiment Results
# For publication: accuracy, confidence intervals, statistical tests

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

results_dir <- "20260110-165547"

cat(strrep("=", 70), "\n")
cat("STATISTICAL ANALYSIS OF EXPERIMENT RESULTS\n")
cat(strrep("=", 70), "\n\n")

# Helper function to load and flatten results from JSON
load_results <- function(filename) {
  path <- file.path(results_dir, filename)
  if (!file.exists(path)) {
    cat("File not found:", path, "\n")
    return(NULL)
  }
  data <- fromJSON(path)
  results <- data$results

  # Flatten the config column
  if ("config" %in% names(results)) {
    config_df <- as.data.frame(results$config)
    results <- cbind(results[, !names(results) %in% "config"], config_df)
  }
  return(results)
}

# Helper function to parse log files for results from killed experiments
parse_log_results <- function(logfile) {
  path <- file.path(results_dir, "logs", logfile)
  if (!file.exists(path)) {
    cat("Log file not found:", path, "\n")
    return(NULL)
  }

  lines <- readLines(path)

  results <- data.frame(
    strategy = character(),
    solved = logical(),
    stringsAsFactors = FALSE
  )

  current_strategy <- NA
  current_solved <- FALSE

  for (line in lines) {
    # Match "Starting experiment" to get current strategy
    if (grepl("Starting experiment", line)) {
      strat_match <- regmatches(line, regexpr('strategy[^"]*"([^"]+)"', line, perl = TRUE))
      if (length(strat_match) > 0) {
        current_strategy <- gsub('.*"([^"]+)".*', '\\1', strat_match)
        current_solved <- FALSE
      }
    }

    # Match "Puzzle solved"
    if (grepl("Puzzle solved", line)) {
      current_solved <- TRUE
    }

    # Match "Completed run" to record result
    if (grepl("Completed run", line) && !is.na(current_strategy)) {
      results <- rbind(results, data.frame(
        strategy = current_strategy,
        solved = current_solved,
        stringsAsFactors = FALSE
      ))
      current_solved <- FALSE
    }
  }

  return(results)
}

# Function to compute confidence interval for proportion (Wilson score)
prop_ci <- function(successes, trials, conf_level = 0.95) {
  if (trials == 0) return(c(NA, NA, NA))
  prop <- successes / trials
  z <- qnorm(1 - (1 - conf_level) / 2)
  denom <- 1 + z^2 / trials
  center <- (prop + z^2 / (2 * trials)) / denom
  margin <- z * sqrt((prop * (1 - prop) + z^2 / (4 * trials)) / trials) / denom
  c(prop, max(0, center - margin), min(1, center + margin))
}

# ============================================================
# 1. ABLATION STUDY (Primary mechanism validation)
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("1. ABLATION STUDY (5x5, 5 empty cells)\n")
cat(strrep("=", 60), "\n\n")

ablation <- load_results("ablation-20260110-171559.json")
if (!is.null(ablation)) {
  cat("Total observations:", nrow(ablation), "\n\n")

  # Create config label
  ablation$config_label <- paste0(
    "D=", ifelse(ablation$decay_enabled, "T", "F"),
    " I=", ifelse(ablation$inhibition_enabled, "T", "F"),
    " E=", ifelse(ablation$examples_enabled, "T", "F")
  )

  cat("RESULTS BY CONFIGURATION:\n")
  cat(sprintf("%-20s %8s %12s %15s %12s\n",
              "Config", "Solved/N", "Rate", "Final Pressure", "SD"))
  cat(strrep("-", 70), "\n")

  configs <- unique(ablation$config_label)
  for (cfg in sort(configs)) {
    subset <- ablation[ablation$config_label == cfg, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    rate <- solved / total * 100
    mean_p <- mean(subset$final_pressure, na.rm = TRUE)
    sd_p <- sd(subset$final_pressure, na.rm = TRUE)
    cat(sprintf("%-20s %4d/%-4d %10.1f%% %15.2f %12.2f\n",
                cfg, solved, total, rate, mean_p, sd_p))
  }

  # T-test: decay effect on final pressure
  cat("\n\nT-TEST: Effect of DECAY on final pressure:\n")
  with_decay <- ablation[ablation$decay_enabled == TRUE, "final_pressure"]
  without_decay <- ablation[ablation$decay_enabled == FALSE, "final_pressure"]
  cat(sprintf("With decay:    mean = %.2f, sd = %.2f, n = %d\n",
              mean(with_decay), sd(with_decay), length(with_decay)))
  cat(sprintf("Without decay: mean = %.2f, sd = %.2f, n = %d\n",
              mean(without_decay), sd(without_decay), length(without_decay)))
  t_result <- t.test(with_decay, without_decay)
  cat(sprintf("\nt = %.2f, df = %.1f, p-value = %.2e\n",
              t_result$statistic, t_result$parameter, t_result$p.value))
  cat(sprintf("95%% CI for difference: [%.2f, %.2f]\n",
              t_result$conf.int[1], t_result$conf.int[2]))

  # Effect size (Cohen's d)
  pooled_sd <- sqrt(((length(with_decay)-1)*var(with_decay) +
                     (length(without_decay)-1)*var(without_decay)) /
                    (length(with_decay) + length(without_decay) - 2))
  cohens_d <- (mean(without_decay) - mean(with_decay)) / pooled_sd
  cat(sprintf("Cohen's d = %.2f (effect size)\n", cohens_d))

  # Summary stats for paper
  cat("\n\nKEY FINDINGS FOR PAPER:\n")
  cat(strrep("-", 50), "\n")
  decay_ratio <- mean(without_decay) / mean(with_decay)
  cat(sprintf("Decay disabled increases pressure %.1fx (%.2f vs %.2f)\n",
              decay_ratio, mean(without_decay), mean(with_decay)))
  cat(sprintf("Effect size: Cohen's d = %.2f (%s effect)\n",
              cohens_d,
              ifelse(abs(cohens_d) > 0.8, "large",
                     ifelse(abs(cohens_d) > 0.5, "medium", "small"))))
}

# ============================================================
# 2. SCALING ANALYSIS (Agent count scaling)
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("2. SCALING ANALYSIS (7x7, 8 empty cells)\n")
cat(strrep("=", 60), "\n\n")

scaling <- load_results("scaling-20260110-181839.json")
if (!is.null(scaling)) {
  cat("Total observations:", nrow(scaling), "\n\n")

  # Summary by strategy
  cat("SOLVE RATES BY STRATEGY (with 95% Wilson CI):\n")
  cat(sprintf("%-20s %8s %12s %20s\n", "Strategy", "Solved/N", "Rate", "95% CI"))
  cat(strrep("-", 65), "\n")

  strategies <- unique(scaling$strategy)
  for (strat in sort(strategies)) {
    subset <- scaling[scaling$strategy == strat, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    ci <- prop_ci(solved, total)
    cat(sprintf("%-20s %4d/%-4d %10.1f%% %8.1f%% - %5.1f%%\n",
                strat, solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
  }

  # By strategy and agent count
  cat("\n\nSOLVE RATES BY STRATEGY AND AGENT COUNT:\n")
  cat(sprintf("%-18s", "Strategy"))
  agent_counts <- sort(unique(scaling$agent_count))
  for (a in agent_counts) cat(sprintf("%7d", a))
  cat("    AVG\n")
  cat(strrep("-", 18 + 8 * length(agent_counts) + 8), "\n")

  for (strat in sort(strategies)) {
    cat(sprintf("%-18s", strat))
    rates <- c()
    for (a in agent_counts) {
      subset <- scaling[scaling$strategy == strat & scaling$agent_count == a, ]
      if (nrow(subset) > 0) {
        rate <- mean(subset$solved, na.rm = TRUE) * 100
        rates <- c(rates, rate)
        cat(sprintf("%6.1f%%", rate))
      } else {
        cat(sprintf("%7s", "N/A"))
      }
    }
    cat(sprintf("  %5.1f%%\n", mean(rates)))
  }

  # Pressure field detailed analysis
  cat("\n\nPRESSURE_FIELD SCALING ANALYSIS:\n")
  pf <- scaling[scaling$strategy == "pressure_field", ]
  cat(sprintf("%-10s %8s %12s %15s\n", "Agents", "Solved/N", "Rate", "95% CI"))
  cat(strrep("-", 50), "\n")
  for (a in agent_counts) {
    subset <- pf[pf$agent_count == a, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    ci <- prop_ci(solved, total)
    cat(sprintf("%-10d %4d/%-4d %10.1f%% %6.1f%% - %5.1f%%\n",
                a, solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
  }

  # Final pressure analysis
  cat("\n\nFINAL PRESSURE BY STRATEGY (mean +/- SD):\n")
  cat(sprintf("%-20s %12s %12s %10s %10s\n", "Strategy", "Mean", "SD", "Min", "Max"))
  cat(strrep("-", 65), "\n")
  for (strat in sort(strategies)) {
    subset <- scaling[scaling$strategy == strat, ]
    p <- subset$final_pressure
    cat(sprintf("%-20s %12.2f %12.2f %10.2f %10.2f\n",
                strat, mean(p, na.rm=TRUE), sd(p, na.rm=TRUE),
                min(p, na.rm=TRUE), max(p, na.rm=TRUE)))
  }

  # Kruskal-Wallis test for final pressure
  cat("\n\nKRUSKAL-WALLIS TEST (Strategy effect on final pressure):\n")
  kw_test <- kruskal.test(final_pressure ~ strategy, data = scaling)
  cat(sprintf("Chi-square = %.2f, df = %d, p-value = %.2e\n",
              kw_test$statistic, kw_test$parameter, kw_test$p.value))

  # Chi-square test: strategy vs solved
  cat("\n\nCHI-SQUARE TEST (Strategy effect on solve rate):\n")
  contingency <- table(scaling$strategy, scaling$solved)
  print(contingency)
  chi_test <- chisq.test(contingency)
  cat(sprintf("\nChi-square = %.2f, df = %d, p-value = %.2e\n",
              chi_test$statistic, chi_test$parameter, chi_test$p.value))

  # Model escalation analysis (from tick_metrics if available)
  cat("\n\nMODEL ESCALATION EVENTS:\n")
  cat(strrep("-", 50), "\n")
  for (strat in sort(strategies)) {
    subset <- scaling[scaling$strategy == strat, ]
    # Count escalation events
    total_escalations <- sum(sapply(subset$escalation_events, function(x) {
      if (is.null(x) || length(x) == 0) return(0)
      if (is.data.frame(x)) return(nrow(x))
      return(length(x))
    }))
    cat(sprintf("%-20s: %d escalation events across %d trials\n",
                strat, total_escalations, nrow(subset)))
  }

  # Token usage analysis (new metrics)
  if ("total_prompt_tokens" %in% names(scaling)) {
    cat("\n\nTOKEN USAGE BY STRATEGY:\n")
    cat(sprintf("%-20s %15s %15s %15s\n", "Strategy", "Prompt Tokens", "Completion", "Total"))
    cat(strrep("-", 70), "\n")
    for (strat in sort(strategies)) {
      subset <- scaling[scaling$strategy == strat, ]
      prompt_t <- sum(subset$total_prompt_tokens, na.rm = TRUE)
      completion_t <- sum(subset$total_completion_tokens, na.rm = TRUE)
      cat(sprintf("%-20s %15d %15d %15d\n",
                  strat, prompt_t, completion_t, prompt_t + completion_t))
    }
  }
}

# ============================================================
# 3. STRATEGY COMPARISON (All 5 strategies from logs)
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("3. STRATEGY COMPARISON (All experiments, including partial)\n")
cat(strrep("=", 60), "\n\n")

# Parse log files from killed experiments
main_grid <- parse_log_results("main-grid.log")
escalation_log <- parse_log_results("escalation.log")
difficulty <- parse_log_results("difficulty.log")

# Report data availability
cat("DATA SOURCES:\n")
cat(strrep("-", 50), "\n")
if (!is.null(main_grid)) {
  cat(sprintf("Main-grid (7x7, 7 empty): %d trials from logs\n", nrow(main_grid)))
}
if (!is.null(scaling)) {
  cat(sprintf("Scaling (7x7, 8 empty):   %d trials from JSON (complete)\n", nrow(scaling)))
}
if (!is.null(escalation_log)) {
  cat(sprintf("Escalation (harder):      %d trials from logs\n", nrow(escalation_log)))
}
if (!is.null(difficulty)) {
  cat(sprintf("Difficulty (easier):      %d trials from logs\n", nrow(difficulty)))
}

# Combine all strategy data
all_strategies <- c("pressure_field", "hierarchical", "sequential", "random", "conversation")

# Function to summarize by strategy
summarize_experiment <- function(data, exp_name) {
  if (is.null(data) || nrow(data) == 0) return(NULL)

  results <- data.frame(
    experiment = character(),
    strategy = character(),
    solved = integer(),
    total = integer(),
    rate = numeric(),
    ci_lower = numeric(),
    ci_upper = numeric(),
    stringsAsFactors = FALSE
  )

  for (strat in all_strategies) {
    subset <- data[data$strategy == strat, ]
    if (nrow(subset) > 0) {
      solved <- sum(subset$solved)
      total <- nrow(subset)
      ci <- prop_ci(solved, total)
      results <- rbind(results, data.frame(
        experiment = exp_name,
        strategy = strat,
        solved = solved,
        total = total,
        rate = ci[1] * 100,
        ci_lower = ci[2] * 100,
        ci_upper = ci[3] * 100,
        stringsAsFactors = FALSE
      ))
    }
  }
  return(results)
}

# Summarize each experiment
results_list <- list()
if (!is.null(difficulty)) {
  results_list$difficulty <- summarize_experiment(difficulty, "Difficulty (easy)")
}
if (!is.null(main_grid)) {
  results_list$main_grid <- summarize_experiment(main_grid, "Main-grid (7x7/7)")
}
if (!is.null(scaling)) {
  # Convert scaling to same format
  scaling_simple <- data.frame(strategy = scaling$strategy, solved = scaling$solved)
  results_list$scaling <- summarize_experiment(scaling_simple, "Scaling (7x7/8)")
}
if (!is.null(escalation_log)) {
  results_list$escalation <- summarize_experiment(escalation_log, "Escalation (hard)")
}

# Combine all results
all_results <- do.call(rbind, results_list)

# Print comparison table
cat("\n\nSTRATEGY COMPARISON ACROSS ALL EXPERIMENTS:\n")
cat(strrep("=", 85), "\n")
cat(sprintf("%-22s | %-15s | %8s | %8s | %18s\n",
            "Experiment", "Strategy", "Solved/N", "Rate", "95% Wilson CI"))
cat(strrep("-", 85), "\n")

current_exp <- ""
for (i in seq_len(nrow(all_results))) {
  row <- all_results[i, ]
  exp_label <- ifelse(row$experiment != current_exp, row$experiment, "")
  current_exp <- row$experiment

  cat(sprintf("%-22s | %-15s | %4d/%-3d | %6.1f%% | [%5.1f%%, %5.1f%%]\n",
              exp_label, row$strategy, row$solved, row$total,
              row$rate, row$ci_lower, row$ci_upper))

  # Add separator between experiments
  if (i < nrow(all_results) && all_results[i+1, "experiment"] != current_exp) {
    cat(strrep("-", 85), "\n")
  }
}
cat(strrep("=", 85), "\n")

# Aggregate across all experiments for overall comparison
cat("\n\nAGGREGATE PERFORMANCE (All experiments combined):\n")
cat(strrep("-", 70), "\n")
cat(sprintf("%-15s | %10s | %8s | %18s\n",
            "Strategy", "Solved/N", "Rate", "95% Wilson CI"))
cat(strrep("-", 70), "\n")

for (strat in all_strategies) {
  strat_data <- all_results[all_results$strategy == strat, ]
  if (nrow(strat_data) > 0) {
    total_solved <- sum(strat_data$solved)
    total_n <- sum(strat_data$total)
    ci <- prop_ci(total_solved, total_n)
    cat(sprintf("%-15s | %5d/%-4d | %6.1f%% | [%5.1f%%, %5.1f%%]\n",
                strat, total_solved, total_n, ci[1]*100, ci[2]*100, ci[3]*100))
  }
}

# Chi-square test for strategy effect (using aggregate data)
cat("\n\nCHI-SQUARE TEST (Strategy effect on solve rate - all data):\n")
# Build contingency table
agg_data <- data.frame(
  strategy = character(),
  solved = integer(),
  failed = integer(),
  stringsAsFactors = FALSE
)
for (strat in all_strategies) {
  strat_data <- all_results[all_results$strategy == strat, ]
  if (nrow(strat_data) > 0) {
    total_solved <- sum(strat_data$solved)
    total_n <- sum(strat_data$total)
    agg_data <- rbind(agg_data, data.frame(
      strategy = strat,
      solved = total_solved,
      failed = total_n - total_solved,
      stringsAsFactors = FALSE
    ))
  }
}

contingency <- matrix(c(agg_data$solved, agg_data$failed), ncol = 2)
rownames(contingency) <- agg_data$strategy
colnames(contingency) <- c("Solved", "Failed")
print(contingency)

chi_test <- chisq.test(contingency)
cat(sprintf("\nChi-square = %.2f, df = %d, p-value = %.2e\n",
            chi_test$statistic, chi_test$parameter, chi_test$p.value))

# Pairwise comparisons (Fisher's exact test)
cat("\n\nPAIRWISE COMPARISONS (Fisher's exact test):\n")
cat(strrep("-", 60), "\n")
cat(sprintf("%-20s vs %-20s %10s\n", "Strategy A", "Strategy B", "p-value"))
cat(strrep("-", 60), "\n")

# Key comparisons
key_pairs <- list(
  c("pressure_field", "hierarchical"),
  c("pressure_field", "sequential"),
  c("pressure_field", "random"),
  c("pressure_field", "conversation"),
  c("hierarchical", "sequential"),
  c("hierarchical", "conversation")
)

for (pair in key_pairs) {
  strat_a <- pair[1]
  strat_b <- pair[2]

  data_a <- all_results[all_results$strategy == strat_a, ]
  data_b <- all_results[all_results$strategy == strat_b, ]

  if (nrow(data_a) > 0 && nrow(data_b) > 0) {
    solved_a <- sum(data_a$solved)
    total_a <- sum(data_a$total)
    solved_b <- sum(data_b$solved)
    total_b <- sum(data_b$total)

    cont_2x2 <- matrix(c(solved_a, total_a - solved_a,
                         solved_b, total_b - solved_b), nrow = 2, byrow = TRUE)
    fisher_result <- fisher.test(cont_2x2)

    sig <- ifelse(fisher_result$p.value < 0.001, "***",
                  ifelse(fisher_result$p.value < 0.01, "**",
                         ifelse(fisher_result$p.value < 0.05, "*", "")))

    cat(sprintf("%-20s vs %-20s %9.2e %s\n",
                strat_a, strat_b, fisher_result$p.value, sig))
  }
}
cat("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n")

# ============================================================
# 4. SUMMARY FOR PAPER
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("4. SUMMARY FOR PAPER\n")
cat(strrep("=", 60), "\n\n")

cat("DATA SUMMARY:\n")
cat("- Ablation study: 8 configurations x 30 trials = 240 runs (complete)\n")
cat("- Scaling: 2 strategies x 5 agent counts x 30 trials = 300 runs (complete)\n")
cat("- Main-grid: 5 strategies x ~120 trials = ~498 runs (partial, from logs)\n")
cat("- Escalation: 5 strategies x ~30 trials = ~140 runs (partial, from logs)\n")
cat("- Difficulty: 5 strategies x ~30 trials = ~140 runs (partial, from logs)\n")
cat("\nNOTE: Some experiments terminated early due to Conversation strategy runtime.\n")
cat("Partial results extracted from log files.\n")

if (!is.null(scaling)) {
  cat("\n\nKEY RESULTS FROM SCALING (7x7, 8 empty, with escalation):\n")
  cat(strrep("-", 50), "\n")

  pf <- scaling[scaling$strategy == "pressure_field", ]
  hi <- scaling[scaling$strategy == "hierarchical", ]

  pf_rate <- mean(pf$solved) * 100
  hi_rate <- mean(hi$solved) * 100

  pf_ci <- prop_ci(sum(pf$solved), nrow(pf))
  hi_ci <- prop_ci(sum(hi$solved), nrow(hi))

  cat(sprintf("Pressure-field: %.1f%% (95%% CI: %.1f%%-%.1f%%)\n",
              pf_rate, pf_ci[2]*100, pf_ci[3]*100))
  cat(sprintf("Hierarchical:   %.1f%% (95%% CI: %.1f%%-%.1f%%)\n",
              hi_rate, hi_ci[2]*100, hi_ci[3]*100))

  if (hi_rate > 0) {
    cat(sprintf("\nRelative improvement: %.1fx\n", pf_rate / hi_rate))
  } else {
    cat(sprintf("\nRelative improvement: Inf (hierarchical = 0%%)\n"))
  }
}

if (!is.null(ablation)) {
  cat("\n\nKEY RESULTS FROM ABLATION:\n")
  cat(strrep("-", 50), "\n")

  with_decay <- ablation[ablation$decay_enabled == TRUE, "final_pressure"]
  without_decay <- ablation[ablation$decay_enabled == FALSE, "final_pressure"]

  cat(sprintf("Decay effect: %.1fx pressure increase when disabled\n",
              mean(without_decay) / mean(with_decay)))
  cat(sprintf("  With decay:    %.2f +/- %.2f\n", mean(with_decay), sd(with_decay)))
  cat(sprintf("  Without decay: %.2f +/- %.2f\n", mean(without_decay), sd(without_decay)))
}

cat("\n\n", strrep("=", 70), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 70), "\n")
