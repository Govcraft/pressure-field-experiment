#!/usr/bin/env Rscript
# Statistical Analysis of Latin Square Experiment Results
# For publication: accuracy, confidence intervals, statistical tests
#
# Data sources:
# - Baselines (hierarchical, sequential, random, conversation): results/20260114-221958/
# - Pressure field (6 combined runs): results/pressure_field_combined/

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

# Directory configuration
baseline_dir <- "results/20260114-221958"
pf_dir <- "results/pressure_field_combined"

cat(strrep("=", 70), "\n")
cat("STATISTICAL ANALYSIS OF EXPERIMENT RESULTS\n")
cat(strrep("=", 70), "\n\n")

cat("Data sources:\n")
cat(sprintf("  Baselines:      %s\n", baseline_dir))
cat(sprintf("  Pressure field: %s (6 combined runs)\n\n", pf_dir))

# Helper function to load and flatten results from JSON
# Only keeps columns needed for analysis (drops complex nested structures)
load_results <- function(filename, dir = baseline_dir) {
  path <- file.path(dir, filename)
  if (!file.exists(path)) {
    cat("File not found:", path, "\n")
    return(NULL)
  }
  data <- fromJSON(path)
  results <- data$results

  # Flatten the config column if present
  if ("config" %in% names(results)) {
    config_df <- as.data.frame(results$config)
    results <- cbind(results[, !names(results) %in% "config"], config_df)
  }

  # Keep only columns needed for analysis (avoid nested data.frames that cause rbind issues)
  keep_cols <- c("solved", "final_pressure", "total_ticks", "final_model",
                 "total_prompt_tokens", "total_completion_tokens",
                 "strategy", "agent_count", "n", "empty_cells",
                 "decay_enabled", "inhibition_enabled", "examples_enabled",
                 "trial", "run_id", "source_dir")
  keep_cols <- intersect(keep_cols, names(results))
  results <- results[, keep_cols, drop = FALSE]

  # Reset row names
  rownames(results) <- NULL

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

# Function to safely combine dataframes with different columns
safe_rbind <- function(df1, df2) {
  # Get all unique column names
  all_cols <- union(names(df1), names(df2))

  # Add missing columns to df1
  for (col in setdiff(all_cols, names(df1))) {
    df1[[col]] <- NA
  }

  # Add missing columns to df2
  for (col in setdiff(all_cols, names(df2))) {
    df2[[col]] <- NA
  }

  # Reset row names BEFORE column reorder to avoid duplicate issues
  rownames(df1) <- paste0("a", seq_len(nrow(df1)))
  rownames(df2) <- paste0("b", seq_len(nrow(df2)))

  # Reorder columns to match
  df1 <- df1[, all_cols]
  df2 <- df2[, all_cols]

  # Combine and reset row names
  result <- rbind(df1, df2)
  rownames(result) <- NULL
  result
}

# ============================================================
# 1. MAIN GRID ANALYSIS (5 strategies comparison)
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("1. MAIN GRID ANALYSIS (7x7, 7 empty cells)\n")
cat(strrep("=", 60), "\n\n")

# Load baselines
main_grid_baselines <- load_results("main-grid-20260115-024718.json", baseline_dir)
# Load pressure_field
main_grid_pf <- load_results("main-grid.json", pf_dir)

if (!is.null(main_grid_baselines) && !is.null(main_grid_pf)) {
  # Combine (using safe_rbind to handle column differences)
  main_grid <- safe_rbind(main_grid_baselines, main_grid_pf)

  cat("Total observations:", nrow(main_grid), "\n")
  cat(sprintf("  Baselines:      %d results (4 strategies)\n", nrow(main_grid_baselines)))
  cat(sprintf("  Pressure field: %d results (6 runs combined)\n\n", nrow(main_grid_pf)))

  # Summary by strategy
  cat("SOLVE RATES BY STRATEGY (with 95% Wilson CI):\n")
  cat(sprintf("%-20s %10s %12s %20s\n", "Strategy", "Solved/N", "Rate", "95% CI"))
  cat(strrep("-", 65), "\n")

  strategies <- unique(main_grid$strategy)
  for (strat in sort(strategies)) {
    subset <- main_grid[main_grid$strategy == strat, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    ci <- prop_ci(solved, total)
    cat(sprintf("%-20s %5d/%-4d %10.1f%% %8.1f%% - %5.1f%%\n",
                strat, solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
  }

  # By strategy and agent count
  cat("\n\nSOLVE RATES BY STRATEGY AND AGENT COUNT:\n")
  cat(sprintf("%-18s", "Strategy"))
  agent_counts <- sort(unique(main_grid$agent_count))
  for (a in agent_counts) cat(sprintf("%7d", a))
  cat("    AVG\n")
  cat(strrep("-", 18 + 8 * length(agent_counts) + 8), "\n")

  for (strat in sort(strategies)) {
    cat(sprintf("%-18s", strat))
    rates <- c()
    for (a in agent_counts) {
      subset <- main_grid[main_grid$strategy == strat & main_grid$agent_count == a, ]
      if (nrow(subset) > 0) {
        rate <- mean(subset$solved, na.rm = TRUE) * 100
        rates <- c(rates, rate)
        cat(sprintf("%6.1f%%", rate))
      } else {
        cat(sprintf("%7s", "N/A"))
      }
    }
    if (length(rates) > 0) {
      cat(sprintf("  %5.1f%%\n", mean(rates)))
    } else {
      cat("\n")
    }
  }

  # Final pressure analysis
  cat("\n\nFINAL PRESSURE BY STRATEGY (mean +/- SD):\n")
  cat(sprintf("%-20s %12s %12s %10s %10s\n", "Strategy", "Mean", "SD", "Min", "Max"))
  cat(strrep("-", 65), "\n")
  for (strat in sort(strategies)) {
    subset <- main_grid[main_grid$strategy == strat, ]
    p <- subset$final_pressure
    cat(sprintf("%-20s %12.2f %12.2f %10.2f %10.2f\n",
                strat, mean(p, na.rm=TRUE), sd(p, na.rm=TRUE),
                min(p, na.rm=TRUE), max(p, na.rm=TRUE)))
  }

  # Chi-square test: strategy vs solved
  cat("\n\nCHI-SQUARE TEST (Strategy effect on solve rate):\n")
  contingency <- table(main_grid$strategy, main_grid$solved)
  print(contingency)
  chi_test <- chisq.test(contingency)
  cat(sprintf("\nChi-square = %.2f, df = %d, p-value = %.2e\n",
              chi_test$statistic, chi_test$parameter, chi_test$p.value))
}

# ============================================================
# 2. SCALING ANALYSIS (Agent count scaling)
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("2. SCALING ANALYSIS (7x7, variable empty cells)\n")
cat(strrep("=", 60), "\n\n")

# Load baselines (only has hierarchical from original run)
scaling_baselines <- load_results("scaling-20260114-231013.json", baseline_dir)
# Load pressure_field
scaling_pf <- load_results("scaling.json", pf_dir)

if (!is.null(scaling_pf)) {
  # Note: scaling baselines only has hierarchical, so we primarily analyze pressure_field
  cat("Pressure field scaling analysis (6 combined runs):\n")
  cat(sprintf("  Total observations: %d\n\n", nrow(scaling_pf)))

  # Summary by agent count
  cat("PRESSURE_FIELD SOLVE RATES BY AGENT COUNT:\n")
  cat(sprintf("%-10s %10s %12s %20s\n", "Agents", "Solved/N", "Rate", "95% CI"))
  cat(strrep("-", 55), "\n")

  agent_counts <- sort(unique(scaling_pf$agent_count))
  for (a in agent_counts) {
    subset <- scaling_pf[scaling_pf$agent_count == a, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    ci <- prop_ci(solved, total)
    cat(sprintf("%-10d %5d/%-4d %10.1f%% %8.1f%% - %5.1f%%\n",
                a, solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
  }

  # Run-to-run variance analysis
  cat("\n\nRUN-TO-RUN VARIANCE (pressure_field solve rate):\n")
  cat(sprintf("%-10s", "Run"))
  for (a in agent_counts) cat(sprintf("%8d", a))
  cat("   Overall\n")
  cat(strrep("-", 10 + 8 * length(agent_counts) + 10), "\n")

  runs <- sort(unique(scaling_pf$run_id))
  for (run in runs) {
    cat(sprintf("%-10d", run))
    overall_solved <- 0
    overall_total <- 0
    for (a in agent_counts) {
      subset <- scaling_pf[scaling_pf$run_id == run & scaling_pf$agent_count == a, ]
      if (nrow(subset) > 0) {
        rate <- mean(subset$solved, na.rm = TRUE) * 100
        overall_solved <- overall_solved + sum(subset$solved, na.rm = TRUE)
        overall_total <- overall_total + nrow(subset)
        cat(sprintf("%7.1f%%", rate))
      } else {
        cat(sprintf("%8s", "N/A"))
      }
    }
    overall_rate <- overall_solved / overall_total * 100
    cat(sprintf("  %6.1f%%\n", overall_rate))
  }

  # Compute mean and SD across runs
  cat(strrep("-", 10 + 8 * length(agent_counts) + 10), "\n")
  cat(sprintf("%-10s", "Mean"))
  for (a in agent_counts) {
    run_rates <- sapply(runs, function(r) {
      subset <- scaling_pf[scaling_pf$run_id == r & scaling_pf$agent_count == a, ]
      if (nrow(subset) > 0) mean(subset$solved, na.rm = TRUE) * 100 else NA
    })
    cat(sprintf("%7.1f%%", mean(run_rates, na.rm = TRUE)))
  }
  cat("\n")
  cat(sprintf("%-10s", "SD"))
  for (a in agent_counts) {
    run_rates <- sapply(runs, function(r) {
      subset <- scaling_pf[scaling_pf$run_id == r & scaling_pf$agent_count == a, ]
      if (nrow(subset) > 0) mean(subset$solved, na.rm = TRUE) * 100 else NA
    })
    cat(sprintf("%7.1f%%", sd(run_rates, na.rm = TRUE)))
  }
  cat("\n")
}

# ============================================================
# 3. DIFFICULTY COMPARISON
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("3. DIFFICULTY COMPARISON (Easy vs Hard)\n")
cat(strrep("=", 60), "\n\n")

# Load baselines
diff_easy_baselines <- load_results("difficulty-easy-20260115-012838.json", baseline_dir)
# Load pressure_field
diff_easy_pf <- load_results("difficulty-easy.json", pf_dir)
diff_hard_pf <- load_results("difficulty-hard.json", pf_dir)

if (!is.null(diff_easy_baselines) && !is.null(diff_easy_pf)) {
  diff_easy <- safe_rbind(diff_easy_baselines, diff_easy_pf)

  cat("EASY DIFFICULTY (5x5, 5 empty cells):\n")
  cat(sprintf("  Baselines:      %d results\n", nrow(diff_easy_baselines)))
  cat(sprintf("  Pressure field: %d results\n\n", nrow(diff_easy_pf)))

  cat(sprintf("%-20s %10s %12s %20s\n", "Strategy", "Solved/N", "Rate", "95% CI"))
  cat(strrep("-", 65), "\n")

  strategies <- unique(diff_easy$strategy)
  for (strat in sort(strategies)) {
    subset <- diff_easy[diff_easy$strategy == strat, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    ci <- prop_ci(solved, total)
    cat(sprintf("%-20s %5d/%-4d %10.1f%% %8.1f%% - %5.1f%%\n",
                strat, solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
  }
}

if (!is.null(diff_hard_pf)) {
  cat("\n\nHARD DIFFICULTY (9x9, 12 empty cells) - pressure_field only:\n")
  cat(sprintf("  Pressure field: %d results (6 runs)\n\n", nrow(diff_hard_pf)))

  solved <- sum(diff_hard_pf$solved, na.rm = TRUE)
  total <- nrow(diff_hard_pf)
  ci <- prop_ci(solved, total)
  cat(sprintf("%-20s %5d/%-4d %10.1f%% %8.1f%% - %5.1f%%\n",
              "pressure_field", solved, total, ci[1]*100, ci[2]*100, ci[3]*100))
}

# ============================================================
# 4. ESCALATION ANALYSIS
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("4. MODEL ESCALATION ANALYSIS\n")
cat(strrep("=", 60), "\n\n")

# Load baselines
esc_with_baselines <- load_results("escalation-with-20260115-002328.json", baseline_dir)
esc_without_baselines <- load_results("escalation-without-20260114-224448.json", baseline_dir)
# Load pressure_field
esc_with_pf <- load_results("escalation-with.json", pf_dir)
esc_without_pf <- load_results("escalation-without.json", pf_dir)

if (!is.null(esc_with_baselines) && !is.null(esc_with_pf) &&
    !is.null(esc_without_baselines) && !is.null(esc_without_pf)) {

  esc_with <- safe_rbind(esc_with_baselines, esc_with_pf)
  esc_without <- safe_rbind(esc_without_baselines, esc_without_pf)

  cat("WITH ESCALATION (can upgrade models):\n")
  cat(sprintf("%-20s %10s %12s\n", "Strategy", "Solved/N", "Rate"))
  cat(strrep("-", 45), "\n")

  for (strat in sort(unique(esc_with$strategy))) {
    subset <- esc_with[esc_with$strategy == strat, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    cat(sprintf("%-20s %5d/%-4d %10.1f%%\n", strat, solved, total, solved/total*100))
  }

  cat("\n\nWITHOUT ESCALATION (single model only):\n")
  cat(sprintf("%-20s %10s %12s\n", "Strategy", "Solved/N", "Rate"))
  cat(strrep("-", 45), "\n")

  for (strat in sort(unique(esc_without$strategy))) {
    subset <- esc_without[esc_without$strategy == strat, ]
    solved <- sum(subset$solved, na.rm = TRUE)
    total <- nrow(subset)
    cat(sprintf("%-20s %5d/%-4d %10.1f%%\n", strat, solved, total, solved/total*100))
  }

  # Escalation benefit analysis for pressure_field
  cat("\n\nESCALATION BENEFIT (pressure_field):\n")
  pf_with <- esc_with[esc_with$strategy == "pressure_field", ]
  pf_without <- esc_without[esc_without$strategy == "pressure_field", ]

  with_rate <- mean(pf_with$solved, na.rm = TRUE) * 100
  without_rate <- mean(pf_without$solved, na.rm = TRUE) * 100

  cat(sprintf("  With escalation:    %.1f%% (n=%d)\n", with_rate, nrow(pf_with)))
  cat(sprintf("  Without escalation: %.1f%% (n=%d)\n", without_rate, nrow(pf_without)))
  cat(sprintf("  Improvement:        +%.1f percentage points\n", with_rate - without_rate))
}

# ============================================================
# 5. AGGREGATE STATISTICS & TESTS
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("5. AGGREGATE STATISTICS & STATISTICAL TESTS\n")
cat(strrep("=", 60), "\n\n")

# Combine main-grid data for comprehensive comparison
if (exists("main_grid") && !is.null(main_grid)) {
  all_strategies <- c("pressure_field", "hierarchical", "sequential", "random", "conversation")

  cat("OVERALL PERFORMANCE (Main Grid, all strategies):\n")
  cat(strrep("-", 70), "\n")
  cat(sprintf("%-15s | %10s | %8s | %18s\n",
              "Strategy", "Solved/N", "Rate", "95% Wilson CI"))
  cat(strrep("-", 70), "\n")

  for (strat in all_strategies) {
    strat_data <- main_grid[main_grid$strategy == strat, ]
    if (nrow(strat_data) > 0) {
      total_solved <- sum(strat_data$solved, na.rm = TRUE)
      total_n <- nrow(strat_data)
      ci <- prop_ci(total_solved, total_n)
      cat(sprintf("%-15s | %5d/%-4d | %6.1f%% | [%5.1f%%, %5.1f%%]\n",
                  strat, total_solved, total_n, ci[1]*100, ci[2]*100, ci[3]*100))
    }
  }

  # Pairwise comparisons (Fisher's exact test)
  cat("\n\nPAIRWISE COMPARISONS (Fisher's exact test):\n")
  cat(strrep("-", 60), "\n")
  cat(sprintf("%-20s vs %-20s %10s\n", "Strategy A", "Strategy B", "p-value"))
  cat(strrep("-", 60), "\n")

  key_pairs <- list(
    c("pressure_field", "hierarchical"),
    c("pressure_field", "sequential"),
    c("pressure_field", "random"),
    c("pressure_field", "conversation"),
    c("hierarchical", "sequential")
  )

  for (pair in key_pairs) {
    strat_a <- pair[1]
    strat_b <- pair[2]

    data_a <- main_grid[main_grid$strategy == strat_a, ]
    data_b <- main_grid[main_grid$strategy == strat_b, ]

    if (nrow(data_a) > 0 && nrow(data_b) > 0) {
      solved_a <- sum(data_a$solved, na.rm = TRUE)
      total_a <- nrow(data_a)
      solved_b <- sum(data_b$solved, na.rm = TRUE)
      total_b <- nrow(data_b)

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
}

# ============================================================
# 6. SUMMARY FOR PAPER
# ============================================================
cat("\n\n", strrep("=", 60), "\n")
cat("6. SUMMARY FOR PAPER\n")
cat(strrep("=", 60), "\n\n")

cat("DATA SUMMARY:\n")
cat("- Baselines: 4 strategies from original full experiment\n")
cat("- Pressure field: 6 independent runs (combined for increased power)\n")
cat(sprintf("- Main grid: %d total observations\n",
            ifelse(exists("main_grid"), nrow(main_grid), 0)))
cat(sprintf("- Scaling: %d pressure_field observations\n",
            ifelse(exists("scaling_pf") && !is.null(scaling_pf), nrow(scaling_pf), 0)))

if (exists("main_grid") && !is.null(main_grid)) {
  cat("\n\nKEY RESULTS (Main Grid):\n")
  cat(strrep("-", 50), "\n")

  pf <- main_grid[main_grid$strategy == "pressure_field", ]
  hi <- main_grid[main_grid$strategy == "hierarchical", ]

  pf_rate <- mean(pf$solved, na.rm = TRUE) * 100
  hi_rate <- mean(hi$solved, na.rm = TRUE) * 100

  pf_ci <- prop_ci(sum(pf$solved, na.rm = TRUE), nrow(pf))
  hi_ci <- prop_ci(sum(hi$solved, na.rm = TRUE), nrow(hi))

  cat(sprintf("Pressure-field: %.1f%% (95%% CI: %.1f%%-%.1f%%, n=%d)\n",
              pf_rate, pf_ci[2]*100, pf_ci[3]*100, nrow(pf)))
  cat(sprintf("Hierarchical:   %.1f%% (95%% CI: %.1f%%-%.1f%%, n=%d)\n",
              hi_rate, hi_ci[2]*100, hi_ci[3]*100, nrow(hi)))

  if (hi_rate > 0) {
    cat(sprintf("\nRelative improvement: %.1fx\n", pf_rate / hi_rate))
  } else {
    cat(sprintf("\nRelative improvement: Inf (hierarchical = 0%%)\n"))
  }
}

cat("\n\n", strrep("=", 70), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 70), "\n")
