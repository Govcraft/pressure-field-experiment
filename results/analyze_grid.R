#!/usr/bin/env Rscript

# Grid Experiment Analysis - All Strategies Comparison
# =====================================================
# Compares pressure_field, conversation, sequential, random, hierarchical
# across difficulties and agent configurations.

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

# Wilson score interval function
wilson_ci <- function(successes, total, z = 1.96) {
  if (total == 0) return(c(0, 0))
  p <- successes / total
  denom <- 1 + z^2/total
  center <- (p + z^2/(2*total)) / denom
  spread <- z * sqrt((p*(1-p) + z^2/(4*total)) / total) / denom
  return(c(max(0, center - spread), min(1, center + spread)))
}

# Cohen's h for proportions
cohens_h <- function(p1, p2) {
  phi1 <- 2 * asin(sqrt(p1))
  phi2 <- 2 * asin(sqrt(p2))
  return(phi1 - phi2)
}

# Load all strategy results
cat("Loading experiment results...\n")

files <- list(
  pressure_field = "pressure-field-results.json",
  conversation = "conversation-results.json",
  sequential = "sequential-results.json",
  random = "random-results.json",
  hierarchical = "hierarchical-results.json"
)

all_results <- list()
for (name in names(files)) {
  if (file.exists(files[[name]])) {
    data <- fromJSON(files[[name]])
    results <- data$results

    df <- data.frame(
      strategy = name,
      agent_count = results$config$agent_count,
      empty_cells = results$config$empty_cells,
      trial = results$config$trial,
      solved = results$solved,
      total_ticks = results$total_ticks,
      final_pressure = results$final_pressure,
      stringsAsFactors = FALSE
    )

    # Add token data if available
    if ("total_prompt_tokens" %in% names(results)) {
      df$prompt_tokens <- results$total_prompt_tokens
      df$completion_tokens <- results$total_completion_tokens
    }

    all_results[[name]] <- df
    cat(sprintf("  Loaded %s: %d trials\n", name, nrow(df)))
  } else {
    cat(sprintf("  MISSING: %s\n", files[[name]]))
  }
}

# Combine all results
df <- do.call(rbind, all_results)
rownames(df) <- NULL

# Map empty_cells to difficulty
df$difficulty <- factor(
  ifelse(df$empty_cells == 6, "easy",
         ifelse(df$empty_cells == 20, "medium", "hard")),
  levels = c("easy", "medium", "hard")
)

df$agent_count <- factor(df$agent_count)
df$strategy <- factor(df$strategy,
                      levels = c("pressure_field", "conversation", "hierarchical", "sequential", "random"))

cat("\n")
cat(paste(rep("=", 80), collapse=""), "\n")
cat("GRID EXPERIMENT ANALYSIS - ALL STRATEGIES\n")
cat(paste(rep("=", 80), collapse=""), "\n")

# ============================================================
# 1. OVERALL SUMMARY
# ============================================================
cat("\n### 1. Overall Summary ###\n\n")
cat(sprintf("Total experiments: %d\n", nrow(df)))
cat(sprintf("Strategies: %s\n", paste(levels(df$strategy), collapse=", ")))
cat(sprintf("Difficulties: %s\n", paste(levels(df$difficulty), collapse=", ")))
cat(sprintf("Agent counts: %s\n", paste(levels(df$agent_count), collapse=", ")))

# ============================================================
# 2. SOLVE RATES BY STRATEGY
# ============================================================
cat("\n### 2. Solve Rates by Strategy ###\n\n")

cat(sprintf("%-20s %8s %8s %8s %20s\n", "Strategy", "Solved", "Total", "Rate", "95% Wilson CI"))
cat(paste(rep("-", 70), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  subset_df <- df[df$strategy == strat, ]
  solved <- sum(subset_df$solved)
  total <- nrow(subset_df)
  rate <- solved / total
  ci <- wilson_ci(solved, total)
  cat(sprintf("%-20s %8d %8d %7.1f%% [%5.1f%%, %5.1f%%]\n",
              strat, solved, total, rate * 100, ci[1] * 100, ci[2] * 100))
}

# ============================================================
# 3. SOLVE RATES BY STRATEGY × DIFFICULTY
# ============================================================
cat("\n### 3. Solve Rates: Strategy × Difficulty ###\n\n")

cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "Easy", "Medium", "Hard"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  rates <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    if (nrow(subset_df) > 0) {
      sprintf("%.1f%% (%d/%d)",
              mean(subset_df$solved) * 100,
              sum(subset_df$solved),
              nrow(subset_df))
    } else {
      "N/A"
    }
  })
  cat(sprintf("%-20s %12s %12s %12s\n", strat, rates[1], rates[2], rates[3]))
}

# ============================================================
# 4. SOLVE RATES BY STRATEGY × AGENTS (Easy Only)
# ============================================================
cat("\n### 4. Solve Rates: Strategy × Agents (Easy Difficulty) ###\n\n")

cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "1 Agent", "2 Agents", "4 Agents"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  rates <- sapply(levels(df$agent_count), function(a) {
    subset_df <- df[df$strategy == strat & df$difficulty == "easy" & df$agent_count == a, ]
    if (nrow(subset_df) > 0) {
      sprintf("%.1f%%", mean(subset_df$solved) * 100)
    } else {
      "N/A"
    }
  })
  cat(sprintf("%-20s %12s %12s %12s\n", strat, rates[1], rates[2], rates[3]))
}

# ============================================================
# 5. EFFECT SIZES (Cohen's h) - pressure_field vs others
# ============================================================
cat("\n### 5. Effect Sizes (Cohen's h): pressure_field vs Others ###\n")
cat("By difficulty level\n\n")

pf_rates <- sapply(levels(df$difficulty), function(d) {
  subset_df <- df[df$strategy == "pressure_field" & df$difficulty == d, ]
  sum(subset_df$solved) / nrow(subset_df)
})

cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "Easy", "Medium", "Hard"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  if (strat == "pressure_field") {
    cat(sprintf("%-20s %12s %12s %12s\n", strat, "—", "—", "—"))
    next
  }

  h_values <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    if (nrow(subset_df) > 0) {
      other_rate <- sum(subset_df$solved) / nrow(subset_df)
      pf_rate <- pf_rates[d]
      h <- cohens_h(pf_rate, other_rate)
      sprintf("%+.2f", h)
    } else {
      "N/A"
    }
  })
  cat(sprintf("%-20s %12s %12s %12s\n", strat, h_values[1], h_values[2], h_values[3]))
}

cat("\nPositive h = pressure_field better; |h| > 0.8 = large effect\n")

# ============================================================
# 6. STATISTICAL TESTS
# ============================================================
cat("\n### 6. Statistical Tests (Fisher's Exact) ###\n")
cat("pressure_field vs each strategy, by difficulty\n\n")

cat(sprintf("%-20s %15s %15s %15s\n", "Strategy", "Easy (p)", "Medium (p)", "Hard (p)"))
cat(paste(rep("-", 70), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  if (strat == "pressure_field") next

  p_values <- sapply(levels(df$difficulty), function(d) {
    pf_df <- df[df$strategy == "pressure_field" & df$difficulty == d, ]
    other_df <- df[df$strategy == strat & df$difficulty == d, ]

    if (nrow(pf_df) == 0 || nrow(other_df) == 0) return("N/A")

    contingency <- matrix(c(
      sum(pf_df$solved), nrow(pf_df) - sum(pf_df$solved),
      sum(other_df$solved), nrow(other_df) - sum(other_df$solved)
    ), nrow = 2, byrow = TRUE)

    test <- fisher.test(contingency)
    sig <- ifelse(test$p.value < 0.001, "***",
                  ifelse(test$p.value < 0.01, "**",
                         ifelse(test$p.value < 0.05, "*", "")))
    sprintf("%.4f%s", test$p.value, sig)
  })
  cat(sprintf("%-20s %15s %15s %15s\n", strat, p_values[1], p_values[2], p_values[3]))
}

cat("\n* p<0.05, ** p<0.01, *** p<0.001\n")

# ============================================================
# 7. TICKS TO SOLUTION
# ============================================================
cat("\n### 7. Average Ticks to Solution (Solved Only) ###\n\n")

cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "Easy", "Medium", "Hard"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  ticks <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$strategy == strat & df$difficulty == d & df$solved, ]
    if (nrow(subset_df) > 0) {
      sprintf("%.1f", mean(subset_df$total_ticks))
    } else {
      "N/A"
    }
  })
  cat(sprintf("%-20s %12s %12s %12s\n", strat, ticks[1], ticks[2], ticks[3]))
}

# ============================================================
# 8. TOKEN USAGE COMPARISON
# ============================================================
cat("\n### 8. Average Token Usage per Trial ###\n\n")

if ("prompt_tokens" %in% names(df)) {
  cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "Prompt", "Completion", "Total"))
  cat(paste(rep("-", 60), collapse=""), "\n")

  for (strat in levels(df$strategy)) {
    subset_df <- df[df$strategy == strat, ]
    if (nrow(subset_df) > 0 && !all(is.na(subset_df$prompt_tokens))) {
      cat(sprintf("%-20s %12s %12s %12s\n",
                  strat,
                  format(round(mean(subset_df$prompt_tokens, na.rm=TRUE)), big.mark=","),
                  format(round(mean(subset_df$completion_tokens, na.rm=TRUE)), big.mark=","),
                  format(round(mean(subset_df$prompt_tokens + subset_df$completion_tokens, na.rm=TRUE)), big.mark=",")))
    }
  }
} else {
  cat("Token usage data not available.\n")
}

# ============================================================
# 9. PRESSURE REDUCTION
# ============================================================
cat("\n### 9. Final Pressure (Lower = Better) ###\n\n")

cat(sprintf("%-20s %12s %12s %12s\n", "Strategy", "Easy", "Medium", "Hard"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (strat in levels(df$strategy)) {
  pressures <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    if (nrow(subset_df) > 0) {
      sprintf("%.2f", mean(subset_df$final_pressure))
    } else {
      "N/A"
    }
  })
  cat(sprintf("%-20s %12s %12s %12s\n", strat, pressures[1], pressures[2], pressures[3]))
}

# ============================================================
# 10. SUMMARY TABLE FOR PAPER
# ============================================================
cat("\n### 10. Summary Table (Paper Ready) ###\n\n")

cat("| Strategy | Easy | Medium | Hard | Overall |\n")
cat("|----------|------|--------|------|--------|\n")

for (strat in levels(df$strategy)) {
  rates <- sapply(c(levels(df$difficulty), "overall"), function(d) {
    if (d == "overall") {
      subset_df <- df[df$strategy == strat, ]
    } else {
      subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    }
    if (nrow(subset_df) > 0) {
      sprintf("%.1f%%", mean(subset_df$solved) * 100)
    } else {
      "—"
    }
  })
  cat(sprintf("| %s | %s | %s | %s | %s |\n", strat, rates[1], rates[2], rates[3], rates[4]))
}

# ============================================================
# 11. KEY FINDINGS SUMMARY
# ============================================================
cat("\n### 11. Key Findings ###\n\n")

# Best strategy per difficulty
for (d in levels(df$difficulty)) {
  best_rate <- 0
  best_strat <- ""
  for (strat in levels(df$strategy)) {
    subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    if (nrow(subset_df) > 0) {
      rate <- mean(subset_df$solved)
      if (rate > best_rate) {
        best_rate <- rate
        best_strat <- strat
      }
    }
  }
  cat(sprintf("Best on %s: %s (%.1f%%)\n", d, best_strat, best_rate * 100))
}

# Overall winner
overall_rates <- sapply(levels(df$strategy), function(strat) {
  subset_df <- df[df$strategy == strat, ]
  mean(subset_df$solved)
})
best_overall <- names(which.max(overall_rates))
cat(sprintf("\nOverall best: %s (%.1f%%)\n", best_overall, max(overall_rates) * 100))

# Largest effect size
cat("\nLargest improvements (pressure_field vs baseline strategies):\n")
for (d in levels(df$difficulty)) {
  pf_rate <- pf_rates[d]

  # Compare to worst baseline
  worst_rate <- 1
  worst_strat <- ""
  for (strat in c("sequential", "random", "hierarchical")) {
    subset_df <- df[df$strategy == strat & df$difficulty == d, ]
    if (nrow(subset_df) > 0) {
      rate <- mean(subset_df$solved)
      if (rate < worst_rate) {
        worst_rate <- rate
        worst_strat <- strat
      }
    }
  }

  if (worst_strat != "") {
    h <- cohens_h(pf_rate, worst_rate)
    cat(sprintf("  %s: pressure_field vs %s = %.1f%% vs %.1f%% (h=%+.2f)\n",
                d, worst_strat, pf_rate * 100, worst_rate * 100, h))
  }
}

cat("\n")
cat(paste(rep("=", 80), collapse=""), "\n")
cat("Analysis complete.\n")
