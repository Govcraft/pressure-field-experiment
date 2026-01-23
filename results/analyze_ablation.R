#!/usr/bin/env Rscript

# Ablation Study Analysis
# =======================
# Analyzes the contribution of decay, inhibition, and examples
# to the pressure field coordination mechanism.

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

# Load the data
cat("Loading ablation results...\n")
data <- fromJSON("schedule-ablation.json")
results <- data$results

# Extract key fields into a data frame
# jsonlite flattens nested objects, so config fields are in results$config as a data frame
config_df <- results$config

# Build config names from flags
get_config_name <- function(decay, inhibition, examples) {
  if (decay && inhibition && examples) return("full")
  if (!decay && inhibition && examples) return("no_decay")
  if (decay && !inhibition && examples) return("no_inhibition")
  if (decay && inhibition && !examples) return("no_examples")
  if (!decay && !inhibition && examples) return("no_decay_no_inhibition")
  if (!decay && inhibition && !examples) return("no_decay_no_examples")
  if (decay && !inhibition && !examples) return("no_inhibition_no_examples")
  if (!decay && !inhibition && !examples) return("baseline")
  return("unknown")
}

df <- data.frame(
  config = mapply(get_config_name,
                  config_df$decay_enabled,
                  config_df$inhibition_enabled,
                  config_df$examples_enabled),
  decay_enabled = config_df$decay_enabled,
  inhibition_enabled = config_df$inhibition_enabled,
  examples_enabled = config_df$examples_enabled,
  trial = config_df$trial,
  seed = config_df$seed,
  solved = results$solved,
  total_ticks = results$total_ticks,
  final_pressure = results$final_pressure
)

# Order configs logically
config_order <- c("full", "no_decay", "no_inhibition", "no_examples",
                  "no_decay_no_inhibition", "no_decay_no_examples",
                  "no_inhibition_no_examples", "baseline")
df$config <- factor(df$config, levels = config_order)

cat("\n")
cat(paste(rep("=", 70), collapse=""), "\n")
cat("ABLATION STUDY ANALYSIS\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# ============================================================
# 1. SOLVE RATES BY CONFIG
# ============================================================
cat("\n### 1. Solve Rates by Configuration ###\n\n")

# Wilson score interval function
wilson_ci <- function(successes, total, z = 1.96) {
  if (total == 0) return(c(0, 0))
  p <- successes / total
  denom <- 1 + z^2/total
  center <- (p + z^2/(2*total)) / denom
  spread <- z * sqrt((p*(1-p) + z^2/(4*total)) / total) / denom
  return(c(max(0, center - spread), min(1, center + spread)))
}

cat(sprintf("%-28s %8s %8s %8s %20s\n", "Config", "Solved", "Total", "Rate", "95% Wilson CI"))
cat(paste(rep("-", 70), collapse=""), "\n")

for (cfg in config_order) {
  subset_df <- df[df$config == cfg, ]
  if (nrow(subset_df) > 0) {
    solved <- sum(subset_df$solved)
    total <- nrow(subset_df)
    rate <- solved / total
    ci <- wilson_ci(solved, total)
    cat(sprintf("%-28s %8d %8d %7.1f%% [%5.1f%%, %5.1f%%]\n",
                cfg, solved, total, rate * 100, ci[1] * 100, ci[2] * 100))
  }
}

# ============================================================
# 2. EFFECT SIZES (Cohen's h)
# ============================================================
cat("\n### 2. Effect Sizes (Cohen's h vs Full) ###\n")
cat("Interpretation: |h| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large\n\n")

cohens_h <- function(p1, p2) {
  phi1 <- 2 * asin(sqrt(p1))
  phi2 <- 2 * asin(sqrt(p2))
  return(phi1 - phi2)
}

interpret_h <- function(h) {
  abs_h <- abs(h)
  if (abs_h < 0.2) return("negligible")
  if (abs_h < 0.5) return("small")
  if (abs_h < 0.8) return("medium")
  return("large")
}

full_df <- df[df$config == "full", ]
full_rate <- sum(full_df$solved) / nrow(full_df)

cat(sprintf("%-28s %10s %15s\n", "Config", "Cohen's h", "Interpretation"))
cat(paste(rep("-", 55), collapse=""), "\n")

for (cfg in config_order) {
  if (cfg == "full") next
  subset_df <- df[df$config == cfg, ]
  if (nrow(subset_df) > 0) {
    rate <- sum(subset_df$solved) / nrow(subset_df)
    h <- cohens_h(full_rate, rate)
    cat(sprintf("%-28s %+10.3f %15s\n", cfg, h, interpret_h(h)))
  }
}

# ============================================================
# 3. FEATURE CONTRIBUTION ANALYSIS
# ============================================================
cat("\n### 3. Individual Feature Contributions ###\n\n")

get_rate <- function(cfg) {
  subset_df <- df[df$config == cfg, ]
  if (nrow(subset_df) == 0) return(NA)
  return(sum(subset_df$solved) / nrow(subset_df))
}

decay_contrib <- get_rate("full") - get_rate("no_decay")
inhibition_contrib <- get_rate("full") - get_rate("no_inhibition")
examples_contrib <- get_rate("full") - get_rate("no_examples")

cat(sprintf("Decay contribution:      %+6.1f%% (full %.1f%% vs no_decay %.1f%%)\n",
            decay_contrib * 100, get_rate("full") * 100, get_rate("no_decay") * 100))
cat(sprintf("Inhibition contribution: %+6.1f%% (full %.1f%% vs no_inhibition %.1f%%)\n",
            inhibition_contrib * 100, get_rate("full") * 100, get_rate("no_inhibition") * 100))
cat(sprintf("Examples contribution:   %+6.1f%% (full %.1f%% vs no_examples %.1f%%)\n",
            examples_contrib * 100, get_rate("full") * 100, get_rate("no_examples") * 100))

# ============================================================
# 4. INTERACTION EFFECTS
# ============================================================
cat("\n### 4. Interaction Effects ###\n")
cat("Testing whether features compensate for each other\n\n")

# Expected vs actual for two-feature removal
# If features are independent: P(no_A_no_B) ≈ P(full) - contrib_A - contrib_B
# If synergistic: actual < expected (worse than sum of parts)
# If redundant: actual > expected (better than sum of parts)

pairs <- list(
  c("no_decay_no_inhibition", "no_decay", "no_inhibition", "decay+inhibition"),
  c("no_decay_no_examples", "no_decay", "no_examples", "decay+examples"),
  c("no_inhibition_no_examples", "no_inhibition", "no_examples", "inhibition+examples")
)

cat(sprintf("%-25s %10s %10s %10s\n", "Interaction", "Expected", "Actual", "Difference"))
cat(paste(rep("-", 60), collapse=""), "\n")

for (pair in pairs) {
  combined <- pair[1]
  single_a <- pair[2]
  single_b <- pair[3]
  label <- pair[4]

  rate_full <- get_rate("full")
  rate_a <- get_rate(single_a)
  rate_b <- get_rate(single_b)
  rate_combined <- get_rate(combined)

  # Expected under independence (additive model)
  contrib_a <- rate_full - rate_a
  contrib_b <- rate_full - rate_b
  expected <- rate_full - contrib_a - contrib_b

  diff <- rate_combined - expected

  cat(sprintf("%-25s %9.1f%% %9.1f%% %+9.1f%%\n",
              label, expected * 100, rate_combined * 100, diff * 100))
}

cat("\nPositive difference = features are redundant (compensate for each other)\n")
cat("Negative difference = features are synergistic (need both)\n")

# ============================================================
# 5. STATISTICAL TESTS
# ============================================================
cat("\n### 5. Statistical Tests ###\n\n")

# Fisher's exact test for each config vs full
cat("Fisher's Exact Test (each config vs full):\n")
cat(sprintf("%-28s %12s %12s\n", "Config", "Odds Ratio", "p-value"))
cat(paste(rep("-", 55), collapse=""), "\n")

for (cfg in config_order) {
  if (cfg == "full") next
  subset_df <- df[df$config == cfg, ]
  if (nrow(subset_df) > 0) {
    # Build 2x2 table
    full_solved <- sum(full_df$solved)
    full_failed <- nrow(full_df) - full_solved
    cfg_solved <- sum(subset_df$solved)
    cfg_failed <- nrow(subset_df) - cfg_solved

    contingency <- matrix(c(full_solved, full_failed, cfg_solved, cfg_failed),
                          nrow = 2, byrow = TRUE)

    test <- fisher.test(contingency)
    cat(sprintf("%-28s %12.2f %12.4f%s\n",
                cfg, test$estimate, test$p.value,
                ifelse(test$p.value < 0.05, " *", "")))
  }
}
cat("\n* = significant at p < 0.05\n")

# ============================================================
# 6. TICKS TO SOLUTION
# ============================================================
cat("\n### 6. Ticks to Solution (Solved Only) ###\n\n")

cat(sprintf("%-28s %8s %8s %8s %8s\n", "Config", "Mean", "Median", "Min", "Max"))
cat(paste(rep("-", 65), collapse=""), "\n")

for (cfg in config_order) {
  subset_df <- df[df$config == cfg & df$solved, ]
  if (nrow(subset_df) > 0) {
    cat(sprintf("%-28s %8.1f %8.1f %8d %8d\n",
                cfg,
                mean(subset_df$total_ticks),
                median(subset_df$total_ticks),
                min(subset_df$total_ticks),
                max(subset_df$total_ticks)))
  }
}

# ============================================================
# 7. TRIAL-BY-TRIAL COMPARISON
# ============================================================
cat("\n### 7. Trial-by-Trial Analysis ###\n")
cat("Comparing outcomes on identical problems (same seed)\n\n")

# Find trials where full succeeded but ablated failed
cat("Trials where FULL succeeded but ablated FAILED:\n")
full_solved_trials <- df[df$config == "full" & df$solved, "trial"]

for (cfg in config_order) {
  if (cfg == "full") next
  subset_df <- df[df$config == cfg, ]
  failed_when_full_succeeded <- subset_df[subset_df$trial %in% full_solved_trials & !subset_df$solved, "trial"]
  if (length(failed_when_full_succeeded) > 0) {
    cat(sprintf("  %s: trials %s\n", cfg, paste(failed_when_full_succeeded, collapse=", ")))
  }
}

# Find trials where ablated succeeded but full failed
cat("\nTrials where ABLATED succeeded but full FAILED:\n")
full_failed_trials <- df[df$config == "full" & !df$solved, "trial"]

for (cfg in config_order) {
  if (cfg == "full") next
  subset_df <- df[df$config == cfg, ]
  succeeded_when_full_failed <- subset_df[subset_df$trial %in% full_failed_trials & subset_df$solved, "trial"]
  if (length(succeeded_when_full_failed) > 0) {
    cat(sprintf("  %s: trials %s\n", cfg, paste(succeeded_when_full_failed, collapse=", ")))
  }
}

# ============================================================
# 8. SUMMARY TABLE FOR PAPER
# ============================================================
cat("\n### 8. Summary Table (LaTeX/Typst Ready) ###\n\n")

cat("| Config | Decay | Inhib | Examples | Solve Rate | 95% CI | Cohen's h |\n")
cat("|--------|-------|-------|----------|------------|--------|----------|\n")

for (cfg in config_order) {
  subset_df <- df[df$config == cfg, ]
  if (nrow(subset_df) > 0) {
    solved <- sum(subset_df$solved)
    total <- nrow(subset_df)
    rate <- solved / total
    ci <- wilson_ci(solved, total)

    decay <- ifelse(subset_df$decay_enabled[1], "✓", "✗")
    inhib <- ifelse(subset_df$inhibition_enabled[1], "✓", "✗")
    examples <- ifelse(subset_df$examples_enabled[1], "✓", "✗")

    if (cfg == "full") {
      h_str <- "—"
    } else {
      h <- cohens_h(full_rate, rate)
      h_str <- sprintf("%+.2f", h)
    }

    cat(sprintf("| %s | %s | %s | %s | %.1f%% | [%.1f%%, %.1f%%] | %s |\n",
                cfg, decay, inhib, examples, rate * 100, ci[1] * 100, ci[2] * 100, h_str))
  }
}

# ============================================================
# 9. TOKEN USAGE ANALYSIS
# ============================================================
cat("\n### 9. Token Usage by Configuration ###\n\n")

# Check if token data exists
if ("total_prompt_tokens" %in% names(results) && "total_completion_tokens" %in% names(results)) {
  df$prompt_tokens <- results$total_prompt_tokens
  df$completion_tokens <- results$total_completion_tokens
  df$total_tokens <- df$prompt_tokens + df$completion_tokens

  cat(sprintf("%-28s %12s %12s %12s\n", "Config", "Prompt", "Completion", "Total"))
  cat(paste(rep("-", 70), collapse=""), "\n")

  for (cfg in config_order) {
    subset_df <- df[df$config == cfg, ]
    if (nrow(subset_df) > 0) {
      cat(sprintf("%-28s %12s %12s %12s\n",
                  cfg,
                  format(round(mean(subset_df$prompt_tokens)), big.mark=","),
                  format(round(mean(subset_df$completion_tokens)), big.mark=","),
                  format(round(mean(subset_df$total_tokens)), big.mark=",")))
    }
  }

  cat("\n(Average tokens per trial)\n")
} else {
  cat("Token usage data not available in results.\n")
}

# ============================================================
# 10. MODEL ESCALATION ANALYSIS
# ============================================================
cat("\n### 10. Model Escalation Patterns ###\n\n")

if ("final_model" %in% names(results)) {
  df$final_model <- results$final_model

  cat(sprintf("%-28s %10s %10s %10s\n", "Config", "0.5b", "1.5b", "3b"))
  cat(paste(rep("-", 60), collapse=""), "\n")

  for (cfg in config_order) {
    subset_df <- df[df$config == cfg, ]
    if (nrow(subset_df) > 0) {
      model_counts <- table(factor(subset_df$final_model,
                                   levels = c("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b")))
      cat(sprintf("%-28s %10d %10d %10d\n",
                  cfg,
                  model_counts["qwen2.5:0.5b"],
                  model_counts["qwen2.5:1.5b"],
                  model_counts["qwen2.5:3b"]))
    }
  }

  cat("\n(Count of trials ending on each model)\n")
} else {
  cat("Model escalation data not available in results.\n")
}

cat("\n")
cat(paste(rep("=", 70), collapse=""), "\n")
cat("Analysis complete.\n")
