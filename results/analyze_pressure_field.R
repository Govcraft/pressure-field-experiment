#!/usr/bin/env Rscript

# Analysis of Pressure Field Experiment Results
# =============================================

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

# Load the data
cat("Loading pressure-field results...\n")
data <- fromJSON("pressure-field-results.json")
results <- data$results

# Extract key fields into a data frame
df <- data.frame(
  strategy = results$config$strategy,
  agent_count = results$config$agent_count,
  empty_cells = results$config$empty_cells,
  trial = results$config$trial,
  solved = results$solved,
  total_ticks = results$total_ticks,
  final_pressure = results$final_pressure
)

# Map empty_cells to difficulty
df$difficulty <- factor(
  ifelse(df$empty_cells == 6, "easy",
         ifelse(df$empty_cells == 20, "medium", "hard")),
  levels = c("easy", "medium", "hard")
)

df$agent_count <- factor(df$agent_count)

cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("PRESSURE FIELD EXPERIMENT ANALYSIS\n")
cat(paste(rep("=", 60), collapse=""), "\n")

# Overall summary
cat("\n### Overall Summary ###\n")
cat(sprintf("Total experiments: %d\n", nrow(df)))
cat(sprintf("Overall solve rate: %.1f%% (%d/%d)\n",
            mean(df$solved) * 100, sum(df$solved), nrow(df)))
cat(sprintf("Average ticks (all): %.1f\n", mean(df$total_ticks)))
cat(sprintf("Average ticks (solved only): %.1f\n",
            mean(df$total_ticks[df$solved])))

# Solve rate by agent count
cat("\n### Solve Rate by Agent Count ###\n")
for (a in levels(df$agent_count)) {
  subset_df <- df[df$agent_count == a, ]
  cat(sprintf("  %s agent(s): %.1f%% (%d/%d)\n",
              a,
              mean(subset_df$solved) * 100,
              sum(subset_df$solved),
              nrow(subset_df)))
}

# Solve rate by difficulty
cat("\n### Solve Rate by Difficulty ###\n")
for (d in levels(df$difficulty)) {
  subset_df <- df[df$difficulty == d, ]
  cat(sprintf("  %s: %.1f%% (%d/%d)\n",
              d,
              mean(subset_df$solved) * 100,
              sum(subset_df$solved),
              nrow(subset_df)))
}

# Cross-tabulation: agents x difficulty
cat("\n### Solve Rate: Agents × Difficulty ###\n")
cat(sprintf("%12s %10s %10s %10s\n", "", "easy", "medium", "hard"))
cat(sprintf("%12s %10s %10s %10s\n", "------------", "----------", "----------", "----------"))

for (a in levels(df$agent_count)) {
  rates <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$agent_count == a & df$difficulty == d, ]
    sprintf("%.1f%%", mean(subset_df$solved) * 100)
  })
  cat(sprintf("%12s %10s %10s %10s\n", paste0(a, " agent(s)"), rates[1], rates[2], rates[3]))
}

# Average ticks to solution by configuration
cat("\n### Average Ticks (Solved Only): Agents × Difficulty ###\n")
cat(sprintf("%12s %10s %10s %10s\n", "", "easy", "medium", "hard"))
cat(sprintf("%12s %10s %10s %10s\n", "------------", "----------", "----------", "----------"))

for (a in levels(df$agent_count)) {
  ticks <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$agent_count == a & df$difficulty == d & df$solved, ]
    if (nrow(subset_df) > 0) {
      sprintf("%.1f", mean(subset_df$total_ticks))
    } else {
      "N/A"
    }
  })
  cat(sprintf("%12s %10s %10s %10s\n", paste0(a, " agent(s)"), ticks[1], ticks[2], ticks[3]))
}

# Statistical test: Does agent count affect solve rate?
cat("\n### Statistical Analysis ###\n")

# Chi-squared test for agent count vs solved
agent_table <- table(df$agent_count, df$solved)
chi_agent <- chisq.test(agent_table)
cat(sprintf("Chi-squared test (agents vs solved): χ²=%.2f, df=%d, p=%.4f\n",
            chi_agent$statistic, chi_agent$parameter, chi_agent$p.value))

# Chi-squared test for difficulty vs solved
diff_table <- table(df$difficulty, df$solved)
chi_diff <- chisq.test(diff_table)
cat(sprintf("Chi-squared test (difficulty vs solved): χ²=%.2f, df=%d, p=%.4f\n",
            chi_diff$statistic, chi_diff$parameter, chi_diff$p.value))

# Token usage analysis (from tick_metrics)
cat("\n### Token Usage Analysis ###\n")
total_prompt_tokens <- 0
total_completion_tokens <- 0

for (i in 1:length(results$tick_metrics)) {
  metrics <- results$tick_metrics[[i]]
  if (!is.null(metrics) && is.data.frame(metrics) && nrow(metrics) > 0) {
    total_prompt_tokens <- total_prompt_tokens + sum(metrics$prompt_tokens, na.rm = TRUE)
    total_completion_tokens <- total_completion_tokens + sum(metrics$completion_tokens, na.rm = TRUE)
  }
}

cat(sprintf("Total prompt tokens: %s\n", format(total_prompt_tokens, big.mark=",")))
cat(sprintf("Total completion tokens: %s\n", format(total_completion_tokens, big.mark=",")))
cat(sprintf("Total tokens: %s\n", format(total_prompt_tokens + total_completion_tokens, big.mark=",")))
cat(sprintf("Avg tokens per experiment: %s\n",
            format(round((total_prompt_tokens + total_completion_tokens) / nrow(df)), big.mark=",")))

# Pressure reduction analysis
cat("\n### Pressure Reduction Analysis ###\n")
cat("Average final pressure by configuration:\n")
cat(sprintf("%12s %10s %10s %10s\n", "", "easy", "medium", "hard"))
cat(sprintf("%12s %10s %10s %10s\n", "------------", "----------", "----------", "----------"))

for (a in levels(df$agent_count)) {
  pressures <- sapply(levels(df$difficulty), function(d) {
    subset_df <- df[df$agent_count == a & df$difficulty == d, ]
    sprintf("%.2f", mean(subset_df$final_pressure))
  })
  cat(sprintf("%12s %10s %10s %10s\n", paste0(a, " agent(s)"), pressures[1], pressures[2], pressures[3]))
}

cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("Analysis complete.\n")
