#!/usr/bin/env Rscript

# Paper Figure Generation
# =======================
# Generates publication-quality figures for the academic paper.
# Run after analyze_grid.R and analyze_ablation.R

.libPaths(c("~/R/library", .libPaths()))

# Install packages if needed
required_packages <- c("ggplot2", "dplyr", "tidyr", "patchwork", "jsonlite")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, lib = "~/R/library", repos = "https://cloud.r-project.org")
  }
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(jsonlite)

cat("Generating publication figures...\n")

# ============================================================
# Helper Functions
# ============================================================

wilson_ci <- function(successes, total, z = 1.96) {
  if (total == 0) return(c(lower = 0, upper = 0))
  p <- successes / total
  denom <- 1 + z^2/total
  center <- (p + z^2/(2*total)) / denom
  spread <- z * sqrt((p*(1-p) + z^2/(4*total)) / total) / denom
  c(lower = max(0, center - spread) * 100,
    upper = min(1, center + spread) * 100)
}

cohens_h <- function(p1, p2) {
  2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))
}

# Consistent color palette
strategy_colors <- c(
  "Pressure Field" = "#2166AC",
  "Conversation"   = "#762A83",
  "Hierarchical"   = "#1B7837",
  "Sequential"     = "#B2182B",
  "Random"         = "#878787"
)

# ============================================================
# Load Data
# ============================================================

cat("Loading grid results...\n")

files <- list(
  "Pressure Field" = "pressure-field-results.json",
  "Conversation" = "conversation-results.json",
  "Sequential" = "sequential-results.json",
  "Random" = "random-results.json",
  "Hierarchical" = "hierarchical-results.json"
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
    all_results[[name]] <- df
  }
}

df_grid <- do.call(rbind, all_results)
rownames(df_grid) <- NULL

df_grid$difficulty <- factor(
  ifelse(df_grid$empty_cells == 6, "Easy",
         ifelse(df_grid$empty_cells == 20, "Medium", "Hard")),
  levels = c("Easy", "Medium", "Hard")
)

df_grid$strategy <- factor(df_grid$strategy,
                           levels = c("Pressure Field", "Conversation",
                                      "Hierarchical", "Sequential", "Random"))

# ============================================================
# FIGURE 1: Strategy Comparison by Difficulty
# ============================================================

cat("Generating Figure 1: Strategy Comparison...\n")

# Aggregate data
df_fig1 <- df_grid %>%
  group_by(strategy, difficulty) %>%
  summarise(
    solved = sum(solved),
    total = n(),
    .groups = "drop"
  ) %>%
  mutate(
    rate = solved / total * 100,
    ci = purrr::map2(solved, total, wilson_ci),
    ci_low = purrr::map_dbl(ci, "lower"),
    ci_high = purrr::map_dbl(ci, "upper")
  )

# Abbreviate strategy names for cleaner x-axis
df_fig1 <- df_fig1 %>%
  mutate(strategy_abbrev = factor(case_when(
    strategy == "Pressure Field" ~ "P-Field",
    strategy == "Conversation" ~ "Conv",
    strategy == "Hierarchical" ~ "Hier",
    strategy == "Sequential" ~ "Seq",
    strategy == "Random" ~ "Rand",
    TRUE ~ as.character(strategy)
  ), levels = c("P-Field", "Conv", "Hier", "Seq", "Rand")))

fig1 <- ggplot(df_fig1, aes(x = strategy_abbrev, y = rate, color = strategy)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.3, linewidth = 0.6) +
  facet_wrap(~difficulty, ncol = 3) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%")
  ) +
  scale_color_manual(values = strategy_colors) +
  labs(x = NULL, y = "Solve Rate") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.5, "lines"),
    plot.margin = margin(10, 15, 10, 10)
  )

ggsave("fig1_strategy_comparison.pdf", fig1, width = 9, height = 4)
ggsave("fig1_strategy_comparison.png", fig1, width = 9, height = 4, dpi = 300)

cat("  Saved: fig1_strategy_comparison.pdf\n")

# ============================================================
# FIGURE 2: Ablation Study
# ============================================================

cat("Generating Figure 2: Ablation Study...\n")

# Load ablation data if available, otherwise use placeholder
if (file.exists("schedule-ablation.json")) {
  abl_data <- fromJSON("schedule-ablation.json")
  abl_results <- abl_data$results

  df_abl <- data.frame(
    decay = abl_results$config$decay_enabled,
    inhibition = abl_results$config$inhibition_enabled,
    examples = abl_results$config$examples_enabled,
    solved = abl_results$solved,
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      config = case_when(
        decay & inhibition & examples ~ "Full",
        !decay & inhibition & examples ~ "No Decay",
        decay & !inhibition & examples ~ "No Inhibition",
        decay & inhibition & !examples ~ "No Examples",
        !decay & !inhibition & examples ~ "No Decay + No Inhib",
        !decay & inhibition & !examples ~ "No Decay + No Ex",
        decay & !inhibition & !examples ~ "No Inhib + No Ex",
        !decay & !inhibition & !examples ~ "Baseline",
        TRUE ~ "Unknown"
      )
    ) %>%
    group_by(config, decay, inhibition, examples) %>%
    summarise(
      solved = sum(solved),
      total = n(),
      rate = sum(solved) / n() * 100,
      .groups = "drop"
    )
} else {
  # Placeholder data from current results
  df_abl <- tribble(
    ~config,                 ~decay, ~inhibition, ~examples, ~rate,
    "Full",                  TRUE,   TRUE,        TRUE,      96.7,
    "No Inhibition",         TRUE,   FALSE,       TRUE,      96.7,
    "No Decay + No Inhib",   FALSE,  FALSE,       TRUE,      93.3,
    "No Examples",           TRUE,   TRUE,        FALSE,     90.0,
    "No Decay",              FALSE,  TRUE,        TRUE,      86.7,
    "No Decay + No Ex",      FALSE,  TRUE,        FALSE,     83.3,
    "No Inhib + No Ex",      TRUE,   FALSE,       FALSE,     80.0,
    "Baseline",              FALSE,  FALSE,       FALSE,     87.5
  )
}

# Order by solve rate
df_abl <- df_abl %>%
  arrange(rate) %>%
  mutate(config = factor(config, levels = config))

# Bar chart
p_bars <- ggplot(df_abl, aes(x = rate, y = config)) +
  geom_col(fill = "#2166AC", alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", rate)),
            hjust = -0.1, size = 3.5, fontface = "bold") +
  scale_x_continuous(
    limits = c(0, 115),
    breaks = seq(0, 100, 25),
    labels = function(x) paste0(x, "%")
  ) +
  labs(x = "Solve Rate", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 10)
  )

# Feature matrix
p_features <- df_abl %>%
  pivot_longer(cols = c(decay, inhibition, examples),
               names_to = "feature", values_to = "enabled") %>%
  mutate(feature = factor(feature, levels = c("decay", "inhibition", "examples"))) %>%
  ggplot(aes(x = feature, y = config, fill = enabled)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = ifelse(enabled, "\u2713", "")), size = 4, color = "white") +
  scale_fill_manual(values = c("TRUE" = "#2166AC", "FALSE" = "#E0E0E0")) +
  scale_x_discrete(labels = c("Decay", "Inhib", "Ex")) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_text(size = 9, angle = 45, hjust = 1)
  )

# Combine
fig2 <- p_features + p_bars + plot_layout(widths = c(1, 4))

ggsave("fig2_ablation.pdf", fig2, width = 7, height = 3.5)
ggsave("fig2_ablation.png", fig2, width = 7, height = 3.5, dpi = 300)

cat("  Saved: fig2_ablation.pdf\n")

# ============================================================
# FIGURE 3: Effect Size Forest Plot
# ============================================================

cat("Generating Figure 3: Effect Sizes...\n")

# Calculate effect sizes for each difficulty
pf_rates <- df_grid %>%
  filter(strategy == "Pressure Field") %>%
  group_by(difficulty) %>%
  summarise(rate = mean(solved), n = n(), .groups = "drop")

effect_data <- list()
for (strat in c("Conversation", "Hierarchical", "Sequential", "Random")) {
  other_rates <- df_grid %>%
    filter(strategy == strat) %>%
    group_by(difficulty) %>%
    summarise(rate = mean(solved), n = n(), .groups = "drop")

  for (d in c("Easy", "Medium", "Hard")) {
    pf_r <- pf_rates$rate[pf_rates$difficulty == d]
    other_r <- other_rates$rate[other_rates$difficulty == d]
    pf_n <- pf_rates$n[pf_rates$difficulty == d]
    other_n <- other_rates$n[other_rates$difficulty == d]

    h <- cohens_h(pf_r, other_r)
    se <- sqrt(1/pf_n + 1/other_n)

    effect_data[[length(effect_data) + 1]] <- data.frame(
      comparison = paste("vs", strat),
      difficulty = d,
      h = h,
      se = se,
      ci_low = h - 1.96 * se,
      ci_high = h + 1.96 * se
    )
  }
}

df_effects <- do.call(rbind, effect_data) %>%
  filter(difficulty == "Easy") %>%  # Focus on Easy for cleaner plot
  mutate(
    comparison = factor(comparison, levels = rev(unique(comparison))),
    strategy = gsub("vs ", "", comparison)
  )

# Colors matching the strategy color scheme
effect_colors <- c(
  "Conversation" = "#762A83",
  "Hierarchical" = "#1B7837",
  "Sequential" = "#B2182B",
  "Random" = "#878787"
)

fig3 <- ggplot(df_effects, aes(x = h, y = comparison, color = strategy)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 0.8, linetype = "dotted", color = "#E31A1C", linewidth = 1) +
  geom_point(size = 3.5) +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high),
                 height = 0.25, linewidth = 0.6) +
  scale_color_manual(values = effect_colors) +
  annotate("text", x = 0.85, y = 1.5, label = "Large effect\nthreshold (0.8)",
           size = 3, color = "#E31A1C", hjust = 0, lineheight = 0.9, fontface = "bold") +
  scale_x_continuous(limits = c(-0.2, 2.8), breaks = seq(0, 2.5, 0.5)) +
  labs(
    x = "Cohen's h (Effect Size)",
    y = NULL,
    subtitle = "Pressure Field vs Baselines (Easy Difficulty)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.subtitle = element_text(size = 10, color = "gray30")
  )

ggsave("fig3_effect_sizes.pdf", fig3, width = 6, height = 3)
ggsave("fig3_effect_sizes.png", fig3, width = 6, height = 3, dpi = 300)

cat("  Saved: fig3_effect_sizes.pdf\n")

# ============================================================
# FIGURE 4: Ticks to Solution (Efficiency)
# ============================================================

cat("Generating Figure 4: Efficiency (Ticks to Solution)...\n")

df_ticks <- df_grid %>%
  filter(solved == TRUE, difficulty == "Easy") %>%
  group_by(strategy) %>%
  summarise(
    mean_ticks = mean(total_ticks),
    sd_ticks = sd(total_ticks),
    n = n(),
    se = sd_ticks / sqrt(n),
    .groups = "drop"
  ) %>%
  filter(n >= 4)  # Include strategies with at least 4 solved trials (includes Hierarchical)

fig4 <- ggplot(df_ticks, aes(x = reorder(strategy, mean_ticks), y = mean_ticks, fill = strategy)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_ticks - se, ymax = mean_ticks + se),
                width = 0.2, linewidth = 0.5) +
  geom_text(aes(y = mean_ticks + se + 2, label = sprintf("%.1f", mean_ticks)),
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = strategy_colors) +
  scale_y_continuous(limits = c(0, 50), breaks = seq(0, 50, 10)) +
  labs(
    x = NULL,
    y = "Mean Ticks to Solution",
    subtitle = "Easy Difficulty (Solved Trials Only)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    plot.subtitle = element_text(size = 10, color = "gray30")
  )

ggsave("fig4_efficiency.pdf", fig4, width = 5, height = 4)
ggsave("fig4_efficiency.png", fig4, width = 5, height = 4, dpi = 300)

cat("  Saved: fig4_efficiency.pdf\n")

# ============================================================
# FIGURE 5: Feature Contribution Summary
# ============================================================

cat("Generating Figure 5: Feature Contributions...\n")

# Calculate contributions
if (exists("df_abl")) {
  full_rate <- df_abl$rate[df_abl$config == "Full"]

  contributions <- data.frame(
    feature = c("Decay", "Inhibition", "Examples"),
    contribution = c(
      full_rate - df_abl$rate[df_abl$config == "No Decay"],
      full_rate - df_abl$rate[df_abl$config == "No Inhibition"],
      full_rate - df_abl$rate[df_abl$config == "No Examples"]
    )
  ) %>%
    mutate(feature = factor(feature, levels = c("Decay", "Examples", "Inhibition")))

  fig5 <- ggplot(contributions, aes(x = feature, y = contribution, fill = contribution > 0)) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_hline(yintercept = 0, linewidth = 0.5) +
    geom_text(aes(label = sprintf("%+.1f%%", contribution),
                  vjust = ifelse(contribution >= 0, -0.5, 1.5)),
              size = 4, fontface = "bold") +
    scale_fill_manual(values = c("TRUE" = "#2166AC", "FALSE" = "#878787")) +
    scale_y_continuous(limits = c(-5, 15), breaks = seq(-5, 15, 5),
                       labels = function(x) paste0(x, "%")) +
    labs(
      x = NULL,
      y = "Contribution to Solve Rate",
      subtitle = "Full Configuration vs Single Feature Removed"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(size = 11, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "gray30")
    )

  ggsave("fig5_contributions.pdf", fig5, width = 5, height = 4)
  ggsave("fig5_contributions.png", fig5, width = 5, height = 4, dpi = 300)

  cat("  Saved: fig5_contributions.pdf\n")
}

# ============================================================
# Summary
# ============================================================

cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("Figure generation complete.\n")
cat("\nGenerated files:\n")
cat("  - fig1_strategy_comparison.pdf/png\n")
cat("  - fig2_ablation.pdf/png\n")
cat("  - fig3_effect_sizes.pdf/png\n")
cat("  - fig4_efficiency.pdf/png\n")
cat("  - fig5_contributions.pdf/png\n")
cat(paste(rep("=", 50), collapse=""), "\n")
