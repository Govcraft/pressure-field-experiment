#!/usr/bin/env Rscript
# Combine 6 pressure_field runs into unified result files
#
# Usage: Rscript combine_pressure_field.R
# Output: results/pressure_field_combined/*.json
#
# Works with nested JSON by processing as raw lists, not data frames

.libPaths(c("~/R/library", .libPaths()))
library(jsonlite)

cat(strrep("=", 60), "\n")
cat("COMBINING PRESSURE_FIELD RESULTS FROM 6 RUNS\n")
cat(strrep("=", 60), "\n\n")

# Configuration - the 6 pressure_field run directories
pf_dirs <- c(
  "20260115-055818",
  "20260115-071439",
  "20260115-083012",
  "20260115-094712",
  "20260115-110643",
  "20260115-122323"
)

# Map experiment name to file glob pattern
experiment_patterns <- list(
  "main-grid"          = "main-grid-pressure_field-*.json",
  "scaling"            = "scaling-pressure_field-*.json",
  "difficulty-easy"    = "difficulty-easy-pressure_field-*.json",
  "difficulty-hard"    = "difficulty-hard-pressure_field-*.json",
  "escalation-with"    = "escalation-with-pressure_field-*.json",
  "escalation-without" = "escalation-without-pressure_field-*.json"
)

output_dir <- "results/pressure_field_combined"

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
cat("Output directory:", output_dir, "\n\n")

# Track totals for summary
totals <- list()

# Process each experiment type
for (exp_name in names(experiment_patterns)) {
  pattern <- experiment_patterns[[exp_name]]
  all_results <- list()

  cat(sprintf("Processing %s...\n", exp_name))

  for (i in seq_along(pf_dirs)) {
    dir_path <- file.path("results", pf_dirs[i])
    files <- Sys.glob(file.path(dir_path, pattern))

    if (length(files) == 0) {
      cat(sprintf("  Warning: No files matching '%s' in %s\n", pattern, pf_dirs[i]))
      next
    }

    # Load as raw JSON (simplifyVector = FALSE keeps it as nested lists)
    json_text <- readLines(files[1], warn = FALSE)
    data <- fromJSON(paste(json_text, collapse = "\n"), simplifyVector = FALSE)

    n_results <- length(data$results)
    cat(sprintf("  Run %d (%s): %d results\n", i, pf_dirs[i], n_results))

    # Add run tracking to each result and collect
    for (j in seq_along(data$results)) {
      result <- data$results[[j]]
      result$run_id <- i
      result$source_dir <- pf_dirs[i]
      all_results <- c(all_results, list(result))
    }
  }

  if (length(all_results) == 0) {
    cat(sprintf("  ERROR: No results found for %s\n\n", exp_name))
    next
  }

  total_results <- length(all_results)
  totals[[exp_name]] <- total_results

  # Write combined results as JSON
  combined <- list(results = all_results)
  output_path <- file.path(output_dir, paste0(exp_name, ".json"))

  # Use write_json with auto_unbox for clean output
  write_json(combined, output_path, auto_unbox = TRUE, pretty = TRUE,
             na = "null", null = "null")

  cat(sprintf("  -> Combined %d results -> %s\n\n", total_results, output_path))
}

# Write metadata
metadata <- list(
  combined_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  source_directories = pf_dirs,
  runs = length(pf_dirs),
  experiments = names(experiment_patterns),
  result_counts = totals
)
write_json(metadata, file.path(output_dir, "metadata.json"),
           auto_unbox = TRUE, pretty = TRUE)

# Summary
cat(strrep("=", 60), "\n")
cat("COMBINATION COMPLETE\n")
cat(strrep("=", 60), "\n\n")

cat("Results per experiment:\n")
for (exp_name in names(totals)) {
  cat(sprintf("  %-20s: %d results\n", exp_name, totals[[exp_name]]))
}
cat(sprintf("\nTotal results combined: %d\n", sum(unlist(totals))))
cat(sprintf("Output directory: %s\n", output_dir))
