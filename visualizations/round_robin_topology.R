# Round-Robin Dispatch Topology Visualization
# Shows how high-pressure regions are assigned to patch actors each tick
# Using base R graphics (no external packages required)

set.seed(42)

# Example scenario: 6 regions, 2 patch actors
regions <- data.frame(
  id = paste0("R", 1:6),
  pressure = c(0.85, 0.72, 0.45, 0.91, 0.33, 0.78),
  inhibited = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)

threshold <- 0.5
n_actors <- 2

# Filter eligible regions (high pressure + not inhibited)
eligible_mask <- regions$pressure >= threshold & !regions$inhibited
eligible <- regions[eligible_mask, ]
eligible$eligible_idx <- seq_len(nrow(eligible)) - 1  # 0-indexed
eligible$assigned_actor <- (eligible$eligible_idx %% n_actors) + 1

# Layout coordinates
region_y <- seq(6, 1, length.out = 6)
actor_y <- c(4.5, 2.5)

# Colors
col_eligible <- "#3498db"
col_below <- "#95a5a6"
col_inhibited <- "#e74c3c"
col_actor <- "#9b59b6"
col_edge <- "#2ecc71"

# Determine region colors
region_colors <- ifelse(regions$inhibited, col_inhibited,
                        ifelse(regions$pressure >= threshold, col_eligible, col_below))

# Create PNG
png("round_robin_topology.png", width = 800, height = 800, res = 100)

par(mar = c(4, 2, 4, 2), bg = "white")
plot(NULL, xlim = c(-1, 5), ylim = c(0, 8),
     xlab = "", ylab = "", xaxt = "n", yaxt = "n",
     main = "Round-Robin Dispatch Topology (Single Tick)",
     cex.main = 1.4, font.main = 2)
mtext("Only eligible regions (pressure >= threshold, not inhibited) receive proposals",
      side = 3, line = 0.3, cex = 0.9, col = "gray40")

# Draw edges (proposals) first
for (i in seq_len(nrow(eligible))) {
  region_idx <- which(regions$id == eligible$id[i])
  actor_idx <- eligible$assigned_actor[i]

  x0 <- 0.6  # end of region circle
  y0 <- region_y[region_idx]
  x1 <- 3.4  # start of actor circle
  y1 <- actor_y[actor_idx]

  arrows(x0, y0, x1, y1, col = col_edge, lwd = 2.5, length = 0.12)
}

# Draw region nodes
for (i in 1:6) {
  symbols(0, region_y[i], circles = 0.4, inches = FALSE,
          add = TRUE, bg = region_colors[i], fg = "white", lwd = 2)

  # Label
  label <- paste0(regions$id[i], "\np=", sprintf("%.2f", regions$pressure[i]))
  text(0, region_y[i], label, cex = 0.7, col = "white", font = 2)

  # Status annotation
  status <- if (regions$inhibited[i]) "(inhibited)"
            else if (regions$pressure[i] < threshold) "(< threshold)"
            else ""
  if (status != "") {
    text(-0.8, region_y[i], status, cex = 0.6, col = region_colors[i], adj = 1)
  }
}

# Draw actor nodes
for (i in 1:n_actors) {
  symbols(4, actor_y[i], circles = 0.5, inches = FALSE,
          add = TRUE, bg = col_actor, fg = "white", lwd = 2)
  text(4, actor_y[i], paste0("Actor", i), cex = 0.8, col = "white", font = 2)
}

# Column labels
text(0, 7.3, "Regions", font = 2, cex = 1.2)
text(4, 7.3, "Patch Actors", font = 2, cex = 1.2)

# Legend
legend("bottom",
       legend = c("Eligible (high pressure)", "Below threshold", "Inhibited (cooldown)", "Patch Actor"),
       fill = c(col_eligible, col_below, col_inhibited, col_actor),
       border = "white", bty = "n", horiz = FALSE, ncol = 2, cex = 0.9)

# Formula annotation
text(2, 0.3,
     paste0("Threshold = ", threshold, "  |  Assignment: region[i] -> actor[i mod ", n_actors, "]"),
     cex = 0.85, col = "gray40")

dev.off()

# Print summary
cat("\n=== Round-Robin Dispatch Summary ===\n")
cat("Threshold:", threshold, "\n")
cat("Total regions:", nrow(regions), "\n")
cat("Eligible regions:", nrow(eligible), "\n")
cat("Patch actors:", n_actors, "\n\n")

cat("Region Status:\n")
print(transform(regions,
  status = ifelse(inhibited, "INHIBITED",
           ifelse(pressure < threshold, "below threshold", "ELIGIBLE"))))

cat("\nAssignments (eligible -> actor):\n")
print(eligible[, c("id", "pressure", "assigned_actor")])

cat("\nVisualization saved to: round_robin_topology.png\n")
