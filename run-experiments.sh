#!/usr/bin/env bash
#
# Experimental Protocol Runner
# Runs all experiments defined in the paper appendix
#
# Usage: ./run-experiments.sh [--dry-run] [--parallel N] [--experiment NAME]
#
# Experiments: main-grid, ablation, scaling, escalation, difficulty, all
#

set -euo pipefail

# Get script's absolute path for sourcing in subshells
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Configuration
LATIN_EXPERIMENT="${LATIN_EXPERIMENT:-./latin-experiment}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
TRIALS="${TRIALS:-30}"  # 30 trials for publication-quality statistics
DRY_RUN="${DRY_RUN:-false}"
PARALLEL="${PARALLEL:-10}"

# Model chains for escalation - load balanced across vLLM servers
# Chain A: 3B → 7B → 14B (ports 8003 → 8004 → 8005) - for main-grid, escalation
MODEL_CHAIN_A="Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B"
VLLM_HOST_A="http://localhost:8003"
VLLM_HOSTS_A="http://localhost:8003,http://localhost:8004,http://localhost:8005"

# Chain B: 7B → 14B (ports 8004 → 8005) - for scaling, difficulty
MODEL_CHAIN_B="Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B"
VLLM_HOST_B="http://localhost:8004"
VLLM_HOSTS_B="http://localhost:8004,http://localhost:8005"

# Single model for ablation (no escalation) - uses 14B to avoid contention
MODEL_SINGLE="Qwen/Qwen2.5-14B"
VLLM_HOST_SINGLE="http://localhost:8005"

# Default chain (used if not overridden)
MODEL_CHAIN="$MODEL_CHAIN_A"
ESCALATION_THRESHOLD=10
NUM_MODELS=3  # 3B, 7B, 14B (skip 0.5B, 1.5B for efficiency)
MAX_TICKS=$((NUM_MODELS * ESCALATION_THRESHOLD))  # 30 ticks = room for all 3 models

# Strategy sets for different experiments
ALL_STRATEGIES="pressure_field,hierarchical,sequential,random,conversation"
MAIN_STRATEGIES="pressure_field,hierarchical"  # For scaling/difficulty (others proven to fail)

# vLLM server configuration
VLLM_HOST="${VLLM_HOST:-http://localhost:8000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Job tracking (initialize empty arrays)
declare -A JOB_PIDS=()
declare -A JOB_NAMES=()
declare -A JOB_LOGS=()
declare -a FAILED_JOBS=()

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_job() {
    local name="$1"
    local msg="$2"
    echo -e "${CYAN}[JOB:${name}]${NC} $msg"
}

run_cmd() {
    local desc="$1"
    shift
    log_info "Running: $desc"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "Dry-run mode: skipping prerequisite checks"
        return 0
    fi

    if [[ ! -x "$LATIN_EXPERIMENT" ]]; then
        log_error "latin-experiment binary not found at: $LATIN_EXPERIMENT"
        exit 1
    fi

    # Check vLLM server is running
    if ! curl -s "${VLLM_HOST}/health" > /dev/null 2>&1; then
        log_error "vLLM server not responding at: $VLLM_HOST"
        log_error "Start it with: vllm serve Qwen/Qwen2.5-7B --dtype bfloat16 --port 8000"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

setup_output_dir() {
    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    RESULTS_DIR="${OUTPUT_DIR}/${timestamp}"
    LOGS_DIR="${RESULTS_DIR}/logs"
    mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
    log_info "Results will be saved to: $RESULTS_DIR"

    # Save experiment metadata
    cat > "$RESULTS_DIR/metadata.json" <<EOF
{
    "timestamp": "$timestamp",
    "trials": $TRIALS,
    "parallel": $PARALLEL,
    "model_chain": "$MODEL_CHAIN",
    "escalation_threshold": $ESCALATION_THRESHOLD,
    "hostname": "$(hostname 2>/dev/null || echo 'unknown')",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
}
EOF
}

# Wait for a job slot to become available
wait_for_slot() {
    while [[ ${#JOB_PIDS[@]} -ge $PARALLEL ]]; do
        for name in "${!JOB_PIDS[@]}"; do
            local pid="${JOB_PIDS[$name]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, check exit status
                wait "$pid" && local status=0 || local status=$?
                if [[ $status -ne 0 ]]; then
                    log_error "Job '$name' failed with exit code $status"
                    FAILED_JOBS+=("$name")
                else
                    log_success "Job '$name' completed successfully"
                fi
                unset "JOB_PIDS[$name]"
                break
            fi
        done
        sleep 1
    done
}

# Wait for all remaining jobs with progress monitoring
wait_all_jobs() {
    log_info "Waiting for ${#JOB_PIDS[@]} job(s) to complete..."
    log_info "Monitor progress with: tail -f $LOGS_DIR/*.log"
    echo ""

    local check_interval=5
    local last_status_time=0

    while [[ ${#JOB_PIDS[@]} -gt 0 ]]; do
        local current_time
        current_time=$(date +%s)

        # Check for completed jobs
        for name in "${!JOB_PIDS[@]}"; do
            local pid="${JOB_PIDS[$name]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, check exit status
                wait "$pid" && local status=0 || local status=$?
                if [[ $status -ne 0 ]]; then
                    log_error "Job '$name' FAILED (exit code $status)"
                    FAILED_JOBS+=("$name")
                else
                    log_success "Job '$name' completed"
                fi
                unset "JOB_PIDS[$name]"
            fi
        done

        # Show status every 30 seconds
        if [[ $((current_time - last_status_time)) -ge 30 ]] && [[ ${#JOB_PIDS[@]} -gt 0 ]]; then
            last_status_time=$current_time
            echo ""
            log_info "=== Status Update ($(date +%H:%M:%S)) ==="
            log_info "Running: ${!JOB_PIDS[*]}"
            for name in "${!JOB_PIDS[@]}"; do
                local log="${JOB_LOGS[$name]}"
                local last_line
                last_line=$(tail -1 "$log" 2>/dev/null | cut -c1-80 || echo "(no output)")
                echo -e "  ${CYAN}$name${NC}: $last_line"
            done
            echo ""
        fi

        sleep "$check_interval"
    done
}

# Launch a job (respects parallel limit)
launch_job() {
    local name="$1"
    local log_file="$2"
    shift 2

    wait_for_slot

    log_job "$name" "Starting (log: $log_file)"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] $*"
        echo "  [DRY-RUN] Would log to: $log_file"
    else
        # Run in background, redirect output to log file
        "$@" > "$log_file" 2>&1 &
        local pid=$!
        JOB_PIDS[$name]=$pid
        JOB_LOGS[$name]="$log_file"
        log_job "$name" "Launched with PID $pid"
    fi
}

# Show live status of running jobs
show_status() {
    if [[ ${#JOB_PIDS[@]} -eq 0 ]]; then
        return
    fi
    echo ""
    log_info "=== Running Jobs Status ==="
    for name in "${!JOB_PIDS[@]}"; do
        local pid="${JOB_PIDS[$name]}"
        local log="${JOB_LOGS[$name]}"
        if kill -0 "$pid" 2>/dev/null; then
            local last_line
            last_line=$(tail -1 "$log" 2>/dev/null || echo "(no output yet)")
            echo -e "  ${CYAN}$name${NC} (PID $pid): $last_line"
        fi
    done
    echo ""
}

# Experiment 1: Main Grid (Strategy Comparison)
# Purpose: Validate that pressure-field coordination outperforms baselines
# Tests ALL 5 strategies - this is the comprehensive comparison
run_main_grid() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Experiment 1: Main Grid (Strategy Comparison)"
    log_info "5 strategies × 3 agent counts × $TRIALS trials = $((5 * 3 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Uses Chain A: 3B → 7B → 14B (ports 8003-8005)
    run_cmd "Main Grid Experiment" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --vllm-hosts "$VLLM_HOSTS_A" \
        --model-chain "$MODEL_CHAIN_A" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 7 \
        --max-ticks $MAX_TICKS \
        --agents 2,4,8 \
        --strategies "$ALL_STRATEGIES" \
        --output "$RESULTS_DIR/main-grid.json"

    log_success "Main Grid experiment complete"
}

# Experiment 2: Ablation Study
# Purpose: Validate that each mechanism contributes to performance
# Note: Runs on EASY grid (5×5) to show solve rate differences
# Note: Runs WITHOUT escalation to isolate mechanism effects
run_ablation() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Experiment 2: Ablation Study"
    log_info "8 configurations × $TRIALS trials = $((8 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Uses single 14B model (port 8005) - no escalation, avoids contention
    run_cmd "Ablation Study" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_SINGLE" \
        --model-chain "$MODEL_SINGLE" \
        ablation \
        --trials "$TRIALS" \
        --n 5 \
        --empty 5 \
        --max-ticks $MAX_TICKS \
        --output "$RESULTS_DIR/ablation.json"

    log_success "Ablation study complete"
}

# Experiment 3: Scaling Analysis
# Purpose: Validate Theorem 3 (linear scaling)
# Only tests pressure_field vs hierarchical (others proven to fail in main grid)
run_scaling() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Experiment 3: Scaling Analysis"
    log_info "2 strategies × 4 agent counts × $TRIALS trials = $((2 * 4 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Uses Chain B: 7B → 14B (ports 8004-8005) - load balanced with main-grid
    run_cmd "Scaling Analysis" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 2,4,8,16 \
        --strategies "$MAIN_STRATEGIES" \
        --output "$RESULTS_DIR/scaling.json"

    log_success "Scaling analysis complete"
}

# Experiment 4: Model Escalation Impact
# Purpose: Validate that model escalation improves solve rate
# Tests ALL 5 strategies to prove baselines fail regardless of escalation
run_escalation() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Experiment 4: Model Escalation Impact"
    log_info "2 configs × 5 strategies × $TRIALS trials = $((2 * 5 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Without escalation - uses single 3B model (port 8003)
    log_info "Running WITHOUT escalation (3B only)..."
    run_cmd "Escalation: Single Model" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --model-chain "Qwen/Qwen2.5-3B" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "$ALL_STRATEGIES" \
        --output "$RESULTS_DIR/escalation-without.json"

    # With escalation - uses Chain A: 3B → 7B → 14B (ports 8003-8005)
    log_info "Running WITH escalation (3B → 7B → 14B)..."
    run_cmd "Escalation: Model Chain" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --vllm-hosts "$VLLM_HOSTS_A" \
        --model-chain "$MODEL_CHAIN_A" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "$ALL_STRATEGIES" \
        --output "$RESULTS_DIR/escalation-with.json"

    log_success "Escalation impact experiment complete"
}

# Experiment 5: Difficulty Scaling
# Purpose: Show framework handles increasing difficulty
# Tests ALL 5 strategies to prove baselines fail regardless of difficulty
run_difficulty() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Experiment 5: Difficulty Scaling"
    log_info "2 difficulty levels × 5 strategies × $TRIALS trials = $((2 * 5 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Easy: 5x5, 5 empty (20%) - uses Chain B: 7B → 14B (ports 8004-8005)
    log_info "Running Easy difficulty (5x5, 5 empty)..."
    run_cmd "Difficulty: Easy" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 5 \
        --empty 5 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "$ALL_STRATEGIES" \
        --output "$RESULTS_DIR/difficulty-easy.json"

    # Hard: 7x7, 7 empty - uses Chain B: 7B → 14B (ports 8004-8005)
    log_info "Running Hard difficulty (7x7, 7 empty)..."
    run_cmd "Difficulty: Hard" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 7 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "$ALL_STRATEGIES" \
        --output "$RESULTS_DIR/difficulty-hard.json"

    log_success "Difficulty scaling experiment complete"
}

# Re-run: Pressure Field Only
# Purpose: Re-run only pressure_field experiments after bug fix
# Bug: SharedGrid was not updated in kernel path, causing stale validation
run_pressure_field_only() {
    local log_prefix="${1:-}"

    [[ -n "$log_prefix" ]] || log_info "========================================"
    [[ -n "$log_prefix" ]] || log_info "Re-run: Pressure Field Only (Bug Fix)"
    log_info "Re-running pressure_field across all experiments"
    log_info "Main: 3 agents × $TRIALS = $((3 * TRIALS)) runs"
    log_info "Scaling: 4 agents × $TRIALS = $((4 * TRIALS)) runs"
    log_info "Escalation: 2 configs × $TRIALS = $((2 * TRIALS)) runs"
    log_info "Difficulty: 2 levels × $TRIALS = $((2 * TRIALS)) runs"
    log_info "Total: $((3 * TRIALS + 4 * TRIALS + 2 * TRIALS + 2 * TRIALS)) runs"
    [[ -n "$log_prefix" ]] || log_info "========================================"

    # Main Grid - pressure_field only (Chain A: 3B → 7B → 14B)
    log_info "Running Main Grid (pressure_field only)..."
    run_cmd "Main Grid (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --vllm-hosts "$VLLM_HOSTS_A" \
        --model-chain "$MODEL_CHAIN_A" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 7 \
        --max-ticks $MAX_TICKS \
        --agents 2,4,8 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/main-grid-pressure_field.json"

    # Scaling - pressure_field only (Chain B: 7B → 14B)
    log_info "Running Scaling (pressure_field only)..."
    run_cmd "Scaling (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 2,4,8,16 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/scaling-pressure_field.json"

    # Escalation without - pressure_field only (3B only)
    log_info "Running Escalation WITHOUT (pressure_field only)..."
    run_cmd "Escalation without (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --model-chain "Qwen/Qwen2.5-3B" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/escalation-without-pressure_field.json"

    # Escalation with - pressure_field only (Chain A: 3B → 7B → 14B)
    log_info "Running Escalation WITH (pressure_field only)..."
    run_cmd "Escalation with (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_A" \
        --vllm-hosts "$VLLM_HOSTS_A" \
        --model-chain "$MODEL_CHAIN_A" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 8 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/escalation-with-pressure_field.json"

    # Difficulty Easy - pressure_field only (Chain B: 7B → 14B)
    log_info "Running Difficulty Easy (pressure_field only)..."
    run_cmd "Difficulty Easy (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 5 \
        --empty 5 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/difficulty-easy-pressure_field.json"

    # Difficulty Hard - pressure_field only (Chain B: 7B → 14B)
    log_info "Running Difficulty Hard (pressure_field only)..."
    run_cmd "Difficulty Hard (pressure_field)" \
        "$LATIN_EXPERIMENT" \
        --vllm-host "$VLLM_HOST_B" \
        --vllm-hosts "$VLLM_HOSTS_B" \
        --model-chain "$MODEL_CHAIN_B" \
        --escalation-threshold "$ESCALATION_THRESHOLD" \
        grid \
        --trials "$TRIALS" \
        --n 7 \
        --empty 7 \
        --max-ticks $MAX_TICKS \
        --agents 4 \
        --strategies "pressure_field" \
        --output "$RESULTS_DIR/difficulty-hard-pressure_field.json"

    log_success "Pressure field re-run complete"
}

run_all_sequential() {
    local start_time
    start_time=$(date +%s)

    log_info "========================================"
    log_info "Running FULL Experimental Protocol (SEQUENTIAL)"
    log_info "========================================"

    run_main_grid
    run_ablation
    run_scaling
    run_escalation
    run_difficulty

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    log_success "========================================"
    log_success "All experiments complete!"
    log_success "Total runtime: ${hours}h ${minutes}m"
    log_success "Results saved to: $RESULTS_DIR"
    log_success "========================================"
}

run_all_parallel() {
    local start_time
    start_time=$(date +%s)

    log_info "========================================"
    log_info "Running FULL Experimental Protocol (PARALLEL: $PARALLEL jobs)"
    log_info "========================================"

    # Export variables needed by subprocesses
    export RESULTS_DIR LOGS_DIR TRIALS MODEL_CHAIN MODEL_SINGLE ESCALATION_THRESHOLD DRY_RUN LATIN_EXPERIMENT VLLM_HOST VLLM_HOSTS ALL_STRATEGIES MAIN_STRATEGIES MODEL_CHAIN_A MODEL_CHAIN_B VLLM_HOST_A VLLM_HOST_B VLLM_HOSTS_A VLLM_HOSTS_B VLLM_HOST_SINGLE

    # Launch all experiments as background jobs
    # These are independent and can run in any order

    launch_job "main-grid" "$LOGS_DIR/main-grid.log" \
        "$SCRIPT_PATH" --internal main-grid

    launch_job "ablation" "$LOGS_DIR/ablation.log" \
        "$SCRIPT_PATH" --internal ablation

    launch_job "scaling" "$LOGS_DIR/scaling.log" \
        "$SCRIPT_PATH" --internal scaling

    launch_job "escalation" "$LOGS_DIR/escalation.log" \
        "$SCRIPT_PATH" --internal escalation

    launch_job "difficulty" "$LOGS_DIR/difficulty.log" \
        "$SCRIPT_PATH" --internal difficulty

    # Wait for all jobs to complete
    wait_all_jobs

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    log_success "========================================"
    if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
        log_error "Some experiments failed: ${FAILED_JOBS[*]}"
        log_info "Check logs in: $LOGS_DIR"
    else
        log_success "All experiments complete!"
    fi
    log_success "Total runtime: ${hours}h ${minutes}m"
    log_success "Results saved to: $RESULTS_DIR"
    log_success "Logs saved to: $LOGS_DIR"
    log_success "========================================"

    # Return failure if any jobs failed
    [[ ${#FAILED_JOBS[@]} -eq 0 ]]
}

run_all() {
    if [[ $PARALLEL -gt 1 ]]; then
        run_all_parallel
    else
        run_all_sequential
    fi
}

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [EXPERIMENT]

Options:
    --dry-run       Print commands without executing
    --parallel N    Run up to N experiments in parallel (default: 1)
                    Recommended: 3-4 for A100 80GB, 2 for A100 40GB
    --trials N      Number of trials per config (default: 30)
    --output DIR    Output directory (default: ./results)
    -h, --help      Show this help message

Experiments (Optimized Protocol - ~1,740 total runs, down from ~3,240):
    main-grid       Strategy comparison: 5 strategies × 3 agents × 30 = 450 runs
    ablation        Mechanism ablation: 8 configs × 30 = 240 runs
    scaling         Agent scaling: 2 strategies × 4 agents × 30 = 240 runs
    escalation      Escalation impact: 2 configs × 5 strategies × 30 = 300 runs
    difficulty      Difficulty scaling: 2 levels × 5 strategies × 30 = 300 runs
    pressure-field-only  Re-run pressure_field only (bug fix): ~330 runs
    all             Run all experiments (default)

Strategies:
    All experiments test ALL 5: pressure_field, hierarchical, sequential, random, conversation
    Exception: scaling only tests pressure_field vs hierarchical (agent count focus)

Parallelization:
    With --parallel N, up to N experiments run concurrently.
    Each experiment logs to $OUTPUT_DIR/<timestamp>/logs/<experiment>.log

    GPU Memory Guidelines:
      - A100 80GB: --parallel 5 (run all experiments simultaneously)
      - A100 40GB: --parallel 3
      - RTX 4090:  --parallel 2

    Estimated Runtime (30 trials, A100 80GB):
      - Sequential:     ~4-5 hours
      - --parallel 5:   ~1 hour (limited by longest experiment)

Examples:
    $0                          # Run all experiments (~1,740 runs)
    $0 --parallel 5 all         # Run all 5 experiments in parallel (A100 80GB)
    $0 main-grid                # Run only main grid experiment
    $0 --dry-run all            # Preview what would run
    $0 --trials 10 ablation     # Run ablation with 10 trials (quick test)

Environment Variables:
    VLLM_HOST           vLLM server URL (default: http://localhost:8000)
    LATIN_EXPERIMENT    Path to binary (default: ./latin-experiment)
    PARALLEL            Default parallel jobs (overridden by --parallel)

vLLM Setup:
    # Install vLLM
    pip install vllm

    # Start server with a single model (simplest)
    vllm serve Qwen/Qwen2.5-7B --dtype bfloat16 --gpu-memory-utilization 0.95

    # Or for smaller GPUs
    vllm serve Qwen/Qwen2.5-3B --dtype float16
EOF
}

# Allow sourcing this script for parallel execution
if [[ "${1:-}" == "--source-only" ]]; then
    return 0 2>/dev/null || exit 0
fi

# Internal mode: run a single experiment function directly (used by parallel jobs)
# Expects RESULTS_DIR to be set via environment
run_internal() {
    local experiment="$1"
    case $experiment in
        main-grid)
            run_main_grid parallel
            ;;
        ablation)
            run_ablation parallel
            ;;
        scaling)
            run_scaling parallel
            ;;
        escalation)
            run_escalation parallel
            ;;
        difficulty)
            run_difficulty parallel
            ;;
        pressure-field-only)
            run_pressure_field_only parallel
            ;;
        *)
            log_error "Unknown internal experiment: $experiment"
            exit 1
            ;;
    esac
}

if [[ "${1:-}" == "--internal" ]]; then
    run_internal "${2:-}"
    exit $?
fi

main() {
    local experiment="all"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --parallel)
                PARALLEL="$2"
                if [[ ! "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]]; then
                    log_error "--parallel requires a positive integer"
                    exit 1
                fi
                shift 2
                ;;
            --trials)
                TRIALS="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            main-grid|ablation|scaling|escalation|difficulty|pressure-field-only|all)
                experiment="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    check_prerequisites
    setup_output_dir

    case $experiment in
        main-grid)
            run_main_grid
            ;;
        ablation)
            run_ablation
            ;;
        scaling)
            run_scaling
            ;;
        escalation)
            run_escalation
            ;;
        difficulty)
            run_difficulty
            ;;
        pressure-field-only)
            run_pressure_field_only
            ;;
        all)
            run_all
            ;;
    esac
}

main "$@"
