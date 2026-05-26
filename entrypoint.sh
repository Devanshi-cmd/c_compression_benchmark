#!/usr/bin/env bash
# entrypoint.sh — Salad platform entrypoint
# Handles: data prep → train → benchmark
# Auto-resumes: train.py detects checkpoints on restart
# No interactive TTY required

set -euo pipefail
cd /workspace

echo "================================================================"
echo "BitNet C Compression Benchmark — Starting"
echo "Date: $(date -u)"
echo "================================================================"

# Step 1: Fetch data (skip if repos already exist)
if [ ! -d "data/repos/linux" ] && [ ! -d "data/repos/git" ]; then
    echo ""
    echo "[1/3] Fetching training data..."
    bash data/fetch_data.sh
else
    echo "[1/3] Training repos already present, skipping fetch."
fi

# Step 2: Prepare data (skip if encoded.dat exists)
if [ ! -f "data/encoded.dat" ]; then
    echo ""
    echo "[2/3] Preparing training corpus..."
    python data/prepare_data.py
else
    echo "[2/3] encoded.dat already exists, skipping preparation."
fi

# Step 3: Run arithmetic coder self-test
echo ""
echo "[self-test] Arithmetic coder..."
python arithmetic_coder.py

# Step 4: Train both models (auto-resumes from checkpoints)
echo ""
echo "[3/3] Training models (auto-resumes if checkpoints exist)..."
python train.py --model both

# Step 5: Benchmark
echo ""
echo "[4/4] Running benchmark..."
python benchmark.py --model both

echo ""
echo "================================================================"
echo "Complete. Results in /workspace/results/"
echo "================================================================"
