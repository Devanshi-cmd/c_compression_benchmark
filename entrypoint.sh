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

# Step 1: Fetch training data (R2 first, then git clone fallback)
if [ ! -f "data/encoded.dat" ]; then
    echo ""
    echo "[1/3] Training data not found locally. Trying R2..."
    python - <<'PYEOF'
import os, sys
sys.path.insert(0, '/workspace')
try:
    from dotenv import load_dotenv
    load_dotenv('/workspace/.env')
except ImportError:
    pass
if os.environ.get('R2_ACCESS_KEY_ID') and os.environ.get('R2_BUCKET'):
    from r2_storage import download_training_data
    ok = download_training_data('/workspace/data')
    sys.exit(0 if ok else 1)
else:
    print("[R2] No credentials — falling back to git clone.")
    sys.exit(1)
PYEOF
    R2_EXIT=$?
    if [ $R2_EXIT -ne 0 ]; then
        echo "[1/3] R2 unavailable or missing. Fetching repos from git..."
        if [ ! -d "data/repos/linux" ] && [ ! -d "data/repos/git" ]; then
            bash data/fetch_data.sh
        fi
        echo "[2/3] Preparing training corpus..."
        python data/prepare_data.py
    else
        echo "[1/3] Training data downloaded from R2."
    fi
else
    echo "[1/3] Training data already present locally, skipping."
fi

# Step 2: Run arithmetic coder self-test
echo ""
echo "[self-test] Arithmetic coder..."
python arithmetic_coder.py

# Step 3: Train both models (auto-resumes from checkpoints; auto-shuts down when done)
echo ""
echo "[2/3] Training models (auto-resumes if checkpoints exist)..."
python train.py --model both

# Step 4: Benchmark
echo ""
echo "[3/3] Running benchmark..."
python benchmark.py --model both

echo ""
echo "================================================================"
echo "Complete. Results in /workspace/results/"
echo "================================================================"
