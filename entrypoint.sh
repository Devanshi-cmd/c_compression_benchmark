#!/usr/bin/env bash
# entrypoint.sh — Salad platform entrypoint
# Training data is prepared locally and uploaded to R2 before running this.
# This script: fetches data from R2 → trains → benchmarks.
# Set EPOCHS env var in Salad to control training length (default: 20).

set -euo pipefail
cd /workspace

echo "================================================================"
echo "BitNet C Compression Benchmark — Starting"
echo "Date: $(date -u)"
echo "EPOCHS=${EPOCHS:-20}"
echo "================================================================"

# Step 1: Fetch training data from R2
echo ""
echo "[1/3] Fetching training data from R2..."
python - <<'PYEOF'
import sys, os
sys.path.insert(0, '/workspace')
try:
    from dotenv import load_dotenv
    load_dotenv('/workspace/.env')
except ImportError:
    pass

if not (os.environ.get('R2_ACCESS_KEY_ID') and os.environ.get('R2_BUCKET')):
    print("ERROR: R2 credentials not set. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT as env vars in Salad.")
    sys.exit(1)

from r2_storage import download_training_data
ok = download_training_data('/workspace/data')
if not ok:
    print("ERROR: Failed to download training data from R2.")
    sys.exit(1)
print("Training data ready.")
PYEOF

# Step 2: Run arithmetic coder self-test
echo ""
echo "[self-test] Arithmetic coder..."
python arithmetic_coder.py

# Step 3: Train both models (auto-resumes from checkpoints; exits 0 when all epochs done)
echo ""
echo "[2/3] Training models..."
python train.py --model both

# Step 4: Benchmark
echo ""
echo "[3/3] Running benchmark..."
python benchmark.py --model both

echo ""
echo "================================================================"
echo "Complete. Results in /workspace/results/"
echo "================================================================"
