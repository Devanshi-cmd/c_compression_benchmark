# Dockerfile — BitNet C Compression Benchmark
# Base: CUDA 12.8 + Ubuntu 22.04
# RTX 5090 (Blackwell, sm_120) requires PyTorch 2.6+ with CUDA 12.6+
# Salad platform compatible: no TTY, auto-resume on spot interruption

FROM --platform=linux/amd64 nvidia/cuda:12.8.1-runtime-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        git \
        curl \
        wget \
        zstd \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.8) ───────────────────────────────────────────────────────
# PyTorch 2.6+ required for RTX 5090 (Blackwell sm_120) support
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# ── Project dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Environment variables ─────────────────────────────────────────────────────
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=8
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
# Salad: write results to /workspace/results (mount a volume here to persist)
ENV RESULTS_DIR=/workspace/results

# ── Working directory & project files ────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace/

# Runtime directories (data/repos and checkpoints come from mounted volumes)
RUN mkdir -p /workspace/data/repos \
             /workspace/results \
             /workspace/checkpoints/model1 \
             /workspace/checkpoints/model2 \
 && chmod +x /workspace/data/fetch_data.sh \
             /workspace/entrypoint.sh

# ── Health check ─────────────────────────────────────────────────────────────
# Salad uses this to verify the container started correctly
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# ── Default command ───────────────────────────────────────────────────────────
# entrypoint.sh: fetch data → prepare → train → benchmark (auto-resumes)
CMD ["/workspace/entrypoint.sh"]
