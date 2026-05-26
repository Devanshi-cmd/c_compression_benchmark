#!/usr/bin/env bash
# fetch_data.sh — clone C source repositories for training corpus
# Usage: bash data/fetch_data.sh
# Clones into data/repos/ with --depth=1 to minimize disk usage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_DIR="${SCRIPT_DIR}/repos"

mkdir -p "${REPOS_DIR}"

REPOS=(
    "https://github.com/torvalds/linux"
    "https://github.com/git/git"
    "https://github.com/redis/redis"
    "https://github.com/nginx/nginx"
    "https://github.com/sqlite/sqlite"
    "https://github.com/curl/curl"
    "https://github.com/FFmpeg/FFmpeg"
    "https://github.com/postgres/postgres"
    "https://github.com/python/cpython"
    "https://github.com/facebook/zstd"
)

total=${#REPOS[@]}
success=0
failed=0

for repo_url in "${REPOS[@]}"; do
    repo_name=$(basename "${repo_url}")
    target_dir="${REPOS_DIR}/${repo_name}"

    if [ -d "${target_dir}/.git" ]; then
        echo "[SKIP] ${repo_name} already cloned at ${target_dir}"
        ((success++)) || true
        continue
    fi

    echo "[CLONE] ${repo_url} → ${target_dir}"
    if git clone --depth=1 --single-branch "${repo_url}" "${target_dir}" 2>&1; then
        echo "[OK] ${repo_name} cloned successfully"
        ((success++)) || true
    else
        echo "[FAIL] Failed to clone ${repo_url}" >&2
        ((failed++)) || true
    fi
done

echo ""
echo "================================================================"
echo "Fetch complete: ${success}/${total} repos cloned, ${failed} failed"
echo "Repos directory: ${REPOS_DIR}"
echo "================================================================"

if [ "${failed}" -gt 0 ]; then
    echo "WARNING: Some repos failed to clone. Continuing with available data."
    exit 0
fi
