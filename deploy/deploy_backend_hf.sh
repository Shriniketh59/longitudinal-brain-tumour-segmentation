#!/usr/bin/env bash
# Deploy the backend to a Hugging Face Space (Docker).
# Usage:  hf auth login          (once, with a token that has write access)
#         deploy/deploy_backend_hf.sh <hf-username>/<space-name>
set -euo pipefail

SPACE="${1:?usage: deploy_backend_hf.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Stage on the main disk: /tmp is often a small RAM disk and the weights are ~850 MB.
mkdir -p "$ROOT/backend/data"
STAGE="$(mktemp -d -p "$ROOT/backend/data" hf-stage.XXXX)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging backend in $STAGE"
cp -r "$ROOT/backend/app" "$STAGE/app"
mkdir -p "$STAGE/scripts" "$STAGE/weights"
cp "$ROOT/backend/scripts/__init__.py" "$ROOT/backend/scripts/download_brats.py" "$STAGE/scripts/"
cp "$ROOT/backend/requirements.txt" "$ROOT/backend/Dockerfile" "$ROOT/backend/.dockerignore" "$STAGE/"
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"
# Weights + their metrics files; skip training resume checkpoints.
find "$ROOT/backend/weights" -maxdepth 1 -type f \( -name '*.pt' -o -name '*.json' \) ! -name '*resume*' -exec cp {} "$STAGE/weights/" \;
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +

echo "Files to upload:"
du -sh "$STAGE"/* | sed 's/^/  /'

hf repos create "$SPACE" --repo-type space --space-sdk docker --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy NeuroScan AI backend"

echo
echo "Uploaded. The Space now builds (about 5-10 minutes): https://huggingface.co/spaces/$SPACE"
echo "API URL once running: https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_' '[:lower:]-').hf.space"
