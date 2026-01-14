#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ -d "models/sam2/segment-anything-2-real-time" ]; then
  python -m pip install -e models/sam2/segment-anything-2-real-time
else
  echo "SAM2 repo not found. Run scripts/download_assets.sh first."
fi
