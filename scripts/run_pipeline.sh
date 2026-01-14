#!/usr/bin/env bash
set -euo pipefail
python -m src.pipeline --config configs/default.yaml "$@"
