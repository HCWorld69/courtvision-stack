# CourtVision Stack: Basketball AI End-to-End Pipeline

CourtVision Stack is a production-style project built from the Roboflow Colab notebook:
https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb

It turns the notebook into a structured repo with configs, scripts, and modular code so a
cloned copy can install dependencies, download assets, and run inference end to end.

## What this pipeline does
- Player detection with RF-DETR
- SAM2 mask-based tracking across video
- Team clustering from jersey colors
- Jersey number OCR with validation
- Court keypoint detection and coordinate mapping
- Shot event detection (jump shot, layup/dunk, ball-in-basket)

## Project layout
```text
courtvision-stack/
  assets/
    fonts/
    examples/
  configs/
    default.yaml
    rosters.yaml
  data/
    raw/videos/
    raw/rosters/
    interim/frames/
    processed/
  models/
    sam2/
  notebooks/
    basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
  outputs/
    detections/
    tracks/
    teams/
    ocr/
    court/
    shots/
  scripts/
    download_assets.py
    download_assets.ps1
    download_assets.sh
    setup_env.ps1
    setup_env.sh
    run_pipeline.ps1
    run_pipeline.sh
  src/
    detection/
    tracking/
    team/
    ocr/
    court/
    shots/
    utils/
    config.py
    pipeline.py
```

## Quickstart
1. Copy the environment file and set your Roboflow key:
   - `copy .env.example .env` (Windows)
   - `cp .env.example .env` (macOS/Linux)

2. Install dependencies:
   - Windows: `scripts\setup_env.ps1`
   - macOS/Linux: `bash scripts/setup_env.sh`

3. Download assets, sample videos, and SAM2 checkpoint:
   - Windows: `scripts\download_assets.ps1`
   - macOS/Linux: `bash scripts/download_assets.sh`

4. Install SAM2 after the repo is downloaded:
   - `python -m pip install -e models/sam2/segment-anything-2-real-time`

5. Run the pipeline:
   - Windows: `scripts\run_pipeline.ps1 --step all`
   - macOS/Linux: `bash scripts/run_pipeline.sh --step all`

## Steps and outputs
- detection: `outputs/detections/<video>-detection.mp4`
- tracking: `outputs/tracks/<video>-mask.mp4`
- teams: `outputs/teams/<video>-teams.mp4`
- ocr: `outputs/ocr/<video>-validated-numbers.mp4`
- court: `outputs/court/<video>-map.mp4`
- shots: `outputs/shots/shot_events.jsonl`

## Configuration
- `configs/default.yaml` controls model IDs, thresholds, and paths.
- `configs/rosters.yaml` stores team rosters and colors.

## Notes
- Heavy files are not stored in the repo. The download script fetches SAM2 weights, fonts,
  and sample videos when you run it.
- `inference-gpu` requires a CUDA-capable GPU. For CPU-only machines, replace it with
  `inference` in `requirements.txt`.
- `ffmpeg` is required for optional video compression; install it if needed.

## Credits
- Original notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb
- SAM2: https://github.com/facebookresearch/sam2
- Real-time SAM2: https://github.com/Gy920/segment-anything-2-real-time
- Roboflow Sports: https://github.com/roboflow/sports
