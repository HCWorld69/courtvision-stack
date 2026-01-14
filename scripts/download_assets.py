import argparse
import subprocess
import sys
from pathlib import Path
import urllib.request

SAM2_REPO = "https://github.com/Gy920/segment-anything-2-real-time.git"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
VIDEOS_GDRIVE = "https://drive.google.com/drive/folders/1eDJYqQ77Fytz15tKGdJCMeYSgmoQ-2-H"
FONTS_GDRIVE = "https://drive.google.com/drive/folders/1RBjpI5Xleb58lujeusxH0W5zYMMA4ytO"


def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Skipping existing file: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-fonts", action="store_true")
    args = parser.parse_args()

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is required. Run scripts/setup_env.ps1 or scripts/setup_env.sh first.")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    sam2_dir = repo_root / "models" / "sam2" / "segment-anything-2-real-time"
    checkpoints_dir = sam2_dir / "checkpoints"
    videos_dir = repo_root / "data" / "raw" / "videos"
    fonts_dir = repo_root / "assets" / "fonts"

    if not args.skip_sam2:
        if not sam2_dir.exists():
            sam2_dir.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", SAM2_REPO, str(sam2_dir)])
        download_file(SAM2_CHECKPOINT_URL, checkpoints_dir / "sam2.1_hiera_large.pt")

    python = sys.executable
    if not args.skip_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)
        run([python, "-m", "gdown", VIDEOS_GDRIVE, "--folder", "-O", str(videos_dir)])

    if not args.skip_fonts:
        fonts_dir.mkdir(parents=True, exist_ok=True)
        run([python, "-m", "gdown", FONTS_GDRIVE, "--folder", "-O", str(fonts_dir)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
