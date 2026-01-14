from __future__ import annotations

from pathlib import Path
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict:
    root = project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (root / path).resolve()
