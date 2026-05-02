#!/usr/bin/env python3
"""
Small project-wide utility helpers for the modular EyeMap MVP.
These functions keep directory creation, configuration loading, logging, and timestamps out of the computer vision modules.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.utils import timestamp_id


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file and return an empty dict if it is blank."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def make_run_dir(output_root: str | Path) -> Path:
    """Create a timestamped output run directory."""
    return ensure_dir(Path(output_root) / timestamp_id())


def append_log(path: str | Path, message: str) -> None:
    """Append one text line to a log file."""
    log_path = Path(path)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")
