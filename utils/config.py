"""Helpers for loading project configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def load_config(path: str | None = None) -> dict:
    """Load the YAML config file."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def get_config_value(config: dict, *keys, default=None):
    """Read a nested config value and fall back to default if it is missing."""
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
