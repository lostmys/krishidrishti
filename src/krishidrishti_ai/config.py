from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration, optionally extending another project config."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    parent_path = config.pop("extends", None)
    if parent_path:
        base = load_config(path.parent / parent_path)
        config = _merge(base, config)
    project_root = next((candidate for candidate in (path.parent, *path.parents) if (candidate / "pyproject.toml").is_file()), path.parent.parent)
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        if not candidate.is_absolute():
            config["paths"][key] = str((project_root / candidate).resolve())
    return config
