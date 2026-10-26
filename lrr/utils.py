from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def load_bundle(path: str | Path) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("bundle root must be a dictionary")
    schedule = value.get("training", {}).get("retain_negative_schedule")
    if isinstance(schedule, dict):
        from lrr.sampling.homogeneous import RetainNegativeSchedule
        value["training"]["retain_negative_schedule"] = RetainNegativeSchedule(
            schedule, base_path=Path(path).resolve().parent)
    return value


def save_result(path: str | Path, result: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, target)


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
