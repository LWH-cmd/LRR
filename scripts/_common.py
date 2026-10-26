from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lrr.utils import load_bundle, load_config


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML run configuration")
    parser.add_argument("--bundle", help="Override configured tensor bundle path")
    parser.add_argument("--checkpoint", help="Override configured pretrained checkpoint path")
    parser.add_argument("--output", help="Override configured output path")
    return parser.parse_args()


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    bundle_path = _resolve(args.bundle or config["bundle_path"], config_path.parent)
    output_path = _resolve(args.output or config.get("output_path", "outputs/run.pt"), config_path.parent)
    bundle = load_bundle(bundle_path)
    checkpoint = args.checkpoint or config.get("pretrained_checkpoint_path")
    if checkpoint:
        checkpoint_path = _resolve(checkpoint, config_path.parent)
        if bundle.get('checkpoint_sha256'):
            import hashlib
            digest=hashlib.sha256()
            with checkpoint_path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=bundle['checkpoint_sha256']:
                raise ValueError('checkpoint differs from cached bundle teacher targets; rebuild the bundle')
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError("pretrained checkpoint must be a state dictionary or payload dictionary")
        state = loaded.get("model_state", loaded.get("state_dict", loaded))
        if not isinstance(state, dict) or not state:
            raise KeyError("checkpoint must contain model_state or state_dict")
        bundle["backbone_state_dict"] = state
    if "backbone_state_dict" not in bundle:
        raise KeyError("provide pretrained_checkpoint_path or backbone_state_dict in the bundle")
    return config, bundle, output_path
