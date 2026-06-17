#!/usr/bin/env python
"""Train a LeRobot ACT policy on prepared CALVIN data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config_snapshot
from src.train_utils import summarize_training_config, train_act


def main() -> None:
    """Run ACT training from a YAML config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_A_only.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    args = parser.parse_args()

    overrides = list(args.override)
    if args.device:
        overrides.append(f"device={args.device}")

    config = load_config(args.config, overrides=overrides)
    snapshot_path = save_config_snapshot(config)
    print(
        json.dumps(
            {
                "status": "starting_training",
                "config_snapshot": str(snapshot_path),
                **summarize_training_config(config),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    try:
        result = train_act(config)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    result["config_snapshot"] = str(snapshot_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
