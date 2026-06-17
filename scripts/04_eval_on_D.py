#!/usr/bin/env python
"""Evaluate a trained ACT policy on unseen CALVIN environment D."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, save_config_snapshot
from src.eval_utils import evaluate_act, summarize_eval_config


def main() -> None:
    """Run zero-shot evaluation on D."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_A_on_D.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--mode", choices=["offline", "rollout"], default="offline")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_steps_per_episode", type=int, default=360)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    args = parser.parse_args()

    overrides = list(args.override)
    if args.checkpoint_path:
        overrides.append(f"checkpoint_path={args.checkpoint_path}")
    if args.dataset_path:
        overrides.append(f"dataset_path={args.dataset_path}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.device:
        overrides.append(f"device={args.device}")
    if args.num_episodes is not None:
        overrides.append(f"max_eval_episodes={args.num_episodes}")

    config = load_config(args.config, overrides=overrides)
    snapshot_path = save_config_snapshot(config)
    print(
        json.dumps(
            {
                "status": "starting_eval",
                "mode": args.mode,
                "config_snapshot": str(snapshot_path),
                **summarize_eval_config(config),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    try:
        result = evaluate_act(
            config,
            mode=args.mode,
            num_episodes=int(args.num_episodes or config["max_eval_episodes"]),
            max_steps_per_episode=int(args.max_steps_per_episode),
        )
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    result["config_snapshot"] = str(snapshot_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
