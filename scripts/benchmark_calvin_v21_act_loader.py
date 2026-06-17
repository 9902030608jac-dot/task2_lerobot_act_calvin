#!/usr/bin/env python
"""Benchmark the lightweight CALVIN v2.1 loader with LeRobot ACTPolicy."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader

from src.calvin_v21_act_dataset import CalvinV21ActDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        action="append",
        default=None,
        help="Raw v2.1 split directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        default=None,
        help="Environment id for each split directory. Defaults to split directory names.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=128)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--shuffle-frames", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_policy(chunk_size: int, device: str) -> ACTPolicy:
    config = ACTConfig(
        input_features={
            "observation.images.rgb_static": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 200, 200),
            ),
            "observation.images.rgb_gripper": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 84, 84),
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config).to(device).train()


def move_train_batch_to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key.startswith("observation.") or key in {"action", "action_is_pad"}
    }


def main() -> None:
    args = parse_args()
    split_dirs = args.split_dir or ["data/raw/xiaoma26_calvin_lerobot/splitA"]
    persistent_workers = args.num_workers > 0 and not args.no_persistent_workers

    dataset_start = time.perf_counter()
    dataset = CalvinV21ActDataset(
        split_dirs,
        env_ids=args.env_id,
        chunk_size=args.chunk_size,
        max_episodes_per_split=args.max_episodes,
        cache_size=args.cache_size,
    )
    print(
        json.dumps(
            {
                "status": "dataset_ready",
                "split_dirs": split_dirs,
                "num_samples": len(dataset),
                "num_episodes": len(dataset.episode_to_indices),
                "dataset_init_sec": round(time.perf_counter() - dataset_start, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle_frames,
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": args.device == "cuda",
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = args.device

    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    policy = make_policy(args.chunk_size, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-5)

    fetch_times: list[float] = []
    step_times: list[float] = []
    iterator = iter(loader)
    max_steps = min(args.steps, len(loader))
    print(
        json.dumps(
            {
                "status": "training_steps_start",
                "steps": max_steps,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "shuffle_frames": args.shuffle_frames,
                "persistent_workers": persistent_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "device": device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for step in range(max_steps):
        fetch_start = time.perf_counter()
        batch = next(iterator)
        fetch_times.append(time.perf_counter() - fetch_start)

        batch = move_train_batch_to_device(batch, device)
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _metrics = policy(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        print(
            json.dumps(
                {
                    "step": step + 1,
                    "fetch_sec": round(fetch_times[-1], 4),
                    "gpu_step_sec": round(step_times[-1], 4),
                    "loss": round(float(loss.detach().cpu()), 6),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    denom = max(1, len(step_times) - 1)
    total_ex_first = sum(fetch_times[1:]) + sum(step_times[1:])
    result = {
        "status": "ok",
        "avg_fetch_sec_excl_first": round(sum(fetch_times[1:]) / denom, 4),
        "avg_gpu_step_sec_excl_first": round(sum(step_times[1:]) / denom, 4),
        "avg_total_sec_excl_first": round(total_ex_first / denom, 4),
        "samples_per_sec_excl_first": round(args.batch_size * denom / total_ex_first, 2)
        if total_ex_first > 0
        else None,
    }
    if device == "cuda":
        result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
