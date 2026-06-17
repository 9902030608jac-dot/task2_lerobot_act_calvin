#!/usr/bin/env python
"""Inspect raw CALVIN data directories without assuming one fixed format."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import ensure_dir, resolve_path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
KNOWN_IMAGE_KEYS = {"rgb_static", "rgb_gripper", "image", "observation.image", "observation.images.rgb_static"}
KNOWN_STATE_KEYS = {"robot_obs", "state", "observation.state"}
KNOWN_ACTION_KEYS = {"rel_actions", "actions", "action"}
KNOWN_TASK_KEYS = {"task", "language", "lang", "instruction"}


def jsonable(value: Any) -> Any:
    """Convert numpy/python values to JSON-friendly summaries."""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in list(value)[:10]]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:200]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


def shallow_read(path: Path) -> dict[str, Any]:
    """Shallow-read a supported file and return structural information."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                keys = list(data.keys())
                return {
                    "path": str(path),
                    "type": "npz",
                    "keys": keys,
                    "fields": {key: jsonable(data[key]) for key in keys[:20]},
                }
        if suffix == ".npy":
            array = np.load(path, allow_pickle=True)
            return {"path": str(path), "type": "npy", "array": jsonable(array)}
        if suffix in {".json", ".jsonl"}:
            with path.open("r", encoding="utf-8") as file:
                if suffix == ".jsonl":
                    rows = [json.loads(line) for _, line in zip(range(5), file) if line.strip()]
                    keys = sorted({key for row in rows if isinstance(row, dict) for key in row})
                    return {"path": str(path), "type": "jsonl", "num_preview_rows": len(rows), "keys": keys}
                data = json.load(file)
                keys = list(data.keys()) if isinstance(data, dict) else []
                return {"path": str(path), "type": "json", "keys": keys, "preview": jsonable(data)}
        if suffix in {".pkl", ".pickle"}:
            with path.open("rb") as file:
                data = pickle.load(file)
            keys = list(data.keys()) if isinstance(data, dict) else []
            return {"path": str(path), "type": "pkl", "keys": keys, "preview": jsonable(data)}
        if suffix in IMAGE_EXTS:
            return {"path": str(path), "type": "image", "suffix": suffix}
    except Exception as exc:
        return {"path": str(path), "type": suffix.lstrip("."), "error": str(exc)}
    return {"path": str(path), "type": suffix.lstrip(".") or "unknown"}


def estimate_episodes(env_dir: Path, files: list[Path]) -> dict[str, Any]:
    """Estimate episode count from common CALVIN/LeRobot structures."""
    ep_start_end = env_dir / "ep_start_end_ids.npy"
    if ep_start_end.exists():
        try:
            starts = np.load(ep_start_end, allow_pickle=True)
            return {"method": "ep_start_end_ids.npy", "episode_count": int(len(starts))}
        except Exception as exc:
            return {"method": "ep_start_end_ids.npy", "error": str(exc)}
    episodes_jsonl = env_dir / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        count = sum(1 for line in episodes_jsonl.open("r", encoding="utf-8") if line.strip())
        return {"method": "meta/episodes.jsonl", "episode_count": count}
    episode_dirs = [path for path in env_dir.iterdir() if path.is_dir() and "episode" in path.name.lower()]
    if episode_dirs:
        return {"method": "episode directories", "episode_count": len(episode_dirs)}
    episode_files = [path for path in files if "episode" in path.stem.lower()]
    return {"method": "episode-like filenames", "episode_count": len(episode_files)}


def classify_fields(keys: list[str]) -> dict[str, list[str]]:
    """Classify possible image/state/action/task fields."""
    key_set = set(keys)
    return {
        "possible_image_fields": sorted(key_set & KNOWN_IMAGE_KEYS),
        "possible_state_fields": sorted(key_set & KNOWN_STATE_KEYS),
        "possible_action_fields": sorted(key_set & KNOWN_ACTION_KEYS),
        "possible_task_fields": sorted(key_set & KNOWN_TASK_KEYS),
    }


def inspect_env(data_root: Path, env: str, max_files: int) -> dict[str, Any]:
    """Inspect one environment directory."""
    candidates = [data_root / f"calvin_{env}", data_root / f"split{env}"]
    env_dir = next((path for path in candidates if path.exists()), candidates[0])
    result: dict[str, Any] = {
        "env": env,
        "path": str(env_dir),
        "exists": env_dir.exists(),
        "candidate_paths": [str(path) for path in candidates],
    }
    if not env_dir.exists():
        result["status"] = "missing"
        return result

    files = sorted(path for path in env_dir.rglob("*") if path.is_file())
    suffix_counts = Counter(path.suffix.lower() or "<no_suffix>" for path in files)
    preview_files = files[:max_files]
    samples = [shallow_read(path) for path in preview_files]
    all_keys = sorted({key for sample in samples for key in sample.get("keys", [])})
    result.update(
        {
            "status": "empty" if len(files) == 0 else "ok",
            "num_files": len(files),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "episode_estimate": estimate_episodes(env_dir, files),
            "sampled_files": samples,
            "discovered_keys": all_keys,
            **classify_fields(all_keys),
        }
    )
    return result


def main() -> None:
    """Inspect CALVIN raw directories and save JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--envs", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--max_files_per_env", type=int, default=20)
    parser.add_argument("--output", default="report_assets/calvin_inspection.json")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    report = {
        "data_root": str(data_root),
        "envs": [inspect_env(data_root, env, args.max_files_per_env) for env in args.envs],
    }
    output = resolve_path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CALVIN inspection saved: {output}")
    for env_report in report["envs"]:
        status = env_report["status"]
        episode_count = env_report.get("episode_estimate", {}).get("episode_count", "NA")
        print(
            f"env {env_report['env']}: {status}, files={env_report.get('num_files', 0)}, "
            f"episodes~={episode_count}, image={env_report.get('possible_image_fields', [])}, "
            f"state={env_report.get('possible_state_fields', [])}, "
            f"action={env_report.get('possible_action_fields', [])}, "
            f"task={env_report.get('possible_task_fields', [])}"
        )


if __name__ == "__main__":
    main()
