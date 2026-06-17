"""Lightweight CALVIN v2.1 parquet dataset adapter for LeRobot ACT."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from src.paths import resolve_path


@dataclass(frozen=True)
class FrameRef:
    """One training sample location inside a CALVIN episode parquet."""

    env_id: str
    episode_index: int
    frame_index: int
    parquet_path: Path


class CalvinV21ActDataset:
    """Read CALVIN/LeRobot v2.1 parquet episodes and emit ACTPolicy batches.

    This adapter intentionally keeps the ACT algorithm in LeRobot while avoiding
    the heavy HuggingFace Datasets cache/index path for the course-provided
    `xiaoma26/calvin-lerobot` v2.1 splits.
    """

    def __init__(
        self,
        split_dirs: str | Path | Iterable[str | Path],
        *,
        env_ids: Iterable[str] | None = None,
        chunk_size: int = 100,
        max_episodes_per_split: int | None = None,
        cache_size: int = 32,
    ) -> None:
        if isinstance(split_dirs, (str, Path)):
            split_paths = [resolve_path(split_dirs)]
        else:
            split_paths = [resolve_path(path) for path in split_dirs]
        if not split_paths:
            raise ValueError("At least one split directory is required.")

        env_list = list(env_ids or [path.name for path in split_paths])
        if len(env_list) != len(split_paths):
            raise ValueError("env_ids must have the same length as split_dirs.")

        self.split_paths = split_paths
        self.env_ids = env_list
        self.chunk_size = int(chunk_size)
        self.cache_size = int(cache_size)
        self.features = _policy_features()
        self.frames: list[FrameRef] = []
        self.episode_to_indices: dict[tuple[str, int], list[int]] = {}
        self._episode_cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self._task_maps: dict[str, dict[int, str]] = {}

        for split_path, env_id in zip(self.split_paths, self.env_ids, strict=True):
            self._task_maps[env_id] = _load_task_map(split_path)
            self._index_split(split_path, env_id, max_episodes_per_split)
        if not self.frames:
            raise ValueError(f"No frames indexed from split dirs: {self.split_paths}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.frames[index]
        episode = self._load_episode(ref.parquet_path)
        row = episode.iloc[ref.frame_index]
        action, action_is_pad = self._build_action_chunk(episode, ref.frame_index)
        task_index = _row_int(row, "task_index", -1)
        task = self._task_maps.get(ref.env_id, {}).get(task_index, "")
        return {
            "observation.images.rgb_static": _decode_image(row["image"]),
            "observation.images.rgb_gripper": _decode_image(row["wrist_image"]),
            "observation.state": np.asarray(row["state"], dtype=np.float32).copy(),
            "action": action,
            "action_is_pad": action_is_pad,
            "env_id": ref.env_id,
            "episode_index": np.int64(ref.episode_index),
            "frame_index": np.int64(ref.frame_index),
            "task_index": np.int64(task_index),
            "task": task,
        }

    def _index_split(
        self,
        split_path: Path,
        env_id: str,
        max_episodes_per_split: int | None,
    ) -> None:
        episodes_path = split_path / "meta" / "episodes.jsonl"
        data_dir = split_path / "data"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing v2.1 episodes metadata: {episodes_path}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing v2.1 data directory: {data_dir}")

        with episodes_path.open("r", encoding="utf-8") as file:
            for count, line in enumerate(file):
                if max_episodes_per_split is not None and count >= max_episodes_per_split:
                    break
                episode = json.loads(line)
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                parquet_path = _find_episode_parquet(data_dir, episode_index)
                start_index = len(self.frames)
                key = (env_id, episode_index)
                indices = list(range(start_index, start_index + length))
                self.episode_to_indices[key] = indices
                self.frames.extend(
                    FrameRef(env_id, episode_index, frame_index, parquet_path)
                    for frame_index in range(length)
                )

    def _load_episode(self, parquet_path: Path) -> pd.DataFrame:
        cached = self._episode_cache.get(parquet_path)
        if cached is not None:
            self._episode_cache.move_to_end(parquet_path)
            return cached

        frame = _read_episode_frame(parquet_path)
        self._episode_cache[parquet_path] = frame
        if len(self._episode_cache) > self.cache_size:
            self._episode_cache.popitem(last=False)
        return frame

    def _build_action_chunk(self, episode: pd.DataFrame, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        actions = episode["actions"].to_numpy()
        end = min(len(actions), frame_index + self.chunk_size)
        chunk = [np.asarray(action, dtype=np.float32) for action in actions[frame_index:end]]
        pad_count = self.chunk_size - len(chunk)
        if pad_count > 0:
            chunk.extend([chunk[-1].copy() for _ in range(pad_count)])
        action_is_pad = np.zeros((self.chunk_size,), dtype=bool)
        if pad_count > 0:
            action_is_pad[-pad_count:] = True
        return np.stack(chunk, axis=0).copy(), action_is_pad


def _find_episode_parquet(data_dir: Path, episode_index: int) -> Path:
    filename = f"episode_{episode_index:06d}.parquet"
    matches = list(data_dir.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"Missing episode parquet under {data_dir}: {filename}")
    if len(matches) > 1:
        raise ValueError(f"Multiple parquet files found for episode {episode_index}: {matches}")
    return matches[0]


def _load_task_map(split_path: Path) -> dict[int, str]:
    tasks_path = split_path / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    task_map: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            task = json.loads(line)
            task_map[int(task["task_index"])] = str(task["task"])
    return task_map


def _read_episode_frame(parquet_path: Path) -> pd.DataFrame:
    required = ["image", "wrist_image", "state", "actions"]
    optional = ["task_index", "timestamp", "source_frame_index", "source_episode_index"]
    try:
        return pd.read_parquet(parquet_path, columns=required + optional)
    except Exception:
        return pd.read_parquet(parquet_path, columns=required)


def _row_int(row: Any, key: str, default: int) -> int:
    if key not in row:
        return default
    value = row[key]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            image = Image.open(BytesIO(value["bytes"])).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        elif value.get("path"):
            image = Image.open(value["path"]).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        else:
            raise ValueError("Image dict has neither bytes nor path.")
    else:
        array = np.asarray(value, dtype=np.float32)
        if array.max(initial=0) > 1.0:
            array = array / 255.0

    if array.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 0, 1)).copy()


def _policy_features() -> dict[str, Any]:
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
    except ImportError:
        return {}
    return {
        "observation.images.rgb_static": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 200, 200),
        ),
        "observation.images.rgb_gripper": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 84, 84),
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
