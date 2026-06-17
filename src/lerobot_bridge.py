"""Bridge utilities between prepared CALVIN indexes and LeRobot-style training."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calvin_data import ACTION_KEYS, IMAGE_KEYS, STATE_KEYS, _first_existing_key
from src.paths import resolve_path


def is_lerobot_available() -> bool:
    """Return whether the lerobot package can be imported."""
    return find_spec("lerobot") is not None


class CalvinLeRobotDataset:
    """Dataset-compatible wrapper over `sample_index.jsonl`.

    The class intentionally implements the minimal PyTorch/LeRobot dataset
    protocol: `__len__` and `__getitem__`.

    `action_chunk` is formed by taking actions from the current frame forward
    inside the same episode. If fewer than `chunk_size` actions remain at the
    episode end, the final available action is repeated. This padding strategy
    keeps a fixed tensor shape without leaking actions from the next episode.
    """

    def __init__(self, dataset_dir: str | Path, chunk_size: int = 100) -> None:
        self.dataset_dir = resolve_path(dataset_dir)
        if (self.dataset_dir / "source_datasets.json").exists():
            raise ValueError(
                f"{self.dataset_dir} is an HF LeRobot manifest dataset. "
                "Use load_lerobot_source_manifest() and instantiate the official LeRobotDataset "
                "for each listed source in the training script."
            )
        self.index_path = self.dataset_dir / "sample_index.jsonl"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing sample index: {self.index_path}")
        self.chunk_size = chunk_size
        self.records = self._load_records()
        self.episode_to_indices = self._build_episode_index()

    def __len__(self) -> int:
        """Return number of frame samples."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one frame sample and its fixed-length future action chunk."""
        record = self.records[index]
        frame = self._load_frame(record)
        action_chunk, action_chunk_is_padded = self._build_action_chunk(index)
        return {
            "image": frame["image"],
            "state": frame["state"],
            "action": frame["action"],
            "action_chunk": action_chunk,
            "action_chunk_is_padded": action_chunk_is_padded,
            "task": record.get("task", ""),
            "env_id": record["env_id"],
        }

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"Index contains no samples: {self.index_path}")
        return records

    def _build_episode_index(self) -> dict[int, list[int]]:
        episode_to_indices: dict[int, list[int]] = {}
        for index, record in enumerate(self.records):
            episode_to_indices.setdefault(int(record["episode_index"]), []).append(index)
        for indices in episode_to_indices.values():
            indices.sort(key=lambda idx: int(self.records[idx]["frame_index"]))
        return episode_to_indices

    def _build_action_chunk(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        episode_indices = self.episode_to_indices[int(record["episode_index"])]
        position = episode_indices.index(index)
        future_indices = episode_indices[position : position + self.chunk_size]
        actions = [self._load_action(self.records[future_index]) for future_index in future_indices]

        pad_count = self.chunk_size - len(actions)
        if pad_count > 0:
            actions.extend([actions[-1].copy() for _ in range(pad_count)])

        padded_mask = np.zeros((self.chunk_size,), dtype=bool)
        if pad_count > 0:
            padded_mask[-pad_count:] = True
        return np.stack(actions, axis=0), padded_mask

    def _load_action(self, record: dict[str, Any]) -> np.ndarray:
        return self._load_frame(record)["action"]

    def _load_frame(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        source_path = resolve_path(record["source_path"])
        source_format = record.get("source_format")
        if source_format == "episode_npz":
            return self._load_episode_npz_frame(source_path, int(record["frame_index"]))
        if source_format == "lerobot_parquet":
            return self._load_lerobot_parquet_frame(
                source_path,
                int(record["source_episode_index"]),
                int(record["frame_index"]),
            )
        return self._load_frame_npz(source_path)

    def _load_frame_npz(self, source_path: Path) -> dict[str, np.ndarray]:
        with np.load(source_path, allow_pickle=True) as data:
            image_key = _first_existing_key(data, IMAGE_KEYS, source_path)
            state_key = _first_existing_key(data, STATE_KEYS, source_path)
            action_key = _first_existing_key(data, ACTION_KEYS, source_path)
            return {
                "image": np.asarray(data[image_key]),
                "state": np.asarray(data[state_key], dtype=np.float32),
                "action": np.asarray(data[action_key], dtype=np.float32).reshape(-1),
            }

    def _load_episode_npz_frame(self, source_path: Path, frame_index: int) -> dict[str, np.ndarray]:
        with np.load(source_path, allow_pickle=True) as data:
            image_key = _first_existing_key(data, IMAGE_KEYS, source_path)
            state_key = _first_existing_key(data, STATE_KEYS, source_path)
            action_key = _first_existing_key(data, ACTION_KEYS, source_path)
            return {
                "image": np.asarray(data[image_key][frame_index]),
                "state": np.asarray(data[state_key][frame_index], dtype=np.float32),
                "action": np.asarray(data[action_key][frame_index], dtype=np.float32).reshape(-1),
            }

    def _load_lerobot_parquet_frame(
        self,
        source_path: Path,
        episode_index: int,
        frame_index: int,
    ) -> dict[str, np.ndarray]:
        table = pd.read_parquet(source_path)
        rows = table[table["episode_index"] == episode_index]
        if "frame_index" in rows.columns:
            rows = rows[rows["frame_index"] == frame_index]
        else:
            rows = rows.iloc[[frame_index]]
        if rows.empty:
            raise IndexError(
                f"Missing parquet row: path={source_path}, episode={episode_index}, frame={frame_index}"
            )
        row = rows.iloc[0]
        image_key = _first_existing_key(table, IMAGE_KEYS, source_path)
        state_key = _first_existing_key(table, STATE_KEYS, source_path)
        action_key = _first_existing_key(table, ACTION_KEYS, source_path)
        return {
            "image": np.asarray(row[image_key]),
            "state": np.asarray(row[state_key], dtype=np.float32),
            "action": np.asarray(row[action_key], dtype=np.float32).reshape(-1),
        }


def load_lerobot_source_manifest(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load source LeRobotDataset directories from a processed manifest."""
    manifest_path = resolve_path(dataset_dir) / "source_datasets.json"
    if not manifest_path.exists():
        return [{"path": str(resolve_path(dataset_dir)), "env_id": None}]
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    sources = manifest.get("sources", [])
    if not sources:
        raise ValueError(f"Manifest contains no source datasets: {manifest_path}")
    return sources


def build_act_training_command(config: dict[str, Any]) -> list[str]:
    """Build the future LeRobot training command from project config."""
    return [
        "lerobot-train",
        f"--policy.type={config['policy_type']}",
        f"--dataset.path={config['dataset_path']}",
        f"--output_dir={config['output_dir']}",
    ]
