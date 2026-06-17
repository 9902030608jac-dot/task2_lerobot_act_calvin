"""CALVIN data readers and preparation utilities.

The processed dataset used by this project is intentionally simple:

- `sample_index.jsonl` stores one JSON record per frame.
- `dataset_stats.json` stores dimensions and counts needed by training/eval.

Each index record keeps the raw source path and `env_id`, so A/B/C/D provenance
is never lost after mixing datasets.
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from src.paths import ensure_dir, resolve_path


IMAGE_KEYS = ("rgb_static", "image", "observation.image", "observation.images.rgb_static")
STATE_KEYS = ("robot_obs", "state", "observation.state")
ACTION_KEYS = ("rel_actions", "actions", "action")
TASK_KEYS = ("task", "language", "lang", "instruction")


@dataclass(frozen=True)
class EpisodeSpec:
    """Description of one readable episode."""

    env_id: str
    episode_index: int
    frame_paths: tuple[Path, ...] = ()
    episode_file: Path | None = None
    start: int | None = None
    end: int | None = None
    source_format: str = "unknown"

    @property
    def length(self) -> int:
        """Return the number of frames in the episode."""
        if self.frame_paths:
            return len(self.frame_paths)
        if self.start is not None and self.end is not None:
            return self.end - self.start + 1
        if self.episode_file is not None:
            with np.load(self.episode_file, allow_pickle=True) as data:
                key = _first_existing_key(data, ACTION_KEYS, self.episode_file)
                return int(np.asarray(data[key]).shape[0])
        return 0


def _first_existing_key(data: Any, candidates: Iterable[str], source: Path) -> str:
    """Return the first candidate key present in a mapping-like npz object."""
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"{source} is missing required field. Expected one of: {list(candidates)}")


def _to_jsonable(value: Any) -> Any:
    """Convert numpy values to JSON-serializable Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    return value


class CalvinEpisodeReader:
    """Read one CALVIN environment directory frame by frame.

    Supported raw layouts:

    1. CALVIN frame files with `ep_start_end_ids.npy` and `episode_XXXXXXX.npz`.
    2. Episode directories that contain ordered `.npz` frame files.
    3. Episode-level `.npz` files where image/state/action arrays have shape
       `[T, ...]`.

    A processed LeRobot-like directory with `sample_index.jsonl` is detected by
    `is_lerobot_dataset`, but this reader is mainly for raw CALVIN conversion.
    """

    def __init__(self, env_dir: str | Path, env_id: str) -> None:
        self.env_dir = resolve_path(env_dir)
        self.env_id = env_id
        if not self.env_dir.exists():
            raise FileNotFoundError(f"CALVIN env directory does not exist: {self.env_dir}")
        if not self.env_dir.is_dir():
            raise NotADirectoryError(f"CALVIN env path is not a directory: {self.env_dir}")

    @property
    def is_lerobot_dataset(self) -> bool:
        """Return whether the directory already looks like an indexed dataset."""
        return (self.env_dir / "sample_index.jsonl").exists() or (self.env_dir / "meta").exists()

    def enumerate_episodes(self, max_episodes: int | None = None) -> list[EpisodeSpec]:
        """Enumerate episodes in deterministic temporal order."""
        if self.is_lerobot_dataset and (self.env_dir / "sample_index.jsonl").exists():
            episodes = self._enumerate_indexed_episodes()
        elif self.is_lerobot_dataset and list(self.env_dir.glob("data/**/*.parquet")):
            episodes = self._enumerate_lerobot_parquet_episodes()
        elif (self.env_dir / "ep_start_end_ids.npy").exists():
            episodes = self._enumerate_calvin_frame_episodes()
        else:
            episodes = self._enumerate_directory_or_file_episodes()

        if not episodes:
            raise FileNotFoundError(
                f"No readable episodes found in {self.env_dir}. Expected CALVIN npz files, "
                "episode directories, or an existing sample_index.jsonl."
            )
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        return episodes

    def iter_frames(self, episode: EpisodeSpec) -> Iterator[dict[str, Any]]:
        """Yield normalized frame dictionaries in temporal order."""
        if episode.source_format == "indexed":
            yield from self._iter_indexed_frames(episode)
            return
        if episode.source_format == "lerobot_parquet":
            yield from self._iter_lerobot_parquet_frames(episode)
            return
        if episode.episode_file is not None and episode.source_format == "episode_npz":
            yield from self._iter_episode_npz_frames(episode)
            return

        frame_paths = episode.frame_paths
        if episode.source_format == "calvin_frame_npz":
            assert episode.start is not None and episode.end is not None
            frame_paths = tuple(self._frame_path(i) for i in range(episode.start, episode.end + 1))

        for frame_index, frame_path in enumerate(frame_paths):
            frame = self._read_frame_npz(frame_path, episode.episode_index, frame_index)
            yield frame

    def read_frame(self, episode: EpisodeSpec, frame_index: int) -> dict[str, Any]:
        """Read a single frame by index from an episode."""
        for frame in self.iter_frames(episode):
            if frame["frame_index"] == frame_index:
                return frame
        raise IndexError(f"Frame {frame_index} not found in episode {episode.episode_index}")

    def _enumerate_indexed_episodes(self) -> list[EpisodeSpec]:
        episode_ids: list[int] = []
        with (self.env_dir / "sample_index.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                episode_ids.append(int(record["episode_index"]))
        return [
            EpisodeSpec(self.env_id, episode_index, source_format="indexed")
            for episode_index in sorted(set(episode_ids))
        ]

    def _enumerate_calvin_frame_episodes(self) -> list[EpisodeSpec]:
        start_end = np.load(self.env_dir / "ep_start_end_ids.npy")
        episodes: list[EpisodeSpec] = []
        for episode_index, pair in enumerate(start_end):
            start, end = int(pair[0]), int(pair[1])
            missing = [str(self._frame_path(i)) for i in range(start, end + 1) if not self._frame_path(i).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Episode {episode_index} in {self.env_dir} references missing frame files: "
                    f"{missing[:3]}"
                )
            episodes.append(
                EpisodeSpec(
                    env_id=self.env_id,
                    episode_index=episode_index,
                    start=start,
                    end=end,
                    source_format="calvin_frame_npz",
                )
            )
        return episodes

    def _enumerate_lerobot_parquet_episodes(self) -> list[EpisodeSpec]:
        episodes: list[EpisodeSpec] = []
        for parquet_file in sorted(self.env_dir.glob("data/**/*.parquet")):
            frame = pd.read_parquet(parquet_file)
            if "episode_index" not in frame.columns:
                raise KeyError(f"{parquet_file} is missing required LeRobot column: episode_index")
            for episode_index in sorted(frame["episode_index"].unique()):
                episodes.append(
                    EpisodeSpec(
                        env_id=self.env_id,
                        episode_index=int(episode_index),
                        episode_file=parquet_file,
                        source_format="lerobot_parquet",
                    )
                )
        return episodes

    def _enumerate_directory_or_file_episodes(self) -> list[EpisodeSpec]:
        episodes: list[EpisodeSpec] = []
        episode_dirs = sorted(path for path in self.env_dir.iterdir() if path.is_dir())
        for episode_index, episode_dir in enumerate(episode_dirs):
            frame_paths = tuple(sorted(episode_dir.glob("*.npz")))
            if frame_paths:
                episodes.append(
                    EpisodeSpec(
                        env_id=self.env_id,
                        episode_index=episode_index,
                        frame_paths=frame_paths,
                        source_format="frame_npz_dir",
                    )
                )

        if episodes:
            return episodes

        episode_files = sorted(self.env_dir.glob("*.npz"))
        return [
            EpisodeSpec(
                env_id=self.env_id,
                episode_index=episode_index,
                episode_file=episode_file,
                source_format="episode_npz",
            )
            for episode_index, episode_file in enumerate(episode_files)
        ]

    def _frame_path(self, frame_id: int) -> Path:
        return self.env_dir / f"episode_{frame_id:07d}.npz"

    def _read_frame_npz(self, frame_path: Path, episode_index: int, frame_index: int) -> dict[str, Any]:
        with np.load(frame_path, allow_pickle=True) as data:
            image_key = _first_existing_key(data, IMAGE_KEYS, frame_path)
            state_key = _first_existing_key(data, STATE_KEYS, frame_path)
            action_key = _first_existing_key(data, ACTION_KEYS, frame_path)
            task = data[_first_existing_key(data, TASK_KEYS, frame_path)] if any(k in data for k in TASK_KEYS) else ""
            return {
                "observation.image": np.asarray(data[image_key]),
                "observation.state": np.asarray(data[state_key]),
                "action": np.asarray(data[action_key]),
                "task": _to_jsonable(task),
                "env_id": self.env_id,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "source_path": str(frame_path),
                "source_format": "frame_npz",
            }

    def _iter_episode_npz_frames(self, episode: EpisodeSpec) -> Iterator[dict[str, Any]]:
        assert episode.episode_file is not None
        with np.load(episode.episode_file, allow_pickle=True) as data:
            image_key = _first_existing_key(data, IMAGE_KEYS, episode.episode_file)
            state_key = _first_existing_key(data, STATE_KEYS, episode.episode_file)
            action_key = _first_existing_key(data, ACTION_KEYS, episode.episode_file)
            images = np.asarray(data[image_key])
            states = np.asarray(data[state_key])
            actions = np.asarray(data[action_key])
            if images.shape[0] != actions.shape[0] or states.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"{episode.episode_file} has inconsistent sequence lengths: "
                    f"image={images.shape[0]}, state={states.shape[0]}, action={actions.shape[0]}"
                )
            task_values = data[_first_existing_key(data, TASK_KEYS, episode.episode_file)] if any(k in data for k in TASK_KEYS) else ""
            for frame_index in range(actions.shape[0]):
                task = task_values[frame_index] if np.asarray(task_values).ndim > 0 else task_values
                yield {
                    "observation.image": images[frame_index],
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                    "task": _to_jsonable(task),
                    "env_id": self.env_id,
                    "episode_index": episode.episode_index,
                    "frame_index": frame_index,
                    "source_path": str(episode.episode_file),
                    "source_format": "episode_npz",
                }

    def _iter_lerobot_parquet_frames(self, episode: EpisodeSpec) -> Iterator[dict[str, Any]]:
        assert episode.episode_file is not None
        table = pd.read_parquet(episode.episode_file)
        rows = table[table["episode_index"] == episode.episode_index].copy()
        if "frame_index" in rows.columns:
            rows = rows.sort_values("frame_index")
        else:
            rows = rows.reset_index(drop=True)
            rows["frame_index"] = np.arange(len(rows))

        image_key = _first_existing_key(rows, IMAGE_KEYS, episode.episode_file)
        state_key = _first_existing_key(rows, STATE_KEYS, episode.episode_file)
        action_key = _first_existing_key(rows, ACTION_KEYS, episode.episode_file)
        task_key = next((key for key in TASK_KEYS if key in rows), None)
        for _, row in rows.iterrows():
            image = row[image_key]
            if isinstance(image, dict) and "path" in image:
                raise ValueError(
                    f"{episode.episode_file} stores image observations as external media paths. "
                    "This scaffold currently supports direct image arrays in parquet or CALVIN npz frames."
                )
            yield {
                "observation.image": np.asarray(image),
                "observation.state": np.asarray(row[state_key]),
                "action": np.asarray(row[action_key]),
                "task": _to_jsonable(row[task_key]) if task_key else "",
                "env_id": self.env_id,
                "episode_index": episode.episode_index,
                "frame_index": int(row["frame_index"]),
                "source_path": str(episode.episode_file),
                "source_format": "lerobot_parquet",
            }

    def _iter_indexed_frames(self, episode: EpisodeSpec) -> Iterator[dict[str, Any]]:
        with (self.env_dir / "sample_index.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["episode_index"]) != episode.episode_index:
                    continue
                yield {
                    "observation.image": np.empty(record["image_shape"], dtype=np.uint8),
                    "observation.state": np.empty(record["state_shape"], dtype=np.float32),
                    "action": np.empty(record["action_shape"], dtype=np.float32),
                    "task": record.get("task", ""),
                    "env_id": record["env_id"],
                    "episode_index": int(record["episode_index"]),
                    "frame_index": int(record["frame_index"]),
                    "source_path": record["source_path"],
                    "source_format": record.get("source_format", "indexed"),
                }


def get_dataset_paths(config: dict[str, Any]) -> list[Path]:
    """Return configured dataset paths as resolved Path objects."""
    dataset_path = config["dataset_path"]
    if isinstance(dataset_path, list):
        return [resolve_path(path) for path in dataset_path]
    return [resolve_path(dataset_path)]


def inspect_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect configured dataset paths and report whether they exist."""
    paths = get_dataset_paths(config)
    return {
        "environments": config.get("train_envs", [config.get("test_env")]),
        "paths": [{"path": str(path), "exists": path.exists()} for path in paths],
    }


def raw_env_path(env_id: str) -> Path:
    """Return the raw data path for a CALVIN environment id."""
    return resolve_path(f"data/raw/calvin_{env_id}")


def hf_subdir_for_env(env_id: str) -> str:
    """Return the Hugging Face split subdir for an environment id."""
    if env_id not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported CALVIN env id: {env_id}")
    return f"split{env_id}"


def download_hf_calvin_splits(
    *,
    repo_id: str,
    revision: str = "main",
    subdirs: Iterable[str],
    local_dir: str | Path = "data/raw/xiaoma26_calvin_lerobot",
    endpoint: str | None = None,
    max_workers: int = 8,
    resume: bool = True,
    force_download: bool = False,
    max_retries: int = 10,
    retry_sleep_seconds: float = 5.0,
    etag_timeout: float = 60.0,
    download_timeout: float = 60.0,
    downloader: str = "hf_cli",
    hf_cli_executable: str = "hf",
    hub_pub_executable: str = "hub-pub",
    hub_pub_command_template: str | None = None,
) -> list[Path]:
    """Download selected LeRobot split folders from the assistant-provided HF repo.

    The class assignment update provides `xiaoma26/calvin-lerobot` with splitA,
    splitB, splitC, and splitD already separated. We download only requested
    split folders so D stays isolated from training data. Hugging Face Hub
    stores downloads through resumable cache files; this function keeps that
    behavior enabled and exposes worker/mirror knobs for large datasets.
    """
    local_dir = ensure_dir(local_dir)
    subdirs = list(subdirs)
    requested_dirs = [local_dir / subdir for subdir in subdirs]
    complete_dirs = [split_dir for split_dir in requested_dirs if _hf_lerobot_split_is_complete(split_dir)]
    if len(complete_dirs) == len(requested_dirs):
        print(
            "HF dataset split(s) already present locally; skipping remote download: "
            + ", ".join(str(path) for path in requested_dirs),
            file=sys.stderr,
        )
        return requested_dirs

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(int(etag_timeout))
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(download_timeout))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    allow_patterns = [f"{subdir}/**" for subdir in subdirs]
    if downloader == "hf_cli":
        executable = shutil.which(hf_cli_executable)
        if executable is None:
            raise FileNotFoundError(
                f"`{hf_cli_executable}` was requested for dataset download, but it is not installed or not on PATH. "
                "Install/upgrade huggingface_hub so the `hf` CLI is available."
            )
        command = [
            executable,
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(local_dir),
            "--max-workers",
            str(max_workers),
        ]
        for pattern in allow_patterns:
            command.extend(["--include", pattern])
        print(
            "Starting hf CLI download: " + shlex.join(command),
            file=sys.stderr,
        )
        subprocess.run(command, check=True)
        return requested_dirs

    if downloader == "hub_pub":
        executable = shutil.which(hub_pub_executable)
        if executable is None:
            raise FileNotFoundError(
                f"`{hub_pub_executable}` was requested for dataset download, but it is not installed or not on PATH. "
                "Install the platform hub-pub downloader first, or pass --hf_downloader hf_cli "
                "to use the Python fallback."
            )
        if hub_pub_command_template:
            format_values = {
                "executable": executable,
                "repo_id": repo_id,
                "revision": revision,
                "repo_type": "dataset",
                "local_dir": str(local_dir),
                "endpoint": endpoint or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
                "allow_patterns": " ".join(allow_patterns),
            }
            command = shlex.split(hub_pub_command_template.format(**format_values))
        else:
            command = [
                executable,
                "download",
                repo_id,
                "--repo-type",
                "dataset",
                "--revision",
                revision,
                "--local-dir",
                str(local_dir),
                "--endpoint",
                endpoint or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
            ]
            for pattern in allow_patterns:
                command.extend(["--include", pattern])
        print(
            "Starting hub-pub download: " + shlex.join(command),
            file=sys.stderr,
        )
        subprocess.run(command, check=True)
        return requested_dirs

    if downloader != "huggingface_hub":
        raise ValueError("downloader must be one of: hf_cli, hub_pub, huggingface_hub")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the provided dataset. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "local_dir": str(local_dir),
        "allow_patterns": allow_patterns,
    }
    signature = inspect.signature(snapshot_download)
    if "etag_timeout" in signature.parameters:
        download_kwargs["etag_timeout"] = etag_timeout
    if "max_workers" in signature.parameters:
        download_kwargs["max_workers"] = max_workers
    if "force_download" in signature.parameters:
        download_kwargs["force_download"] = force_download
    if "resume_download" in signature.parameters:
        download_kwargs["resume_download"] = resume
    if endpoint and "endpoint" in signature.parameters:
        download_kwargs["endpoint"] = endpoint

    missing_dirs = [path for path in requested_dirs if path not in complete_dirs]
    print(
        "Starting HF snapshot_download for "
        f"{repo_id}@{revision}; endpoint={endpoint or os.environ.get('HF_ENDPOINT') or 'default'}; "
        f"subdirs={list(subdirs)}; local_dir={local_dir}; max_workers={max_workers}; "
        f"resume={resume}; missing_or_incomplete={missing_dirs}",
        file=sys.stderr,
    )

    for retry_index in range(max_retries + 1):
        try:
            snapshot_download(**download_kwargs)
            break
        except Exception as exc:
            if retry_index >= max_retries:
                raise
            sleep_seconds = min(60.0, retry_sleep_seconds * (retry_index + 1))
            print(
                f"HF download failed ({exc}). Retrying "
                f"{retry_index + 1}/{max_retries} in {sleep_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
    return [local_dir / subdir for subdir in subdirs]


def _hf_lerobot_split_is_complete(split_dir: Path) -> bool:
    """Return whether a downloaded HF LeRobot split has all parquet episodes."""
    info_path = split_dir / "meta" / "info.json"
    if not info_path.exists() or not (split_dir / "data").exists():
        return False
    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
        expected_episodes = int(info.get("total_episodes", 0) or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if expected_episodes <= 0:
        return False
    parquet_count = sum(1 for _ in split_dir.glob("data/**/*.parquet"))
    return parquet_count >= expected_episodes


def split_to_envs(split: str) -> list[str]:
    """Map a prepare split name to allowed environment ids."""
    if split == "A_only":
        return ["A"]
    if split == "ABC":
        return ["A", "B", "C"]
    if split == "D":
        return ["D"]
    raise ValueError("split must be one of: A_only, ABC, D")


def split_to_output_path(split: str) -> Path:
    """Map a prepare split name to its processed output directory."""
    mapping = {
        "A_only": "data/processed/calvin_A_lerobot",
        "ABC": "data/processed/calvin_ABC_lerobot",
        "D": "data/processed/calvin_D_lerobot",
    }
    return resolve_path(mapping[split])


def split_to_hf_subdirs(split: str) -> list[str]:
    """Return HF dataset subdirectories for a prepare split."""
    return [hf_subdir_for_env(env_id) for env_id in split_to_envs(split)]


def write_hf_manifest_dataset(
    *,
    split: str,
    source_dirs: list[Path],
    output_dir: str | Path,
    chunk_size: int,
) -> dict[str, Any]:
    """Create a processed manifest that points to already split LeRobot datasets.

    This path is used for the course-provided HF dataset. It avoids rewriting
    videos/parquet files and keeps the original LeRobotDataset split structure.
    """
    envs = split_to_envs(split)
    if split in {"A_only", "ABC"} and any("splitD" in str(path) for path in source_dirs):
        raise ValueError("Environment D must never be referenced by a training split.")
    if split == "ABC" and set(path.name for path in source_dirs) - {"splitA", "splitB", "splitC"}:
        raise ValueError("ABC split may only reference splitA/splitB/splitC.")

    resolved_output = ensure_dir(output_dir)
    manifest_path = resolved_output / "source_datasets.json"
    stats_path = resolved_output / "dataset_stats.json"
    index_path = resolved_output / "sample_index.jsonl"

    sources: list[dict[str, Any]] = []
    total_episodes = 0
    total_samples = 0
    for env_id, source_dir in zip(envs, source_dirs, strict=True):
        source_dir = resolve_path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"HF split directory is missing: {source_dir}")
        if not (source_dir / "meta").exists() or not (source_dir / "data").exists():
            raise FileNotFoundError(
                f"{source_dir} does not look like a LeRobotDataset split. Expected meta/ and data/."
            )

        info_path = source_dir / "meta" / "info.json"
        episodes_path = source_dir / "meta" / "episodes.jsonl"
        source_info: dict[str, Any] = {}
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as file:
                source_info = json.load(file)
        episode_count = 0
        if episodes_path.exists():
            with episodes_path.open("r", encoding="utf-8") as file:
                episode_count = sum(1 for line in file if line.strip())
        sample_count = int(source_info.get("total_frames", 0) or source_info.get("num_frames", 0) or 0)
        total_episodes += episode_count
        total_samples += sample_count
        sources.append(
            {
                "env_id": env_id,
                "hf_subdir": source_dir.name,
                "path": str(source_dir),
                "num_episodes": episode_count,
                "num_samples": sample_count,
                "info_path": str(info_path) if info_path.exists() else None,
                "episodes_path": str(episodes_path) if episodes_path.exists() else None,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "format": "hf_lerobot_manifest",
                "split": split,
                "envs": envs,
                "chunk_size": chunk_size,
                "sources": sources,
                "note": "Use these source LeRobotDataset directories directly in training/eval.",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Keep sample_index.jsonl present for downstream scripts. For HF LeRobot
    # video/parquet datasets, actual samples should be read through LeRobotDataset.
    with index_path.open("w", encoding="utf-8") as file:
        for source in sources:
            file.write(json.dumps({"env_id": source["env_id"], "source_dataset": source["path"]}) + "\n")

    stats = {
        "split": split,
        "envs": envs,
        "source_format": "hf_lerobot_manifest",
        "num_samples": total_samples,
        "num_episodes": total_episodes,
        "samples_per_env": {source["env_id"]: source["num_samples"] for source in sources},
        "action_dim": None,
        "state_dim": None,
        "image_shape": None,
        "chunk_size": chunk_size,
        "manifest_path": str(manifest_path),
        "index_path": str(index_path),
        "note": "Dimensions are left null because the provided LeRobotDataset may store image observations as videos.",
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "output_dir": str(resolved_output),
        "sample_index": str(index_path),
        "dataset_stats": str(stats_path),
        **stats,
    }


def prepare_processed_dataset(
    *,
    split: str,
    output_dir: str | Path | None = None,
    max_episodes: int | None = None,
    chunk_size: int = 100,
) -> dict[str, Any]:
    """Create a processed sample index and statistics file for one split."""
    envs = split_to_envs(split)
    if split in {"A_only", "ABC"} and "D" in envs:
        raise ValueError("Environment D must never be included in a training split.")
    if split == "ABC" and set(envs) != {"A", "B", "C"}:
        raise ValueError("ABC split must contain only A/B/C.")

    resolved_output = ensure_dir(output_dir or split_to_output_path(split))
    index_path = resolved_output / "sample_index.jsonl"
    stats_path = resolved_output / "dataset_stats.json"

    sample_count = 0
    episode_count = 0
    action_dim: int | None = None
    state_dim: int | None = None
    image_shape: list[int] | None = None
    env_counts: dict[str, int] = {}
    episode_lengths: list[int] = []
    global_episode_index = 0

    with index_path.open("w", encoding="utf-8") as index_file:
        for env_id in envs:
            reader = CalvinEpisodeReader(raw_env_path(env_id), env_id)
            episodes = reader.enumerate_episodes(max_episodes=max_episodes)
            env_counts[env_id] = 0

            for episode in episodes:
                frames = list(reader.iter_frames(episode))
                if not frames:
                    raise ValueError(f"Empty episode found: env={env_id}, episode={episode.episode_index}")

                episode_count += 1
                episode_lengths.append(len(frames))

                for frame in frames:
                    image = np.asarray(frame["observation.image"])
                    state = np.asarray(frame["observation.state"])
                    action = np.asarray(frame["action"])
                    if action.ndim == 0:
                        raise ValueError(f"Action must be vector-like in {frame['source_path']}")

                    current_action_dim = int(action.reshape(-1).shape[0])
                    current_state_dim = int(state.reshape(-1).shape[0])
                    current_image_shape = list(image.shape)
                    action_dim = action_dim or current_action_dim
                    state_dim = state_dim or current_state_dim
                    image_shape = image_shape or current_image_shape
                    if action_dim != current_action_dim:
                        raise ValueError(f"Inconsistent action dim in {frame['source_path']}")
                    if state_dim != current_state_dim:
                        raise ValueError(f"Inconsistent state dim in {frame['source_path']}")
                    if image_shape != current_image_shape:
                        raise ValueError(f"Inconsistent image shape in {frame['source_path']}")

                    record = {
                        "env_id": env_id,
                        "episode_index": global_episode_index,
                        "source_episode_index": frame["episode_index"],
                        "frame_index": frame["frame_index"],
                        "episode_frame_count": len(frames),
                        "source_path": frame["source_path"],
                        "source_format": frame["source_format"],
                        "image_shape": current_image_shape,
                        "state_shape": list(state.shape),
                        "action_shape": list(action.shape),
                        "task": _to_jsonable(frame.get("task", "")),
                        "chunk_size": chunk_size,
                    }
                    index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sample_count += 1
                    env_counts[env_id] += 1

                global_episode_index += 1

    if sample_count == 0:
        raise ValueError(f"No samples were written for split={split}.")

    stats = {
        "split": split,
        "envs": envs,
        "num_samples": sample_count,
        "num_episodes": episode_count,
        "samples_per_env": env_counts,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "image_shape": image_shape,
        "chunk_size": chunk_size,
        "episode_length_min": min(episode_lengths),
        "episode_length_max": max(episode_lengths),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "index_path": str(index_path),
    }
    with stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "output_dir": str(resolved_output),
        "sample_index": str(index_path),
        "dataset_stats": str(stats_path),
        **stats,
    }


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare a dataset using config fields when called by older scripts."""
    train_envs = config.get("train_envs")
    test_env = config.get("test_env")
    if train_envs == ["A"]:
        split = "A_only"
    elif train_envs == ["A", "B", "C"]:
        split = "ABC"
    elif test_env == "D":
        split = "D"
    else:
        raise ValueError(f"Cannot infer split from config: train_envs={train_envs}, test_env={test_env}")
    return prepare_processed_dataset(
        split=split,
        output_dir=config["dataset_path"],
        chunk_size=int(config.get("chunk_size", 100)),
    )
