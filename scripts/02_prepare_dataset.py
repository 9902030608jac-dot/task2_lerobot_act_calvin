#!/usr/bin/env python
"""Prepare CALVIN A-only, ABC, or D data into indexed LeRobot-readable folders."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calvin_data import (
    download_hf_calvin_splits,
    prepare_processed_dataset,
    split_to_hf_subdirs,
    split_to_output_path,
    write_hf_manifest_dataset,
)
from src.config import load_config, save_config_snapshot


def infer_split(config: dict[str, object]) -> str:
    """Infer prepare split from a merged train/eval config."""
    train_envs = config.get("train_envs")
    test_env = config.get("test_env")
    if train_envs == ["A"]:
        return "A_only"
    if train_envs == ["A", "B", "C"]:
        return "ABC"
    if test_env == "D":
        return "D"
    raise ValueError(f"Cannot infer split from train_envs={train_envs}, test_env={test_env}")


def main() -> None:
    """Create `sample_index.jsonl` and `dataset_stats.json` for one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional train/eval YAML config.")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--split", choices=["A_only", "ABC", "D"], default=None)
    parser.add_argument("--source", choices=["hf", "raw"], default="hf")
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--hf_local_dir", default=None)
    parser.add_argument(
        "--hf_downloader",
        choices=["hf_cli", "hub_pub", "huggingface_hub"],
        default=None,
        help="Downloader backend. Defaults to hf_cli, the official Hugging Face CLI.",
    )
    parser.add_argument(
        "--hf_cli_executable",
        default=None,
        help="Name or path of the Hugging Face `hf` executable.",
    )
    parser.add_argument(
        "--hub_pub_executable",
        default=None,
        help="Name or path of the hub-pub executable.",
    )
    parser.add_argument(
        "--hub_pub_command_template",
        default=None,
        help=(
            "Optional command template for nonstandard hub-pub CLIs. Available fields: "
            "{executable}, {repo_id}, {repo_type}, {revision}, {local_dir}, {endpoint}, {allow_patterns}."
        ),
    )
    parser.add_argument(
        "--hf_endpoint",
        default=None,
        help="Optional Hugging Face mirror endpoint, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--hf_max_workers",
        type=int,
        default=None,
        help="Number of parallel Hugging Face download workers.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resumable Hugging Face downloads. Resume is enabled by default.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download files even if a local cached copy exists.",
    )
    parser.add_argument(
        "--hf_max_retries",
        type=int,
        default=None,
        help="Number of retry attempts after a Hugging Face download failure.",
    )
    parser.add_argument(
        "--hf_retry_sleep",
        type=float,
        default=None,
        help="Base seconds to wait between Hugging Face download retries.",
    )
    parser.add_argument(
        "--hf_etag_timeout",
        type=float,
        default=None,
        help="Seconds to wait for Hugging Face metadata requests before retrying.",
    )
    parser.add_argument(
        "--hf_download_timeout",
        type=float,
        default=None,
        help="Seconds to wait for Hugging Face file download chunks before retrying.",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override) if args.config else None
    split = args.split or (infer_split(config) if config else "A_only")
    chunk_size = args.chunk_size or (int(config["chunk_size"]) if config else 100)
    output_dir = args.output_dir or str(split_to_output_path(split))

    if args.source == "hf":
        repo_id = args.hf_repo_id or (config.get("hf_dataset_repo") if config else "xiaoma26/calvin-lerobot")
        revision = args.hf_revision or (config.get("hf_dataset_revision") if config else "main")
        local_dir = args.hf_local_dir or (
            config.get("hf_dataset_local_dir") if config else "data/raw/xiaoma26_calvin_lerobot"
        )
        endpoint = args.hf_endpoint or (config.get("hf_dataset_endpoint") if config else "https://hf-mirror.com")
        downloader = args.hf_downloader or str(config.get("hf_dataset_downloader", "hf_cli") if config else "hf_cli")
        hf_cli_executable = args.hf_cli_executable or str(
            config.get("hf_cli_executable", "hf") if config else "hf"
        )
        hub_pub_executable = args.hub_pub_executable or str(
            config.get("hub_pub_executable", "hub-pub") if config else "hub-pub"
        )
        hub_pub_command_template = args.hub_pub_command_template or (
            config.get("hub_pub_command_template") if config else None
        )
        max_workers = args.hf_max_workers or int(config.get("hf_dataset_max_workers", 1) if config else 1)
        if max_workers <= 0:
            raise ValueError("--hf_max_workers must be a positive integer.")
        max_retries = args.hf_max_retries if args.hf_max_retries is not None else int(
            config.get("hf_dataset_max_retries", 30) if config else 30
        )
        retry_sleep = args.hf_retry_sleep if args.hf_retry_sleep is not None else float(
            config.get("hf_dataset_retry_sleep", 10.0) if config else 10.0
        )
        etag_timeout = args.hf_etag_timeout if args.hf_etag_timeout is not None else float(
            config.get("hf_dataset_etag_timeout", 300.0) if config else 300.0
        )
        download_timeout = args.hf_download_timeout if args.hf_download_timeout is not None else float(
            config.get("hf_dataset_download_timeout", 300.0) if config else 300.0
        )
        if max_retries < 0:
            raise ValueError("--hf_max_retries must be non-negative.")
        if retry_sleep < 0:
            raise ValueError("--hf_retry_sleep must be non-negative.")
        if etag_timeout <= 0:
            raise ValueError("--hf_etag_timeout must be positive.")
        if download_timeout <= 0:
            raise ValueError("--hf_download_timeout must be positive.")
        subdirs = config.get("hf_dataset_subdirs") if config else split_to_hf_subdirs(split)
        source_dirs = download_hf_calvin_splits(
            repo_id=str(repo_id),
            revision=str(revision),
            subdirs=list(subdirs),
            local_dir=str(local_dir),
            endpoint=str(endpoint) if endpoint else None,
            max_workers=max_workers,
            resume=not args.no_resume,
            force_download=args.force_download,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep,
            etag_timeout=etag_timeout,
            download_timeout=download_timeout,
            downloader=downloader,
            hf_cli_executable=hf_cli_executable,
            hub_pub_executable=hub_pub_executable,
            hub_pub_command_template=str(hub_pub_command_template) if hub_pub_command_template else None,
        )
        result = write_hf_manifest_dataset(
            split=split,
            source_dirs=source_dirs,
            output_dir=output_dir,
            chunk_size=chunk_size,
        )
        result["hf_dataset_repo"] = repo_id
        result["hf_dataset_subdirs"] = list(subdirs)
        result["hf_dataset_endpoint"] = endpoint
        result["hf_dataset_max_workers"] = max_workers
        result["hf_dataset_downloader"] = downloader
        result["hf_cli_executable"] = hf_cli_executable
        result["hub_pub_executable"] = hub_pub_executable
        result["hf_dataset_resume"] = not args.no_resume
        result["hf_dataset_max_retries"] = max_retries
        result["hf_dataset_retry_sleep"] = retry_sleep
        result["hf_dataset_etag_timeout"] = etag_timeout
        result["hf_dataset_download_timeout"] = download_timeout
    else:
        result = prepare_processed_dataset(
            split=split,
            output_dir=output_dir,
            max_episodes=args.max_episodes,
            chunk_size=chunk_size,
        )
    if config:
        result["config_snapshot"] = str(save_config_snapshot(config, output_dir=output_dir))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
