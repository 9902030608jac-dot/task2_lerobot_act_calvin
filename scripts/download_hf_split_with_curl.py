#!/usr/bin/env python
"""Download selected HF dataset subdirectories through hf-mirror with curl.

This avoids `hf download --include split/**` recursively listing the full repo
tree, which can hang on some cloud hosts. It lists each directory one level at
a time through the Hub API, then downloads concrete file URLs with curl resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def curl_json(url: str, retries: int, retry_sleep: float) -> list[dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = run(
                [
                    "curl",
                    "-L",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "120",
                    "-sS",
                    url,
                ],
                capture=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}") from last_error


def list_files(endpoint: str, repo_id: str, revision: str, root: str, retries: int, retry_sleep: float) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    queue: deque[str] = deque([root.strip("/")])
    while queue:
        path = queue.popleft()
        encoded_path = quote(path, safe="/")
        url = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/tree/{revision}/{encoded_path}?recursive=false"
        entries = curl_json(url, retries=retries, retry_sleep=retry_sleep)
        print(f"listed {path}: {len(entries)} entries", file=sys.stderr, flush=True)
        for entry in entries:
            entry_type = entry.get("type")
            entry_path = str(entry["path"])
            if entry_type == "directory":
                queue.append(entry_path)
            elif entry_type == "file":
                files.append(entry)
            else:
                print(f"Skipping unknown entry type {entry_type!r}: {entry}", file=sys.stderr)
    return files


def download_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    file_path: str,
    size: int | None,
    local_dir: Path,
    retries: int,
) -> str:
    destination = local_dir / file_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if size is not None and destination.exists() and destination.stat().st_size == size:
        return "skip"

    url_path = quote(file_path, safe="/")
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{url_path}"
    command = [
        "curl",
        "-L",
        "-C",
        "-",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--silent",
        "--show-error",
        "-o",
        str(destination),
        url,
    ]
    run(command)
    if size is not None and destination.stat().st_size != size:
        raise RuntimeError(f"Downloaded size mismatch for {destination}: got {destination.stat().st_size}, expected {size}")
    return "download"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local_dir", default="data/raw/xiaoma26_calvin_lerobot")
    parser.add_argument("--subdir", action="append", required=True)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    local_dir = Path(args.local_dir)
    all_files: list[dict[str, object]] = []
    for subdir in args.subdir:
        all_files.extend(
            list_files(
                args.endpoint,
                args.repo_id,
                args.revision,
                subdir,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
        )

    manifest_path = Path(args.manifest) if args.manifest else local_dir / "curl_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(all_files, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest written: {manifest_path} ({len(all_files)} files)", file=sys.stderr, flush=True)

    downloaded = 0
    skipped = 0
    completed = 0

    def fetch(file_entry: dict[str, object]) -> tuple[str, str]:
        file_path = str(file_entry["path"])
        size = file_entry.get("size")
        status = download_file(
            args.endpoint,
            args.repo_id,
            args.revision,
            file_path,
            int(size) if isinstance(size, int) else None,
            local_dir,
            retries=args.retries,
        )
        return status, file_path

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, file_entry) for file_entry in all_files]
        for future in as_completed(futures):
            status, file_path = future.result()
            completed += 1
            if status == "skip":
                skipped += 1
            else:
                downloaded += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(all_files),
                        "status": status,
                        "path": file_path,
                        "downloaded": downloaded,
                        "skipped": skipped,
                        "workers": args.workers,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
