#!/usr/bin/env python
"""Check whether the current environment can run this project."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value: str) -> bool:
    """Parse command-line boolean values."""
    return value.lower() in {"1", "true", "yes", "y"}


def module_available(name: str) -> bool:
    """Return whether a Python module can be imported."""
    return importlib.util.find_spec(name) is not None


def check_torch() -> tuple[bool, bool, str | None]:
    """Check torch import and CUDA availability."""
    if not module_available("torch"):
        return False, False, None
    import torch

    return True, bool(torch.cuda.is_available()), getattr(torch, "__version__", None)


def path_status(paths: list[str]) -> tuple[list[str], dict[str, bool]]:
    """Return missing paths and per-path status."""
    status = {path: (PROJECT_ROOT / path).exists() for path in paths}
    missing = [path for path, exists in status.items() if not exists]
    return missing, status


def overall_status(failures: list[str], warnings: list[str]) -> str:
    """Return FAIL, WARN, or PASS."""
    if failures:
        return "FAIL"
    if warnings:
        return "WARN"
    return "PASS"


def main() -> None:
    """Run environment checks and print a concise report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--require_cuda", default="false")
    args = parser.parse_args()

    require_cuda = parse_bool(args.require_cuda)
    required_packages = ["numpy", "pandas", "yaml", "tqdm", "matplotlib"]
    optional_packages = ["wandb", "swanlab"]
    torch_available, cuda_available, torch_version = check_torch()
    lerobot_available = module_available("lerobot")

    missing_packages = [pkg for pkg in required_packages if not module_available(pkg)]
    if not torch_available:
        missing_packages.append("torch")
    if not lerobot_available:
        missing_packages.append("lerobot")

    config_paths = [
        "configs/base_act.yaml",
        "configs/train_A_only.yaml",
        "configs/train_ABC.yaml",
        "configs/eval_A_on_D.yaml",
        "configs/eval_ABC_on_D.yaml",
    ]
    data_paths = [f"{args.data_root}/calvin_{env}" for env in ["A", "B", "C", "D"]]
    output_paths = ["outputs", "logs", "report_assets"]
    missing_paths, path_checks = path_status(config_paths + data_paths + output_paths)

    warnings: list[str] = []
    failures: list[str] = []
    if sys.version_info < (3, 12):
        failures.append("Python >= 3.12 is required by LeRobot 0.5.x.")
    if missing_packages:
        failures.append("Missing required packages: " + ", ".join(missing_packages))
    if require_cuda and not cuda_available:
        failures.append("CUDA is required by --require_cuda true but is not available.")
    elif not cuda_available:
        warnings.append("CUDA is not available; CPU/smoke tests can still run, full training may be slow.")
    if missing_paths:
        warnings.append("Some expected paths are missing: " + ", ".join(missing_paths))

    optional_status = {pkg: module_available(pkg) for pkg in optional_packages}
    if not optional_status["wandb"]:
        warnings.append("wandb is not installed; install it if you need WandB visualizations.")

    status = overall_status(failures, warnings)
    report: dict[str, Any] = {
        "status": status,
        "python_version": sys.version.split()[0],
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "lerobot_available": lerobot_available,
        "missing_packages": missing_packages,
        "optional_packages": optional_status,
        "missing_paths": missing_paths,
        "path_checks": path_checks,
        "warnings": warnings,
        "failures": failures,
        "recommendation": "可以继续" if status in {"PASS", "WARN"} else "需要先修复环境",
    }

    print(f"Environment status: {status}")
    print(f"Python: {report['python_version']}")
    print(f"CUDA available: {cuda_available}")
    print(f"LeRobot available: {lerobot_available}")
    if missing_packages:
        print("Missing packages: " + ", ".join(missing_packages))
    if missing_paths:
        print("Missing paths: " + ", ".join(missing_paths))
    for message in warnings:
        print(f"WARN: {message}")
    for message in failures:
        print(f"FAIL: {message}")
    print("Recommendation: " + report["recommendation"])
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
