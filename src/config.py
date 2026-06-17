"""Configuration loading, validation, overrides, and audit helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.paths import ensure_dir, resolve_path


COMMON_HYPERPARAM_KEYS = [
    "project_name",
    "seed",
    "device",
    "policy_type",
    "network_architecture",
    "image_keys",
    "state_key",
    "action_key",
    "task_key",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "loss_function",
    "gradient_clip_norm",
    "num_train_steps",
    "log_interval",
    "save_interval",
    "eval_interval",
    "validation_split",
    "max_val_batches",
    "chunk_size",
    "temporal_ensemble",
    "num_workers",
    "use_wandb",
    "use_swanlab",
    "offline_eval_only",
    "max_eval_episodes",
]

OPTIONAL_DATA_SOURCE_KEYS = [
    "hf_dataset_repo",
    "hf_dataset_revision",
    "hf_dataset_local_dir",
    "hf_dataset_endpoint",
    "hf_dataset_max_workers",
    "hf_dataset_max_retries",
    "hf_dataset_retry_sleep",
    "hf_dataset_etag_timeout",
    "hf_dataset_download_timeout",
    "hf_dataset_subdirs",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base dictionary."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with resolve_path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_override_value(raw_value: str) -> Any:
    """Parse a command-line override value using YAML scalar/list syntax."""
    return yaml.safe_load(raw_value)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a config value using a dotted key such as training.batch_size."""
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set '{dotted_key}': '{key}' is not a mapping.")
        cursor = existing
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply key=value command-line overrides to a config dictionary."""
    merged = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid override '{item}'. Override key is empty.")
        set_by_dotted_key(merged, key, parse_override_value(raw_value))
    return merged


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load an experiment config, merge its base config, and apply overrides."""
    config = load_yaml(path)
    base_config = config.get("base_config")
    if base_config:
        config = deep_merge(load_yaml(base_config), config)
    config = apply_overrides(config, overrides)
    if validate:
        validate_config(config)
    return config


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Save the fully merged config to output_dir/config_snapshot.yaml."""
    resolved_output = ensure_dir(output_dir or config["output_dir"])
    snapshot_path = resolved_output / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return snapshot_path


def common_hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of config fields that must match across fair runs."""
    return {key: deepcopy(config.get(key)) for key in COMMON_HYPERPARAM_KEYS}


def compare_common_hyperparams(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return differing common hyperparameters between two merged configs."""
    left_common = common_hyperparams(left)
    right_common = common_hyperparams(right)
    return {
        key: (left_common.get(key), right_common.get(key))
        for key in COMMON_HYPERPARAM_KEYS
        if left_common.get(key) != right_common.get(key)
    }


def validate_config(config: dict[str, Any]) -> None:
    """Validate key fields needed by train and eval entrypoints."""
    required_common = {
        "project_name": str,
        "seed": int,
        "device": str,
        "policy_type": str,
        "network_architecture": str,
        "image_keys": list,
        "state_key": str,
        "action_key": str,
        "task_key": str,
        "batch_size": int,
        "learning_rate": (int, float),
        "weight_decay": (int, float),
        "optimizer": str,
        "loss_function": str,
        "gradient_clip_norm": (int, float),
        "num_train_steps": int,
        "log_interval": int,
        "save_interval": int,
        "eval_interval": int,
        "validation_split": (int, float),
        "max_val_batches": int,
        "chunk_size": int,
        "temporal_ensemble": bool,
        "num_workers": int,
        "use_wandb": bool,
        "use_swanlab": bool,
        "offline_eval_only": bool,
        "max_eval_episodes": int,
    }
    for key, expected_type in required_common.items():
        if key not in config:
            raise ValueError(f"Missing required config field: {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(
                f"Config field '{key}' has type {type(config[key]).__name__}, "
                f"expected {expected_type}."
            )

    if config["policy_type"] != "act":
        raise ValueError("This project expects policy_type: act.")
    if config["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps.")
    if not config["image_keys"]:
        raise ValueError("image_keys must contain at least one image observation key.")

    positive_int_keys = [
        "batch_size",
        "num_train_steps",
        "log_interval",
        "save_interval",
        "eval_interval",
        "max_val_batches",
        "chunk_size",
        "max_eval_episodes",
    ]
    for key in positive_int_keys:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if config["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative.")
    if "hf_dataset_max_workers" in config:
        if not isinstance(config["hf_dataset_max_workers"], int):
            raise TypeError("hf_dataset_max_workers must be an integer.")
        if config["hf_dataset_max_workers"] <= 0:
            raise ValueError("hf_dataset_max_workers must be positive.")
    if "hf_dataset_max_retries" in config:
        if not isinstance(config["hf_dataset_max_retries"], int):
            raise TypeError("hf_dataset_max_retries must be an integer.")
        if config["hf_dataset_max_retries"] < 0:
            raise ValueError("hf_dataset_max_retries must be non-negative.")
    if "hf_dataset_retry_sleep" in config and config["hf_dataset_retry_sleep"] < 0:
        raise ValueError("hf_dataset_retry_sleep must be non-negative.")
    if "hf_dataset_etag_timeout" in config and config["hf_dataset_etag_timeout"] <= 0:
        raise ValueError("hf_dataset_etag_timeout must be positive.")
    if "hf_dataset_download_timeout" in config and config["hf_dataset_download_timeout"] <= 0:
        raise ValueError("hf_dataset_download_timeout must be positive.")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative.")
    if not 0 <= config["validation_split"] < 1:
        raise ValueError("validation_split must be in [0, 1).")
    if config["optimizer"] != "adamw":
        raise ValueError("This training scaffold currently expects optimizer: adamw.")
    if config["loss_function"] != "action_l1":
        raise ValueError("This training scaffold currently expects loss_function: action_l1.")
    if config["gradient_clip_norm"] < 0:
        raise ValueError("gradient_clip_norm must be non-negative.")

    required_experiment = ["run_name", "dataset_path", "output_dir"]
    for key in required_experiment:
        if key not in config:
            raise ValueError(f"Missing required experiment field: {key}")

    is_eval = "checkpoint_path" in config or "test_env" in config
    if is_eval:
        for key in ["checkpoint_path", "test_env"]:
            if key not in config:
                raise ValueError(f"Missing required evaluation field: {key}")
    else:
        if "train_envs" not in config:
            raise ValueError("Missing required training field: train_envs")
        if not isinstance(config["train_envs"], list) or not config["train_envs"]:
            raise TypeError("train_envs must be a non-empty list.")
