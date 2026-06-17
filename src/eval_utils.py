"""Evaluation helpers for zero-shot tests on CALVIN D."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.paths import ensure_dir, resolve_path
from src.train_utils import (
    build_act_policy,
    import_required_runtime,
    load_training_dataset,
    move_batch_to_device,
    normalize_act_batch,
    resolve_device,
)
from src.wandb_utils import init_wandb, wandb_finish, wandb_log, wandb_log_artifact


@dataclass
class EvalArtifacts:
    """Paths created for one evaluation run."""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    failure_cases_path: Path
    eval_log_path: Path
    task_metrics_path: Path
    episode_metrics_path: Path
    chunk_horizon_metrics_path: Path


def summarize_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract evaluation settings for logging and reporting."""
    ensure_dir(config["output_dir"])
    return {
        "run_name": config["run_name"],
        "test_env": config["test_env"],
        "dataset_path": str(resolve_path(config["dataset_path"])),
        "checkpoint_path": str(resolve_path(config["checkpoint_path"])),
        "checkpoint_exists": checkpoint_file(config["checkpoint_path"]).exists(),
        "max_eval_episodes": config["max_eval_episodes"],
        "offline_eval_only": config["offline_eval_only"],
        "output_dir": config["output_dir"],
    }


def prepare_eval_artifacts(config: dict[str, Any]) -> EvalArtifacts:
    """Create evaluation output files and directories."""
    output_dir = ensure_dir(config["output_dir"])
    return EvalArtifacts(
        output_dir=output_dir,
        metrics_path=output_dir / "metrics.json",
        predictions_path=output_dir / "predictions.jsonl",
        failure_cases_path=output_dir / "failure_cases.json",
        eval_log_path=output_dir / "eval_log.txt",
        task_metrics_path=output_dir / "task_metrics.csv",
        episode_metrics_path=output_dir / "episode_metrics.csv",
        chunk_horizon_metrics_path=output_dir / "chunk_horizon_metrics.csv",
    )


def checkpoint_file(path: str | Path) -> Path:
    """Resolve checkpoint dir/file into an actual .pt file path."""
    resolved = resolve_path(path)
    if resolved.is_dir():
        candidate = resolved / "checkpoint.pt"
        if candidate.exists():
            return candidate
    return resolved


def write_log(path: Path, message: str) -> None:
    """Append one line to eval_log.txt."""
    with path.open("a", encoding="utf-8") as file:
        file.write(message.rstrip() + "\n")


def load_checkpoint(torch: Any, checkpoint_path: str | Path, device: Any) -> dict[str, Any]:
    """Load a checkpoint payload from a directory or file."""
    path = checkpoint_file(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {path}")
    return payload


def load_eval_policy(
    eval_config: dict[str, Any],
    dataset: Any,
    torch: Any,
    lerobot_api: dict[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build ACT policy and load checkpoint weights."""
    payload = load_checkpoint(torch, eval_config["checkpoint_path"], device)
    train_config = payload.get("config", {})
    policy_config = dict(eval_config)
    eval_identity_keys = {
        "dataset_path",
        "output_dir",
        "run_name",
        "checkpoint_path",
        "test_env",
        "hf_dataset_subdirs",
    }
    policy_config.update({key: value for key, value in train_config.items() if key not in eval_identity_keys})
    policy = build_act_policy(policy_config, dataset, lerobot_api).to(device)
    missing, unexpected = policy.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        # Keep evaluation possible across minor LeRobot version differences, but log the mismatch.
        payload["state_dict_load_note"] = {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, payload


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def batch_size_from_batch(batch: dict[str, Any]) -> int:
    """Infer batch size from a collated batch dictionary."""
    for value in batch.values():
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from batch.")


def remove_action_targets(batch: dict[str, Any]) -> dict[str, Any]:
    """Drop supervision and metadata keys before calling action-selection APIs."""
    metadata_keys = {
        "env_id",
        "episode_index",
        "frame_index",
        "task",
        "task_index",
        "source_frame_index",
        "source_episode_index",
    }
    return {
        key: value
        for key, value in batch.items()
        if key not in {"action", "action_chunk", "action_chunk_is_padded", *metadata_keys}
    }


def extract_prediction(output: Any) -> Any:
    """Extract predicted action tensor from common policy outputs."""
    if isinstance(output, dict):
        for key in ["action_pred", "pred_action", "actions_pred", "pred_actions", "action"]:
            if key in output:
                return output[key]
        raise RuntimeError(
            f"Policy output did not contain predicted actions. Available keys: {list(output.keys())}"
        )
    return output


def predict_actions(policy: Any, batch: dict[str, Any]) -> Any:
    """Predict actions with ACT without updating parameters."""
    if hasattr(policy, "model") and hasattr(policy, "config"):
        metadata_keys = {
            "env_id",
            "episode_index",
            "frame_index",
            "task",
            "task_index",
            "source_frame_index",
            "source_episode_index",
        }
        model_batch = {key: value for key, value in batch.items() if key not in metadata_keys}
        image_features = getattr(policy.config, "image_features", None)
        if image_features:
            model_batch["observation.images"] = [model_batch[key] for key in image_features]
        output = policy.model(model_batch)
        if isinstance(output, tuple) and output:
            return output[0]
        return output

    observation_batch = remove_action_targets(batch)
    if hasattr(policy, "select_action"):
        try:
            return policy.select_action(observation_batch)
        except Exception:
            # Some LeRobot versions expect a full batch in forward instead.
            pass
    output = policy(batch)
    return extract_prediction(output)


def target_actions_from_batch(batch: dict[str, Any]) -> Any:
    """Return target actions from an eval batch."""
    if "action_chunk" in batch:
        return batch["action_chunk"]
    if "action" in batch:
        return batch["action"]
    raise KeyError("Evaluation batch is missing target `action` or `action_chunk`.")


def align_prediction_and_target(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted and target actions for L1/error computation."""
    pred = tensor_to_numpy(prediction)
    tgt = tensor_to_numpy(target)
    if pred.ndim == 1:
        pred = pred[None, :]
    if tgt.ndim == 1:
        tgt = tgt[None, :]
    if pred.ndim == 2 and tgt.ndim == 3:
        # select_action commonly predicts one action while target is an action chunk.
        tgt = tgt[:, 0, :]
    if pred.ndim == 3 and tgt.ndim == 2:
        pred = pred[:, 0, :]
    pred = pred.reshape(pred.shape[0], -1)
    tgt = tgt.reshape(tgt.shape[0], -1)
    dim = min(pred.shape[1], tgt.shape[1])
    if dim == 0:
        raise ValueError("Predicted or target action has zero dimensions.")
    return pred[:, :dim], tgt[:, :dim]


def jsonable_batch_value(value: Any, index: int) -> Any:
    """Extract a JSON-friendly per-sample metadata value from a batch."""
    try:
        if isinstance(value, (list, tuple)):
            return value[index]
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] > index:
            item = value[index]
            if hasattr(item, "detach"):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray):
                return item.tolist()
            return item.item() if hasattr(item, "item") else item
    except Exception:
        return None
    return value if isinstance(value, (str, int, float, bool)) else None


def sample_metadata(batch: dict[str, Any], index: int) -> dict[str, Any]:
    """Extract per-sample metadata used for grouped offline analysis."""
    task_index = jsonable_batch_value(batch.get("task_index", -1), index)
    task = jsonable_batch_value(batch.get("task", ""), index) or ""
    env_id = jsonable_batch_value(batch.get("env_id", "D"), index) or "D"
    episode_index = jsonable_batch_value(batch.get("episode_index", -1), index)
    frame_index = jsonable_batch_value(batch.get("frame_index", -1), index)
    return {
        "task_index": int(task_index) if task_index is not None else -1,
        "task": str(task),
        "env_id": str(env_id),
        "episode_index": int(episode_index) if episode_index is not None else -1,
        "frame_index": int(frame_index) if frame_index is not None else -1,
    }


def update_group_stats(
    stats: dict[str, dict[str, Any]],
    key: str,
    *,
    error: float,
    metadata: dict[str, Any],
) -> None:
    """Accumulate count and L1 sums for a task or episode group."""
    row = stats.setdefault(
        key,
        {
            "env_id": metadata["env_id"],
            "task_index": metadata["task_index"],
            "task": metadata["task"],
            "episode_index": metadata["episode_index"],
            "num_samples": 0,
            "action_l1_sum": 0.0,
        },
    )
    row["num_samples"] += 1
    row["action_l1_sum"] += error


def write_group_metrics_csv(path: Path, rows: list[dict[str, Any]], *, include_episode: bool) -> None:
    """Write task-wise or episode-wise action L1 metrics."""
    fields = ["env_id", "task_index", "task"]
    if include_episode:
        fields.append("episode_index")
    fields.extend(["num_samples", "action_l1_loss"])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {
                "env_id": row["env_id"],
                "task_index": row["task_index"],
                "task": row["task"],
                "num_samples": row["num_samples"],
                "action_l1_loss": row["action_l1_sum"] / max(1, row["num_samples"]),
            }
            if include_episode:
                output["episode_index"] = row["episode_index"]
            writer.writerow(output)


def write_chunk_horizon_csv(path: Path, values: list[float]) -> None:
    """Write mean action error for each ACT chunk horizon step."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["chunk_step", "mean_abs_error"])
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({"chunk_step": index + 1, "mean_abs_error": value})


def offline_evaluate_act(config: dict[str, Any]) -> dict[str, Any]:
    """Run offline D evaluation and compute action-error metrics."""
    if config.get("test_env") != "D":
        raise ValueError("Zero-shot evaluation config must use test_env: D.")

    torch, lerobot_api = import_required_runtime()
    artifacts = prepare_eval_artifacts(config)
    device = resolve_device(torch, config["device"])
    wandb_run = init_wandb(config, job_type="eval")
    dataset = load_training_dataset(config, torch, lerobot_api)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=False,
    )
    policy, checkpoint_payload = load_eval_policy(config, dataset, torch, lerobot_api, device)
    train_config = checkpoint_payload.get("config", {})

    write_log(
        artifacts.eval_log_path,
        "Offline evaluation on D: this measures action prediction error only, not rollout success.",
    )
    if checkpoint_payload.get("state_dict_load_note"):
        write_log(artifacts.eval_log_path, json.dumps(checkpoint_payload["state_dict_load_note"]))

    num_samples = 0
    abs_error_sum: np.ndarray | None = None
    l1_sum = 0.0
    task_error_sum: dict[str, float] = {}
    task_count: dict[str, int] = {}
    task_stats: dict[str, dict[str, Any]] = {}
    episode_stats: dict[str, dict[str, Any]] = {}
    chunk_error_sum: np.ndarray | None = None
    chunk_error_count = 0
    max_prediction_records = int(config.get("max_prediction_records", 1000))

    try:
        start = time.perf_counter()
        with artifacts.predictions_path.open("w", encoding="utf-8") as predictions_file:
            with torch.no_grad():
                for batch in dataloader:
                    batch = normalize_act_batch(batch)
                    batch = move_batch_to_device(batch, torch, device)
                    prediction = predict_actions(policy, batch)
                    target = target_actions_from_batch(batch)
                    target_np_raw = tensor_to_numpy(target)
                    pred_np, tgt_np = align_prediction_and_target(prediction, target)
                    abs_error = np.abs(pred_np - tgt_np)
                    sample_l1 = abs_error.mean(axis=1)
                    action_dim = int(target_np_raw.shape[-1]) if target_np_raw.ndim >= 2 else pred_np.shape[1]
                    if action_dim > 0 and abs_error.shape[1] % action_dim == 0:
                        chunk_error = abs_error.reshape(abs_error.shape[0], -1, action_dim).mean(axis=2)
                        if chunk_error_sum is None:
                            chunk_error_sum = chunk_error.sum(axis=0)
                        else:
                            chunk_error_sum += chunk_error.sum(axis=0)
                        chunk_error_count += int(chunk_error.shape[0])

                    if abs_error_sum is None:
                        abs_error_sum = abs_error.sum(axis=0)
                    else:
                        abs_error_sum += abs_error.sum(axis=0)
                    l1_sum += float(sample_l1.sum())

                    batch_n = pred_np.shape[0]
                    for i in range(batch_n):
                        metadata = sample_metadata(batch, i)
                        task = metadata["task"] or f"task_index={metadata['task_index']}"
                        env_id = metadata["env_id"]
                        if env_id != "D":
                            raise ValueError(f"Offline D evaluation received non-D sample env_id={env_id!r}.")
                        task_error_sum[task] = task_error_sum.get(task, 0.0) + float(sample_l1[i])
                        task_count[task] = task_count.get(task, 0) + 1
                        task_key = f"{metadata['task_index']}::{task}"
                        episode_key = f"{metadata['env_id']}::{metadata['episode_index']}"
                        update_group_stats(task_stats, task_key, error=float(sample_l1[i]), metadata=metadata)
                        update_group_stats(episode_stats, episode_key, error=float(sample_l1[i]), metadata=metadata)
                        if num_samples + i < max_prediction_records:
                            predictions_file.write(
                                json.dumps(
                                    {
                                        "sample_index": num_samples + i,
                                        "env_id": env_id,
                                        "episode_index": metadata["episode_index"],
                                        "frame_index": metadata["frame_index"],
                                        "task_index": metadata["task_index"],
                                        "task": task,
                                        "prediction": pred_np[i].tolist(),
                                        "target": tgt_np[i].tolist(),
                                        "abs_error": abs_error[i].tolist(),
                                        "action_l1_error": float(sample_l1[i]),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    num_samples += batch_n

        if num_samples == 0 or abs_error_sum is None:
            raise RuntimeError("Offline evaluation produced zero samples.")

        action_error_by_dim = (abs_error_sum / num_samples).tolist()
        chunk_horizon_error = (
            (chunk_error_sum / chunk_error_count).tolist()
            if chunk_error_sum is not None and chunk_error_count > 0
            else []
        )
        action_l1_loss = l1_sum / num_samples
        task_rows = sorted(task_stats.values(), key=lambda row: (row["task_index"], row["task"]))
        episode_rows = sorted(episode_stats.values(), key=lambda row: (row["episode_index"], row["task_index"]))
        write_group_metrics_csv(artifacts.task_metrics_path, task_rows, include_episode=False)
        write_group_metrics_csv(artifacts.episode_metrics_path, episode_rows, include_episode=True)
        write_chunk_horizon_csv(artifacts.chunk_horizon_metrics_path, chunk_horizon_error)
        metrics = {
            "model_name": config["run_name"],
            "train_envs": train_config.get("train_envs"),
            "test_env": "D",
            "mode": "offline",
            "checkpoint_path": str(checkpoint_file(config["checkpoint_path"])),
            "num_samples": num_samples,
            "num_episodes": None,
            "action_l1_loss": action_l1_loss,
            "action_error_mean": action_l1_loss,
            "action_error_by_dim": action_error_by_dim,
            "chunk_horizon_error": chunk_horizon_error,
            "action_error_by_task": {
                task: task_error_sum[task] / task_count[task] for task in sorted(task_error_sum)
            },
            "task_metrics_path": str(artifacts.task_metrics_path),
            "episode_metrics_path": str(artifacts.episode_metrics_path),
            "chunk_horizon_metrics_path": str(artifacts.chunk_horizon_metrics_path),
            "success_rate": None,
            "avg_episode_length": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - start,
            "max_prediction_records": max_prediction_records,
            "note": "Offline evaluation measures action prediction error on D and is not equivalent to closed-loop task success.",
        }
        artifacts.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.failure_cases_path.write_text("[]\n", encoding="utf-8")
        wandb_log(
            wandb_run,
            {
                "eval/action_l1_loss_on_D": action_l1_loss,
                "eval/action_error_mean_on_D": action_l1_loss,
                "eval/num_samples": num_samples,
            },
        )
        for dim, value in enumerate(action_error_by_dim):
            wandb_log(wandb_run, {f"eval/action_error_dim_{dim}_on_D": value})
        wandb_log_artifact(
            wandb_run,
            artifacts.metrics_path,
            name=f"{config['run_name']}_eval_metrics",
            artifact_type="eval_metrics",
        )
        wandb_log_artifact(
            wandb_run,
            artifacts.predictions_path,
            name=f"{config['run_name']}_predictions",
            artifact_type="predictions",
        )
        return {
            "status": "ok",
            "metrics_path": str(artifacts.metrics_path),
            "predictions_path": str(artifacts.predictions_path),
            "failure_cases_path": str(artifacts.failure_cases_path),
            "eval_log_path": str(artifacts.eval_log_path),
            "task_metrics_path": str(artifacts.task_metrics_path),
            "episode_metrics_path": str(artifacts.episode_metrics_path),
            "chunk_horizon_metrics_path": str(artifacts.chunk_horizon_metrics_path),
            "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
            **metrics,
        }
    finally:
        wandb_finish(wandb_run)


def rollout_evaluate_act(config: dict[str, Any], num_episodes: int, max_steps_per_episode: int) -> dict[str, Any]:
    """Rollout evaluation placeholder for local CALVIN simulator setups."""
    artifacts = prepare_eval_artifacts(config)
    wandb_run = init_wandb(config, job_type="eval")
    metrics = {
        "model_name": config["run_name"],
        "train_envs": None,
        "test_env": "D",
        "mode": "rollout",
        "checkpoint_path": str(checkpoint_file(config["checkpoint_path"])),
        "num_samples": None,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "action_l1_loss": None,
        "action_error_mean": None,
        "action_error_by_dim": None,
        "success_rate": None,
        "avg_episode_length": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Rollout evaluation requires a local CALVIN D simulator. It is not implemented in this scaffold; success_rate is intentionally null.",
    }
    artifacts.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.predictions_path.write_text("", encoding="utf-8")
    artifacts.failure_cases_path.write_text("[]\n", encoding="utf-8")
    write_log(
        artifacts.eval_log_path,
        "Rollout mode requested, but no CALVIN simulator adapter is implemented. success_rate is null, not estimated.",
    )
    wandb_log(wandb_run, {"eval/success_rate_on_D": None, "eval/rollout_available": 0})
    wandb_log_artifact(
        wandb_run,
        artifacts.metrics_path,
        name=f"{config['run_name']}_rollout_metrics",
        artifact_type="eval_metrics",
    )
    wandb_finish(wandb_run)
    return {
        "status": "rollout_unavailable",
        "metrics_path": str(artifacts.metrics_path),
        "predictions_path": str(artifacts.predictions_path),
        "failure_cases_path": str(artifacts.failure_cases_path),
        "eval_log_path": str(artifacts.eval_log_path),
        **metrics,
    }


def evaluate_act(
    config: dict[str, Any],
    *,
    mode: str,
    num_episodes: int,
    max_steps_per_episode: int,
) -> dict[str, Any]:
    """Run ACT evaluation in offline or rollout mode."""
    if config.get("test_env") != "D":
        raise ValueError("This entrypoint is only for zero-shot evaluation on test_env: D.")
    if mode == "offline":
        return offline_evaluate_act(config)
    if mode == "rollout":
        return rollout_evaluate_act(config, num_episodes, max_steps_per_episode)
    raise ValueError("mode must be one of: offline, rollout")
