"""Training helpers for ACT experiments."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.calvin_v21_act_dataset import CalvinV21ActDataset
from src.lerobot_bridge import CalvinLeRobotDataset, load_lerobot_source_manifest
from src.paths import ensure_dir, resolve_path
from src.wandb_utils import init_wandb, wandb_finish, wandb_log, wandb_log_artifact


@dataclass
class TrainArtifacts:
    """Paths created for one training run."""

    output_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_path: Path


def summarize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract key hyperparameters for logging and reporting."""
    ensure_dir(config["output_dir"])
    return {
        "run_name": config["run_name"],
        "policy_type": config["policy_type"],
        "train_envs": config["train_envs"],
        "dataset_path": config["dataset_path"],
        "output_dir": config["output_dir"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "optimizer": config["optimizer"],
        "loss_function": config["loss_function"],
        "num_train_steps": config["num_train_steps"],
        "validation_split": config["validation_split"],
        "max_val_batches": config["max_val_batches"],
        "chunk_size": config["chunk_size"],
        "temporal_ensemble": config["temporal_ensemble"],
    }


def prepare_train_artifacts(config: dict[str, Any]) -> TrainArtifacts:
    """Create training output directories."""
    output_dir = ensure_dir(config["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")
    metrics_path = output_dir / "train_metrics.csv"
    return TrainArtifacts(output_dir, checkpoints_dir, logs_dir, metrics_path)


def import_required_runtime() -> tuple[Any, dict[str, Any]]:
    """Import torch and LeRobot classes, raising a clear error if unavailable."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ACT training/evaluation but is not installed. "
            "Install dependencies first, e.g. `pip install -r requirements.txt`."
        ) from exc

    candidates = [
        {
            "dataset_module": "lerobot.datasets.lerobot_dataset",
            "policy_module": "lerobot.policies.act.modeling_act",
            "config_module": "lerobot.policies.act.configuration_act",
        },
        {
            "dataset_module": "lerobot.datasets",
            "policy_module": "lerobot.policies.act",
            "config_module": "lerobot.policies.act.configuration_act",
        },
        {
            "dataset_module": "lerobot.datasets.lerobot_dataset",
            "policy_module": "lerobot.policies.act",
            "config_module": "lerobot.policies.act.configuration_act",
        },
        {
            "dataset_module": "lerobot.common.datasets.lerobot_dataset",
            "policy_module": "lerobot.common.policies.act.modeling_act",
            "config_module": "lerobot.common.policies.act.configuration_act",
        },
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            dataset_module = importlib.import_module(candidate["dataset_module"])
            policy_module = importlib.import_module(candidate["policy_module"])
            try:
                config_module = importlib.import_module(candidate["config_module"])
                act_config = getattr(config_module, "ACTConfig")
            except (ImportError, AttributeError):
                act_config = getattr(policy_module, "ACTConfig")
            return torch, {
                "LeRobotDataset": getattr(dataset_module, "LeRobotDataset"),
                "ACTPolicy": getattr(policy_module, "ACTPolicy"),
                "ACTConfig": act_config,
            }
        except (ImportError, AttributeError) as exc:
            last_error = exc

    raise RuntimeError(
        "LeRobot is required for ACT training/evaluation, but compatible ACT APIs were not found. "
        "Tried `lerobot.datasets`/`lerobot.policies.act` and legacy `lerobot.common.*` paths. "
        "Install or update LeRobot, e.g. `pip install git+https://github.com/huggingface/lerobot.git`."
    ) from last_error


def resolve_device(torch: Any, device_name: str) -> Any:
    """Resolve auto/cuda/cpu/mps into a torch.device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    return torch.device(device_name)


def instantiate_lerobot_dataset(dataset_cls: Any, source: dict[str, Any]) -> Any:
    """Instantiate LeRobotDataset across common constructor signatures."""
    path = resolve_path(source["path"])
    attempts = [
        lambda: dataset_cls(root=path),
        lambda: dataset_cls(str(path)),
        lambda: dataset_cls(repo_id=str(path)),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        f"Could not instantiate LeRobotDataset for {path}. Constructor attempts failed: {errors}"
    )


def load_training_dataset(config: dict[str, Any], torch: Any, lerobot_api: dict[str, Any]) -> Any:
    """Load the processed dataset for training."""
    dataset_dir = resolve_path(config["dataset_path"])
    sources = load_lerobot_source_manifest(dataset_dir)
    if (dataset_dir / "source_datasets.json").exists():
        if config.get("dataset_adapter", "calvin_v21_parquet") == "calvin_v21_parquet":
            return CalvinV21ActDataset(
                [source["path"] for source in sources],
                env_ids=[source["env_id"] for source in sources],
                chunk_size=int(config["chunk_size"]),
                max_episodes_per_split=config.get("max_train_episodes_per_split"),
                cache_size=int(config.get("dataset_cache_size", 32)),
            )
        datasets = [
            EnvTaggedDataset(instantiate_lerobot_dataset(lerobot_api["LeRobotDataset"], source), source["env_id"])
            for source in sources
        ]
        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)
    return CalvinLeRobotDataset(dataset_dir, chunk_size=int(config["chunk_size"]))


class EnvTaggedDataset:
    """Attach env_id to samples returned by an official LeRobotDataset."""

    def __init__(self, dataset: Any, env_id: str) -> None:
        self.dataset = dataset
        self.env_id = env_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if isinstance(item, dict):
            item = dict(item)
            item.setdefault("env_id", self.env_id)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def infer_features_from_dataset(dataset: Any) -> tuple[Any, Any]:
    """Extract input/output features if the LeRobot dataset exposes them."""
    source = dataset
    if hasattr(dataset, "datasets") and dataset.datasets:
        source = dataset.datasets[0]
    if isinstance(source, EnvTaggedDataset):
        source = source.dataset

    features = getattr(source, "features", None)
    meta = getattr(source, "meta", None)
    if features is None and meta is not None:
        features = getattr(meta, "features", None)
    if features is None:
        return None, None

    input_features: dict[str, Any] = {}
    output_features: dict[str, Any] = {}
    for key, value in dict(features).items():
        if key.startswith("observation."):
            input_features[key] = value
        if key == "action" or key.startswith("action."):
            output_features[key] = value
    return input_features or None, output_features or None


def build_act_policy(config: dict[str, Any], dataset: Any, lerobot_api: dict[str, Any]) -> Any:
    """Build a LeRobot ACT policy with version-tolerant config arguments."""
    act_config_cls = lerobot_api["ACTConfig"]
    act_policy_cls = lerobot_api["ACTPolicy"]
    input_features, output_features = infer_features_from_dataset(dataset)

    requested = {
        "input_features": input_features,
        "output_features": output_features,
        "chunk_size": config["chunk_size"],
        "n_action_steps": config["chunk_size"],
        "pretrained_backbone_weights": config.get("pretrained_backbone_weights"),
    }
    signature = inspect.signature(act_config_cls)
    kwargs = {
        key: value
        for key, value in requested.items()
        if key in signature.parameters and (value is not None or key == "pretrained_backbone_weights")
    }
    try:
        policy_config = act_config_cls(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Could not construct LeRobot ACTConfig from dataset features. "
            f"Attempted kwargs: {kwargs}. Original error: {exc}"
        ) from exc

    try:
        return act_policy_cls(policy_config)
    except TypeError:
        return act_policy_cls(policy_config, dataset_stats=getattr(dataset, "stats", None))


def move_batch_to_device(batch: Any, torch: Any, device: Any) -> Any:
    """Move tensor values in a nested batch to device."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, torch, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, torch, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, torch, device) for value in batch)
    return batch


def normalize_act_batch(batch: Any) -> Any:
    """Normalize project-local batch keys before feeding ACTPolicy."""
    if isinstance(batch, dict) and "action_chunk" in batch:
        normalized = dict(batch)
        normalized["action"] = normalized["action_chunk"]
        return normalized
    return batch


def extract_loss(output: Any) -> tuple[Any, Any]:
    """Extract total loss and action L1 loss from ACT policy output."""
    if isinstance(output, dict):
        total_loss = output.get("loss")
        if total_loss is None:
            total_loss = output.get("total_loss")
        action_l1 = output.get("l1_loss")
        if action_l1 is None:
            action_l1 = output.get("action_l1_loss")
        if action_l1 is None:
            action_l1 = output.get("train_action_l1_loss")
        if action_l1 is None:
            action_l1 = total_loss
        if total_loss is None:
            scalar_values = [value for value in output.values() if hasattr(value, "ndim") and value.ndim == 0]
            if scalar_values:
                total_loss = scalar_values[0]
                action_l1 = action_l1 or total_loss
    elif isinstance(output, tuple) and output:
        total_loss = output[0]
        metrics = output[1] if len(output) > 1 and isinstance(output[1], dict) else {}
        action_l1 = metrics.get("l1_loss") or metrics.get("train_action_l1_loss") or total_loss
    else:
        total_loss = output
        action_l1 = output
    if total_loss is None:
        raise RuntimeError(
            "ACT policy output did not contain a usable loss. Expected a tensor or dict with `loss`."
        )
    return total_loss, action_l1


def scalar_to_float(value: Any) -> float:
    """Convert tensor-like or Python scalar values to float."""
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def save_checkpoint(policy: Any, optimizer: Any, step: int, checkpoint_dir: Path, config: dict[str, Any]) -> Path:
    """Save a training checkpoint in a directory."""
    torch = importlib.import_module("torch")
    ensure_dir(checkpoint_dir)
    payload = {
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(payload, checkpoint_dir / "checkpoint.pt")
    return checkpoint_dir


def write_metrics_header(metrics_path: Path) -> None:
    """Initialize the train metrics CSV."""
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "train_action_l1_loss",
                "total_loss",
                "val_action_l1_loss",
                "val_total_loss",
                "learning_rate",
                "time_per_step",
            ],
        )
        writer.writeheader()


def append_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    """Append one metrics row to CSV."""
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writerow(row)


def split_dataset_for_validation(dataset: Any, torch: Any, config: dict[str, Any]) -> tuple[Any, Any | None]:
    """Split dataset into train/validation subsets with a fixed seed."""
    validation_split = float(config.get("validation_split", 0.0))
    if validation_split <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * validation_split))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset


def compute_validation_metrics(
    policy: Any,
    dataloader: Any,
    torch: Any,
    device: Any,
    max_batches: int,
) -> dict[str, float]:
    """Compute validation action L1 and total loss without updating parameters."""
    was_training = policy.training
    policy.train()
    total_loss_sum = 0.0
    action_l1_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            batch = normalize_act_batch(batch)
            batch = move_batch_to_device(batch, torch, device)
            output = policy(batch)
            total_loss, action_l1_loss = extract_loss(output)
            total_loss_sum += scalar_to_float(total_loss)
            action_l1_sum += scalar_to_float(action_l1_loss)
            count += 1
    policy.train(was_training)
    if count == 0:
        return {"val_action_l1_loss": float("nan"), "val_total_loss": float("nan")}
    return {
        "val_action_l1_loss": action_l1_sum / count,
        "val_total_loss": total_loss_sum / count,
    }


def train_act(config: dict[str, Any]) -> dict[str, Any]:
    """Train LeRobot ACT with a stable project-local loop."""
    torch, lerobot_api = import_required_runtime()
    artifacts = prepare_train_artifacts(config)
    device = resolve_device(torch, config["device"])
    wandb_run = init_wandb(config, job_type="train")

    dataset = load_training_dataset(config, torch, lerobot_api)
    train_dataset, val_dataset = split_dataset_for_validation(dataset, torch, config)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        drop_last=True,
        pin_memory=str(device) == "cuda",
        persistent_workers=int(config["num_workers"]) > 0
        and bool(config.get("persistent_workers", True)),
        prefetch_factor=int(config.get("prefetch_factor", 4))
        if int(config["num_workers"]) > 0
        else None,
    )
    if len(dataloader) == 0:
        raise RuntimeError(
            "Training dataloader is empty. Reduce batch_size or verify dataset preparation."
        )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            drop_last=False,
            pin_memory=str(device) == "cuda",
            persistent_workers=int(config["num_workers"]) > 0
            and bool(config.get("persistent_workers", True)),
            prefetch_factor=int(config.get("prefetch_factor", 4))
            if int(config["num_workers"]) > 0
            else None,
        )

    policy = build_act_policy(config, dataset, lerobot_api).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    write_metrics_header(artifacts.metrics_path)
    step = 0
    data_iter = iter(dataloader)
    last_checkpoint: Path | None = None
    try:
        while step < int(config["num_train_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            step_start = time.perf_counter()
            batch = normalize_act_batch(batch)
            batch = move_batch_to_device(batch, torch, device)
            optimizer.zero_grad(set_to_none=True)
            output = policy(batch)
            total_loss, action_l1_loss = extract_loss(output)
            total_loss.backward()
            if float(config["gradient_clip_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            elapsed = time.perf_counter() - step_start
            step += 1

            row = {
                "step": step,
                "train_action_l1_loss": scalar_to_float(action_l1_loss),
                "total_loss": scalar_to_float(total_loss),
                "val_action_l1_loss": "",
                "val_total_loss": "",
                "learning_rate": optimizer.param_groups[0]["lr"],
                "time_per_step": elapsed,
            }
            if val_dataloader is not None and (step % int(config["eval_interval"]) == 0 or step == 1):
                val_metrics = compute_validation_metrics(
                    policy,
                    val_dataloader,
                    torch,
                    device,
                    int(config["max_val_batches"]),
                )
                row.update(val_metrics)
                wandb_log(
                    wandb_run,
                    {
                        "val/action_l1_loss": row["val_action_l1_loss"],
                        "val/total_loss": row["val_total_loss"],
                    },
                    step=step,
                )
            append_metrics(artifacts.metrics_path, row)
            wandb_log(
                wandb_run,
                {
                    "train/train_action_l1_loss": row["train_action_l1_loss"],
                    "train/total_loss": row["total_loss"],
                    "train/learning_rate": row["learning_rate"],
                    "train/time_per_step": row["time_per_step"],
                },
                step=step,
            )

            if step % int(config["log_interval"]) == 0 or step == 1:
                print(json.dumps(row, ensure_ascii=False))
            if step % int(config["save_interval"]) == 0:
                last_checkpoint = save_checkpoint(
                    policy,
                    optimizer,
                    step,
                    artifacts.checkpoints_dir / f"step_{step:06d}",
                    config,
                )

        final_checkpoint = save_checkpoint(
            policy,
            optimizer,
            step,
            artifacts.checkpoints_dir / "final",
            config,
        )
        wandb_log_artifact(
            wandb_run,
            artifacts.metrics_path,
            name=f"{config['run_name']}_train_metrics",
            artifact_type="train_metrics",
        )
        wandb_log_artifact(
            wandb_run,
            final_checkpoint / "checkpoint.pt",
            name=f"{config['run_name']}_final_checkpoint",
            artifact_type="checkpoint",
        )
        return {
            "status": "ok",
            "run_name": config["run_name"],
            "num_train_steps": step,
            "metrics_path": str(artifacts.metrics_path),
            "last_periodic_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "final_checkpoint": str(final_checkpoint),
            "device": str(device),
            "dataset_size": len(dataset),
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
            "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
        }
    finally:
        wandb_finish(wandb_run)
