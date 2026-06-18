# task2_lerobot_act_calvin

本仓库用于完成课程题目二：基于 LeRobot ACT 的 CALVIN 跨环境泛化实验。实验采用统一的训练、评估和统计协议，比较只在环境 A 训练的 ACT policy 与在环境 A/B/C 联合训练的 ACT policy 在未见环境 D 上的 zero-shot 表现。

## Research Objective

实验回答两个问题：

1. 多环境训练是否比单环境训练带来更好的 zero-shot 泛化。
2. ACT 的 Action Chunking 在跨环境视觉分布偏移下是否保持稳定的动作序列预测。

核心对比：

| Model | Training Environments | Test Environment | Policy |
| --- | --- | --- | --- |
| ACT-A-only | A | D | LeRobot ACT |
| ACT-ABC | A/B/C | D | LeRobot ACT |

环境 D 只用于最终 zero-shot evaluation，不参与训练、验证、调参或模型选择。

## Experimental Design

本项目的默认正式实验是四 seed 受控对比：

```text
seeds = 42, 43, 44, 45
models = ACT-A-only, ACT-ABC
training jobs = 2 models x 4 seeds = 8
evaluation jobs = 8 checkpoints x D offline evaluation
```

每一个 seed 训练出的 checkpoint 都必须参与完整 D 评估。最终报告使用 `mean ± std`，并保留 per-seed 指标。

受控变量：

| Category | Controlled Setting |
| --- | --- |
| Algorithm | LeRobot ACT |
| Inputs | static RGB image, wrist RGB image, robot state |
| Output | action chunk |
| Test split | CALVIN D |
| Metrics | offline Action L1, task-wise Action L1, episode-wise distribution, chunk-horizon Action L1 |
| Difference | training data source only |

ACT-A-only 与 ACT-ABC 必须保持相同网络结构、chunk size、batch size、learning rate、optimizer、loss、训练步数、数据预处理和评估脚本。

## Evaluation Protocol

主评估指标是 D 环境 offline Action L1。该指标在 D 专家轨迹上比较模型预测动作与专家动作：

```text
offline_action_l1 = mean(abs(predicted_action_chunk - expert_action_chunk))
```

该评估是 open-loop / offline evaluation，不等价于 closed-loop Success Rate。题目允许评估 Success Rate 或动作误差，本仓库默认采用动作误差，并将 online rollout Success Rate 作为探索性扩展。

正式 D 评估输出四类结果：

| Output | Purpose |
| --- | --- |
| `metrics.json` | 整体 D Action L1 和 action-dimension error |
| `task_metrics.csv` | 按 `task_index` / task description 聚合的 D Action L1 |
| `episode_metrics.csv` | 按 episode 聚合的轨迹级 Action L1 |
| `chunk_horizon_metrics.csv` | ACT chunk 内第 1 到第 `chunk_size` 步的 Action L1 |

这四类指标共同支撑报告分析：

- 整体 D Action L1：回答哪个训练设置整体更好。
- task-wise Action L1：回答哪些任务更受益于多环境训练。
- episode-wise distribution：观察轨迹级误差分布和高误差 episode。
- chunk-horizon Action L1：分析 ACT Action Chunking 在未来动作序列上的稳定性。

## Dataset Interface

推荐数据源：

```text
xiaoma26/calvin-lerobot
```

数据放置路径：

```text
data/raw/xiaoma26_calvin_lerobot/
├── splitA/
├── splitB/
├── splitC/
└── splitD/
```

每个 split 采用 LeRobot v2.1 风格目录：

```text
splitD/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
└── data/
    └── chunk-xxx/
        └── episode_xxxxxx.parquet
```

D split 中的层级关系：

```text
task description
  -> multiple episodes
episode
  -> continuous trajectory
frame
  -> one timestep
```

`episodes.jsonl` 示例：

```json
{
  "episode_index": 0,
  "tasks": ["move_slider_left: move the door to the left side"],
  "length": 65,
  "source_episode_index": 0,
  "source_start_frame": 315660,
  "source_end_frame": 315724,
  "scene": "D"
}
```

字段含义：

| Field | Meaning |
| --- | --- |
| `episode_index` | 当前 split 内的 episode ID |
| `tasks` | 该 episode 对应的任务描述 |
| `length` | episode 内 frame 数 |
| `source_*` | 原始 CALVIN 数据流中的来源位置 |
| `scene` | 环境 ID |

parquet frame 字段：

```text
image
wrist_image
state
actions
timestamp
frame_index
episode_index
index
task_index
source_frame_index
source_episode_index
```

## ACT Dataset Adapter

仓库中的 `CalvinV21ActDataset` 负责把 CALVIN v2.1 parquet 数据映射为 LeRobot ACTPolicy batch。模型、loss、optimizer 和反向传播仍由 LeRobot ACT 实现；adapter 只负责数据接口。

字段映射：

| CALVIN v2.1 Field | ACT Batch Field |
| --- | --- |
| `image` | `observation.images.rgb_static` |
| `wrist_image` | `observation.images.rgb_gripper` |
| `state` | `observation.state` |
| future `actions` | `action` |
| episode-end padding | `action_is_pad` |
| `task_index` + `tasks.jsonl` | `task_index`, `task` |
| episode metadata | `episode_index`, `frame_index`, `env_id` |

Action chunk 构造规则：

```text
sample at frame t -> actions[t : t + chunk_size]
```

chunk 只在同一个 episode 内构造。若 episode 剩余长度不足 `chunk_size`，则使用末尾动作 padding，并通过 `action_is_pad` 标记。

## Repository Structure

```text
task2_lerobot_act_calvin/
├── configs/              # Training and evaluation configs
├── scripts/              # Command entrypoints
├── src/                  # Dataset, training, evaluation, plotting utilities
├── data/                 # Ignored by git
├── outputs/              # Ignored by git
├── logs/                 # Ignored by git
├── checkpoints/          # Ignored by git
└── report_assets/        # Generated report tables and notes
```

`.gitignore` excludes datasets, checkpoints, logs, WandB cache, and generated outputs.

## Environment Setup

Recommended environment:

```bash
conda create -n lerobot-act-calvin python=3.12 -y
conda activate lerobot-act-calvin
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If the machine requires a different PyTorch CUDA build, install PyTorch first, then install the remaining dependencies:

```bash
pip install -r requirements-no-torch.txt
```

Environment check:

```bash
python scripts/00_check_env.py --data_root data/raw --require_cuda true
```

For remote GPU runs, use `tmux`, `screen`, or `nohup` so that jobs survive SSH disconnects.

## Data Acquisition

The raw CALVIN LeRobot dataset should be downloaded before manifest generation. The default target directory is:

```text
data/raw/xiaoma26_calvin_lerobot/
```

The repository provides two supported download paths.

### Option A: Resumable multi-thread curl downloader

This is the recommended path for large remote GPU machines. It lists the requested split directories through the Hugging Face Hub API, downloads concrete files with `curl`, uses `curl -C -` for resume, skips files whose local size already matches the remote size, and runs multiple workers in parallel.

Download all four splits:

```bash
python scripts/download_hf_split_with_curl.py \
  --repo_id xiaoma26/calvin-lerobot \
  --revision main \
  --endpoint https://hf-mirror.com \
  --local_dir data/raw/xiaoma26_calvin_lerobot \
  --subdir splitA \
  --subdir splitB \
  --subdir splitC \
  --subdir splitD \
  --workers 16 \
  --retries 30 \
  --retry_sleep 10
```

For a smaller first-stage setup, download only the required splits:

```bash
# A-only training plus D evaluation
python scripts/download_hf_split_with_curl.py \
  --endpoint https://hf-mirror.com \
  --local_dir data/raw/xiaoma26_calvin_lerobot \
  --subdir splitA \
  --subdir splitD \
  --workers 16 \
  --retries 30 \
  --retry_sleep 10

# Add B/C before ACT-ABC training
python scripts/download_hf_split_with_curl.py \
  --endpoint https://hf-mirror.com \
  --local_dir data/raw/xiaoma26_calvin_lerobot \
  --subdir splitB \
  --subdir splitC \
  --workers 16 \
  --retries 30 \
  --retry_sleep 10
```

The downloader writes:

```text
data/raw/xiaoma26_calvin_lerobot/curl_download_manifest.json
```

It can be rerun safely after interruption. Existing complete files are marked as `skip`, incomplete files resume from the previous byte offset.

### Option B: HF CLI through prepare script

The config also supports Hugging Face CLI download through the prepare script:

```bash
python scripts/02_prepare_dataset.py \
  --config configs/train_A_only.yaml \
  --hf_downloader hf_cli \
  --hf_endpoint https://hf-mirror.com \
  --hf_max_workers 8

python scripts/02_prepare_dataset.py \
  --config configs/train_ABC.yaml \
  --hf_downloader hf_cli \
  --hf_endpoint https://hf-mirror.com \
  --hf_max_workers 8

python scripts/02_prepare_dataset.py \
  --config configs/eval_A_on_D.yaml \
  --hf_downloader hf_cli \
  --hf_endpoint https://hf-mirror.com \
  --hf_max_workers 8
```

`configs/base_act.yaml` defaults to:

```yaml
hf_dataset_endpoint: https://hf-mirror.com
hf_dataset_downloader: hf_cli
hf_dataset_max_workers: 8
hf_dataset_max_retries: 30
hf_dataset_retry_sleep: 10.0
```

If `hf` is unavailable or recursively listing a large dataset hangs on the host, use Option A.

## Manifest Preparation

After raw split download, create processed manifests:

```bash
python scripts/02_prepare_dataset.py --config configs/train_A_only.yaml
python scripts/02_prepare_dataset.py --config configs/train_ABC.yaml
python scripts/02_prepare_dataset.py --config configs/eval_A_on_D.yaml
```

The processed directories store lightweight manifests pointing back to the raw v2.1 split directories. Formal training and evaluation read parquet files through `CalvinV21ActDataset`.

## Smoke Tests

Before formal training:

```bash
python scripts/03_train_act.py \
  --config configs/train_A_only.yaml \
  --device cuda \
  --override num_train_steps=10 save_interval=5 log_interval=1 use_wandb=false max_train_episodes_per_split=128

python scripts/03_train_act.py \
  --config configs/train_ABC.yaml \
  --device cuda \
  --override num_train_steps=10 save_interval=5 log_interval=1 use_wandb=false max_train_episodes_per_split=64
```

Smoke tests verify data loading, ACT forward/backward, checkpoint writing, and basic metric logging. They are not reported as final experiments.

## Formal Multi-seed Training

`configs/base_act.yaml` 默认启用 WandB：

```yaml
use_wandb: true
project_name: task2_lerobot_act_calvin
```

因此 `bash scripts/run_multiseed_4gpu.sh` 会为 8 个训练任务分别创建 WandB run。脚本额外设置：

```bash
WANDB_GROUP=multiseed_act_calvin
```

用于在 WandB 中把这些 seed run 归为同一组。运行前应先完成：

```bash
wandb login
```

如果远程网络暂时不稳定，可以使用离线记录：

```bash
WANDB_MODE=offline bash scripts/run_multiseed_4gpu.sh
wandb sync wandb/
```

训练过程记录的关键指标包括 `train/train_action_l1_loss`、`train/total_loss`、`val/action_l1_loss`、`val/total_loss`、learning rate 和 step time；训练结束会上传 `train_metrics.csv` 和 final checkpoint artifact。

Four-GPU schedule for four seeds per model:

```bash
bash scripts/run_multiseed_4gpu.sh
```

Schedule:

```text
Wave 1:
GPU0 -> act_A_only seed42
GPU1 -> act_ABC    seed42
GPU2 -> act_A_only seed43
GPU3 -> act_ABC    seed43

Wave 2:
GPU0 -> act_A_only seed44
GPU1 -> act_ABC    seed44
GPU2 -> act_A_only seed45
GPU3 -> act_ABC    seed45
```

Outputs:

```text
outputs/train/act_A_only_seed42/
outputs/train/act_A_only_seed43/
outputs/train/act_A_only_seed44/
outputs/train/act_A_only_seed45/
outputs/train/act_ABC_seed42/
outputs/train/act_ABC_seed43/
outputs/train/act_ABC_seed44/
outputs/train/act_ABC_seed45/
```

Each training output should contain:

```text
checkpoints/final/checkpoint.pt
train_metrics.csv
config_snapshot.yaml
```

## Formal D Evaluation

Evaluate every trained checkpoint on D:

```bash
bash scripts/run_multiseed_eval_4gpu.sh
```

This produces one D evaluation directory per checkpoint:

```text
outputs/eval/act_A_only_seed42_on_D/
outputs/eval/act_A_only_seed43_on_D/
outputs/eval/act_A_only_seed44_on_D/
outputs/eval/act_A_only_seed45_on_D/
outputs/eval/act_ABC_seed42_on_D/
outputs/eval/act_ABC_seed43_on_D/
outputs/eval/act_ABC_seed44_on_D/
outputs/eval/act_ABC_seed45_on_D/
```

Each directory includes:

```text
metrics.json
task_metrics.csv
episode_metrics.csv
chunk_horizon_metrics.csv
predictions.jsonl
failure_cases.json
eval_log.txt
```

`predictions.jsonl` is capped by `max_prediction_records` to control file size. Aggregate metrics use the full D split.

## Metric Aggregation

Aggregate per-seed and cross-seed metrics:

```bash
python scripts/10_collect_multiseed_metrics.py --seeds 42 43 44 45
```

Generated tables:

```text
report_assets/result_tables/multiseed_per_seed_metrics.csv
report_assets/result_tables/multiseed_summary_metrics.csv
report_assets/result_tables/multiseed_summary_metrics.md
report_assets/result_tables/multiseed_task_metrics.csv
report_assets/result_tables/multiseed_task_metrics.md
report_assets/result_tables/multiseed_episode_distribution_metrics.csv
report_assets/result_tables/multiseed_episode_distribution_metrics.md
report_assets/result_tables/multiseed_chunk_horizon_metrics.csv
```

Interpretation:

| Table | Interpretation |
| --- | --- |
| `multiseed_summary_metrics.*` | overall D Action L1 mean/std |
| `multiseed_task_metrics.*` | per-task Action L1 mean/std |
| `multiseed_episode_distribution_metrics.*` | episode-level distribution summary |
| `multiseed_chunk_horizon_metrics.csv` | chunk step error mean/std |

## Diagnostic Plots

For a selected seed or representative pair of evaluation directories:

```bash
python scripts/09_plot_d_diagnostics.py \
  --a-dir outputs/eval/act_A_only_seed42_on_D \
  --abc-dir outputs/eval/act_ABC_seed42_on_D \
  --output-dir outputs/figures
```

Outputs:

```text
outputs/figures/task_wise_action_l1_delta_on_D.png
outputs/figures/episode_wise_action_l1_boxplot_on_D.png
outputs/figures/action_error_by_chunk_step_on_D.png
outputs/figures/task_wise_delta.csv
```

The chunk-horizon plot is the main figure for Action Chunking analysis. The x-axis is future chunk step; the y-axis is mean absolute action error.

## Online Rollout Success Rate

Online rollout is defined as an exploratory extension:

```text
policy observes current simulator state
-> predicts action chunk
-> action adapter selects executable action(s)
-> simulator steps
-> task oracle checks success
-> repeat until success or max steps
```

Required components:

| Component | Requirement |
| --- | --- |
| CALVIN simulator | reset and step environment D |
| observation adapter | match training inputs: static image, wrist image, state |
| action adapter | map ACT action chunk to simulator actions |
| task oracle | compute task success |
| rollout scheduler | decide chunk refresh and temporal ensemble behavior |

Until these components are implemented and verified, `success_rate` remains null. Offline Action L1 is the formal metric for this repository.

## Report Checklist

The final report should include:

- Experimental setting: ACT-A-only vs ACT-ABC, zero-shot D.
- Dataset split and D isolation statement.
- Fair comparison table for fixed hyperparameters.
- Overall D Action L1 mean/std across seeds.
- Task-wise D Action L1 analysis.
- Episode-wise error distribution.
- Chunk-horizon error curve for Action Chunking.
- WandB training and validation curves.
- Limitation: offline action error is not closed-loop Success Rate.
- Optional extension plan for online rollout Success Rate.

Recommended conclusion language:

```text
We evaluate zero-shot generalization on held-out environment D using offline Action L1, comparing predicted ACT action chunks against expert action chunks. This metric measures imitation accuracy under visual distribution shift and should not be interpreted as closed-loop task success. Multi-seed results are reported as mean ± standard deviation.
```

## GitHub

Initialize and push:

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/9902030608jac-dot/task2_lerobot_act_calvin.git
git push -u origin main
```
