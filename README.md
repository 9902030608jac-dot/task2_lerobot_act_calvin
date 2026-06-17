# task2_lerobot_act_calvin

本仓库用于重建课程题目二：基于 LeRobot ACT 在 CALVIN 数据集上的跨环境泛化实验。目标不是堆结果，而是建立一套可复现、可审计、可扩展的实验流程：先完成合规的 zero-shot D offline action error 评估，再补充 task-wise、episode-wise、chunk-horizon 和 multi-seed 分析；online rollout success rate 作为探索性任务保留。

## 实验问题

课程要求是在 CALVIN 数据集上使用 LeRobot 框架中集成的 ACT 算法：

- 仅使用环境 A 训练一个基础视觉-动作策略模型 ACT-A-only。
- 使用环境 A/B/C 训练一个多环境策略模型 ACT-ABC。
- 在未见过的环境 D 上做 zero-shot 测试。
- 评估 Success Rate 或动作误差。
- 重点分析 ACT 的 Action Chunking 在跨环境视觉分布偏移下的鲁棒性。

本仓库采用如下分层实验设计：

| Priority | Task | Requires Retraining | Metric/Artifact | Role |
| --- | --- | --- | --- | --- |
| P0 | A-only vs ABC 主实验 | Yes | D offline Action L1 | 满足题目基本要求 |
| P1 | task-wise offline Action L1 | No, after checkpoints exist | `task_metrics.csv` | 分析哪些任务更鲁棒 |
| P2 | episode-wise error distribution | No, after checkpoints exist | `episode_metrics.csv`, boxplot | 分析轨迹级误差分布和失败案例 |
| P3 | chunk-horizon error analysis | No, after checkpoints exist | `chunk_horizon_metrics.csv`, horizon curve | 对应 ACT Action Chunking 机制 |
| P4 | multi-seed | Yes | mean/std across seeds | 增强可信度 |
| P5 | online rollout Success Rate | Extra environment integration | closed-loop success rate | 探索性任务 |

`P5` 被放在最后不是因为它不重要，而是因为它需要完整 CALVIN simulator、task oracle、observation/action adapter 和 ACT action chunk 执行协议。没有这些接口时，不应伪造 Success Rate。

## 核心假设

训练环境和测试环境的关系：

```text
ACT-A-only: train on A       -> zero-shot test on D
ACT-ABC:    train on A/B/C   -> zero-shot test on D
```

环境 D 必须保持未见状态，不得参与训练、调参或模型选择。A-only 和 ABC 除训练数据来源外，ACT 架构、chunk size、batch size、学习率、训练 step、optimizer、loss、图像/状态/动作预处理和评估脚本都应保持一致。

## 数据集结构

推荐使用 Hugging Face 数据集：

```text
xiaoma26/calvin-lerobot
```

远程实测该数据集是 LeRobot v2.1 风格。每个 split 的典型结构如下：

```text
data/raw/xiaoma26_calvin_lerobot/
├── splitA/
├── splitB/
├── splitC/
└── splitD/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   ├── episodes.jsonl
    │   └── episodes_stats.jsonl
    └── data/
        └── chunk-xxx/
            └── episode_xxxxxx.parquet
```

D split 中存在任务和轨迹结构：

```text
task_index/task descriptions: 389
episodes:                     5124
frames:                       308918
```

关系是：

```text
task description
  -> multiple episodes
episode
  -> continuous trajectory for one task
frame
  -> one timestep inside an episode
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

这里的 `episode_index` 是当前 split 内重新编号后的轨迹 ID；`source_start_frame/source_end_frame` 是原始 CALVIN 数据流里的来源位置。两者是追溯关系，不要求按大小顺序对应。

每个 parquet frame 包含：

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

本仓库的 v2.1 adapter 会读取 `task_index` 并通过 `meta/tasks.jsonl` 映射到自然语言任务描述，用于 task-wise offline action L1 分析。

## 为什么使用 v2.1 Adapter

远程环境中 `lerobot==0.5.1` 默认偏向 v3.0 数据格式，直接使用官方 `LeRobotDataset` 加载 v2.1 CALVIN 数据会遇到版本兼容和 HF Datasets cache 性能问题。之前验证过 v2.1 到 v3.0 转换路径，但完整转换会占用额外磁盘，并且在 `episodes_stats.jsonl` 字段兼容性上需要修补。

本仓库保留 LeRobot 的 ACTPolicy/ACTConfig，数据层使用轻量 adapter 直接读取 v2.1 parquet：

| Raw Field | ACT Batch Field |
| --- | --- |
| `image` | `observation.images.rgb_static` |
| `wrist_image` | `observation.images.rgb_gripper` |
| `state` | `observation.state` |
| future `actions` chunk | `action` |
| episode end padding | `action_is_pad` |
| `task_index` + `tasks.jsonl` | `task_index`, `task` |
| episode metadata | `episode_index`, `frame_index`, `env_id` |

这仍然满足“使用 LeRobot 框架中集成 ACT 算法”的要求：模型、loss、optimizer 前后向都由 LeRobot ACT policy 完成；自定义部分只是把课程数据集映射成 ACT batch。

## 目录结构

```text
task2_lerobot_act_calvin/
├── configs/
├── scripts/
├── src/
├── data/
│   ├── raw/             # 不进 git
│   └── processed/       # 不进 git
├── outputs/             # 不进 git
├── logs/                # 不进 git
├── checkpoints/         # 不进 git
└── report_assets/
```

本重建仓库不包含旧实验结果、旧 checkpoint、原始数据集或远程缓存。所有大文件通过 `.gitignore` 排除。

## 环境安装

建议 Python 3.12：

```bash
conda create -n lerobot-act-calvin python=3.12 -y
conda activate lerobot-act-calvin
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果 GPU driver 与默认 PyTorch CUDA wheel 不匹配，先按 PyTorch 官方说明安装对应版本，再安装：

```bash
pip install -r requirements-no-torch.txt
```

检查环境：

```bash
python scripts/00_check_env.py --data_root data/raw --require_cuda true
```

长期任务建议使用 `nohup`、`tmux` 或 `screen`，避免 SSH 断开导致训练中断。

## 数据准备

推荐先将 HF 数据下载到：

```text
data/raw/xiaoma26_calvin_lerobot/
```

如果网络不稳定，可使用 `HF_ENDPOINT=https://hf-mirror.com` 或本项目下载脚本。数据下载完成后生成 processed manifest：

```bash
python scripts/02_prepare_dataset.py --config configs/train_A_only.yaml
python scripts/02_prepare_dataset.py --config configs/train_ABC.yaml
python scripts/02_prepare_dataset.py --config configs/eval_A_on_D.yaml
```

`data/processed/*/source_datasets.json` 会记录每个 processed split 对应的原始 v2.1 split 路径。正式训练时实际由 `CalvinV21ActDataset` 直接读取原始 parquet。

## 单 Seed 主实验

先跑短 smoke test：

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

正式单 seed：

```bash
python scripts/03_train_act.py --config configs/train_A_only.yaml --device cuda
python scripts/03_train_act.py --config configs/train_ABC.yaml --device cuda
```

## D Offline Evaluation

本项目主评估是 D 环境 offline Action L1。它不需要 simulator，读取 D 专家轨迹，比较模型预测 action chunk 与专家 action chunk：

```bash
python scripts/04_eval_on_D.py \
  --config configs/eval_A_on_D.yaml \
  --mode offline \
  --device cuda \
  --override use_wandb=false batch_size=128 num_workers=8 max_prediction_records=1000

python scripts/04_eval_on_D.py \
  --config configs/eval_ABC_on_D.yaml \
  --mode offline \
  --device cuda \
  --override use_wandb=false batch_size=128 num_workers=8 max_prediction_records=1000
```

输出：

```text
outputs/eval/A_only_on_D/
├── metrics.json
├── predictions.jsonl
├── task_metrics.csv
├── episode_metrics.csv
├── chunk_horizon_metrics.csv
├── failure_cases.json
└── eval_log.txt
```

`metrics.json` 是整体 D 指标；`task_metrics.csv` 是按任务描述聚合；`episode_metrics.csv` 是按轨迹聚合；`chunk_horizon_metrics.csv` 是 ACT action chunk 的每个未来步平均误差。

`max_prediction_records` 只限制逐样本 prediction 明细保存数量，不影响整体、task-wise、episode-wise 或 chunk-horizon 指标计算。

## 诊断图

单 seed 或 multi-seed 中任意一对 A-only/ABC 评估完成后，可以画 D 诊断图：

```bash
python scripts/09_plot_d_diagnostics.py \
  --a-dir outputs/eval/A_only_on_D \
  --abc-dir outputs/eval/ABC_on_D \
  --output-dir outputs/figures
```

主要输出：

```text
outputs/figures/task_wise_action_l1_delta_on_D.png
outputs/figures/episode_wise_action_l1_boxplot_on_D.png
outputs/figures/action_error_by_chunk_step_on_D.png
outputs/figures/task_wise_delta.csv
```

报告优先使用 `action_error_by_chunk_step_on_D.png` 来分析 ACT Action Chunking。横轴为未来 chunk step，纵轴为该未来步 7 维动作平均误差。

## 4 GPU Multi-seed 计划

如果有四张 32GB GPU，推荐每个模型跑 4 个 seed：

```text
seeds = 42, 43, 44, 45
models = ACT-A-only, ACT-ABC
total training jobs = 8
```

执行：

```bash
bash scripts/run_multiseed_4gpu.sh
```

调度策略：

```text
Wave 1:
GPU0 -> A-only seed42
GPU1 -> ABC    seed42
GPU2 -> A-only seed43
GPU3 -> ABC    seed43

Wave 2:
GPU0 -> A-only seed44
GPU1 -> ABC    seed44
GPU2 -> A-only seed45
GPU3 -> ABC    seed45
```

对应评估：

```bash
bash scripts/run_multiseed_eval_4gpu.sh
```

汇总 mean/std：

```bash
python scripts/10_collect_multiseed_metrics.py --seeds 42 43 44 45
```

输出：

```text
report_assets/result_tables/multiseed_per_seed_metrics.csv
report_assets/result_tables/multiseed_summary_metrics.csv
report_assets/result_tables/multiseed_summary_metrics.md
```

报告中如果完成 multi-seed，应使用 `mean ± std`，避免对单 seed 的小幅差异做过强结论。

## Online Rollout Success Rate 探索任务

Success Rate 是最完整指标，但它不是离线数据集直接能算出的字段。它需要：

- CALVIN D simulator 能 reset/step。
- D 环境资产和相机观测可用。
- task initialization 和 task oracle 可用。
- simulator observation 转换成训练时的 `rgb_static/rgb_gripper/state`。
- ACT 输出的 `[chunk_size, 7]` action chunk 转换成环境逐步执行动作。
- 明确每几步重新预测 chunk，是否使用 temporal ensemble。

因此，本仓库当前 rollout 模式只写出 `success_rate: null`，不会伪造结果。若时间允许，应新增一个 `calvin_rollout_adapter`，在官方 CALVIN benchmark 协议下做 task-wise success rate。

## 报告写法建议

严谨表述：

```text
This work reports offline action error on the held-out D environment, which is allowed by the assignment. The metric evaluates imitation accuracy under visual distribution shift by comparing predicted action chunks with expert action chunks. It should not be interpreted as closed-loop task success.
```

如果只有单 seed：

```text
Under a single-seed offline evaluation, ACT-ABC achieves lower D action error than ACT-A-only. This suggests improved offline generalization, but statistical significance requires multi-seed training.
```

如果完成 multi-seed：

```text
We report mean and standard deviation across four seeds. This reduces the risk that the observed A-only vs ABC difference is caused by random initialization or data order.
```

Action Chunking 分析应结合：

- 整体 D Action L1。
- task-wise delta。
- episode-wise distribution。
- chunk-horizon curve。

## GitHub 上传

这是一个重建仓库。确认不含 `data/`、`outputs/`、`logs/`、`checkpoints/`、`wandb/` 等大文件后：

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/9902030608jac-dot/task2_lerobot_act_calvin.git
git push -u origin main
```

如果远程仓库已有历史且需要覆盖，应先确认后再处理，避免误删已有代码。
