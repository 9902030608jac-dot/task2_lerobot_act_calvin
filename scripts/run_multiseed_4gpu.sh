#!/usr/bin/env bash
set -euo pipefail

# Run 4 seeds for ACT-A-only and ACT-ABC on four 32GB GPUs.
# Each wave launches one process per GPU, then waits before launching the next wave.

mkdir -p logs/multiseed

run_train() {
  local gpu="$1"
  local model="$2"
  local config="$3"
  local seed="$4"
  local output_dir="outputs/train/${model}_seed${seed}"
  local run_name="${model}_seed${seed}"
  local log_path="logs/multiseed/train_${model}_seed${seed}.log"

  echo "Launching ${run_name} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  WANDB_GROUP="multiseed_act_calvin" \
  nohup python scripts/03_train_act.py \
    --config "${config}" \
    --device cuda \
    --override seed="${seed}" run_name="${run_name}" output_dir="${output_dir}" \
    > "${log_path}" 2>&1 &
}

# Wave 1: two seeds, both model families.
run_train 0 "act_A_only" "configs/train_A_only.yaml" 42
run_train 1 "act_ABC" "configs/train_ABC.yaml" 42
run_train 2 "act_A_only" "configs/train_A_only.yaml" 43
run_train 3 "act_ABC" "configs/train_ABC.yaml" 43
wait

# Wave 2: the remaining two seeds.
run_train 0 "act_A_only" "configs/train_A_only.yaml" 44
run_train 1 "act_ABC" "configs/train_ABC.yaml" 44
run_train 2 "act_A_only" "configs/train_A_only.yaml" 45
run_train 3 "act_ABC" "configs/train_ABC.yaml" 45
wait

echo "All multi-seed training jobs finished."
