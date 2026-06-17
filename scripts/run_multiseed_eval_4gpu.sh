#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 4-seed ACT-A-only and ACT-ABC checkpoints on held-out D.
# Outputs include overall metrics plus task-wise, episode-wise, and chunk-horizon CSVs.

mkdir -p logs/multiseed

run_eval() {
  local gpu="$1"
  local model="$2"
  local config="$3"
  local seed="$4"
  local train_dir="outputs/train/${model}_seed${seed}"
  local checkpoint_path="${train_dir}/checkpoints/final"
  local output_dir="outputs/eval/${model}_seed${seed}_on_D"
  local run_name="eval_${model}_seed${seed}_on_D"
  local log_path="logs/multiseed/eval_${model}_seed${seed}.log"

  echo "Launching ${run_name} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  WANDB_GROUP="multiseed_act_calvin_eval" \
  nohup python scripts/04_eval_on_D.py \
    --config "${config}" \
    --mode offline \
    --device cuda \
    --override seed="${seed}" run_name="${run_name}" checkpoint_path="${checkpoint_path}" output_dir="${output_dir}" max_prediction_records=1000 \
    > "${log_path}" 2>&1 &
}

run_eval 0 "act_A_only" "configs/eval_A_on_D.yaml" 42
run_eval 1 "act_ABC" "configs/eval_ABC_on_D.yaml" 42
run_eval 2 "act_A_only" "configs/eval_A_on_D.yaml" 43
run_eval 3 "act_ABC" "configs/eval_ABC_on_D.yaml" 43
wait

run_eval 0 "act_A_only" "configs/eval_A_on_D.yaml" 44
run_eval 1 "act_ABC" "configs/eval_ABC_on_D.yaml" 44
run_eval 2 "act_A_only" "configs/eval_A_on_D.yaml" 45
run_eval 3 "act_ABC" "configs/eval_ABC_on_D.yaml" 45
wait

echo "All multi-seed D evaluation jobs finished."
