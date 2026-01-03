#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
target_updates=${6:-}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

extra_args=()
ckpt_suffix=""
if [ -n "${target_updates}" ]; then
  extra_args+=(--target_updates "${target_updates}")
  ckpt_suffix="-u${target_updates}"
fi

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}${ckpt_suffix}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 14 \
    --seed ${seed} \
    "${extra_args[@]}"
