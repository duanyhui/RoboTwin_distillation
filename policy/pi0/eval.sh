#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7}

# 第 7 个参数 checkpoint_id 是可选的：
# - 不传：使用 policy/pi0/deploy_policy.yml 里的默认值（当前是 3000）
# - 传具体数字：评测该步数 ckpt（例如 6000）
# - 传 -1 或 latest：自动选择当前实验目录下最大的数字 ckpt

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

XLA_FLAGS="--xla_gpu_autotune_level=0" PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    ${checkpoint_id:+--checkpoint_id ${checkpoint_id}} \
    --policy_name ${policy_name} 
