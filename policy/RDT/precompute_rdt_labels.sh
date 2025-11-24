#!/bin/bash

# 预计算RDT标签的脚本示例
# 用法: bash precompute_rdt_labels.sh

# ============ 配置参数 ============

# 任务配置
TASK_NAME="lift_pot"        # 修改为你的任务名
TASK_CONFIG="demo_clean"             # 修改为你的配置
NUM_EPISODES=50                   # 数据集的episode数量

# RDT模型配置
RDT_CKPT="checkpoints/RDT_lift_pot_demo_clean_1b_pretrain/checkpoint-20000"  # 修改为实际的checkpoint路径
INSTRUCTION="lift_pot"   # 可选: 语言指令

# 数据路径
DP_DATA_PATH="../DP/data/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}.zarr"
OUTPUT_PATH="./rdt_labels/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}_with_rdt.zarr"

# 机器人配置
# ⚠️ 重要: 这里的维度必须与RDT训练时使用的维度一致！
# 如果数据集是14维 (7+1+7+1)，但RDT期望16维，说明RDT训练时用的是8+8
# 方案1: 修改为RDT训练时的维度 (推荐)
LEFT_ARM_DIM=6
RIGHT_ARM_DIM=6
RDT_STEP=64

# 方案2: 如果确定数据集是7+7维，需要填充到8+8维
# 请根据实际情况选择

# 标签提取策略
USE_FIRST_STEP="--use_first_step"        # 只使用第1步
# USE_MEAN_STEPS="--use_mean_steps 4"   # 或者使用前4步的平均

# ============ 运行预计算 ============

echo "开始预计算RDT标签..."
echo "  任务: ${TASK_NAME}"
echo "  数据: ${DP_DATA_PATH}"
echo "  输出: ${OUTPUT_PATH}"
echo ""

python precompute_rdt_labels.py \
    --rdt_ckpt ${RDT_CKPT} \
    --task_name ${TASK_NAME} \
    --instruction "${INSTRUCTION}" \
    --data_path ${DP_DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --left_arm_dim ${LEFT_ARM_DIM} \
    --right_arm_dim ${RIGHT_ARM_DIM} \
    --rdt_step ${RDT_STEP} \
    ${USE_FIRST_STEP}

echo ""
echo "✅ 预计算完成!"
echo "生成的标签文件: ${OUTPUT_PATH}"
