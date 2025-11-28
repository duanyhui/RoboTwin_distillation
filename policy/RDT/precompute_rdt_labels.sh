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
# ⚠️ 维度改为自动推断：设为 -1 会根据 zarr 的 state 维度自动推断左右臂。
# 若需要强行指定，可改成具体数字，比如 LEFT_ARM_DIM=6 RIGHT_ARM_DIM=6
LEFT_ARM_DIM=-1
RIGHT_ARM_DIM=-1
RDT_STEP=64
STATIONARY_MASK_EPS=0.01   # 静止掩码阈值: 专家速度<eps则用专家动作覆盖

# 标签提取策略
# 默认使用前4步均值平滑；若想只用第1步，将 USE_FIRST_STEP 取消注释并注释掉 USE_MEAN_STEPS
#USE_FIRST_STEP="--use_first_step"        # 只使用第1步
USE_MEAN_STEPS="--use_mean_steps 4"      # 使用前N步平均

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
    --stationary_mask_eps ${STATIONARY_MASK_EPS} \
    ${USE_FIRST_STEP} \
    ${USE_MEAN_STEPS}

echo ""
echo "✅ 预计算完成!"
echo "生成的标签文件: ${OUTPUT_PATH}"
