#!/bin/bash

# 可视化RDT动作质量的脚本
# 用法: bash visualize_actions.sh

# ============ 配置参数 ============

TASK_NAME="lift_pot"
TASK_CONFIG="demo_clean"
NUM_EPISODES=50

# 数据路径
ZARR_PATH="./rdt_labels/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}_with_rdt.zarr"
OUTPUT_DIR="./action_visualizations/${TASK_NAME}"

# 要可视化的episode列表
EPISODE_IDS=(0 1 2 3 4 5 6 7 8 9 10)  # 可以添加更多

# 是否创建动画 (耗时)
CREATE_ANIMATION=false

# ============ 运行可视化 ============

echo "=========================================="
echo "RDT动作质量可视化"
echo "=========================================="
echo "任务: ${TASK_NAME}"
echo "数据: ${ZARR_PATH}"
echo "输出: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 对每个episode进行可视化
for EP_ID in "${EPISODE_IDS[@]}"; do
    echo "----------------------------------------"
    echo "可视化 Episode ${EP_ID}..."
    echo "----------------------------------------"
    
    if [ "$CREATE_ANIMATION" = true ]; then
        python visualize_rdt_actions.py \
            --zarr_path ${ZARR_PATH} \
            --episode_id ${EP_ID} \
            --output_dir ${OUTPUT_DIR} \
            --create_animation \
            --fps 10
    else
        python visualize_rdt_actions.py \
            --zarr_path ${ZARR_PATH} \
            --episode_id ${EP_ID} \
            --output_dir ${OUTPUT_DIR}
    fi
    
    echo ""
done

echo "=========================================="
echo "✅ 可视化完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  cd ${OUTPUT_DIR}"
echo "  ls -lh"
echo ""
echo "生成的文件包括:"
echo "  - episode_X_trajectory_comparison.png  (轨迹对比)"
echo "  - episode_X_error_analysis.png         (误差分析)"
echo "  - episode_X_correlation.png            (相关性分析)"
if [ "$CREATE_ANIMATION" = true ]; then
    echo "  - episode_X_animation_dim0.gif         (动作动画)"
fi
echo ""
echo "根据质量评级决定是否使用RDT标签:"
echo "  ⭐⭐⭐⭐⭐ 优秀 → 直接使用"
echo "  ⭐⭐⭐⭐   良好 → 直接使用"  
echo "  ⭐⭐⭐     中等 → 混合使用"
echo "  ⭐⭐       差   → 改进RDT模型"
