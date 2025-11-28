#!/usr/bin/env python3
"""
预计算RDT模型的推理输出用于指导DP训练

功能:
1. 加载训练数据集(与DP相同的zarr格式)
2. 使用训练好的RDT模型对每个时间步进行推理
3. 保存RDT的预测结果,用于后续DP训练时加载

使用方法:
python precompute_rdt_labels.py \
    --rdt_ckpt checkpoints/your_model/checkpoint-10000 \
    --data_path ../DP/data/task_name-config-50.zarr \
    --output_path ./rdt_labels/task_name_labels.zarr \
    --task_name your_task \
    --instruction "your instruction text"
"""

import argparse
import os
import sys
import time
from pathlib import Path
import yaml
import zarr
import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image as PImage

# 添加RDT路径
current_file = Path(__file__)
parent_dir = current_file.parent
sys.path.append(str(parent_dir))

from model import RDT


def load_zarr_dataset(zarr_path):
    """加载DP格式的zarr数据集"""
    print(f"加载数据集: {zarr_path}")
    
    zarr_root = zarr.open(zarr_path, mode='r')
    data = zarr_root['data']
    meta = zarr_root['meta']
    
    # 读取所有数据
    head_camera = np.array(data['head_camera'])  # [N, C, H, W]
    state = np.array(data['state'])              # [N, D]
    action = np.array(data['action'])            # [N, D] - 原始专家动作
    episode_ends = np.array(meta['episode_ends']) # [num_episodes]
    
    print(f"  - head_camera shape: {head_camera.shape}")
    print(f"  - state shape: {state.shape}")
    print(f"  - action shape: {action.shape}")
    print(f"  - 总episodes数: {len(episode_ends)}")
    print(f"  - 总样本数: {len(state)}")
    
    return {
        'head_camera': head_camera,
        'state': state,
        'action': action,
        'episode_ends': episode_ends
    }


def preprocess_image(img_nchw):
    """
    将NCHW格式的图像转换为RDT需要的PIL格式
    Args:
        img_nchw: [C, H, W] numpy array, 值范围[0, 255]
    Returns:
        PIL Image
    """
    # NCHW -> HWC
    img_hwc = np.transpose(img_nchw, (1, 2, 0))
    
    # 确保是uint8格式
    if img_hwc.dtype != np.uint8:
        img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)
    
    # 应用JPEG编码/解码 (与RDT训练时一致)
    img_encoded = cv2.imencode('.jpg', img_hwc)[1].tobytes()
    img_decoded = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)
    
    # 转为PIL Image
    return PImage.fromarray(img_decoded)


def run_rdt_inference(
    rdt_model,
    dataset,
    use_first_step_only=False,
    use_mean_steps=None,
    instruction=None,
    stationary_mask_eps=None,
):
    """
    对数据集中的每个样本运行RDT推理
    
    Args:
        rdt_model: 已加载的RDT模型
        dataset: 数据集字典
        use_first_step_only: 是否只使用RDT预测的第一步 (默认True)
        use_mean_steps: 如果不为None,使用前N步的平均值作为标签
    
    Returns:
        rdt_predictions: [N, action_dim] numpy array
    """
    head_camera = dataset['head_camera']
    state = dataset['state']
    expert_action = dataset['action']
    episode_ends = dataset['episode_ends']
    instruction_text = instruction if instruction else rdt_model.task_name
    
    num_samples = len(state)
    data_action_dim = dataset['action'].shape[1]  # 数据集的原始动作维度 (例如14)
    
    # 初始化输出数组 - 使用数据集的原始维度
    rdt_predictions = np.zeros((num_samples, data_action_dim), dtype=np.float32)
    
    # 获取RDT模型的动作维度
    model_action_dim = rdt_model.left_arm_dim + 1 + rdt_model.right_arm_dim + 1
    
    print(f"\n开始RDT推理...")
    print(f"  - 数据集动作维度: {data_action_dim}")
    print(f"  - RDT模型动作维度: {model_action_dim}")
    print(f"  - 文本指令: {instruction_text}")
    if stationary_mask_eps is not None:
        print(f"  - 静止掩码: 速度阈值 eps={stationary_mask_eps}")
    
    if data_action_dim != model_action_dim:
        print(f"  ⚠️  动作维度不匹配! 将自动转换 {model_action_dim}维 → {data_action_dim}维")
    
    # 获取episode的起始和结束索引
    episode_starts = [0] + episode_ends[:-1].tolist()
    episode_ends_list = episode_ends.tolist()
    
    if use_first_step_only:
        strategy_desc = "仅使用第1步"
    elif use_mean_steps is not None:
        strategy_desc = f"使用前{use_mean_steps}步平均"
    else:
        strategy_desc = "未启用平滑，默认使用第1步"
    print(f"  - 配置: {strategy_desc}")
    
    sample_idx = 0
    
    # 检查维度是否匹配
    data_state_dim = state.shape[1]  # 数据集的状态维度
    model_state_dim = rdt_model.left_arm_dim + 1 + rdt_model.right_arm_dim + 1  # 模型期望的维度
    
    if data_state_dim != model_state_dim:
        print(f"\n⚠️  警告: 维度不匹配!")
        print(f"  - 数据集状态维度: {data_state_dim}")
        print(f"  - 模型期望维度: {model_state_dim} (left_arm={rdt_model.left_arm_dim}, right_arm={rdt_model.right_arm_dim})")
        print(f"  - 数据集格式假设: [left_arm(7), left_gripper(1), right_arm(6), right_gripper(1)]")
        
        if data_state_dim < model_state_dim:
            print(f"  - 自动填充到 {model_state_dim} 维...")
            
            # 假设数据格式: [left_arm(7), left_gripper(1), right_arm(6), right_gripper(1)] = 15维
            # 或者: [left_arm(7), left_gripper(1), right_arm(5), right_gripper(1)] = 14维
            # 模型期望: [left_arm(8), left_gripper(1), right_arm(8), right_gripper(1)] = 18维
            # 或: [left_arm(8), left_gripper(1), right_arm(7), right_gripper(1)] = 17维
            
            state_padded = np.zeros((num_samples, model_state_dim), dtype=np.float32)
            
            # 计算实际的arm维度 (数据集中)
            data_left_arm_dim = (data_state_dim - 2) // 2  # 减去2个gripper,然后平分
            data_right_arm_dim = data_state_dim - data_left_arm_dim - 2
            
            print(f"  - 检测到数据集: left_arm={data_left_arm_dim}, right_arm={data_right_arm_dim}")
            
            # 复制左臂关节 (尽可能多地复制)
            left_copy_dim = min(data_left_arm_dim, rdt_model.left_arm_dim)
            state_padded[:, :left_copy_dim] = state[:, :left_copy_dim]
            # 如果需要填充,剩余维度保持为0
            
            # 复制左夹爪
            state_padded[:, rdt_model.left_arm_dim] = state[:, data_left_arm_dim]
            
            # 复制右臂关节
            right_copy_dim = min(data_right_arm_dim, rdt_model.right_arm_dim)
            state_padded[:, rdt_model.left_arm_dim+1:rdt_model.left_arm_dim+1+right_copy_dim] = \
                state[:, data_left_arm_dim+1:data_left_arm_dim+1+right_copy_dim]
            
            # 复制右夹爪
            state_padded[:, rdt_model.left_arm_dim+1+rdt_model.right_arm_dim] = state[:, data_state_dim-1]
            
            state = state_padded
            print(f"  ✅ 填充完成! 新维度: {state.shape}")
        else:
            raise ValueError(f"数据维度({data_state_dim})大于模型期望({model_state_dim}), 无法自动处理")
    
    for ep_idx, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends_list)):
        print(f"\n处理 Episode {ep_idx + 1}/{len(episode_ends_list)} (样本 {ep_start} - {ep_end})")
        
        # 每个episode开始时重置模型
        rdt_model.reset_obsrvationwindows()
        rdt_model.set_language_instruction(instruction_text)  # 使用指定指令
   
        # 初始化观察窗口 (需要两帧)
        # 第一帧用dummy
        rdt_model.observation_window = None
        
        for t in tqdm(range(ep_start, ep_end), desc=f"Episode {ep_idx + 1}"):
            # 准备当前帧的图像 (NCHW -> HWC -> BGR uint8)
            current_img_nchw = head_camera[t]  # [3, H, W]
            current_img_hwc = np.transpose(current_img_nchw, (1, 2, 0))  # [H, W, 3]
            
            # 确保是uint8 BGR格式
            if current_img_hwc.dtype != np.uint8:
                current_img_hwc = np.clip(current_img_hwc, 0, 255).astype(np.uint8)
            # 训练期使用 RGB，这里从 BGR 转回 RGB 以减小颜色域偏移
            current_img_hwc = cv2.cvtColor(current_img_hwc, cv2.COLOR_BGR2RGB)
            
            # cv2.imshow("RDT", current_img_hwc)
            # cv2.waitKey(1)
            
            # 准备前一帧的图像
            if t == ep_start:
                # 第一帧: 使用当前帧作为前一帧
                prev_img_hwc = current_img_hwc.copy()
            else:
                prev_img_nchw = head_camera[t-1]
                prev_img_hwc = np.transpose(prev_img_nchw, (1, 2, 0))
                if prev_img_hwc.dtype != np.uint8:
                    prev_img_hwc = np.clip(prev_img_hwc, 0, 255).astype(np.uint8)
                prev_img_hwc = cv2.cvtColor(prev_img_hwc, cv2.COLOR_BGR2RGB)
            
            # 准备图像数组 (RDT的update_observation_window需要3个相机的图像)
            # 但我们只有head_camera，所以复制它来填充3个位置
            img_arr = [
                current_img_hwc,  # head/front camera
                current_img_hwc,  # right camera (用head替代)
                current_img_hwc,  # left camera (用head替代)
            ]
            
            # 更新观察窗口 (这会自动处理第一次调用时的初始化)
            rdt_model.update_observation_window(img_arr, state[t])
            
            # 获取RDT动作预测
            # actions shape: [64, model_action_dim]
            actions = rdt_model.get_action()
            
            # 提取监督标签
            if use_first_step_only:
                # 方案1: 只使用第一步
                predicted_action = actions[0]
            elif use_mean_steps is not None:
                # 方案2: 使用前N步的平均
                predicted_action = actions[:use_mean_steps].mean(axis=0)
            else:
                # 默认使用第一步
                predicted_action = actions[0]
            
            # 转换动作维度 (如果需要)
            if model_action_dim != data_action_dim:
                # 将RDT的预测转换为数据集的格式
                # 假设格式: [left_arm, left_gripper, right_arm, right_gripper]
                
                # 计算数据集的arm维度
                data_left_arm = (data_action_dim - 2) // 2
                data_right_arm = data_action_dim - data_left_arm - 2
                
                converted_action = np.zeros(data_action_dim, dtype=np.float32)
                
                # 复制左臂 (取前data_left_arm维)
                converted_action[:data_left_arm] = predicted_action[:data_left_arm]
                
                # 复制左夹爪
                converted_action[data_left_arm] = predicted_action[rdt_model.left_arm_dim]
                
                # 复制右臂
                converted_action[data_left_arm+1:data_left_arm+1+data_right_arm] = \
                    predicted_action[rdt_model.left_arm_dim+1:rdt_model.left_arm_dim+1+data_right_arm]
                
                # 复制右夹爪
                converted_action[-1] = predicted_action[model_action_dim-1]
                
                rdt_predictions[t] = converted_action
            else:
                # 维度匹配，直接使用
                rdt_predictions[t] = predicted_action
            
            # 方案1: 静止状态掩码 (专家不动时强制对齐专家，抑制RDT漂移/抢跑)
            if stationary_mask_eps is not None:
                prev_idx = max(ep_start, t - 1)
                expert_curr = expert_action[t]
                expert_prev = expert_action[prev_idx]
                vel = np.abs(expert_curr - expert_prev)
                mask = vel < stationary_mask_eps  # 专家维持静止的维度
                if np.any(mask):
                    # 将这些维度的标签替换成专家的动作
                    rdt_predictions[t, mask] = expert_curr[mask]
            
            sample_idx += 1
    
    print(f"\n✅ RDT推理完成! 共处理 {num_samples} 个样本")
    return rdt_predictions


def save_rdt_labels(output_path, dataset, rdt_predictions):
    """
    保存RDT预测的标签
    保持与原始数据集相同的格式,但添加rdt_action字段
    """
    print(f"\n保存RDT标签到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 删除已存在的文件
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    # 创建新的zarr文件
    zarr_root = zarr.group(output_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # 保存原始数据
    zarr_data.create_dataset(
        'head_camera',
        data=dataset['head_camera'],
        chunks=(100, *dataset['head_camera'].shape[1:]),
        overwrite=True,
        compressor=compressor,
    )
    
    zarr_data.create_dataset(
        'state',
        data=dataset['state'],
        chunks=(100, dataset['state'].shape[1]),
        dtype='float32',
        overwrite=True,
        compressor=compressor,
    )
    
    # 保存原始专家动作
    zarr_data.create_dataset(
        'action',
        data=dataset['action'],
        chunks=(100, dataset['action'].shape[1]),
        dtype='float32',
        overwrite=True,
        compressor=compressor,
    )
    
    # 🔥 保存RDT预测的动作 (这是新增的!)
    zarr_data.create_dataset(
        'rdt_action',
        data=rdt_predictions,
        chunks=(100, rdt_predictions.shape[1]),
        dtype='float32',
        overwrite=True,
        compressor=compressor,
    )
    
    zarr_meta.create_dataset(
        'episode_ends',
        data=dataset['episode_ends'],
        dtype='int64',
        overwrite=True,
        compressor=compressor,
    )
    
    print(f"✅ 保存完成!")
    print(f"  - 包含字段: head_camera, state, action, rdt_action, episode_ends")


def main():
    parser = argparse.ArgumentParser(description='预计算RDT推理输出用于DP训练')
    
    # RDT模型参数
    parser.add_argument('--rdt_ckpt', type=str, required=True,
                        help='RDT checkpoint路径')
    parser.add_argument('--task_name', type=str, required=True,
                        help='任务名称')
    parser.add_argument('--instruction', type=str, default=None,
                        help='语言指令 (可选,默认使用task_name)')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='输入数据集路径 (DP格式的zarr)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出标签路径')
    
    # 推理参数
    parser.add_argument('--left_arm_dim', type=int, default=-1,
                        help='左臂维度，-1 时自动根据数据维度推断')
    parser.add_argument('--right_arm_dim', type=int, default=-1,
                        help='右臂维度，-1 时自动根据数据维度推断')
    parser.add_argument('--rdt_step', type=int, default=64,
                        help='RDT chunk size')
    
    # 标签提取策略
    parser.add_argument('--use_first_step', action='store_true', default=False,
                        help='只使用RDT预测的第1步作为标签 (默认关闭，可用 --use_mean_steps 做平滑)')
    parser.add_argument('--use_mean_steps', type=int, default=None,
                        help='使用前N步的平均作为标签')
    parser.add_argument('--stationary_mask_eps', type=float, default=None,
                        help='静止掩码阈值; 若专家速度<eps则用专家动作覆盖 (推荐0.01~0.02)')
    
    args = parser.parse_args()
    
    # 1. 加载数据集
    dataset = load_zarr_dataset(args.data_path)

    # 自动推断臂维度
    data_dim = dataset["state"].shape[1]
    inferred_left = (data_dim - 2) // 2
    inferred_right = data_dim - inferred_left - 2
    if args.left_arm_dim < 0 or args.right_arm_dim < 0:
        args.left_arm_dim = inferred_left
        args.right_arm_dim = inferred_right
        print(f"  - 自动推断关节维度: left_arm={args.left_arm_dim}, right_arm={args.right_arm_dim}")
    else:
        if args.left_arm_dim + args.right_arm_dim + 2 != data_dim:
            print(f"⚠️ 指定的臂维度 ({args.left_arm_dim}+{args.right_arm_dim}+2) 与数据维度 {data_dim} 不一致，请确认！")

    # 处理标签策略冲突
    if args.use_first_step and args.use_mean_steps is not None:
        print("⚠️ 同时指定了 --use_first_step 和 --use_mean_steps，优先使用均值平滑。")
        args.use_first_step = False
    
    # 2. 初始化RDT模型
    print(f"\n初始化RDT模型...")
    print(f"  - Checkpoint: {args.rdt_ckpt}")
    print(f"  - Task: {args.task_name}")
    
    # 检测checkpoint格式并选择正确的路径
    ckpt_dir = args.rdt_ckpt
    ds_checkpoint = os.path.join(ckpt_dir, "pytorch_model", "mp_rank_00_model_states.pt")
    ema_checkpoint = os.path.join(ckpt_dir, "ema", "model.safetensors")
    
    if os.path.isfile(ds_checkpoint):
        pretrained_path = ds_checkpoint
        print(f"  - 使用DeepSpeed checkpoint: {ds_checkpoint}")
    elif any(os.path.isfile(os.path.join(ckpt_dir, fname)) for fname in ("model.safetensors", "pytorch_model.bin")):
        pretrained_path = ckpt_dir  # HuggingFace style checkpoint folder
        print(f"  - 使用HuggingFace checkpoint: {ckpt_dir}")
    elif os.path.isfile(ema_checkpoint):
        pretrained_path = ema_checkpoint
        print(f"  - 使用EMA checkpoint: {ema_checkpoint}")
    else:
        raise FileNotFoundError(f"❌ 无法在 {ckpt_dir} 下找到可用的RDT权重文件")
    
    rdt = RDT(
        pretrained_path,
        args.task_name,
        args.left_arm_dim,
        args.right_arm_dim,
        args.rdt_step
    )
    
    # 设置指令
    instruction = args.instruction if args.instruction else args.task_name
    rdt.set_language_instruction(instruction)
    
    print(f"✅ RDT模型加载完成")
    
    # 3. 运行推理
    rdt_predictions = run_rdt_inference(
        rdt,
        dataset,
        use_first_step_only=args.use_first_step,
        use_mean_steps=args.use_mean_steps,
        instruction=instruction,
        stationary_mask_eps=args.stationary_mask_eps,
    )
    
    # 4. 保存结果
    save_rdt_labels(args.output_path, dataset, rdt_predictions)
    
    print(f"\n{'='*60}")
    print(f"✅ 全部完成!")
    print(f"{'='*60}")
    print(f"现在您可以修改DP的dataset代码,读取 'rdt_action' 作为监督标签")


if __name__ == '__main__':
    main()
