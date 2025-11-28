#!/usr/bin/env python3
"""
可视化RDT标签质量的工具

功能:
1. 对比RDT预测和专家动作的差异
2. 可视化动作轨迹
3. 统计分析

使用方法:
python visualize_rdt_labels.py --zarr_path ./rdt_labels/task_with_rdt.zarr
"""

import argparse
import time

import zarr
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(zarr_path):
    """加载数据"""
    # time.sleep(10)
    zarr_root = zarr.open(zarr_path, mode='r')
    data = zarr_root['data']
    meta = zarr_root['meta']
    
    return {
        'expert_action': np.array(data['action']),
        'rdt_action': np.array(data['rdt_action']),
        'episode_ends': np.array(meta['episode_ends'])
    }


def compute_statistics(data):
    """计算统计指标"""
    expert = data['expert_action']
    rdt = data['rdt_action']
    
    # 计算差异
    diff = expert - rdt
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    
    # 按维度统计
    mse_per_dim = np.mean(diff ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    print("---------------12412jasihfiuasfiuhbah")
    print("="*60)
    print("📊 RDT标签质量统计")
    print("="*60)
    print(f"总样本数: {len(expert)}")
    print(f"动作维度: {expert.shape[1]}")
    print(f"\n整体误差:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"\n各维度MAE:")
    for i, mae_val in enumerate(mae_per_dim):
        print(f"  - Dim {i}: {mae_val:.6f}")
    print("="*60)
    
    return {
        'mse': mse,
        'mae': mae,
        'mse_per_dim': mse_per_dim,
        'mae_per_dim': mae_per_dim,
        'diff': diff
    }


def plot_comparison(data, stats, output_dir='./visualizations'):
    """绘制对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    expert = data['expert_action']
    rdt = data['rdt_action']
    episode_ends = data['episode_ends']
    diff = stats['diff']
    
    # 1. 绘制动作轨迹对比 (第一个episode)
    fig, axes = plt.subplots(7, 2, figsize=(15, 20))
    fig.suptitle('Episode 0: RDT vs Expert Actions', fontsize=16)
    
    ep_start = 0
    ep_end = episode_ends[0]
    
    for dim in range(14):
        row = dim % 7
        col = dim // 7
        ax = axes[row, col]
        
        time_steps = np.arange(ep_start, ep_end)
        ax.plot(time_steps, expert[ep_start:ep_end, dim], 'b-', label='Expert', linewidth=2)
        ax.plot(time_steps, rdt[ep_start:ep_end, dim], 'r--', label='RDT', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Dim {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=150)
    print(f"✅ 保存轨迹对比图: {output_dir}/trajectory_comparison.png")
    plt.close()
    
    # 2. 绘制误差分布
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 整体误差直方图
    axes[0].hist(np.abs(diff).flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Absolute Errors')
    axes[0].axvline(stats['mae'], color='r', linestyle='--', linewidth=2, label=f'MAE={stats["mae"]:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各维度误差对比
    dims = np.arange(14)
    axes[1].bar(dims, stats['mae_per_dim'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Action Dimension')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error per Dimension')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=150)
    print(f"✅ 保存误差分析图: {output_dir}/error_analysis.png")
    plt.close()
    
    # 3. 绘制相关性矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算每个维度的相关系数
    correlations = []
    for dim in range(14):
        corr = np.corrcoef(expert[:, dim], rdt[:, dim])[0, 1]
        correlations.append(corr)
    
    ax.bar(dims, correlations, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation between RDT and Expert Actions')
    ax.axhline(y=0.9, color='r', linestyle='--', label='0.9 threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=150)
    print(f"✅ 保存相关性分析图: {output_dir}/correlation_analysis.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='包含RDT标签的zarr文件路径')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='可视化结果输出目录')
    args = parser.parse_args()
    
    print(f"\n加载数据: {args.zarr_path}")
    data = load_data(args.zarr_path)
    
    print("\n计算统计指标...")
    stats = compute_statistics(data)
    
    print("\n生成可视化...")
    plot_comparison(data, stats, args.output_dir)
    
    print(f"\n✅ 完成! 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
