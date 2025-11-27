#!/usr/bin/env python3
"""
可视化专家动作加高斯噪声前后的差异。

用法示例:
python visualize_expert_noise.py \
    --zarr_path data/task.zarr \
    --output noise_compare.png \
    --noise_std 0.01 \
    --noise_clip 0.05 \
    --seed 42 \
    --max_steps 200
"""

import argparse
import os
import math
import zarr
import numpy as np
import matplotlib.pyplot as plt


def add_noise(expert, std, clip, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=expert.shape).astype(np.float32)
    if clip is not None:
        noise = np.clip(noise, -clip, clip)
    return expert + noise


def plot_compare(expert, noisy, out_path, max_steps):
    T, D = expert.shape
    steps = min(max_steps, T)
    expert = expert[:steps]
    noisy = noisy[:steps]

    n_cols = 2
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
    t = np.arange(steps)
    for i in range(D):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        ax.plot(t, expert[:, i], label="expert", linewidth=1.2)
        ax.plot(t, noisy[:, i], label="noisy", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Joint {i}")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()
    # 清空多余子图
    for i in range(D, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ 已保存对比图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化专家动作加噪声前后的差异")
    parser.add_argument("--zarr_path", type=str, required=True, help="包含expert action的zarr路径")
    parser.add_argument("--output", type=str, default="expert_noise.png", help="输出图片路径")
    parser.add_argument("--noise_std", type=float, default=0.01, help="噪声标准差")
    parser.add_argument("--noise_clip", type=float, default=0.05, help="噪声截断; 设为负值或None表示不截断")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_steps", type=int, default=200, help="可视化的最大时间步数")
    args = parser.parse_args()

    group = zarr.open(args.zarr_path, mode="r")
    expert = np.array(group["data"]["action"])

    clip = args.noise_clip
    if clip is not None and clip < 0:
        clip = None

    noisy = add_noise(expert, args.noise_std, clip, args.seed)
    plot_compare(expert, noisy, args.output, args.max_steps)


if __name__ == "__main__":
    main()
