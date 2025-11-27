#!/usr/bin/env python3
"""
Visualize the joint-wise difference between action and rdt_action in DP/RDT zarr data.

Example:
python visualize_action_diff.py \
    --zarr_path ./rdt_labels/place_container_plate-demo_clean-50_with_rdt.zarr \
    --episode_id 0 \
    --output_dir ./rdt_action_diff_vis
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import zarr
except ImportError as exc:
    raise ImportError("Please install zarr first, e.g. pip install zarr") from exc


def build_joint_labels(dim: int):
    if dim == 14:
        return [
            "L_J1",
            "L_J2",
            "L_J3",
            "L_J4",
            "L_J5",
            "L_J6",
            "L_Grip",
            "R_J1",
            "R_J2",
            "R_J3",
            "R_J4",
            "R_J5",
            "R_J6",
            "R_Grip",
        ]
    if dim == 16:
        return [
            "L_J1",
            "L_J2",
            "L_J3",
            "L_J4",
            "L_J5",
            "L_J6",
            "L_J7",
            "L_Grip",
            "R_J1",
            "R_J2",
            "R_J3",
            "R_J4",
            "R_J5",
            "R_J6",
            "R_J7",
            "R_Grip",
        ]
    return [f"Dim{i}" for i in range(dim)]


def load_actions(zarr_path: Path):
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root.get("meta", None)
    action = np.asarray(data["action"])
    rdt_action = np.asarray(data["rdt_action"])
    episode_ends = np.asarray(meta["episode_ends"]) if meta and "episode_ends" in meta else None
    return action, rdt_action, episode_ends


def episode_slice(episode_ends, episode_id: int, total: int):
    if episode_ends is None:
        return 0, total
    if episode_id < 0 or episode_id >= len(episode_ends):
        raise ValueError(f"episode_id {episode_id} out of range 0~{len(episode_ends) - 1}")
    start = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    end = int(episode_ends[episode_id])
    return start, end


def plot_joint_series(action, rdt_action, labels, output_dir, episode_id, start, end):
    T, dim = action.shape
    cols = 2
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), squeeze=False)
    x = np.arange(T)

    for idx in range(dim):
        ax = axes[idx // cols, idx % cols]
        diff = action[:, idx] - rdt_action[:, idx]
        ax.plot(x, action[:, idx], label="action", color="#1f77b4", linewidth=1.8)
        ax.plot(x, rdt_action[:, idx], label="rdt_action", color="#d62728", linestyle="--", linewidth=1.5)
        ax.fill_between(x, action[:, idx], rdt_action[:, idx], color="gray", alpha=0.18)
        mae = np.mean(np.abs(diff))
        ax.set_title(f"{labels[idx]} | MAE={mae:.4f}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    for idx in range(dim, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"Episode {episode_id} ({start}-{end}) action vs rdt_action", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(output_dir) / f"episode_{episode_id}_joint_trends.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_error_summary(action, rdt_action, labels, output_dir, episode_id):
    diff = np.abs(action - rdt_action)
    mae = diff.mean(axis=0)
    p95 = np.percentile(diff, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, mae, width=width, label="MAE", color="#1f77b4")
    ax.bar(x + width / 2, p95, width=width, label="95th percentile error", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("Error magnitude")
    ax.set_title("Per-joint error statistics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = Path(output_dir) / f"episode_{episode_id}_error_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_diff_heatmap(action, rdt_action, labels, output_dir, episode_id):
    diff = action - rdt_action
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(diff.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Joint")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("action - rdt_action difference heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Difference")

    fig.tight_layout()
    out = Path(output_dir) / f"episode_{episode_id}_diff_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize joint-wise difference between action and rdt_action")
    default_path = Path(__file__).resolve().parent / "rdt_labels" / "place_container_plate-demo_clean-50_with_rdt.zarr"
    parser.add_argument("--zarr_path", type=str, default=str(default_path), help="Target zarr path")
    parser.add_argument("--episode_id", type=int, default=0, help="Episode id to visualize, -1 means whole sequence")
    parser.add_argument("--output_dir", type=str, default="./rdt_action_diff_vis", help="Output directory")
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"zarr file not found: {zarr_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    action, rdt_action, episode_ends = load_actions(zarr_path)
    if args.episode_id == -1:
        start, end = 0, len(action)
    else:
        start, end = episode_slice(episode_ends, args.episode_id, len(action))

    action_ep = action[start:end]
    rdt_ep = rdt_action[start:end]
    labels = build_joint_labels(action_ep.shape[1])
    print(f"Joint dims: {action_ep.shape[1]}")

    total_mae = np.mean(np.abs(action_ep - rdt_ep))
    print(f"Loaded: {zarr_path}")
    print(f"Episode range: {start}~{end} (length {len(action_ep)})")
    print(f"Overall MAE: {total_mae:.6f}")
    print("MAE per joint:")
    for name, val in zip(labels, np.mean(np.abs(action_ep - rdt_ep), axis=0)):
        print(f"  {name}: {val:.6f}")

    saved = [
        plot_joint_series(action_ep, rdt_ep, labels, args.output_dir, args.episode_id, start, end),
        plot_error_summary(action_ep, rdt_ep, labels, args.output_dir, args.episode_id),
        plot_diff_heatmap(action_ep, rdt_ep, labels, args.output_dir, args.episode_id),
    ]

    print("\nSaved figures:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
