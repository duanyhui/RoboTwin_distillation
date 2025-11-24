#!/usr/bin/env python3
"""
Visualize RDT-generated action labels

Features:
1. Compare RDT predictions vs expert action trajectories
2. Analyze prediction error distributions
3. Generate action sequence visualization animations
4. Compute confidence and quality metrics

Usage:
python visualize_rdt_actions.py \
    --zarr_path ./rdt_labels/lift_pot-demo_clean-50_with_rdt.zarr \
    --output_dir ./action_visualizations \
    --episode_id 0
"""

import argparse
import os
import time
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


def load_data(zarr_path):
    """Load data from zarr file"""
    print(f"Loading data: {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode='r')
    data = zarr_root['data']
    meta = zarr_root['meta']
    
    return {
        'expert_action': np.array(data['action']),
        'rdt_action': np.array(data['rdt_action']),
        'episode_ends': np.array(meta['episode_ends'])
    }


def get_episode_data(data, episode_id):
    """Extract data for a single episode"""
    episode_ends = data['episode_ends']
    
    if episode_id == 0:
        start = 0
    else:
        start = episode_ends[episode_id - 1]
    end = episode_ends[episode_id]
    
    return {
        'expert': data['expert_action'][start:end],
        'rdt': data['rdt_action'][start:end],
        'start': start,
        'end': end
    }


def plot_trajectory_comparison(episode_data, output_dir, episode_id):
    """Plot trajectory comparison"""
    expert = episode_data['expert']
    rdt = episode_data['rdt']
    T = len(expert)
    action_dim = expert.shape[1]
    
    # Create figure
    n_rows = (action_dim + 1) // 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    fig.suptitle(f'Episode {episode_id}: RDT Predictions vs Expert Actions', fontsize=16, y=0.995)
    
    axes = axes.flatten()
    
    # Joint labels
    joint_labels = []
    if action_dim == 14:
        joint_labels = [f'L_Arm{i+1}' for i in range(7)] + ['L_Gripper'] + \
                       [f'R_Arm{i+1}' for i in range(6)] + ['R_Gripper']
    elif action_dim == 16:
        joint_labels = [f'L_Arm{i+1}' for i in range(7)] + ['L_Gripper'] + \
                       [f'R_Arm{i+1}' for i in range(7)] + ['R_Gripper']
    else:
        joint_labels = [f'Dim{i}' for i in range(action_dim)]
    
    for dim in range(action_dim):
        ax = axes[dim]
        time_steps = np.arange(T)
        
        # Plot expert and RDT predictions
        ax.plot(time_steps, expert[:, dim], 'b-', label='Expert', linewidth=2, alpha=0.7)
        ax.plot(time_steps, rdt[:, dim], 'r--', label='RDT', linewidth=2, alpha=0.7)
        
        # Fill error region
        ax.fill_between(time_steps, expert[:, dim], rdt[:, dim], alpha=0.2, color='gray')
        
        # Compute error statistics
        mse = np.mean((expert[:, dim] - rdt[:, dim]) ** 2)
        mae = np.mean(np.abs(expert[:, dim] - rdt[:, dim]))
        corr = np.corrcoef(expert[:, dim], rdt[:, dim])[0, 1]
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel(joint_labels[dim], fontsize=10)
        ax.set_title(f'{joint_labels[dim]}\nMAE={mae:.4f}, Corr={corr:.3f}', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for dim in range(action_dim, len(axes)):
        axes[dim].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'episode_{episode_id}_trajectory_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved trajectory comparison: {output_path}")
    plt.close()


def plot_error_analysis(episode_data, output_dir, episode_id):
    """Plot error analysis"""
    expert = episode_data['expert']
    rdt = episode_data['rdt']
    
    error = expert - rdt
    abs_error = np.abs(error)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Error heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(error.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Action Dimension', fontsize=12)
    ax1.set_title(f'Episode {episode_id}: Error Heatmap (Blue=RDT overestimates, Red=RDT underestimates)', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Error Value')
    
    # 2. Error distribution histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(error.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Error', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(abs_error.flatten(), bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax3.axvline(np.mean(abs_error), color='r', linestyle='--', linewidth=2, 
                label=f'MAE={np.mean(abs_error):.4f}')
    ax3.set_xlabel('Absolute Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Absolute Error Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error over time
    ax4 = fig.add_subplot(gs[1, 2])
    mae_over_time = np.mean(abs_error, axis=1)
    ax4.plot(mae_over_time, linewidth=2, color='green')
    ax4.fill_between(range(len(mae_over_time)), 0, mae_over_time, alpha=0.3, color='green')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Mean Absolute Error', fontsize=11)
    ax4.set_title('Error Over Time', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-dimension error boxplot
    ax5 = fig.add_subplot(gs[2, :])
    bp = ax5.boxplot([abs_error[:, i] for i in range(error.shape[1])],
                      labels=[f'D{i}' for i in range(error.shape[1])],
                      patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax5.set_xlabel('Action Dimension', fontsize=12)
    ax5.set_ylabel('Absolute Error', fontsize=12)
    ax5.set_title('Per-Dimension Error Distribution (Boxplot)', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'episode_{episode_id}_error_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved error analysis: {output_path}")
    plt.close()


def plot_correlation_matrix(episode_data, output_dir, episode_id):
    """Plot correlation analysis"""
    expert = episode_data['expert']
    rdt = episode_data['rdt']
    action_dim = expert.shape[1]
    
    # Compute correlation coefficient for each dimension
    correlations = []
    for dim in range(action_dim):
        corr = np.corrcoef(expert[:, dim], rdt[:, dim])[0, 1]
        correlations.append(corr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Correlation bar chart
    colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' for c in correlations]
    bars = ax1.bar(range(action_dim), correlations, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.9, color='g', linestyle='--', linewidth=2, label='Excellent (>0.9)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='Good (>0.7)')
    ax1.set_xlabel('Action Dimension', fontsize=12)
    ax1.set_ylabel('Correlation Coefficient', fontsize=12)
    ax1.set_title(f'Episode {episode_id}: RDT vs Expert Correlation', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Scatter plot (first dimension)
    sample_dim = 0
    ax2.scatter(expert[:, sample_dim], rdt[:, sample_dim], alpha=0.5, s=20)
    
    # Add diagonal line (perfect prediction)
    min_val = min(expert[:, sample_dim].min(), rdt[:, sample_dim].min())
    max_val = max(expert[:, sample_dim].max(), rdt[:, sample_dim].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel(f'Expert Action (Dim {sample_dim})', fontsize=12)
    ax2.set_ylabel(f'RDT Prediction (Dim {sample_dim})', fontsize=12)
    ax2.set_title(f'Scatter Plot: Dim {sample_dim}\nCorrelation={correlations[sample_dim]:.3f}', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'episode_{episode_id}_correlation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved correlation analysis: {output_path}")
    plt.close()


def create_animation(episode_data, output_dir, episode_id, fps=10):
    """Create action sequence animation"""
    expert = episode_data['expert']
    rdt = episode_data['rdt']
    T = len(expert)
    action_dim = expert.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Initialize lines
    time_steps = np.arange(T)
    line_expert, = ax.plot([], [], 'b-', label='Expert', linewidth=2)
    line_rdt, = ax.plot([], [], 'r--', label='RDT', linewidth=2)
    point_expert, = ax.plot([], [], 'bo', markersize=8)
    point_rdt, = ax.plot([], [], 'ro', markersize=8)
    
    ax.set_xlim(0, T)
    ax.set_ylim(expert.min() - 0.1, expert.max() + 0.1)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Action Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Dimension selector
    current_dim = [0]
    
    def init():
        line_expert.set_data([], [])
        line_rdt.set_data([], [])
        point_expert.set_data([], [])
        point_rdt.set_data([], [])
        return line_expert, line_rdt, point_expert, point_rdt
    
    def animate(frame):
        dim = current_dim[0]
        
        # Update lines
        line_expert.set_data(time_steps[:frame+1], expert[:frame+1, dim])
        line_rdt.set_data(time_steps[:frame+1], rdt[:frame+1, dim])
        
        # Update current points
        point_expert.set_data([frame], [expert[frame, dim]])
        point_rdt.set_data([frame], [rdt[frame, dim]])
        
        # Update title
        error = abs(expert[frame, dim] - rdt[frame, dim])
        ax.set_title(f'Episode {episode_id} - Dim {dim} | Step {frame}/{T-1} | Error={error:.4f}',
                    fontsize=14)
        
        return line_expert, line_rdt, point_expert, point_rdt
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=T, interval=1000//fps, blit=True)
    
    output_path = os.path.join(output_dir, f'episode_{episode_id}_animation_dim0.gif')
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"✅ Saved animation: {output_path}")
    plt.close()


def compute_quality_metrics(data):
    """Compute global quality metrics"""
    expert = data['expert_action']
    rdt = data['rdt_action']
    
    # Global metrics
    mse = np.mean((expert - rdt) ** 2)
    mae = np.mean(np.abs(expert - rdt))
    rmse = np.sqrt(mse)
    
    # Per-dimension correlation
    correlations = []
    for dim in range(expert.shape[1]):
        corr = np.corrcoef(expert[:, dim], rdt[:, dim])[0, 1]
        correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    min_corr = np.min(correlations)
    
    print("\n" + "="*60)
    print("📊 RDT Prediction Quality Assessment")
    print("="*60)
    print(f"Total Samples: {len(expert)}")
    print(f"Action Dimensions: {expert.shape[1]}")
    print(f"\nOverall Metrics:")
    print(f"  - MSE:  {mse:.6f}")
    print(f"  - MAE:  {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"\nCorrelation Metrics:")
    print(f"  - Avg Correlation: {avg_corr:.4f}")
    print(f"  - Min Correlation: {min_corr:.4f}")
    print(f"  - Dims >0.9: {sum(c > 0.9 for c in correlations)}/{len(correlations)}")
    print(f"  - Dims >0.7: {sum(c > 0.7 for c in correlations)}/{len(correlations)}")
    
    # Quality rating
    if avg_corr > 0.9 and mae < 0.05:
        quality = "Excellent ⭐⭐⭐⭐⭐"
    elif avg_corr > 0.8 and mae < 0.1:
        quality = "Good ⭐⭐⭐⭐"
    elif avg_corr > 0.7 and mae < 0.15:
        quality = "Fair ⭐⭐⭐"
    else:
        quality = "Needs Improvement ⭐⭐"
    
    print(f"\n✨ Overall Quality Rating: {quality}")
    print("="*60 + "\n")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'avg_corr': avg_corr,
        'min_corr': min_corr,
        'correlations': correlations,
        'quality': quality
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize RDT-generated actions')
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to zarr file containing RDT labels')
    parser.add_argument('--output_dir', type=str, default='./action_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--episode_id', type=int, default=0,
                        help='Episode ID to visualize')
    parser.add_argument('--create_animation', action='store_true',
                        help='Whether to create animation (time-consuming)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation frame rate')
    
    args = parser.parse_args()
    time.sleep(10)  # Delay to allow user to cancel if needed
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_data(args.zarr_path)
    
    # Compute global quality metrics
    metrics = compute_quality_metrics(data)
    
    # Extract single episode
    episode_data = get_episode_data(data, args.episode_id)
    print(f"\nVisualizing Episode {args.episode_id} (samples {episode_data['start']} - {episode_data['end']})")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_trajectory_comparison(episode_data, args.output_dir, args.episode_id)
    plot_error_analysis(episode_data, args.output_dir, args.episode_id)
    plot_correlation_matrix(episode_data, args.output_dir, args.episode_id)
    
    if args.create_animation:
        print("\nCreating animation (may take a few minutes)...")
        create_animation(episode_data, args.output_dir, args.episode_id, args.fps)
    
    print(f"\n✅ All done! Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in os.listdir(args.output_dir):
        if f.startswith(f'episode_{args.episode_id}'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
