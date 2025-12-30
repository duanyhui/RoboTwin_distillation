"""
构建 Plan-STaR 的 Sampling Plan（采样计划文件）。

这个脚本只做“离线生成 plan”：
- 从 LeRobotDataset（repo_id）中选择一部分 demo（episode 子集抽样：50→5/10/20）
- 过滤掉 episode 末尾无法提供完整 action_horizon 的起点（valid_start：f <= L_e - H）
- 在每个 episode 内按 stride 选择锚点（anchor），降低相邻帧强相关（提高短训效率）
- 给每个 anchor 打阶段标签（time_quantile 或 change_point_state）
- 计算每个 anchor 的采样权重（阶段均衡 + 可选夹爪事件加权 + 裁剪/混合）

训练阶段只需要把输出目录通过 `--sampling-plan-path` 传给 train.py，
即可在不改训练循环（单阶段训练）的情况下，改变“数据出现的分布”。
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as _config
from openpi.training.sampling_plan import SamplingPlan


def _stable_hash_u64(text: str) -> int:
    # 用 sha256 做稳定哈希（跨进程/跨平台一致），用于 stride 对齐 offset。
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _sample_without_replacement(items: np.ndarray, k: int, *, seed: int) -> np.ndarray:
    """无放回抽样（用于 episode 子集抽样）。"""
    rng = np.random.default_rng(seed)
    if k > len(items):
        raise ValueError(f"Cannot sample k={k} from {len(items)} items without replacement.")
    return rng.choice(items, size=k, replace=False)


def _episode_lengths(episode_index: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    """根据 episode_index/frame_index 统计每个 episode 的长度（最大 frame + 1）。"""
    if episode_index.size == 0:
        return np.zeros((0, ), dtype=np.int64)
    max_episode = int(episode_index.max())
    max_frame = np.full((max_episode + 1, ), -1, dtype=np.int64)
    np.maximum.at(max_frame, episode_index.astype(np.int64), frame_index.astype(np.int64))
    lengths = max_frame + 1
    if np.any(lengths <= 0):
        bad = np.where(lengths <= 0)[0].tolist()
        raise ValueError(f"Invalid episode lengths for episodes: {bad[:10]}")
    return lengths


def _select_valid_start_mask(
    *,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    episode_lengths: np.ndarray,
    horizon: int,
    tail_keep: int,
) -> np.ndarray:
    """有效起点过滤（可选放宽尾部起点）。

严格规则（tail_keep=0）：
  valid_start(e,f) := (f <= L_e - H)

放宽尾部起点（tail_keep>0）：
  valid_start(e,f) := (f <= min(L_e - H + tail_keep, L_e - 1))

直觉：允许更多“episode 末尾”的起点进入训练，可以显著增大 anchors 数，
减少少样本条件下的重复暴露；但也可能引入更“平稳/退化”的动作序列（如末尾动作重复），
需要在消融中验证（建议先试 tail_keep=H-1 即不做尾部过滤）。
"""
    tail_keep = int(max(0, tail_keep))
    ep = episode_index.astype(np.int64)
    max_start = episode_lengths[ep] - int(horizon) + tail_keep
    max_start = np.minimum(max_start, episode_lengths[ep] - 1)
    return frame_index <= max_start


def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    """简单 1D 中值滤波（抑制夹爪小抖动/噪声）。"""
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError("median filter window must be odd.")
    radius = window // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - radius)
        hi = min(len(x), i + radius + 1)
        out[i] = np.median(x[lo:hi])
    return out


def _local_peaks(values: np.ndarray, *, threshold: float) -> np.ndarray:
    """局部峰值（用于变化点/夹爪事件候选）。"""
    if len(values) < 3:
        return np.zeros((0, ), dtype=np.int64)
    mid = (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] > threshold)
    return np.where(mid)[0].astype(np.int64) + 1


def _nms_1d(indices: np.ndarray, scores: np.ndarray, *, k: int, min_gap: int) -> np.ndarray:
    """1D NMS（非极大值抑制）：避免同一事件被密集重复计数。"""
    if k <= 0 or len(indices) == 0:
        return np.zeros((0, ), dtype=np.int64)
    order = np.argsort(scores[indices])[::-1]
    kept: list[int] = []
    for idx in indices[order]:
        if all(abs(int(idx) - int(j)) >= min_gap for j in kept):
            kept.append(int(idx))
            if len(kept) >= k:
                break
    return np.array(sorted(kept), dtype=np.int64)


def _time_quantile_stage(frame: np.ndarray, denom: np.ndarray, k: int) -> np.ndarray:
    """时间分位数阶段：stage = floor(K * f / (L-H))。"""
    u = frame / denom
    stage = np.floor(k * u).astype(np.int64)
    return np.clip(stage, 0, k - 1).astype(np.int32)


def _compute_change_point_boundaries(
    state: np.ndarray,
    *,
    k: int,
    delta: int,
    min_gap: int,
) -> np.ndarray:
    """用 state 变化幅度（L2 norm）找变化点边界（每条 episode 取 K-1 个边界）。"""
    if k <= 1:
        return np.zeros((0, ), dtype=np.int64)
    if len(state) < (delta + 3):
        return np.array([max(1, int(round((j + 1) * len(state) / k))) for j in range(k - 1)], dtype=np.int64)
    diffs = state[delta:] - state[:-delta]
    score = np.linalg.norm(diffs, axis=1)
    score = np.concatenate([np.zeros((delta, ), dtype=score.dtype), score], axis=0)
    peaks = _local_peaks(score, threshold=float(np.quantile(score, 0.9)))
    peaks = _nms_1d(peaks, score, k=k - 1, min_gap=min_gap)
    if len(peaks) < (k - 1):
        fallback = np.array([int(round((j + 1) * (len(state) - 1) / k)) for j in range(k - 1)], dtype=np.int64)
        for b in fallback:
            if len(peaks) >= (k - 1):
                break
            if b <= 0 or b >= (len(state) - 1):
                continue
            if all(abs(int(b) - int(p)) >= min_gap for p in peaks):
                peaks = np.array(sorted([*peaks.tolist(), int(b)]), dtype=np.int64)
    return peaks.astype(np.int64)


def _compute_gripper_events(
    state: np.ndarray,
    *,
    gripper_dims: list[int],
    smooth_window: int,
    delta1: int,
    delta2: int,
    q: float,
    min_gap: int,
) -> np.ndarray:
    """检测夹爪事件（开/合等关键变化时刻）。

思路：从 state 里挑 gripper_dims，做中值滤波后计算多尺度差分，
用高分位阈值 + NMS 只保留最显著且彼此相隔足够远的峰值。
"""
    if len(state) == 0:
        return np.zeros((0, ), dtype=np.int64)
    g = np.max(np.abs(state[:, gripper_dims]), axis=1)
    g = _median_filter_1d(g, smooth_window)
    d = np.zeros_like(g)
    for delta in sorted({int(delta1), int(delta2)}):
        if delta <= 0:
            continue
        diff = np.zeros_like(g)
        diff[delta:] = np.abs(g[delta:] - g[:-delta])
        d = np.maximum(d, diff)
    base = d[max(delta1, delta2):] if len(d) > max(delta1, delta2) else d
    tau = float(np.quantile(base, q)) if len(base) else 0.0
    peaks = _local_peaks(d, threshold=tau)
    peaks = _nms_1d(peaks, d, k=max(1, len(peaks)), min_gap=min_gap)
    return peaks.astype(np.int64)


def _distance_event_to_window(start_frame: np.ndarray, *, horizon: int, event_frames: np.ndarray) -> np.ndarray:
    """计算事件点到窗口 [f, f+H-1] 的最小距离（窗口内则为 0）。"""
    if len(event_frames) == 0:
        return np.full_like(start_frame, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    f0 = start_frame.astype(np.int64)
    f1 = f0 + (horizon - 1)
    dist = np.full_like(f0, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    for t in event_frames.astype(np.int64):
        d = np.where(t < f0, f0 - t, np.where(t > f1, t - f1, 0))
        dist = np.minimum(dist, d)
    return dist


@dataclasses.dataclass(frozen=True)
class Args:
    """build_sampling_plan.py 的命令行参数（Tyro 会根据这里自动生成 CLI）。"""

    # === 输入数据 / 训练配置 ===
    repo_id: str
    train_config_name: str = "pi0_base_aloha_robotwin_lora"

    # === episode 子集抽样（50→5/10/20）===
    num_episodes_keep: int = 50
    episode_selection_seed: int = 0
    # plan_seed：影响每个 episode 的 stride offset（保证“去相关”同时可复现）
    plan_seed: int = 0
    # sampler_seed：训练时 WeightedRandomSampler(generator=...) 的种子（写入 plan_meta）
    sampler_seed: int = 0
    # replacement：是否“有放回采样”（抽过的样本还可能再次被抽到）
    replacement: bool = True

    # === 去相关锚点（anchor）===
    # stride_S：episode 内隔 S 帧取一个锚点（S 越大越去相关，但 anchors 越少）
    stride_S: int = 2
    # min_anchors：最少锚点数量（默认用 batch_size，保证至少能凑出 1 个 batch）
    min_anchors: int | None = None

    # === 有效起点过滤（尾部放宽）===
    # valid_start_tail_keep：在严格规则 f<=L-H 的基础上，额外允许保留多少帧作为起点（0 表示严格过滤）。
    # 对于 horizon=50，设置为 49 近似等价于“不做尾部过滤”（允许起点到 L-1）。
    valid_start_tail_keep: int = 0

    # === 阶段标注（stage labeling）===
    stage_strategy: Literal["time_quantile", "change_point_state"] = "time_quantile"
    # num_stages_K：阶段数 K
    num_stages_K: int = 4
    # change_point_*：用于 change_point_state 的变化点检测超参
    change_point_delta: int = 3
    change_point_min_gap: int = 10

    # === 阶段均衡/加权（输出 weights 的 base 部分）===
    stage_balance: Literal["uniform_over_stage", "inv_freq"] = "uniform_over_stage"
    inv_freq_power: float = 1.0
    # clip_ratio：把权重裁剪到 [median/clip_ratio, median*clip_ratio]，避免极端权重
    clip_ratio: float = 5.0
    # epsilon_mix：权重与均值的 ε 混合，避免极小/零权重导致有效样本暴露不足
    epsilon_mix: float = 0.01

    # === 夹爪事件加权（可选）===
    gripper_event_enabled: bool = False
    # gripper_dims：夹爪在 observation.state 中的维度索引（需按你的数据集确认）
    gripper_dims: str = "6,13"
    gripper_smooth_window: int = 5
    gripper_delta1: int = 1
    gripper_delta2: int = 5
    gripper_quantile_q: float = 0.95
    gripper_min_gap: int = 10
    # soft_exp：距离事件越近加权越大；hard：距离<=w 才加权
    gripper_event_mode: Literal["soft_exp", "hard"] = "soft_exp"
    # soft_exp 的衰减尺度 σ（越大越“宽”）
    gripper_sigma: float = 10.0
    # hard 模式的窗口阈值 w（单位：帧）
    gripper_window_w: int = 5
    # gamma：事件加权强度，最终倍数为 (1 + gamma * m)
    gripper_gamma: float = 2.0

    # === 输出目录 ===
    out_dir: str = "sampling_plans/plan_star"


def main(args: Args) -> None:
    cfg = _config.get_config(args.train_config_name)
    # H = action_horizon（一个样本窗口需要的动作序列长度）
    horizon = int(cfg.model.action_horizon)
    batch_size = int(cfg.batch_size)
    min_anchors = int(args.min_anchors) if args.min_anchors is not None else batch_size

    # 只读 HuggingFace Dataset 的列（不解码图像），构建 plan 更快更省内存
    ds = LeRobotDataset(args.repo_id)
    hf = ds.hf_dataset
    dataset_fingerprint = str(getattr(hf, "_fingerprint", ""))
    dataset_len = len(hf)

    columns = set(hf.column_names)
    has_episode_fields = ("episode_index" in columns) and ("frame_index" in columns)
    if has_episode_fields:
        episode_index = np.asarray(hf["episode_index"], dtype=np.int64)
        frame_index = np.asarray(hf["frame_index"], dtype=np.int64)
    else:
        # 退化模式：没有 episode 信息时，把全数据视为单一 episode（无法做 50→5/10/20 子集抽样）
        episode_index = np.zeros((dataset_len, ), dtype=np.int64)
        frame_index = np.arange(dataset_len, dtype=np.int64)

    need_state = (args.stage_strategy == "change_point_state") or bool(args.gripper_event_enabled)
    if need_state:
        if "observation.state" not in columns:
            raise ValueError("Dataset missing observation.state, required for stage_strategy=change_point_state or gripper_event_enabled.")
        state = np.asarray(hf["observation.state"], dtype=np.float32)
    else:
        state = None

    episodes_all = np.unique(episode_index)
    if args.num_episodes_keep <= 0:
        raise ValueError("--num_episodes_keep must be positive.")
    if not has_episode_fields and args.num_episodes_keep != 1:
        raise ValueError("Dataset missing episode_index/frame_index; episode subset sampling is unavailable. Use --num_episodes_keep 1 or regenerate the dataset with episode fields.")
    if args.num_episodes_keep > len(episodes_all):
        raise ValueError(f"Requested num_episodes_keep={args.num_episodes_keep} but dataset has {len(episodes_all)} episodes.")

    # === (1) episode 子集抽样：固定 seed 后可复现 ===
    episodes_keep = (
        episodes_all if args.num_episodes_keep == len(episodes_all) else _sample_without_replacement(episodes_all, args.num_episodes_keep, seed=args.episode_selection_seed)
    )
    episodes_keep = np.sort(episodes_keep).astype(np.int64)

    # === (2) 有效起点过滤：f <= L_e - H（可选放宽 tail_keep）===
    tail_keep = int(np.clip(int(args.valid_start_tail_keep), 0, max(0, horizon - 1)))
    lengths = _episode_lengths(episode_index, frame_index)
    valid_start_mask = _select_valid_start_mask(
        episode_index=episode_index,
        frame_index=frame_index,
        episode_lengths=lengths,
        horizon=horizon,
        tail_keep=tail_keep,
    )
    keep_episode_mask = np.isin(episode_index, episodes_keep)
    candidate_mask = valid_start_mask & keep_episode_mask

    # === (3) anchor 选择：episode 内 stride + 稳定 offset（避免每条 demo 都对齐到同一相位） ===
    stride = int(args.stride_S)
    if stride <= 0:
        raise ValueError("--stride_S must be positive.")

    anchor_mask = None
    while True:
        offsets = np.zeros((len(lengths), ), dtype=np.int64)
        for e in episodes_keep.tolist():
            offsets[int(e)] = int(_stable_hash_u64(f"{args.plan_seed}:{int(e)}") % stride)
        anchor_mask = candidate_mask & (((frame_index - offsets[episode_index]) % stride) == 0)
        anchor_indices = np.where(anchor_mask)[0].astype(np.int64)
        if len(anchor_indices) >= min_anchors or stride == 1:
            break
        # anchors 不足时自动减小 stride（保证最少能凑出 1 个 batch）
        stride = max(1, stride // 2)

    if anchor_mask is None:
        raise RuntimeError("Anchor selection failed.")

    anchor_indices = np.where(anchor_mask)[0].astype(np.int64)
    if len(anchor_indices) < min_anchors:
        raise ValueError(f"Not enough anchors for one batch: anchors={len(anchor_indices)}, required>={min_anchors}. Try smaller stride or more demos.")

    denom = (lengths[episode_index[anchor_indices]] - horizon).astype(np.float32)
    if np.any(denom <= 0):
        raise ValueError("Found episodes with length <= horizon; cannot construct valid start frames.")

    # === (4) 阶段标注：time_quantile 或 change_point_state ===
    if args.stage_strategy == "time_quantile":
        stage_ids = _time_quantile_stage(frame_index[anchor_indices].astype(np.float32), denom, int(args.num_stages_K))
    else:
        k = int(args.num_stages_K)
        boundaries_per_episode: dict[int, np.ndarray] = {}
        for e in episodes_keep.tolist():
            idx_e = np.where(episode_index == int(e))[0].astype(np.int64)
            if len(idx_e) == 0:
                boundaries_per_episode[int(e)] = np.zeros((0, ), dtype=np.int64)
                continue
            order = np.argsort(frame_index[idx_e])
            idx_sorted = idx_e[order]
            if state is None:
                raise RuntimeError("state is required for change_point_state but was not loaded.")
            state_e = state[idx_sorted]
            boundaries_per_episode[int(e)] = _compute_change_point_boundaries(
                state_e,
                k=k,
                delta=int(args.change_point_delta),
                min_gap=int(args.change_point_min_gap),
            )
        stage_ids = np.empty((len(anchor_indices), ), dtype=np.int32)
        for j, i in enumerate(anchor_indices.tolist()):
            e = int(episode_index[i])
            b = boundaries_per_episode.get(e, np.zeros((0, ), dtype=np.int64))
            stage_ids[j] = int(np.searchsorted(b, int(frame_index[i]), side="right"))

    k = int(args.num_stages_K)
    stage_ids = np.clip(stage_ids.astype(np.int64), 0, k - 1).astype(np.int32)
    stage_counts = np.bincount(stage_ids.astype(np.int64), minlength=k).astype(np.int64)
    if np.any(stage_counts == 0):
        pass

    stage_count_for_sample = stage_counts[stage_ids.astype(np.int64)]
    if args.stage_balance == "uniform_over_stage":
        # 让每个阶段总体采样概率更接近：每个样本权重与阶段样本数成反比
        w_base = 1.0 / stage_count_for_sample
    else:
        # 更激进的稀缺补偿：1/(count^p)
        w_base = 1.0 / np.power(stage_count_for_sample, float(args.inv_freq_power))

    w_event = np.ones_like(w_base, dtype=np.float64)
    gripper_dims = []
    gripper_events_stats = {"min": 0, "mean": 0.0, "max": 0}
    if args.gripper_event_enabled:
        # === (5) 夹爪事件加权：只加权“显著变化”附近的窗口，抑制小抖动误报 ===
        gripper_dims = [int(x.strip()) for x in args.gripper_dims.split(",") if x.strip() != ""]
        if not gripper_dims:
            raise ValueError("--gripper_dims is empty.")
        if state is None:
            raise RuntimeError("state is required for gripper_event_enabled but was not loaded.")
        state_dim = state.shape[1]
        if any(d < 0 or d >= state_dim for d in gripper_dims):
            raise ValueError(f"Invalid gripper_dims={gripper_dims} for state_dim={state_dim}.")

        events_per_episode: dict[int, np.ndarray] = {}
        counts = []
        for e in episodes_keep.tolist():
            idx_e = np.where(episode_index == int(e))[0].astype(np.int64)
            order = np.argsort(frame_index[idx_e])
            idx_sorted = idx_e[order]
            state_e = state[idx_sorted]
            events = _compute_gripper_events(
                state_e,
                gripper_dims=gripper_dims,
                smooth_window=int(args.gripper_smooth_window),
                delta1=int(args.gripper_delta1),
                delta2=int(args.gripper_delta2),
                q=float(args.gripper_quantile_q),
                min_gap=int(args.gripper_min_gap),
            )
            events_per_episode[int(e)] = events
            counts.append(int(len(events)))
        if counts:
            gripper_events_stats = {"min": int(np.min(counts)), "mean": float(np.mean(counts)), "max": int(np.max(counts))}

        dist = np.empty((len(anchor_indices), ), dtype=np.int64)
        for e in episodes_keep.tolist():
            mask_e = episode_index[anchor_indices] == int(e)
            if not np.any(mask_e):
                continue
            f_start = frame_index[anchor_indices[mask_e]]
            events = events_per_episode.get(int(e), np.zeros((0, ), dtype=np.int64))
            d = _distance_event_to_window(f_start, horizon=horizon, event_frames=events)
            dist[mask_e] = d
        if args.gripper_event_mode == "soft_exp":
            # 平滑衰减：距离越近，加权越大
            sigma = float(args.gripper_sigma)
            m = np.exp(-dist.astype(np.float64) / max(1e-6, sigma))
        else:
            # 硬阈值：距离<=w 才加权
            w = int(args.gripper_window_w)
            m = (dist <= w).astype(np.float64)
        w_event = 1.0 + float(args.gripper_gamma) * m

    # === (6) 合成最终权重 + 裁剪 + ε 混合（避免极端权重）===
    weights = w_base.astype(np.float64) * w_event.astype(np.float64)
    median = float(np.median(weights))
    if median <= 0:
        median = 1.0
    clip_ratio = float(args.clip_ratio)
    w_min = median / clip_ratio
    w_max = median * clip_ratio
    weights = np.clip(weights, w_min, w_max)

    eps = float(args.epsilon_mix)
    if eps > 0:
        weights = (1.0 - eps) * weights + eps * (np.sum(weights) / len(weights))

    weight_stats = {"min": float(np.min(weights)), "median": float(np.median(weights)), "max": float(np.max(weights))}
    p = weights / np.sum(weights)
    ess = float(1.0 / np.sum(np.square(p)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "plan_version": "plan-star-v1",
        "created_at_utc": np.datetime64("now", "s").astype(str),
        "train_config_name": str(args.train_config_name),
        "repo_id": args.repo_id,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_len": int(dataset_len),
        "has_episode_fields": bool(has_episode_fields),
        "num_episodes_total": int(len(episodes_all)),
        "num_episodes_keep": int(len(episodes_keep)),
        "episodes_keep": episodes_keep.tolist(),
        "episode_selection_seed": int(args.episode_selection_seed),
        "plan_seed": int(args.plan_seed),
        "sampler_seed": int(args.sampler_seed),
        "replacement": bool(args.replacement),
        "action_horizon_H": int(horizon),
        "batch_size": int(batch_size),
        # 这里记录的是“有效起点过滤”的协议，方便审稿/复现时明确“哪些帧被允许作为起点”
        "valid_start_rule": {
            "tail_keep": int(tail_keep),
            "formula": "f_i <= min(L_e - H + tail_keep, L_e - 1)",
        },
        "anchor_strategy": {
            "mode": "episode_stride",
            "stride_S": int(stride),
            "offset_mode": "hash(plan_seed, episode) % S",
            "min_anchors": int(min_anchors),
        },
        "stage_strategy": {
            "type": str(args.stage_strategy),
            "num_stages_K": int(args.num_stages_K),
            "delta": int(args.change_point_delta) if args.stage_strategy == "change_point_state" else None,
            "min_gap": int(args.change_point_min_gap) if args.stage_strategy == "change_point_state" else None,
            "peak_mode": "local_max+nms" if args.stage_strategy == "change_point_state" else None,
        },
        "weight_strategy": {
            "stage_balance": str(args.stage_balance),
            "inv_freq_power": float(args.inv_freq_power),
            "clip_ratio": float(args.clip_ratio),
            "epsilon_mix": float(args.epsilon_mix),
        },
        "gripper_event": {
            "enabled": bool(args.gripper_event_enabled),
            "gripper_dims": gripper_dims,
            "smoothing": {"method": "median", "window": int(args.gripper_smooth_window)},
            "diff": {"delta1": int(args.gripper_delta1), "delta2": int(args.gripper_delta2)},
            "threshold": {"method": "quantile", "q": float(args.gripper_quantile_q)},
            "min_gap": int(args.gripper_min_gap),
            "event_window": {"mode": str(args.gripper_event_mode), "sigma_or_w": float(args.gripper_sigma) if args.gripper_event_mode == "soft_exp" else int(args.gripper_window_w)},
            "gamma": float(args.gripper_gamma),
        },
        "summary": {
            "num_valid_frames": int(np.sum(candidate_mask)),
            "num_anchors": int(len(anchor_indices)),
            "stage_counts": stage_counts.tolist(),
            "num_gripper_events_per_episode": gripper_events_stats,
            "weight_stats": weight_stats,
            "ESS_reference": ess,
        },
    }

    plan = SamplingPlan(
        meta=meta,
        anchor_indices=anchor_indices.astype(np.int64, copy=False),
        stage_ids=stage_ids.astype(np.int32, copy=False),
        weights=weights.astype(np.float32, copy=False),
    )
    meta_path, arrays_path = plan.save(out_dir)
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"Wrote plan: {out_dir}")
    print(f"- meta: {meta_path}")
    print(f"- arrays: {arrays_path}")
    print(f"- plan_sha256: {saved_meta.get('plan_sha256')}")
    print(f"- anchors: {saved_meta['summary']['num_anchors']} (min_anchors={min_anchors}, stride_S={saved_meta['anchor_strategy']['stride_S']})")
    print(f"- stage_counts: {saved_meta['summary']['stage_counts']}")
    if saved_meta["gripper_event"]["enabled"]:
        print(f"- gripper_events_per_episode: {saved_meta['summary']['num_gripper_events_per_episode']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
