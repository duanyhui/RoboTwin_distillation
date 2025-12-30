"""
验证 action_horizon 在 episode 末尾的“标签退化”现象。

Pi0 的数据读取使用 LeRobotDataset 的 delta_timestamps 机制来构造 action 序列：
对每个起点 frame，会取一个长度为 H（action_horizon）的动作序列。

当起点靠近 episode 末尾时，如果未来帧不足，序列往往会出现“重复/退化”（例如末尾动作被重复填充），
这会导致训练时看到大量低信息样本。Plan-STaR 因此使用有效起点过滤规则：

  valid_start(e, f) := (f <= L_e - H)

本脚本对比同一 episode 的：
- last_start_frame：最后一帧作为起点（最容易退化）
- max_valid_start_frame：满足 valid_start 的最大起点（理论上仍能得到完整序列）

输出中 `mean_std_over_horizon` 越接近 0，代表动作序列越“近似常数/退化”。
"""

import dataclasses

import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    # repo_id：LeRobotDataset 数据集名（例如 beat_block_hammer-demo_clean-50）
    repo_id: str
    # train_config_name：用于读取 action_horizon（H），应与训练时一致
    train_config_name: str = "pi0_base_aloha_robotwin_lora"
    # episode_id：要检查哪条 demo（episode）的末尾退化现象
    episode_id: int = 0


def main(args: Args) -> None:
    cfg = _config.get_config(args.train_config_name)
    horizon = int(cfg.model.action_horizon)
    meta = LeRobotDatasetMetadata(args.repo_id)
    fps = float(meta.fps)

    # 用 delta_timestamps 构造 action 序列（长度为 horizon）
    ds = LeRobotDataset(args.repo_id, delta_timestamps={"action": [t / fps for t in range(horizon)]})
    hf = ds.hf_dataset

    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)

    e = int(args.episode_id)
    mask = episode == e
    if not np.any(mask):
        raise ValueError(f"episode_id={e} not found in dataset.")

    frames_e = frame[mask]
    last_f = int(frames_e.max())
    safe_f = int(last_f - horizon + 1)
    if safe_f < 0:
        raise ValueError(f"Episode too short for horizon={horizon}: last_f={last_f}")

    idx_last = int(np.where(mask & (frame == last_f))[0][0])
    idx_safe = int(np.where(mask & (frame == safe_f))[0][0])

    for name, idx in [("last_start_frame", idx_last), ("max_valid_start_frame", idx_safe)]:
        x = ds[idx]
        a = np.asarray(x["action"], dtype=np.float32)
        std_mean = float(a.std(axis=0).mean())
        print(f"{name}: global_idx={idx}, episode={int(x['episode_index'])}, frame={int(x['frame_index'])}")
        print(f"- action_shape={tuple(a.shape)} horizon={horizon} fps={fps}")
        print(f"- nan={bool(np.isnan(a).any())} mean_std_over_horizon={std_mean:.6g}")


if __name__ == "__main__":
    main(tyro.cli(Args))
