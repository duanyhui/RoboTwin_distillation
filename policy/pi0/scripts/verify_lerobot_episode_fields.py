"""
快速检查 LeRobotDataset 是否包含 Plan-STaR 依赖的关键字段。

重点字段：
- episode_index / frame_index：用于 demo（episode）子集抽样、有效起点过滤、按 episode 做 stride
- observation.state：用于 change_point_state 与 gripper event（夹爪事件）检测
- action：用于构造 action 序列（delta_timestamps）

该脚本也会输出 state/action 的维度，便于你确定 `--gripper-dims`（夹爪维度索引）。
"""

import dataclasses

import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class Args:
    # repo_id：LeRobotDataset 数据集名
    repo_id: str
    # 打印前几个 episode 的长度，便于确认 episode 分割是否正常
    max_episodes_print: int = 5


def main(args: Args) -> None:
    ds = LeRobotDataset(args.repo_id)
    hf = ds.hf_dataset

    keys = set(hf.column_names)
    required = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    print(f"repo_id: {args.repo_id}")
    print(f"len: {len(hf)}")
    print(f"fingerprint: {getattr(hf, '_fingerprint', '')}")
    print("has_fields:")
    for k in required:
        print(f"- {k}: {k in keys}")

    if "observation.state" in keys:
        try:
            state0 = np.asarray(hf["observation.state"][0])
            print(f"state0_shape: {tuple(state0.shape)}  state_dim: {int(state0.reshape(-1).shape[0])}")
        except Exception:
            print("state0_shape: <unavailable>")

    if "action" in keys:
        try:
            action0 = np.asarray(hf["action"][0])
            print(f"action0_shape: {tuple(action0.shape)}  action_dim: {int(action0.reshape(-1).shape[0])}")
        except Exception:
            print("action0_shape: <unavailable>")

    if "episode_index" not in keys or "frame_index" not in keys:
        return

    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    episodes = np.unique(episode)
    print(f"num_episodes: {len(episodes)}")

    max_episode = int(episode.max())
    max_frame = np.full((max_episode + 1, ), -1, dtype=np.int64)
    np.maximum.at(max_frame, episode, frame)
    lengths = max_frame + 1

    ok = True
    counts = np.bincount(episode, minlength=max_episode + 1)
    mismatch = np.where(counts != lengths)[0]
    if len(mismatch) > 0:
        ok = False
        print(f"WARNING: episode count != max_frame+1 for {len(mismatch)} episodes (show first 10): {mismatch[:10].tolist()}")
    print(f"episode contiguous frames check: {ok}")

    valid_lengths = lengths[lengths > 0]
    print(f"episode_len: min={int(valid_lengths.min())}, mean={float(valid_lengths.mean()):.2f}, max={int(valid_lengths.max())}")
    print("first_episodes:")
    for e in episodes[: args.max_episodes_print]:
        e = int(e)
        print(f"- ep={e}: len={int(lengths[e])}")


if __name__ == "__main__":
    main(tyro.cli(Args))
