"""
校验 Sampling Plan 的完整性与关键信息摘要输出。

用途：
- 训练前：确认 plan 是否包含必要字段、sha256 校验通过、anchors 数足够
- 复现/审计：快速打印 plan_version/repo_id/episodes_keep/阶段统计/权重统计/ESS 等
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from openpi.training.sampling_plan import SamplingPlan


@dataclasses.dataclass(frozen=True)
class Args:
    # plan_path：plan 输出目录 或 plan_meta.json / plan_arrays.npz
    plan_path: str
    # 可选：检查 anchors 是否至少达到某个阈值（例如 >= batch_size）
    check_min_anchors: int | None = None


def main(args: Args) -> None:
    plan = SamplingPlan.load(args.plan_path, verify_sha256=True)
    meta = plan.meta

    required_meta = [
        "plan_version",
        "repo_id",
        "dataset_fingerprint",
        "dataset_len",
        "num_episodes_keep",
        "episodes_keep",
        "plan_seed",
        "sampler_seed",
        "replacement",
        "action_horizon_H",
        "batch_size",
        "anchor_strategy",
        "stage_strategy",
        "weight_strategy",
        "gripper_event",
        "plan_sha256",
        "summary",
    ]
    missing = [k for k in required_meta if k not in meta]
    if missing:
        raise ValueError(f"Missing meta fields: {missing}")

    m = len(plan.anchor_indices)
    print(f"plan_sha256: {meta.get('plan_sha256')}")
    print(f"repo_id: {meta.get('repo_id')}")
    print(f"dataset_len: {meta.get('dataset_len')} fingerprint: {meta.get('dataset_fingerprint')}")
    print(f"episodes_keep: {meta.get('num_episodes_keep')}  anchors: {m}")
    print(f"replacement: {meta.get('replacement')}  sampler_seed: {meta.get('sampler_seed')}")
    print(f"anchor_strategy: {json.dumps(meta.get('anchor_strategy'), ensure_ascii=False)}")
    print(f"stage_strategy: {json.dumps(meta.get('stage_strategy'), ensure_ascii=False)}")
    print(f"gripper_event: {json.dumps(meta.get('gripper_event'), ensure_ascii=False)}")

    if args.check_min_anchors is not None and m < int(args.check_min_anchors):
        raise ValueError(f"anchors={m} < check_min_anchors={args.check_min_anchors}")

    stage_ids = plan.stage_ids.astype(np.int64)
    k = int(meta["stage_strategy"]["num_stages_K"])
    stage_counts = np.bincount(stage_ids, minlength=k)
    w = plan.weights.astype(np.float64)
    p = w / np.sum(w)
    ess = float(1.0 / np.sum(np.square(p)))

    print(f"stage_counts(recomputed): {stage_counts.tolist()}")
    print(f"weight_stats: min={float(w.min()):.6g} median={float(np.median(w)):.6g} max={float(w.max()):.6g}")
    print(f"ESS_reference(recomputed): {ess:.3f}")

    meta_path = Path(args.plan_path)
    if meta_path.is_dir():
        meta_path = meta_path / "plan_meta.json"
    print(f"meta_path: {meta_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
