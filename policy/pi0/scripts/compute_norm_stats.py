"""计算归一化统计（norm stats：均值/方差/分位数等）。

训练数据会在进入模型前做 Normalize（标准化），其参数来自 `norm_stats.json`。
少样本/短训条件下，norm stats 的 reuse（复用）/recompute（重算）往往会显著影响稳定性，
因此本脚本支持：

- 全量重算：对 repo_id 的全数据统计
- 按 Sampling Plan 子集重算（控制变量）：anchors / episodes / episodes_valid

其中 episodes_valid 会额外应用有效起点过滤（valid_start），避免 episode 末尾 action_horizon
退化影响统计。该规则会优先读取 plan_meta.json 的 `valid_start_rule.tail_keep`（若不存在则默认 0）：

- strict（tail_keep=0）：valid_start(e,f) := (f <= L_e - H)
- relaxed（tail_keep>0）：valid_start(e,f) := (f <= min(L_e - H + tail_keep, L_e - 1))
"""

import typing
from typing import Literal

import numpy as np
import tqdm
import torch
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.sampling_plan import SamplingPlan
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        # JAX/统计阶段不需要字符串字段，且字符串无法直接转成数值数组
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig, ) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def _unwrap_lerobot_dataset(dataset: _data_loader.Dataset):
    ds = dataset
    while hasattr(ds, "_dataset"):
        ds = getattr(ds, "_dataset")
    return ds


def _episode_lengths(episode_index: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
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


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    plan_path: str | None = None,
    # plan_subset_mode：
    # - none：不用 plan（默认）
    # - anchors：只用 plan.anchor_indices 对应的样本
    # - episodes：只用 plan.episodes_keep 对应 episode 的所有帧
    # - episodes_valid：episodes + valid_start 过滤（推荐）
    plan_subset_mode: Literal["none", "anchors", "episodes", "episodes_valid"] = "none",
    plan_verify_sha256: bool = True,
    # seed：当 max_frames 触发随机子采样时用于可复现抽样
    seed: int = 0,
    # output_asset_id：输出目录名（避免覆盖默认 repo_id 的 norm_stats.json）
    output_asset_id: str | None = None,
):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    selected_indices = np.arange(len(dataset), dtype=np.int64)

    if plan_path is not None and plan_subset_mode != "none":
        plan = SamplingPlan.load(plan_path, verify_sha256=plan_verify_sha256)
        meta = plan.meta
        if meta.get("repo_id") != data_config.repo_id:
            raise ValueError(f"Plan repo_id mismatch: plan={meta.get('repo_id')} config={data_config.repo_id}")
        if int(meta.get("action_horizon_H", -1)) != int(config.model.action_horizon):
            raise ValueError(
                f"Plan action_horizon_H mismatch: plan={meta.get('action_horizon_H')} model={config.model.action_horizon}"
            )

        if plan_subset_mode == "anchors":
            selected_indices = plan.anchor_indices.astype(np.int64, copy=False)
        elif plan_subset_mode in ("episodes", "episodes_valid"):
            episodes_keep = meta.get("episodes_keep")
            if not isinstance(episodes_keep, list) or len(episodes_keep) == 0:
                raise ValueError("Plan meta missing episodes_keep; cannot use plan_subset_mode=episodes*.")

            lr_ds = _unwrap_lerobot_dataset(dataset)
            hf = getattr(lr_ds, "hf_dataset", None)
            if hf is None:
                raise ValueError("Cannot access hf_dataset; cannot use plan_subset_mode=episodes*.")

            if "episode_index" not in hf.column_names or "frame_index" not in hf.column_names:
                raise ValueError("Dataset missing episode_index/frame_index; cannot use plan_subset_mode=episodes*.")

            episode_index = np.asarray(hf["episode_index"], dtype=np.int64)
            frame_index = np.asarray(hf["frame_index"], dtype=np.int64)
            keep = np.isin(episode_index, np.asarray(episodes_keep, dtype=np.int64))
            if plan_subset_mode == "episodes_valid":
                horizon = int(config.model.action_horizon)
                # 与 build_sampling_plan.py 的 valid_start 规则保持一致：
                # - 新版 plan：meta["valid_start_rule"]["tail_keep"]
                # - 旧版 plan：可能没有该字段，此时默认按 strict（tail_keep=0）
                valid_rule = meta.get("valid_start_rule", {})
                tail_keep = int(valid_rule.get("tail_keep", 0)) if isinstance(valid_rule, dict) else 0
                tail_keep = int(np.clip(tail_keep, 0, max(0, horizon - 1)))
                lengths = _episode_lengths(episode_index, frame_index)
                ep = episode_index.astype(np.int64)
                max_start = lengths[ep] - horizon + tail_keep
                max_start = np.minimum(max_start, lengths[ep] - 1)
                valid = frame_index <= max_start
                keep = keep & valid
            selected_indices = np.where(keep)[0].astype(np.int64)
        else:
            raise ValueError(f"Unknown plan_subset_mode: {plan_subset_mode}")

    if selected_indices.size == 0:
        raise ValueError("No frames selected for norm stats.")

    if max_frames is not None and max_frames < int(selected_indices.size):
        rng = np.random.default_rng(int(seed))
        selected_indices = rng.choice(selected_indices, size=int(max_frames), replace=False).astype(np.int64)

    dataset = torch.utils.data.Subset(
        typing.cast(torch.utils.data.Dataset, dataset),
        selected_indices.tolist(),
    )

    num_frames = len(dataset)
    local_batch_size = min(8, num_frames)
    num_batches = max(1, num_frames // local_batch_size)

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        num_workers=8,
        shuffle=False,
        num_batches=num_batches,
        seed=int(seed),
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = output_asset_id or data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Could not determine output asset_id.")

    output_path = config.assets_dirs / str(asset_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
