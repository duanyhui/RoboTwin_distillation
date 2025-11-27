"""
使用RDT标签的DP数据集类

这个数据集会加载预先计算好的RDT推理输出作为监督标签,而不是使用原始的专家动作
"""

from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDatasetWithRDT(BaseImageDataset):
    """
    使用RDT预测作为监督标签的数据集
    
    与原始RobotImageDataset的区别:
    1. 加载 'rdt_action' 字段作为标签
    2. 可选择是否也保留原始的 'action' 用于对比
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        use_expert_action=False,  # 🔥 是否只用专家动作(False则使用RDT)
        mix_expert_action=False,  # 🔥 同时使用专家与教师, 生成混合标签
        mix_alpha=0.5,            # 🔥 混合权重/概率
        mix_mode="prob",          # 🔥 prob=按概率选RDT/专家; linear=线性加权
        add_expert_noise=False,   # 🔥 是否对专家动作加高斯噪声
        noise_std=0.01,           # 🔥 噪声标准差
        noise_clip=0.05,          # 🔥 噪声截断阈值; 设为None则不截断
    ):

        super().__init__()
        
        # 确定使用哪个action字段
        self.use_expert_action = use_expert_action
        self.mix_expert_action = mix_expert_action
        self.mix_alpha = mix_alpha
        self.mix_mode = mix_mode
        self.add_expert_noise = add_expert_noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        if mix_expert_action:
            mode_tip = "按概率选择 (混合概率=alpha)" if mix_mode == "prob" else "线性加权 (alpha*rdt + (1-alpha)*expert)"
            print(f"🔥 使用混合标签进行训练, mix_alpha={mix_alpha}, 模式={mix_mode} ({mode_tip})")
            action_key = None  # 下方会组合生成
            keys = ["head_camera", "state", "action", "rdt_action"]
        else:
            action_key = 'action' if use_expert_action else 'rdt_action'
            print(f"🔥 使用{'专家动作' if use_expert_action else 'RDT标签'}进行训练")
            keys = ["head_camera", "state", action_key]

        if add_expert_noise and not (use_expert_action or mix_expert_action):
            print("⚠️ add_expert_noise=True 但未使用专家动作，此设置无效")
        elif add_expert_noise:
            print(f"🔊 对专家动作加入高斯噪声: std={noise_std}, clip={noise_clip}")
        
        # 加载数据
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=keys,
        )
        
        # 兼容: 将最终监督标签放到 self.replay_buffer['action']
        final_action = None

        def _noisify(arr):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)
            if self.noise_clip is not None:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            return arr + noise

        if mix_expert_action:
            # 混合教师与专家
            expert = self.replay_buffer['action']
            if self.add_expert_noise:
                expert = _noisify(expert)
            teacher = self.replay_buffer['rdt_action']
            if self.mix_mode == "linear":
                mixed = self.mix_alpha * teacher + (1 - self.mix_alpha) * expert
            elif self.mix_mode == "prob":
                rng = np.random.default_rng(seed)
                # 以mix_alpha为概率选用RDT整步动作，避免夹爪/关节出现在“半开半关”的无效插值
                mask = rng.random((expert.shape[0], 1)) < self.mix_alpha
                mixed = np.where(mask, teacher, expert)
            else:
                raise ValueError(f"不支持的mix_mode: {self.mix_mode}")
            # ReplayBuffer 不支持 __setitem__, 直接写 data
            final_action = mixed.astype(np.float32)
            self.replay_buffer.data['action'] = final_action
            # 保留 teacher 以便可视化/调试需要; 若想省内存可删除:
            # del self.replay_buffer['rdt_action']
        elif not use_expert_action:
            final_action = self.replay_buffer[action_key].astype(np.float32)
            self.replay_buffer.data['action'] = final_action
            del self.replay_buffer.data[action_key]
        else:
            # 只用专家动作
            expert = self.replay_buffer['action']
            if self.add_expert_noise:
                expert = _noisify(expert)
            final_action = expert.astype(np.float32)
            self.replay_buffer.data['action'] = final_action

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
