from collections.abc import Iterator, Sequence
import multiprocessing
import os
import random
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.augmentations as _augmentations
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    image_transforms = None
    if data_config.rgb_augmenter:
        image_transforms = _augmentations.build_image_augmenter(data_config.rgb_augmenter)

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        image_transforms=image_transforms,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _unwrap_lerobot_dataset(dataset: Dataset) -> lerobot_dataset.LeRobotDataset | None:
    """把 TransformedDataset 一层层拆开，拿到最底层的 LeRobotDataset（用于 fingerprint 校验等）。"""
    ds = dataset
    while hasattr(ds, "_dataset"):
        ds = getattr(ds, "_dataset")
    if isinstance(ds, lerobot_dataset.LeRobotDataset):
        return ds
    return None


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. "
                             "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if config.sampling_plan_path:
        # Plan-STaR：启用采样计划文件后
        # 1) 校验 plan 与当前训练数据/模型 horizon 一致
        # 2) 用 plan.anchor_indices 对数据集做 Subset
        # 3) 用 plan.weights 构造 WeightedRandomSampler（可选 replacement）
        from openpi.training.sampling_plan import SamplingPlan

        plan = SamplingPlan.load(config.sampling_plan_path, verify_sha256=True)
        meta = plan.meta
        if meta.get("repo_id") != data_config.repo_id:
            raise ValueError(
                f"Sampling plan repo_id mismatch: plan={meta.get('repo_id')} config={data_config.repo_id}")
        if int(meta.get("dataset_len", -1)) != len(dataset):
            raise ValueError(
                f"Sampling plan dataset_len mismatch: plan={meta.get('dataset_len')} dataset={len(dataset)}")
        if int(meta.get("action_horizon_H", -1)) != int(config.model.action_horizon):
            raise ValueError(
                f"Sampling plan action_horizon_H mismatch: plan={meta.get('action_horizon_H')} model={config.model.action_horizon}"
            )

        lr_ds = _unwrap_lerobot_dataset(dataset)
        if lr_ds is not None:
            fp = getattr(getattr(lr_ds, "hf_dataset", None), "_fingerprint", None)
            plan_fp = meta.get("dataset_fingerprint")
            if fp is not None and plan_fp and str(fp) != str(plan_fp):
                raise ValueError(f"Sampling plan dataset_fingerprint mismatch: plan={plan_fp} dataset={fp}")

        if np.any(plan.anchor_indices < 0) or np.any(plan.anchor_indices >= len(dataset)):
            raise ValueError("Sampling plan anchor_indices out of range for dataset.")

        # 训练数据只保留 anchors（锚点起点）；后续 DataLoader 看到的 index 是 [0..len(anchors)-1]
        dataset = torch.utils.data.Subset(typing.cast(torch.utils.data.Dataset, dataset), plan.anchor_indices.tolist())
        # replacement=True：有放回采样（少样本短训常用）
        # replacement=False：无放回采样（更像按 epoch 覆盖一遍 anchors）
        replacement = config.sampler_replacement_override if config.sampler_replacement_override is not None else bool(
            meta.get("replacement", True))
        # sampler_seed：控制 WeightedRandomSampler 的随机性（独立于 training seed，可用于复现实验）
        sampler_seed = config.sampler_seed_override if config.sampler_seed_override is not None else int(
            meta.get("sampler_seed", config.seed))
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(sampler_seed)
        weights = torch.as_tensor(plan.weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=replacement,
            generator=sampler_generator,
        )

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        # 若启用 sampler，则必须关闭 shuffle（PyTorch 要求 sampler 与 shuffle 互斥）
        shuffle=shuffle if sampler is None else False,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
        sampler=sampler,
    )

    class DataLoaderImpl(DataLoader):

        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        sampler: torch.utils.data.Sampler | None = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B", )),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    # 关键：同步设置 random/numpy/torch 的 worker seed，提升多 worker 场景下的可复现性
    seed = int(worker_info.seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
