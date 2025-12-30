import dataclasses
import functools
import json
import logging
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):

        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict({
        k: v
        for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)
    })


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1, ),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.freeze_mode == "default":
        # default：保持 config 里原本的 freeze_filter（baseline 不变）
        pass
    elif config.freeze_mode == "strong":
        # strong：在 baseline freeze_filter 基础上额外冻结视觉塔（vision tower：图像编码器）
        # 目的：少样本/短训时减少可训练参数自由度，降低过拟合与 seed 方差
        base = config.freeze_filter
        img = nnx_utils.PathRegex(".*PaliGemma/img.*")
        config.freeze_filter = nnx.Any(base, img)
    else:
        raise ValueError(f"Unknown freeze_mode: {config.freeze_mode}")

    if config.budget_aware_schedule:
        # 预算感知 schedule：当你把 30k steps 改成 6k/3k 时，
        # 如果直接沿用 warmup/decay 往往会导致“不收敛/震荡”，从而误判 few-shot 方法无效。
        # 这里按 total_steps 重标定 warmup_steps/decay_steps（不同 schedule 类型分别处理）。
        total_steps = int(config.num_train_steps)
        if isinstance(config.lr_schedule, _optimizer.CosineDecaySchedule):
            warmup = max(int(config.budget_min_warmup_steps), int(round(config.budget_warmup_ratio * total_steps)))
            warmup = min(warmup, max(1, total_steps // 2))
            config.lr_schedule = dataclasses.replace(
                config.lr_schedule,
                warmup_steps=warmup,
                decay_steps=total_steps,
            )
            logging.info(
                "Budget-aware schedule enabled: warmup_steps=%d decay_steps=%d peak_lr=%g decay_lr=%g",
                warmup,
                total_steps,
                config.lr_schedule.peak_lr,
                config.lr_schedule.decay_lr,
            )
        elif isinstance(config.lr_schedule, _optimizer.RsqrtDecaySchedule):
            warmup = max(int(config.budget_min_warmup_steps), int(round(config.budget_warmup_ratio * total_steps)))
            warmup = min(warmup, max(1, total_steps // 2))
            timescale = float(total_steps)
            config.lr_schedule = dataclasses.replace(config.lr_schedule, warmup_steps=warmup, timescale=timescale)
            logging.info(
                "Budget-aware schedule enabled: warmup_steps=%d timescale=%g peak_lr=%g",
                warmup,
                timescale,
                config.lr_schedule.peak_lr,
            )
        else:
            logging.info("Budget-aware schedule enabled but lr_schedule type is not supported: %s", type(config.lr_schedule))

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    sampling_plan_info: dict[str, Any] = {}
    if config.sampling_plan_path:
        from openpi.training.sampling_plan import SamplingPlan

        # 读取 plan（默认校验 sha256），并把 plan 文件复制到 checkpoint 下，便于复现审计
        plan = SamplingPlan.load(config.sampling_plan_path, verify_sha256=True)
        plan_meta = plan.meta
        plan_dir = config.checkpoint_dir / "sampling_plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        try:
            src = Path(config.sampling_plan_path)
            if src.is_dir():
                meta_src = src / "plan_meta.json"
                arrays_src = src / "plan_arrays.npz"
            elif src.suffix == ".json":
                meta_src = src
                arrays_src = src.with_name("plan_arrays.npz")
            elif src.suffix == ".npz":
                meta_src = src.with_name("plan_meta.json")
                arrays_src = src
            else:
                meta_src = src
                arrays_src = src.with_name("plan_arrays.npz")

            if meta_src.exists():
                shutil.copyfile(meta_src, plan_dir / "plan_meta.json")
            if arrays_src.exists():
                shutil.copyfile(arrays_src, plan_dir / "plan_arrays.npz")
        except Exception:
            pass
        sampling_plan_info = {
            "sampling_plan_path": str(config.sampling_plan_path),
            "plan_sha256": plan_meta.get("plan_sha256"),
            "plan_version": plan_meta.get("plan_version"),
            "plan_repo_id": plan_meta.get("repo_id"),
            "plan_dataset_fingerprint": plan_meta.get("dataset_fingerprint"),
            "plan_num_anchors": int(len(plan.anchor_indices)),
            "plan_replacement": bool(plan_meta.get("replacement", True)),
            "plan_sampler_seed": int(plan_meta.get("sampler_seed", config.seed)),
        }
        # === 有效样本暴露量（exposure）指标 ===
        # n：总抽样次数 = steps * batch_size
        # p_i：每个 anchor 的采样概率（按权重归一）
        # ESS（有效样本量）：1 / sum(p_i^2)，越大表示权重越不“尖”
        # expected_unique：期望看到的不同样本数（replacement=True 时按抽样模型估计）
        # expected_coverage：expected_unique / N
        # avg_repeat：平均每个被看到的样本重复次数（n / expected_unique）
        n = int(config.num_train_steps) * int(config.batch_size)
        w = plan.weights.astype("float64")
        p = w / w.sum()
        ess = float(1.0 / (p * p).sum())
        if sampling_plan_info["plan_replacement"]:
            expected_unique = float((1.0 - np.exp(n * np.log1p(-p))).sum())
        else:
            # replacement=False 时本实现使用 num_samples=N（每轮 epoch 覆盖一次），短训可用 min(N, n) 近似。
            expected_unique = float(min(len(p), n))
        expected_coverage = float(expected_unique / len(p))
        avg_repeat = float(n / max(1e-9, expected_unique))
        sampling_plan_info.update({
            "exposure_n": n,
            "exposure_ESS": ess,
            "exposure_expected_unique": expected_unique,
            "exposure_expected_coverage": expected_coverage,
            "exposure_avg_repeat": avg_repeat,
        })

        # === 可选：plan-aware 学习率缩放 ===
        # 当有效重复次数 repeat_eff = n / ESS 很大时，LoRA/PEFT 往往会“训坏/训爆”：
        # - 过拟合：训练样本重复太多，泛化到新 seed 失败
        # - 遗忘：把 base 模型原本的能力（尤其是视觉/语言对齐）训没了
        # 这里提供一个可控、可消融的缩放开关（默认关闭，不影响 baseline）。
        if config.plan_aware_lr_scale:
            repeat_eff = float(n / max(1e-9, ess))
            scale = float((config.plan_target_effective_repeat / max(1e-9, repeat_eff))**float(config.plan_lr_scale_power))
            scale = float(np.clip(scale, float(config.plan_lr_min_scale), 1.0))

            if isinstance(config.lr_schedule, _optimizer.CosineDecaySchedule):
                config.lr_schedule = dataclasses.replace(
                    config.lr_schedule,
                    peak_lr=float(config.lr_schedule.peak_lr) * scale,
                    decay_lr=float(config.lr_schedule.decay_lr) * scale,
                )
            elif isinstance(config.lr_schedule, _optimizer.RsqrtDecaySchedule):
                config.lr_schedule = dataclasses.replace(
                    config.lr_schedule,
                    peak_lr=float(config.lr_schedule.peak_lr) * scale,
                )
            else:
                logging.info("plan_aware_lr_scale enabled but lr_schedule type is not supported: %s", type(config.lr_schedule))

            sampling_plan_info.update({
                "plan_lr_scale_enabled": True,
                "plan_lr_scale": scale,
                "plan_effective_repeat": repeat_eff,
            })
            logging.info(
                "Plan-aware LR scaling: repeat_eff=%.2f target=%.2f power=%.3f scale=%.4f",
                repeat_eff,
                float(config.plan_target_effective_repeat),
                float(config.plan_lr_scale_power),
                scale,
            )
        (config.checkpoint_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "train_config_name": config.name,
                    "exp_name": config.exp_name,
                    "seed": config.seed,
                    "num_train_steps": int(config.num_train_steps),
                    "batch_size": int(config.batch_size),
                    "freeze_mode": config.freeze_mode,
                    "budget_aware_schedule": bool(config.budget_aware_schedule),
                    "plan_aware_lr_scale": bool(config.plan_aware_lr_scale),
                    "plan_target_effective_repeat": float(config.plan_target_effective_repeat),
                    "norm_stats_asset_id": getattr(getattr(config.data, "assets", None), "asset_id", None),
                    **sampling_plan_info,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        # baseline 仍然 shuffle=True；若启用 plan，data_loader 内部会自动关闭 shuffle 并启用 sampler
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1, ),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    start_time = time.time()
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if sampling_plan_info:
                reduced_info = dict(reduced_info)
                reduced_info.update({k: v for k, v in sampling_plan_info.items() if k.startswith("exposure_")})
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            if step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step + 1)
            else:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    elapsed = time.time() - start_time
    logging.info("Training finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main(_config.cli())
