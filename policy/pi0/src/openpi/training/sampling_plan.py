from __future__ import annotations

"""
Plan-STaR 的采样计划文件（Sampling Plan）。

核心目标：把“训练时看哪些样本、每个样本的权重、阶段标签、随机种子”等信息，
固化成一个可审计、可复现的静态协议（plan），训练阶段只读 plan 来构造 sampler，
从而保持单阶段训练流程不变（不引入 curriculum：多阶段接力训练）。

文件格式：
- plan_meta.json：人类可读的元信息（repo_id、episodes_keep、超参、sha256 等）
- plan_arrays.npz：数组（anchor_indices / stage_ids / weights）

完整性校验：
- plan_sha256 = sha256( canonical_json(meta_without_plan_sha256) + arrays_bytes )
  训练与校验脚本可通过 sha256 确认“同一个 plan”。
"""

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PLAN_META_FILENAME = "plan_meta.json"
DEFAULT_PLAN_ARRAYS_FILENAME = "plan_arrays.npz"


def _canonical_json_bytes(obj: Any) -> bytes:
    """生成可复现的 JSON bytes（键排序 + 紧凑分隔符）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _resolve_plan_paths(path: str | Path) -> tuple[Path, Path]:
    """支持三种输入：目录 / plan_meta.json / plan_arrays.npz，并解析成(meta_path, arrays_path)。"""
    path = Path(path)
    if path.is_dir():
        return path / DEFAULT_PLAN_META_FILENAME, path / DEFAULT_PLAN_ARRAYS_FILENAME
    if path.suffix == ".json":
        return path, path.with_name(DEFAULT_PLAN_ARRAYS_FILENAME)
    if path.suffix == ".npz":
        return path.with_name(DEFAULT_PLAN_META_FILENAME), path
    raise ValueError(
        f"Unsupported plan path: {path}. Provide a directory, {DEFAULT_PLAN_META_FILENAME}, or {DEFAULT_PLAN_ARRAYS_FILENAME}."
    )


def compute_plan_sha256(*, meta: dict[str, Any], arrays_bytes: bytes) -> str:
    """计算 plan 的内容哈希；meta 内的 plan_sha256 字段不参与哈希。"""
    meta_for_hash = dict(meta)
    meta_for_hash.pop("plan_sha256", None)
    h = hashlib.sha256()
    h.update(_canonical_json_bytes(meta_for_hash))
    h.update(arrays_bytes)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class SamplingPlan:
    """采样计划：anchors/stage_ids/weights + meta。

    - anchor_indices：在原始 LeRobotDataset（按 repo_id 对应的 dataset）中的全局索引。
    - stage_ids：每个 anchor 对应的阶段标签（用于阶段均衡/统计/消融）。
    - weights：每个 anchor 的采样权重（用于 WeightedRandomSampler）。
    """
    meta: dict[str, Any]
    anchor_indices: np.ndarray
    stage_ids: np.ndarray
    weights: np.ndarray

    @property
    def plan_sha256(self) -> str | None:
        value = self.meta.get("plan_sha256")
        return str(value) if value is not None else None

    def validate(self) -> None:
        """基础一致性检查：形状、dtype、非负、非 NaN 等。"""
        if self.anchor_indices.ndim != 1:
            raise ValueError("anchor_indices must be 1D.")
        if self.stage_ids.ndim != 1:
            raise ValueError("stage_ids must be 1D.")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D.")
        n = len(self.anchor_indices)
        if len(self.stage_ids) != n or len(self.weights) != n:
            raise ValueError(f"Plan arrays length mismatch: anchors={n}, stages={len(self.stage_ids)}, weights={len(self.weights)}")
        if n == 0:
            raise ValueError("Plan has zero anchors.")
        if not np.issubdtype(self.anchor_indices.dtype, np.integer):
            raise ValueError("anchor_indices must be integer dtype.")
        if not np.issubdtype(self.stage_ids.dtype, np.integer):
            raise ValueError("stage_ids must be integer dtype.")
        if not np.issubdtype(self.weights.dtype, np.floating):
            raise ValueError("weights must be floating dtype.")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain non-finite values.")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")

    def _arrays_bytes(self) -> bytes:
        """序列化 arrays 为 npz bytes（用于哈希与 save）。"""
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            anchor_indices=self.anchor_indices.astype(np.int64, copy=False),
            stage_ids=self.stage_ids.astype(np.int32, copy=False),
            weights=self.weights.astype(np.float32, copy=False),
        )
        return buf.getvalue()

    def recompute_sha256(self) -> str:
        return compute_plan_sha256(meta=self.meta, arrays_bytes=self._arrays_bytes())

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        """写出 plan_meta.json 与 plan_arrays.npz，并自动写入 plan_sha256。"""
        self.validate()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / DEFAULT_PLAN_META_FILENAME
        arrays_path = out_dir / DEFAULT_PLAN_ARRAYS_FILENAME

        arrays_bytes = self._arrays_bytes()
        meta = dict(self.meta)
        meta["plan_sha256"] = compute_plan_sha256(meta=meta, arrays_bytes=arrays_bytes)

        arrays_path.write_bytes(arrays_bytes)
        meta_path.write_text(_canonical_json_bytes(meta).decode("utf-8"), encoding="utf-8")
        return meta_path, arrays_path

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: bool = True) -> "SamplingPlan":
        """读取 plan，并可选校验 sha256（默认开启，建议保持开启）。"""
        meta_path, arrays_path = _resolve_plan_paths(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Plan meta not found: {meta_path}")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Plan arrays not found: {arrays_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays_bytes = arrays_path.read_bytes()
        npz = np.load(io.BytesIO(arrays_bytes))
        plan = cls(
            meta=meta,
            anchor_indices=np.asarray(npz["anchor_indices"]),
            stage_ids=np.asarray(npz["stage_ids"]),
            weights=np.asarray(npz["weights"]),
        )
        plan.validate()

        if verify_sha256:
            expected = plan.plan_sha256
            computed = compute_plan_sha256(meta=meta, arrays_bytes=arrays_bytes)
            if expected is None:
                raise ValueError(f"Plan meta missing plan_sha256: {meta_path}")
            if computed != expected:
                raise ValueError(f"Plan sha256 mismatch: expected={expected}, computed={computed}")

        return plan
