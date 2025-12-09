import collections
import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform, functional as F  # noqa: N812


class RandomMask(Transform):
    """在图像上随机应用矩形遮挡（cutout）。"""

    def __init__(self, mask_size=(0.2, 0.2)):
        super().__init__()
        self.mask_size = mask_size

    def make_params(self, flat_inputs):
        h, w = flat_inputs[0].shape[-2:]
        mask_h = int(self.mask_size[0] * h)
        mask_w = int(self.mask_size[1] * w)
        top = torch.randint(0, h - mask_h + 1, (1,)).item()
        left = torch.randint(0, w - mask_w + 1, (1,)).item()
        return {"top": top, "left": left, "mask_h": mask_h, "mask_w": mask_w}

    def transform(self, inpt, params):
        top, left, mask_h, mask_w = params.values()
        mask = torch.ones_like(inpt)
        mask[..., top:top + mask_h, left:left + mask_w] = 0
        return inpt * mask


class RandomBorderCutout(Transform):
    """随机裁掉图像的一侧边框（上、下、左、右）。"""

    def __init__(self, cut_ratio=0.1):
        super().__init__()
        self.cut_ratio = cut_ratio

    def make_params(self, flat_inputs):
        h, w = flat_inputs[0].shape[-2:]
        border = random.choice(["top", "bottom", "left", "right"])
        return {"h": h, "w": w, "border": border}

    def transform(self, inpt, params):
        h, w, border = params["h"], params["w"], params["border"]
        cut_h, cut_w = int(h * self.cut_ratio), int(w * self.cut_ratio)
        mask = torch.ones_like(inpt)
        if border == "top":
            mask[..., :cut_h, :] = 0
        elif border == "bottom":
            mask[..., -cut_h:, :] = 0
        elif border == "left":
            mask[..., :, :cut_w] = 0
        elif border == "right":
            mask[..., :, -cut_w:] = 0
        return inpt * mask


class GaussianNoise(Transform):
    """添加高斯噪声。"""

    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def make_params(self, flat_inputs):
        return {}

    def transform(self, inpt, params):
        noise = torch.randn_like(inpt) * self.std + self.mean
        return torch.clamp(inpt + noise, 0.0, 1.0)


class GammaCorrection(Transform):
    """应用伽马校正。"""

    def __init__(self, gamma=(0.8, 1.2)):
        super().__init__()
        self.gamma = gamma

    def make_params(self, flat_inputs):
        gamma_val = torch.empty(1).uniform_(self.gamma[0], self.gamma[1]).item()
        return {"gamma": gamma_val}

    def transform(self, inpt, params):
        return inpt ** params["gamma"]


class RandomSubsetApply(Transform):
    """从一组变换中无放回随机采样 N 个依次应用。

    参数:
        transforms: 变换列表。
        p: 多项式概率（无放回），权重会自动归一化，None 表示等概率。
        n_subset: 采样并应用的变换数量，None 表示全部，范围 [1, len(transforms)]。
        random_order: 是否打乱应用顺序（否则按索引升序）。"""

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}")

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """随机调整图像或视频的锐度。"""

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """单个变换的配置。"""

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """图像变换配置容器。"""

    enable: bool = False
    max_num_transforms: int = 3
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(default_factory=dict)


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    if cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    if cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    if cfg.type == "RandomRotation":
        return v2.RandomRotation(**cfg.kwargs)
    if cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    if cfg.type == "RandomPerspective":
        return v2.RandomPerspective(**cfg.kwargs)
    if cfg.type == "GaussianBlur":
        return v2.GaussianBlur(**cfg.kwargs)
    if cfg.type == "RandomMask":
        return RandomMask(**cfg.kwargs)
    if cfg.type == "RandomBorderCutout":
        return RandomBorderCutout(**cfg.kwargs)
    if cfg.type == "GaussianNoise":
        return GaussianNoise(**cfg.kwargs)
    if cfg.type == "GammaCorrection":
        return GammaCorrection(**cfg.kwargs)
    raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """根据配置组装图像变换。"""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)


_ACT_RGB_AUGMENTER = {
    "enable": True,
    "max_num_transforms": 1,
    "random_order": True,
    "tfs": {
        "notransform": {"weight": 2.0, "type": "Identity", "kwargs": {}},
        "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.5, 1.5]}},
        "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.5, 1.5]}},
        "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
        "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
        "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
        "random_mask": {"weight": 1.0, "type": "RandomMask", "kwargs": {"mask_size": [0.1, 0.1]}},
        "random_border_cutout": {"weight": 1.0, "type": "RandomBorderCutout", "kwargs": {"cut_ratio": 0.15}},
        "gaussian_noise": {"weight": 1.0, "type": "GaussianNoise", "kwargs": {"mean": 0.0, "std": 0.05}},
        "gamma_correction": {"weight": 1.0, "type": "GammaCorrection", "kwargs": {"gamma": [0.5, 2.0]}},
    },
}


def default_act_rgb_augmenter() -> dict[str, Any]:
    """返回 ACT 数据增广默认配置的深拷贝。"""
    return copy.deepcopy(_ACT_RGB_AUGMENTER)


def build_image_augmenter(cfg: dict[str, Any] | ImageTransformsConfig | None):
    """Build a torchvision v2 transform pipeline from config."""
    if cfg is None:
        return None

    if isinstance(cfg, ImageTransformsConfig):
        img_tf_cfg = cfg
    else:
        tfs = {}
        for name, tf_dict in cfg.get("tfs", {}).items():
            tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )

        max_num = cfg.get("max_num_transforms", None)
        if max_num is None:
            max_num = len(tfs) if len(tfs) > 0 else 0

        img_tf_cfg = ImageTransformsConfig(
            enable=cfg.get("enable", False),
            max_num_transforms=max_num,
            random_order=cfg.get("random_order", False),
            tfs=tfs,
        )

    return ImageTransforms(img_tf_cfg)
