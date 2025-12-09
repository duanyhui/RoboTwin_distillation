import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_kuavo_example() -> dict:
    """Creates a random input example for the Kuavo policy."""
    return {
        "state": np.ones((16, )),
        "images": {
            "head_cam_h": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_cam_l": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_cam_r": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "Pick and Place",
    }


@dataclasses.dataclass(frozen=True)
class KuavoInputs(transforms.DataTransformFn):
    """Inputs for the Kuavo policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [16]
    - actions: [action_horizon, 16]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # The expected cameras names.
    cameras: tuple[str, ...] = (
        "head_cam_h",
        "wrist_cam_l",
        "wrist_cam_r",
    )

    def __call__(self, data: dict) -> dict:
        # Decode images (convert to HWC and uint8 if needed)
        data = _decode_kuavo(data, self.cameras)

        # Get the state. We are padding from 20 to the model action dim (32).
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        
        # Map Kuavo camera names to pi0 expected keys
        # head_cam_h -> base_0_rgb
        # wrist_cam_l -> left_wrist_0_rgb
        # wrist_cam_r -> right_wrist_0_rgb
        
        base_image = in_images.get("head_cam_h")
        if base_image is None:
             # Fallback or error handling if head_cam_h is missing
             # For consistency with Aloha, we might want to raise an error or handle it gracefully
             if "head_cam_h" in self.cameras:
                 raise ValueError(f"Expected images to contain head_cam_h, got {tuple(in_images)}")

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_map = {
            "left_wrist_0_rgb": "wrist_cam_l",
            "right_wrist_0_rgb": "wrist_cam_r",
        }
        
        for dest, source in extra_image_map.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KuavoOutputs(transforms.DataTransformFn):
    """Outputs for the Kuavo policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": actions}


def _decode_kuavo(data: dict, expected_cameras: tuple[str, ...]) -> dict:
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # 如果是浮点类型的图像，就先缩放到 0–255 再转成 uint8。
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # 把图像从 [channel, height, width] 转成 [height, width, channel]。
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {
        name: convert_image(img)
        for name, img in images.items()
        if name in expected_cameras
    }

    data["images"] = images_dict
    data["state"] = state
    return data
