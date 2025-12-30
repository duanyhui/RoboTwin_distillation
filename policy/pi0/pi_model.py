#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from pathlib import Path
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _resolve_checkpoint_id(train_config_name: str, model_name: str, checkpoint_id):
    """解析 checkpoint_id：

    - int/数字字符串：直接使用（例如 3000/6000）
    - -1 / "latest"：自动选择 checkpoints 目录下最大的数字子目录
    """
    if checkpoint_id is None:
        return None
    if isinstance(checkpoint_id, int) and checkpoint_id >= 0:
        return str(checkpoint_id)

    if isinstance(checkpoint_id, str) and checkpoint_id.isdigit():
        return checkpoint_id

    if (isinstance(checkpoint_id, int) and checkpoint_id == -1) or (
        isinstance(checkpoint_id, str) and checkpoint_id.lower() in {"latest", "max", "-1"}
    ):
        base_dir = Path(f"policy/pi0/checkpoints/{train_config_name}/{model_name}")
        candidates = []
        if base_dir.exists():
            for p in base_dir.iterdir():
                if p.is_dir() and p.name.isdigit():
                    candidates.append(int(p.name))
        if not candidates:
            raise FileNotFoundError(f"No numeric checkpoint dirs found under: {base_dir}")
        return str(max(candidates))

    return str(checkpoint_id)


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        # 支持 checkpoint_id=-1/latest，避免 deploy_policy.yml 固定 3000 导致评测错 ckpt
        self.checkpoint_id = _resolve_checkpoint_id(train_config_name, model_name, checkpoint_id)

        config = _config.get_config(self.train_config_name)
        # 加载训练出的 policy：目录结构为 policy/pi0/checkpoints/<train_config>/<exp_name>/<checkpoint_id>
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
