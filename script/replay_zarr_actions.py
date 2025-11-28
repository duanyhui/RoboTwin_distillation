#!/usr/bin/env python3
"""
将 zarr 中的动作（expert 或 rdt_action）在仿真环境中回放，便于对比标签效果。

用法示例：
python script/replay_zarr_actions.py \
  --zarr_path policy/RDT/rdt_labels/lift_pot-demo_clean-50_with_rdt_mask_0.02.zarr \
  --action_key rdt_action \
  --episode 0 \
  --task_name lift_pot \
  --task_config demo_clean \
  --render_freq 0 \
  --video_dir ./replay_videos/rdt_ep0

如果想看专家轨迹，把 --action_key 改成 action；episode 按需调整。
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import yaml
import zarr

# 确保使用仓库本地路径（不依赖外部同名 envs 包）
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# 配置目录指向仓库根下的 task_config
CONFIGS_PATH = str((REPO_ROOT / "task_config").resolve()) + "/"


def class_decorator(task_name):
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_task_and_robot(task_name, task_config):
    # 任务配置
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config

    # 机器人/相机配置
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def build_ffmpeg(video_path, video_size):
    video_path.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path / "replay.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    return ffmpeg


def main():
    parser = argparse.ArgumentParser(description="在仿真中回放 zarr 里的动作")
    parser.add_argument("--zarr_path", type=str, required=True, help="包含动作的 zarr 路径")
    parser.add_argument("--action_key", type=str, default="rdt_action", choices=["action", "rdt_action"],
                        help="选择回放专家动作(action)还是 RDT 标签(rdt_action)")
    parser.add_argument("--episode", type=int, default=0, help="回放第几个 episode（从 0 开始）")
    parser.add_argument("--task_name", type=str, required=True, help="任务名，如 lift_pot")
    parser.add_argument("--task_config", type=str, required=True, help="任务配置，如 demo_clean")
    parser.add_argument("--render_freq", type=int, default=1, help="渲染频率，0 则不渲染")
    parser.add_argument("--video_dir", type=str, default=None, help="保存回放视频的目录，可选")
    parser.add_argument("--seed", type=int, default=0, help="环境随机种子")
    args_cli = parser.parse_args()

    # 载入动作
    z = zarr.open(args_cli.zarr_path, mode="r")
    if f"data/{args_cli.action_key}" not in z:
        raise ValueError(f"{args_cli.action_key} 不在 zarr 数据中")
    actions = np.array(z[f"data/{args_cli.action_key}"])
    episode_ends = np.array(z["meta/episode_ends"])
    if args_cli.episode >= len(episode_ends):
        raise ValueError(f"episode 超出范围，最多 {len(episode_ends)-1}")
    start = 0 if args_cli.episode == 0 else episode_ends[args_cli.episode - 1]
    end = episode_ends[args_cli.episode]
    traj = actions[start:end]
    print(f"回放 episode {args_cli.episode}: steps {start}-{end}, 共 {len(traj)} 步")

    # 环境参数
    args = load_task_and_robot(args_cli.task_name, args_cli.task_config)
    args["render_freq"] = args_cli.render_freq
    args["eval_mode"] = True
    if args_cli.video_dir is not None:
        args["eval_video_log"] = True
        args["eval_video_save_dir"] = Path(args_cli.video_dir)
    else:
        args["eval_video_log"] = False

    # 创建环境
    TASK_ENV = class_decorator(args["task_name"])
    TASK_ENV.setup_demo(now_ep_num=args_cli.episode, seed=args_cli.seed, is_test=True, **args)

    ffmpeg = None
    if args["eval_video_log"]:
        # 视频大小使用头摄配置
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
        with open(camera_config_path, "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_camera_type = args["camera"]["head_camera_type"]
        video_size = f"{_camera_config[head_camera_type]['w']}x{_camera_config[head_camera_type]['h']}"
        ffmpeg = build_ffmpeg(Path(args_cli.video_dir), video_size)
        TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

    # 先获取一次观测，确保 take_action 内的视频写入有可用的 now_obs
    _ = TASK_ENV.get_obs()

    # 回放
    for i, act in enumerate(traj, 1):
        # 每一步都先刷新观测（对应当前画面），再执行动作
        _ = TASK_ENV.get_obs()
        TASK_ENV.take_action(act)
        if TASK_ENV.eval_success:
            break

    if ffmpeg is not None:
        TASK_ENV._del_eval_video_ffmpeg()

    TASK_ENV.close_env(clear_cache=True)
    if TASK_ENV.render_freq and hasattr(TASK_ENV, "viewer") and TASK_ENV.viewer is not None:
        TASK_ENV.viewer.close()
    print("回放完成")


if __name__ == "__main__":
    main()
