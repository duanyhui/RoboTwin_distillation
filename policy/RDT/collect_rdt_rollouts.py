#!/usr/bin/env python3
"""
在仿真环境中用 RDT 策略跑若干成功 episode，并保存成 DP 可直接使用的 zarr 数据集。

特性：
- 使用与 deploy 时一致的 RDT 推理流程（两帧窗口 + chunk 连续执行），避免逐帧重规划带来的抖动。
- 只保留成功的 episode，支持设置最大尝试次数。
- zarr 字段：head_camera(NCHW)、state(当前关节向量)、action(执行后下一时刻的关节向量)、rdt_action(RDT 给出的动作指令)、meta/episode_ends。

用法示例（在仓库根目录下）：
python policy/RDT/collect_rdt_rollouts.py \
  --task_name lift_pot \
  --task_config demo_clean \
  --ckpt_setting RDT_lift_pot_demo_clean_1b_pretrain \
  --checkpoint_id 20000 \
  --num_episodes 50 \
  --output_path policy/RDT/rdt_labels/lift_pot-demo_clean-rdt50.zarr \
  --instruction_type unseen \
  --seed 0 \
  --max_trials 200 \
  --render_freq 0
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess

import numpy as np
import yaml
import zarr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from description.utils.generate_episode_instructions import generate_episode_descriptions

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
        embodiment_args = yaml.safe_load(f)
    return embodiment_args


def load_task_and_robot(task_name, task_config):
    """加载任务/机器人配置，返回 args（与 eval_policy 一致）。"""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)

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


def encode_obs(observation):
    """与 deploy_policy 中一致的观测后处理。"""
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


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


def collect_one_episode(task_env, model, instruction, render_freq):
    """
    在已 setup_demo 的环境上跑一条 RDT 轨迹，返回是否成功及收集到的样本。
    样本对齐 DP 数据：state_t / head_t 对应当前观测，action_t 为执行动作后得到的下一时刻关节向量。
    额外保存 rdt_action 为 RDT 输出的动作指令。
    """
    head_list, state_list, action_list, rdt_action_list = [], [], [], []

    # 初始观测
    obs = task_env.get_obs()
    obs_enc = encode_obs(obs)
    # 设置指令
    model.reset_obsrvationwindows()
    model.set_language_instruction(instruction)

    # 将首帧塞入观察窗口
    img_arr = [
        obs_enc["observation"]["head_camera"]["rgb"],
        obs_enc["observation"].get("right_camera", obs_enc["observation"]["head_camera"])["rgb"],
        obs_enc["observation"].get("left_camera", obs_enc["observation"]["head_camera"])["rgb"],
    ]
    model.update_observation_window(img_arr, obs_enc["agent_pos"])

    chunk_actions = None
    chunk_idx = 0
    success = False

    while task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success:
        if chunk_actions is None or chunk_idx >= len(chunk_actions):
            # 生成新的 chunk
            chunk_actions = model.get_action()[: model.rdt_step, :]
            chunk_idx = 0
        action_cmd = chunk_actions[chunk_idx]
        chunk_idx += 1

        # 记录当前观测与动作（标签为执行后的下一时刻 qpos）
        head_list.append(obs_enc["observation"]["head_camera"]["rgb"])
        state_list.append(np.array(obs_enc["agent_pos"], dtype=np.float32))
        rdt_action_list.append(np.array(action_cmd, dtype=np.float32))

        # 执行动作
        task_env.take_action(action_cmd)
        # 动作后获取新观测
        obs_next = task_env.get_obs()
        obs_next_enc = encode_obs(obs_next)
        next_state = np.array(obs_next_enc["agent_pos"], dtype=np.float32)
        action_list.append(next_state)

        # 更新观察窗口，准备下一个 chunk
        img_arr = [
            obs_next_enc["observation"]["head_camera"]["rgb"],
            obs_next_enc["observation"].get("right_camera", obs_next_enc["observation"]["head_camera"])["rgb"],
            obs_next_enc["observation"].get("left_camera", obs_next_enc["observation"]["head_camera"])["rgb"],
        ]
        model.update_observation_window(img_arr, next_state)
        obs_enc = obs_next_enc

        if task_env.eval_success:
            success = True
            break

    return success, head_list, state_list, action_list, rdt_action_list


def save_zarr(output_path, heads, states, actions, rdt_actions, episode_ends, save_action=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    heads = np.array(heads, dtype=np.uint8)
    # HWC -> NCHW
    heads = np.transpose(heads, (0, 3, 1, 2))
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rdt_actions = np.array(rdt_actions, dtype=np.float32)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    z_root = zarr.group(str(output_path))
    z_data = z_root.create_group("data")
    z_meta = z_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    z_data.create_dataset(
        "head_camera",
        data=heads,
        chunks=(100, *heads.shape[1:]),
        dtype="uint8",
        overwrite=True,
        compressor=compressor,
    )
    z_data.create_dataset(
        "state",
        data=states,
        chunks=(100, states.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    if save_action:
        z_data.create_dataset(
            "action",  # 下一时刻的 qpos（与 DP 原始 expert 数据对齐）
            data=actions,
            chunks=(100, actions.shape[1]),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
    z_data.create_dataset(
        "rdt_action",  # RDT 输出的控制指令（通常与下一时刻 qpos 接近，可作监督标签备选）
        data=rdt_actions,
        chunks=(100, rdt_actions.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    z_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="用 RDT 评估生成成功轨迹，保存为 zarr")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--ckpt_setting", type=str, required=True, help="RDT 模型目录名（checkpoints 下）")
    parser.add_argument("--checkpoint_id", type=str, required=True, help="checkpoint-XXXX 中的数字")
    parser.add_argument("--num_episodes", type=int, default=50, help="需要收集的成功 episode 数")
    parser.add_argument("--max_trials", type=int, default=200, help="最多尝试多少个 seed（成功/失败都算一次尝试）")
    parser.add_argument("--seed", type=int, default=0, help="起始随机 seed（每次尝试自增）")
    parser.add_argument("--instruction_type", type=str, default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--render_freq", type=int, default=0, help="渲染频率，0 表示不渲染")
    parser.add_argument("--output_path", type=str, default=None, help="输出 zarr 路径，未填则自动生成")
    parser.add_argument("--only_rdt_action", action="store_true", help="仅保存 rdt_action，不保存 action(next_state)")
    return parser.parse_args()


def main():
    args_cli = parse_args()
    task_name = args_cli.task_name
    task_config = args_cli.task_config
    ckpt_setting = args_cli.ckpt_setting
    checkpoint_id = args_cli.checkpoint_id

    args_env = load_task_and_robot(task_name, task_config)
    # eval 模式下限制 step 数
    args_env["eval_mode"] = True
    args_env["render_freq"] = args_cli.render_freq
    # 明确关闭视频（避免缺少 eval_video_save_dir 报错）；如需录像可自行改成 True 并指定路径
    args_env["eval_video_log"] = False

    # arm 维度由机器人 config 推断
    left_arm_dim = len(args_env["left_embodiment_config"]["arm_joints_name"][0])
    right_arm_dim = len(args_env["right_embodiment_config"]["arm_joints_name"][1])

    # 加载 RDT 模型
    from policy.RDT.deploy_policy import get_model, reset_model

    rdt_args = {
        "ckpt_setting": ckpt_setting,
        "checkpoint_id": checkpoint_id,
        "task_name": task_name,
        "left_arm_dim": left_arm_dim,
        "right_arm_dim": right_arm_dim,
        "rdt_step": 64,  # 默认 chunk 大小，与模型一致
    }
    model = get_model(rdt_args)

    # 默认输出路径
    if args_cli.output_path is None:
        out_name = f"{task_name}-{task_config}-rdt{args_cli.num_episodes}.zarr"
        output_path = REPO_ROOT / "policy" / "RDT" / "rdt_labels" / out_name
    else:
        output_path = Path(args_cli.output_path)

    TASK_ENV = class_decorator(task_name)

    success_cnt = 0
    trial_cnt = 0
    now_seed = args_cli.seed
    episode_ends = []
    heads_all, states_all, actions_all, rdt_actions_all = [], [], [], []

    print(f"任务: {task_name}, 配置: {task_config}, 目标成功数: {args_cli.num_episodes}, 最多尝试: {args_cli.max_trials}")

    with tqdm(total=args_cli.max_trials, desc=f"成功 {success_cnt}/{args_cli.num_episodes}", dynamic_ncols=True) as pbar:
        while success_cnt < args_cli.num_episodes and trial_cnt < args_cli.max_trials:
            trial_cnt += 1
            pbar.set_postfix(seed=now_seed)
            print(f"\n===== 尝试 {trial_cnt} / {args_cli.max_trials} (seed={now_seed}) =====")

            # 先跑专家规划检查场景是否有效
            try:
                TASK_ENV.setup_demo(now_ep_num=trial_cnt, seed=now_seed, is_test=True, **args_env)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e:
                print(f"环境初始化失败，跳过此 seed，原因: {e}")
                TASK_ENV.close_env()
                now_seed += 1
                pbar.update(1)
                continue

            if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
                print("专家规划失败，换下一个 seed")
                now_seed += 1
                pbar.update(1)
                continue

            # 构造指令
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(task_name, episode_info_list, max_descriptions=1000)
            instr_pool = results[0].get(args_cli.instruction_type, []) if results else []
            if len(instr_pool) == 0:
                instruction = task_name
            else:
                instruction = np.random.choice(instr_pool)

            # 重新搭环境跑策略
            TASK_ENV.setup_demo(now_ep_num=trial_cnt, seed=now_seed, is_test=True, **args_env)
            TASK_ENV.set_instruction(instruction=instruction)

            # 可选视频记录：沿用 eval 的配置（仅当 eval_video_log=true）
            if args_env.get("eval_video_log", False):
                camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
                with open(camera_config_path, "r", encoding="utf-8") as f:
                    _camera_config = yaml.safe_load(f)
                head_camera_type = args_env["camera"]["head_camera_type"]
                video_size = f"{_camera_config[head_camera_type]['w']}x{_camera_config[head_camera_type]['h']}"
                video_dir = args_env.get("eval_video_save_dir", REPO_ROOT / "replay_videos" / "rdt_collect")
                ffmpeg = build_ffmpeg(Path(video_dir), video_size)
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            reset_model(model)
            success, heads, states, actions, rdt_actions = collect_one_episode(
                TASK_ENV, model, instruction, args_cli.render_freq
            )

            if args_env.get("eval_video_log", False):
                TASK_ENV._del_eval_video_ffmpeg()

            TASK_ENV.close_env(clear_cache=((success_cnt + 1) % args_env.get("clear_cache_freq", 50) == 0))
            if TASK_ENV.render_freq and hasattr(TASK_ENV, "viewer") and TASK_ENV.viewer is not None:
                TASK_ENV.viewer.close()

            if success:
                success_cnt += 1
                heads_all.extend(heads)
                states_all.extend(states)
                actions_all.extend(actions)
                rdt_actions_all.extend(rdt_actions)
                episode_ends.append(len(actions_all))
                pbar.set_description(f"成功 {success_cnt}/{args_cli.num_episodes}")
                print(f"✅ 成功 {success_cnt}/{args_cli.num_episodes} ，当前总步数: {len(actions_all)}")
            else:
                print("❌ 失败，丢弃该 episode")

            now_seed += 1
            pbar.update(1)

    if success_cnt == 0:
        raise RuntimeError("未收集到成功轨迹，无法保存。")

    save_zarr(
        output_path,
        heads_all,
        states_all,
        actions_all,
        rdt_actions_all,
        episode_ends,
        save_action=not args_cli.only_rdt_action,
    )
    print("\n保存完成:")
    print(f"  - 输出: {output_path}")
    print(f"  - 成功 episode 数: {success_cnt}")
    print(f"  - 总样本数: {len(actions_all)}")
    if args_cli.only_rdt_action:
        print("字段: data/head_camera (NCHW uint8), data/state, data/rdt_action, meta/episode_ends")
    else:
        print("字段: data/head_camera (NCHW uint8), data/state, data/action(下一步 qpos), data/rdt_action, meta/episode_ends")


if __name__ == "__main__":
    main()
