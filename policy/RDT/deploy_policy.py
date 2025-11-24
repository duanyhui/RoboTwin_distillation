# import packages and module here
import sys, os
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def get_model(usr_args):  # keep
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    ckpt_dir = os.path.join(parent_directory, f"checkpoints/{model_name}/checkpoint-{checkpoint_id}")
    ds_checkpoint = os.path.join(ckpt_dir, "pytorch_model", "mp_rank_00_model_states.pt")
    ema_checkpoint = os.path.join(ckpt_dir, "ema", "model.safetensors")

    if os.path.isfile(ds_checkpoint):
        pretrained_path = ds_checkpoint
    elif any(os.path.isfile(os.path.join(ckpt_dir, fname)) for fname in ("model.safetensors", "pytorch_model.bin")):
        pretrained_path = ckpt_dir  # HuggingFace style checkpoint folder
    elif os.path.isfile(ema_checkpoint):
        pretrained_path = ema_checkpoint
    else:
        raise FileNotFoundError(f"Could not find usable RDT weights under {ckpt_dir}. Please verify the checkpoint.")

    rdt = RDT(pretrained_path, usr_args["task_name"], left_arm_dim, right_arm_dim, rdt_step)
    return rdt


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    
    # Generate blank images for wrist cameras (same shape as head_camera)
    # head_img = obs["observation"]["head_camera"]["rgb"]
    #噪声图像
    # noise_img_1 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
    # noise_img_2 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
    # noise_img_3 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
    
    # blank_img_1 = np.full(head_img.shape, 255, dtype=np.uint8)  # White image
    # blank_img_2 = np.full(head_img.shape, 255, dtype=np.uint8)  # White image
    
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],  # Keep head camera
        obs["observation"]["right_camera"]["rgb"],  # Replace right wrist with blank image
        obs["observation"]["left_camera"]["rgb"],  # Replace left wrist with
        # noise_img_1,  # Replace right wrist with noise image
        # noise_img_2,  # Replace left wrist with noise image
        # noise_img_3,  # Replace left wrist with noise image
    ], obs["agent_pos"]  # TODO
    # cv2.imwrite("debug_head_camera.png", head_img)
    # cv2.imwrite("debug_noise_camera_1.png", noise_img_1)
    # cv2.imwrite("debug_noise_camera_2.png", noise_img_2)
    # print("Saved debug images for head and wrist cameras.")

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.rdt_step, :]  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        
        # # Generate blank images for wrist cameras (same shape as head_camera)
        # head_img = obs["observation"]["head_camera"]["rgb"]
        # blank_img_1 = np.full(head_img.shape, 255, dtype=np.uint8)  # White image
        # blank_img_2 = np.full(head_img.shape, 255, dtype=np.uint8)  # White image
        # noise_img_1 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
        # noise_img_2 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
        # noise_img_3 = np.random.randint(0, 256, head_img.shape, dtype=np.uint8)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        # cv2.imwrite("debug_head_camera.png", head_img)
        # cv2.imwrite("debug_noise_camera_1.png", noise_img_1)
        # cv2.imwrite("debug_noise_camera_2.png", noise_img_2)
        # print("Saved debug images for head and wrist cameras.")   
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
