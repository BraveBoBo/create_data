### File: create_data/collect_metadata.py
# ✓ Successfully rendered corner (camera_id=0): shape (480, 480, 3)
# ✓ Successfully rendered topview (camera_id=1): shape (480, 480, 3)
# ✓ Successfully rendered behindGripper (camera_id=2): shape (480, 480, 3)
# ✓ Successfully rendered gripperPOV (camera_id=3): shape (480, 480, 3)
# ✓ Successfully rendered corner2 (camera_id=4): shape (480, 480, 3)
# ✓ Successfully rendered corner3 (camera_id=5): shape (480, 480, 3)
# ✓ Successfully rendered corner4 (camera_id=6): shape (480, 480, 3)

import os
import numpy as np
import pickle
import cv2
import json

from sympy import true

# Set random seeds for reproducibility
np.random.seed(42)

from Metaworld.metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    MT50_V3,
)
from Metaworld.metaworld.policies import ENV_POLICY_MAP
import gymnasium as gym
from typing import Union

os.environ['MUJOCO_GL'] = 'egl'
debug = False
RANDOM_DIS = True
RENDER_MODE = "rgb_array"
CAMERA_NAMES = ["corner", "topview", "behindGripper", "gripperPOV", "corner2", "corner3", "corner4"]
CAMERA_FIP = False
SEED = 42
EPISODES_NUMBER = 100
MAX_EPISODE_STEPS = 500

SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3"
DEBUG_DIR = "/home/libo/project/cm/trunck-consistency-policy/create_data/datademo"
os.makedirs(DEBUG_DIR, exist_ok=True)


def store_rendered_image(images, img):
    if CAMERA_FIP:
        img = img[::-1]
    images.append(img)
    return images


def get_multiview_images(env, camera_names):
    """Get images from multiple camera angles using camera_id manipulation"""
    multiview_images = {}
    
    for i, camera_name in enumerate(camera_names):
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'mujoco_renderer'):
            renderer = env.unwrapped.mujoco_renderer
            if hasattr(renderer, 'camera_id'):
                original_camera_id = renderer.camera_id
                renderer.camera_id = i
                img = env.render()
                renderer.camera_id = original_camera_id
                
                if CAMERA_FIP:
                    img = img[::-1]
                multiview_images[camera_name] = img

    return multiview_images


def get_video_writer(path, frame_shape):
    VIDEO_FPS = 30  # frames per second for debug video
    """Return an OpenCV VideoWriter given output path and first frame shape."""
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, VIDEO_FPS, (width, height))


def get_task_discriptions():
    """Returns a dictionary mapping environment names to their task descriptions."""
    with open(
        "/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json",
        "r",
    ) as f:
        task_descriptions = json.load(f)
    return task_descriptions


def get_task_name_and_desc(task_list) -> dict:
    def rewrite_task_name(task_name: str) -> str:
        return task_name.replace(" ", "_").lower() + "-v3"

    result = {}
    for item in task_list:
        # env_name = rewrite_task_name(item["env"])
        env_name = item["env"]
        assert env_name in MT50_V3, f"Task {env_name} not found in MT50_V3"
        result[env_name] = item["description"]  # take the first description string
    # print("Task name and descriptions:", result)
    return result


def get_task_expert_policy(env_names):
    """
    Returns a dictionary mapping environment names to their expert policy class names.
    """
    policy_dict = {}
    for env_name in env_names:
        base = "Sawyer"
        res = env_name.split("-")
        for substr in res:
            base += substr.capitalize()
        policy_name = base + "Policy"
        if policy_name == "SawyerPegInsertSideV3Policy":
            policy_name = "SawyerPegInsertionSideV3Policy"
        policy_dict[env_name] = policy_name
    return policy_dict


def _get_task_names(
    envs: Union[gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv],
) -> list[str]:
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


task_ = get_task_discriptions()
task_env_dis = get_task_name_and_desc(task_)


def collect_data(task_env_dis, save_path_root=SAVE_ROOT):
    """
    Collects data for each environment in the task_env_dis dictionary.
    Saves the data to the specified save_path.
    """
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    for env_name, task_descrps in task_env_dis.items():
        if RANDOM_DIS:
            task_description = task_descrps[np.random.randint(len(task_descrps))]
        # env_name = 'push-v3' # debug

        env = gym.make('Meta-World/MT1', env_name=env_name, render_mode=RENDER_MODE, camera_name=CAMERA_NAMES[0])

        policy = ENV_POLICY_MAP[env_name]()
        print(f"Environment: {env_name}")

        task_path = os.path.join(save_path_root, env_name)
        if not os.path.exists(task_path):
            os.makedirs(task_path)

        suc_episode = 0

        while(suc_episode < EPISODES_NUMBER):

            done = False
            step = 0  # step counter for this episode

            obss = []
            acts = []
            rews = []
            dones = []
            images = []
            multiview_images = []  # Store multi-view images

            # Initialize video writers for each camera angle
            if debug:
                video_writers = {}
                for camera_name in CAMERA_NAMES:
                    video_path = os.path.join(DEBUG_DIR, f"{env_name}_episode_{suc_episode:04d}_{camera_name}.mp4")
                    video_writers[camera_name] = None  # lazy init after first frame

            obs, info = env.reset(seed=SEED)

            while not done and step < MAX_EPISODE_STEPS:

                obss.append(obs.copy())
                
                # Get multi-view images
                multiview_imgs = get_multiview_images(env, CAMERA_NAMES)
                multiview_images.append(multiview_imgs)
                
                # Use first camera view for main image storage (backward compatibility)
                main_img = list(multiview_imgs.values())[0] if multiview_imgs else env.render()
                images = store_rendered_image(images, main_img)

                # Write debug video frames for each camera
                if debug:
                    for camera_name, img in multiview_imgs.items():
                        if video_writers[camera_name] is None:
                            video_path = os.path.join(DEBUG_DIR, f"{env_name}_episode_{suc_episode:04d}_{camera_name}.mp4")
                            video_writers[camera_name] = get_video_writer(video_path, img.shape)
                        video_writers[camera_name].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                a = policy.get_action(obs)
                acts.append(a.copy())

                obs, rew, _, _, info = env.step(a)

                rews.append(rew)

                done = int(info["success"]) == 1
                dones.append(done)
                step += 1

                if done:
                    obss.append(obs.copy())
                    # Get final multi-view images
                    multiview_imgs = get_multiview_images(env, CAMERA_NAMES)
                    multiview_images.append(multiview_imgs)
                    
                    # Use first camera view for main image storage
                    main_img = list(multiview_imgs.values())[0] if multiview_imgs else env.render()
                    images = store_rendered_image(images, main_img)
                    
                    # Write final frames to videos
                    if debug:
                        for camera_name, img in multiview_imgs.items():
                            if video_writers[camera_name] is not None:
                                video_writers[camera_name].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                    suc_episode += 1

            # Convert episode buffers to numpy arrays
            obss = np.array(obss)
            acts = np.array(acts)
            rews = np.array(rews)
            images = np.array(images)
            dones = np.array(dones)

            # Package episode dict in Diffusion‑Policy style and save as pkl
            episode_data = {
                "observations": obss,
                "actions": acts,
                "rewards": rews,
                "dones": dones,
                "images": images,
                "multiview_images": multiview_images,  # Add multi-view images
                "description": task_description,
            }
            episode_file = os.path.join(task_path, f"episode_{suc_episode:04d}.pkl")
            with open(episode_file, "wb") as pf:
                pickle.dump(episode_data, pf)

            # Close all video writers
            if debug:
                for camera_name, writer in video_writers.items():
                    if writer is not None and writer.isOpened():
                        writer.release()

        env.close()

if __name__ == "__main__":
    collect_data(task_env_dis, save_path_root=SAVE_ROOT)
