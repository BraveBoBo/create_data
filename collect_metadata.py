# -*- coding: utf-8 -*-
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco as mj
import numpy as np
import zarr
from Metaworld.metaworld.env_dict import MT50_V3
from Metaworld.metaworld.policies import ENV_POLICY_MAP
import warnings
warnings.filterwarnings(
    "ignore", message=".*Overriding environment.*already in registry.*"
)
warnings.filterwarnings("ignore", message=".*Constant\\(s\\) may be too high.*")


# ==============================================================================
# Global Configuration
# ==============================================================================
os.environ["MUJOCO_GL"] = "egl"  # For headless rendering
np.random.seed(42)

DEBUG = False  # If True, suppresses video writing
RANDOM_DIS = True  # If True, uses random task descriptions
RENDER_MODE = "rgb_array"
CAMERA_NAMES = ["corner4", "gripperPOV"]
# CAMERA_NAMES = ["corner", "corner2", "corner3", "corner4", "gripperPOV", "topview", "behindGripper"]
CAMERA_FIP = True  # If True, flips images vertically
SEED = 42
EPISODES_NUMBER = 100
MAX_EPISODE_STEPS = 500

SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3_zarr"
# SAVE_ROOT = "/data/robot_dataset/metaworld/debug/metaworld/mt50_v3_zarr"
os.makedirs(SAVE_ROOT, exist_ok=True)

# New camera mapping using explicit IDs per environment
CAMERA_ID_MAP_PER_ENV: Dict[str, Dict[str, int]] = {
    # Per-task override example (fill with your real ids if they differ)
    # "pick-place-v3": {"topview": 2, "gripperPOV": 5},
    # "door-close-v3": {"topview": 0, "gripperPOV": 3},
    # Fallback default mapping (used if no per-env override is present)
    "__default__": {"topview": 0, "gripperPOV": 6, "corner": 1, "corner2": 2, "corner3": 3, "corner4": 4, "behindGripper": 5}
}


# ==============================================================================
# Utility Functions
# ==============================================================================
def get_task_discriptions() -> dict:
    """Reads the task description file."""
    with open(
        "/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json",
        "r",
    ) as f:
        task_descriptions = json.load(f)
    return task_descriptions


def get_task_name_and_desc(task_list: list) -> dict:
    """Maps environment names to a list of descriptions."""
    result = {}
    for item in task_list:
        env_name = item["env"]
        assert env_name in MT50_V3, f"Task {env_name} not found in MT50_V3"
        result[env_name] = item["description"]
    return result


def sanitize_obs(obs: np.ndarray, space: gym.Space) -> np.ndarray:
    """Sanitizes an observation array."""
    x = np.asarray(obs)
    if hasattr(space, "dtype") and x.dtype != space.dtype:
        x = x.astype(space.dtype, copy=False)
    if np.isnan(x).any() or np.isinf(x).any():
        x = np.nan_to_num(x, copy=False)
    if hasattr(space, "low") and hasattr(space, "high"):
        low, high = space.low, space.high
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            x = np.clip(x, low, high)
    return x


# ==================== NEW RENDERING FUNCTION ====================
def get_images_by_id(
    env: gym.Env,
    env_name: str,
    camera_names: List[str],
    camera_id_map: Dict[str, Dict[str, int]],
) -> Dict[str, np.ndarray]:
    """Renders images from multiple camera views using explicit camera IDs.

    Args:
        env: The Gymnasium environment.
        env_name: The name of the current Meta-World task.
        camera_names: A list of camera names to render.
        camera_id_map: A dictionary mapping env names and camera names to IDs.

    Returns:
        A dictionary mapping camera names to (H, W, 3) uint8 image arrays.
    """
    multiview_images = {}
    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "mujoco_renderer", None)

    if renderer is None or not hasattr(renderer, "camera_id"):
        print("Warning: mujoco_renderer with camera_id not available.")
        return multiview_images

    # Get the specific camera ID mapping for this env, or use the default
    specific_map = camera_id_map.get(env_name, camera_id_map["__default__"])
    
    original_id = renderer.camera_id
    try:
        for name in camera_names:
            camera_id = specific_map.get(name)
            if camera_id is None:
                print(f"Warning: Camera '{name}' not found in ID map for env '{env_name}'.")
                continue
            
            try:
                renderer.camera_id = camera_id
                img = env.render()
                if CAMERA_FIP:
                    img = img[::-1]
                multiview_images[name] = img
                # import cv2
                # cv2.imwrite(f"{camera_id}_{name}.png", img)  # Save image for debugging
            except Exception as e:
                print(f"Error rendering camera '{name}' (ID: {camera_id}): {e}")
    finally:
        # Always restore the original camera ID
        renderer.camera_id = original_id
        
    return multiview_images
# ===============================================================


def resolve_model_data(env: gym.Env) -> Tuple[mj.MjModel, mj.MjData]:
    """Resolves MuJoCo model and data handles from the environment."""
    base = getattr(env, "unwrapped", env)
    rend = getattr(base, "mujoco_renderer", None)
    if rend is not None:
        model = getattr(rend, "model", None)
        data = getattr(rend, "data", None)
        if model is None or data is None:
            sim = getattr(rend, "sim", None)
            if sim is not None:
                model = getattr(sim, "model", None)
                data = getattr(sim, "data", None)
        if model is not None and data is not None:
            return model, data

    model = getattr(base, "model", None)
    data = getattr(base, "data", None)
    if model is not None and data is not None:
        return model, data
    raise AttributeError("Cannot locate MuJoCo model/data from environment.")


def mj_camera_name_to_id(model: mj.MjModel, name: str) -> Optional[int]:
    """Returns the MuJoCo camera ID for a given camera name."""
    try:
        return int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name))
    except Exception:
        return None


def split_arm_qpos(model: mj.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Heuristically extracts arm hinge joint positions from a qpos vector."""
    nj = model.njnt
    adr = model.jnt_qposadr
    jtype = model.jnt_type
    sizes = [
        int((adr[j + 1] - adr[j]) if j < nj - 1 else (model.nq - adr[j]))
        for j in range(nj)
    ]

    def jname(j):
        try:
            return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
        except Exception:
            return ""

    arm = []
    for j in range(nj):
        sl = slice(int(adr[j]), int(adr[j]) + sizes[j])
        jt = int(jtype[j])
        name = jname(j).lower()
        if jt == mj.mjtJoint.mjJNT_HINGE:
            if name.startswith(("right_j", "sawyer", "arm")) and (
                "finger" not in name
            ) and ("grip" not in name):
                arm.append(float(qpos[sl][0]))
    return np.asarray(arm, dtype=np.float32)


def create_or_open_zarr(path: str) -> Tuple[zarr.Group, zarr.Group, zarr.Group]:
    """Creates a Zarr root group and returns the root, data, and meta groups."""
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")
    return root, data, meta


def zarr_create_1d(
    meta_grp: zarr.Group, name: str, dtype, compressor=None
) -> zarr.core.Array:
    """Creates a 1D resizable Zarr array in the meta group."""
    return meta_grp.create(
        name,
        shape=(0,),
        chunks=(1024,),
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )


def zarr_create_timeseries(
    data_grp: zarr.Group,
    name: str,
    per_step_shape: Tuple[int, ...],
    dtype,
    chunks_t: int,
    compressor=None,
) -> zarr.core.Array:
    """Creates a time-series Zarr array with time as the first dimension."""
    shape = (0,) + tuple(per_step_shape)
    chunks = (chunks_t,) + tuple(per_step_shape)
    return data_grp.create(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )


def zarr_append(arr: zarr.core.Array, data_np: np.ndarray):
    """Appends a NumPy array to the end of a Zarr array."""
    assert arr.ndim == data_np.ndim, f"ndim mismatch: {arr.ndim} vs {data_np.ndim}"
    old_len = arr.shape[0]
    new_len = old_len + data_np.shape[0]
    arr.resize((new_len,) + arr.shape[1:])
    arr[old_len:new_len] = data_np


# ==============================================================================
# Main Data Collection Pipeline
# ==============================================================================
def collect_data_to_zarr(
    task_env_dis: Dict[str, List[str]], save_root: str = SAVE_ROOT
):
    """Collects data for each environment and writes it to a Zarr file."""
    os.makedirs(save_root, exist_ok=True)

    for env_name, desc_list in task_env_dis.items():
        task_description = (
            desc_list[np.random.randint(len(desc_list))] if RANDOM_DIS else desc_list[0]
        )
        print(f"\n--- Processing {env_name} ---")

        # 1. Create the environment.
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode=RENDER_MODE,
            camera_name=CAMERA_NAMES[0],
            width=256,
            height=256,
        )
        policy = ENV_POLICY_MAP[env_name]()

        # 2. Detect observation dimensions and camera size after reset.
        obs, _ = env.reset(seed=SEED)
        obs = sanitize_obs(obs, env.observation_space)
        multiview = get_images_by_id(env, env_name, CAMERA_NAMES, CAMERA_ID_MAP_PER_ENV)

        active_cams = [name for name in CAMERA_NAMES if name in multiview]
        if not active_cams:
            raise RuntimeError(f"Could not render any active cameras for env {env_name}")

        H, W, C = list(multiview.values())[0].shape
        assert C == 3, "Expecting RGB images."
        state_dim = int(np.asarray(obs).shape[0])
        action_dim = int(env.action_space.shape[0])

        model, data = resolve_model_data(env)
        qpos_dim = int(data.qpos.shape[0])
        arm_probe = split_arm_qpos(model, data.qpos)
        arm_dim = int(arm_probe.shape[0])

        # 3. Create Zarr directory and arrays for this environment.
        zarr_path = os.path.join(save_root, f"{env_name}.zarr")
        if os.path.exists(zarr_path):
            print(f"  -> Removing existing directory: {zarr_path}")
            shutil.rmtree(zarr_path)

        root, data_grp, meta_grp = create_or_open_zarr(zarr_path)
        root.attrs["env_name"] = env_name
        root.attrs["description"] = task_description
        root.attrs["dataset_version"] = "mw_mt50_v3_multi_cam_chw_uint8_v2"

        # Setup camera metadata.
        camera_map = {}
        for i, cam_name in enumerate(active_cams):
            ds_name = cam_name
            mj_id = mj_camera_name_to_id(model, cam_name)
            camera_map[cam_name] = {
                "dataset": ds_name,
                "index": i,
                "mj_id": mj_id,
                "flip_vertical": bool(CAMERA_FIP),
                "color": "rgb",
            }
        root.attrs["cameras"] = camera_map
        root.attrs["image_layout"] = "NCHW_uint8"

        # Setup low-dimensional arrays.
        arr_action = zarr_create_timeseries(
            data_grp, "action", (action_dim,), np.float32, chunks_t=1024
        )
        arr_state = zarr_create_timeseries(
            data_grp, "state", (state_dim,), np.float32, chunks_t=1024
        )
        arr_qpos = zarr_create_timeseries(
            data_grp, "qpos", (qpos_dim,), np.float32, chunks_t=1024
        )

        arr_proprio = None
        if arm_dim > 0:
            arr_proprio = zarr_create_timeseries(
                data_grp, "proprio", (arm_dim,), np.float32, chunks_t=1024
            )

        # Setup image arrays (CHW uint8).
        cam_arrays: Dict[str, zarr.core.Array] = {}
        for cam_name, meta in camera_map.items():
            ds_name = meta["dataset"]
            cam_arrays[cam_name] = zarr_create_timeseries(
                data_grp, ds_name, (3, H, W), np.uint8, chunks_t=1
            )
            cam_arrays[cam_name].attrs["source_camera_name"] = cam_name
            cam_arrays[cam_name].attrs["flip_vertical"] = bool(CAMERA_FIP)
            cam_arrays[cam_name].attrs["color"] = "rgb"
            cam_arrays[cam_name].attrs["layout"] = "CHW"

        arr_ep_ends = zarr_create_1d(meta_grp, "episode_ends", np.int64)

        # 4. Collection loop.
        total_steps = 0
        successes = 0
        while successes < EPISODES_NUMBER:
            obs, _ = env.reset(seed=SEED)
            obs = sanitize_obs(obs, env.observation_space)
            step_count = 0

            while step_count < MAX_EPISODE_STEPS:
                multiview = get_images_by_id(env, env_name, active_cams, CAMERA_ID_MAP_PER_ENV)

                model, data = resolve_model_data(env)
                qpos_now = data.qpos.copy().astype(np.float32)
                arm_now = (
                    split_arm_qpos(model, qpos_now) if arm_dim > 0 else None
                )

                act = policy.get_action(obs).astype(np.float32)

                # Write low-dimensional data.
                zarr_append(arr_state, obs.reshape(1, -1).astype(np.float32))
                zarr_append(arr_qpos, qpos_now.reshape(1, -1))
                if arr_proprio is not None:
                    zarr_append(arr_proprio, arm_now.reshape(1, -1))

                # Write image frames (convert to CHW uint8).
                for cam_name in active_cams:
                    img_nhwc = multiview[cam_name]  # (H,W,3) uint8
                    assert img_nhwc.dtype == np.uint8 and img_nhwc.ndim == 3
                    img_chw = np.ascontiguousarray(
                        img_nhwc.transpose(2, 0, 1)
                    )
                    zarr_append(cam_arrays[cam_name], img_chw[None, ...])

                # Write action.
                zarr_append(arr_action, act.reshape(1, -1))

                # Step the environment.
                obs, rew, terminated, truncated, info = env.step(act)
                obs = sanitize_obs(obs, env.observation_space)
                step_count += 1
                total_steps += 1

                done = (
                    bool(info.get("success", 0))
                    or bool(terminated)
                    or bool(truncated)
                )
                if done:
                    zarr_append(
                        arr_ep_ends, np.asarray([total_steps], dtype=np.int64)
                    )
                    if info.get("success", 0):
                        successes += 1
                    break

        # Clean up the renderer to release GPU resources.
        try:
            if hasattr(env, "unwrapped") and hasattr(
                env.unwrapped, "mujoco_renderer"
            ):
                renderer = env.unwrapped.mujoco_renderer
                if hasattr(renderer, "close"):
                    renderer.close()
        except Exception:
            pass
        env.close()

        print(f"Saved {successes} episodes ({total_steps} steps) to: {zarr_path}")
        print("Keys under /data:", list(data_grp.array_keys()))
        print("Episode ends:", arr_ep_ends[:])
    print("\nAll tasks done.")


if __name__ == "__main__":
    task_descriptions = get_task_discriptions()
    task_env_map = get_task_name_and_desc(task_descriptions)
    collect_data_to_zarr(task_env_map, save_root=SAVE_ROOT)