# -*- coding: utf-8 -*-
"""
Meta-World MT50 collector using diffusion_policy ReplayBuffer (disk compression):
- Let ReplayBuffer create /meta/episode_ends via create_from_path(mode='a').
- Per-episode accumulation -> ReplayBuffer.add_episode(compressors='disk').
- Images: HWC -> CHW uint8 (copy; break negative strides).
- Low-dim: float32.
- Time-only chunking: low-dim (large), image (smaller).
"""

import os
import json
import inspect
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco as mj
import numpy as np
import zarr
from tqdm import tqdm
import shutil

# Meta-World
from Metaworld.metaworld.env_dict import MT50_V3
from Metaworld.metaworld.policies import ENV_POLICY_MAP

# diffusion_policy ReplayBuffer
import sys
sys.path.append("/home/libo/project/cm/trunck-consistency-policy")  # project root
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa

# ===================== Config =====================
os.environ.setdefault("MUJOCO_GL", "egl")     # offscreen rendering
# 可选：多线程 Blosc，'disk' 使用 zstd+bitshuffle，通常稳定
os.environ.setdefault("BLOSC_NTHREADS", "8")

np.random.seed(42)

# SAVE_ROOT = "/data/robot_dataset/metaworld/debug_rb"
SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3_zarr"

os.makedirs(SAVE_ROOT, exist_ok=True)

CAMERA_NAMES = ["corner4", "gripperPOV"]
CAMERA_FIP = True
RENDER_MODE = "rgb_array"

SEED = 42
EPISODES_NUMBER = 100
MAX_EPISODE_STEPS = 500

# Chunk length (time axis only)
CHUNK_T_IMAGE = 64
CHUNK_T_LOWDIM = 4096

CAMERA_ID_MAP_PER_ENV: Dict[str, Dict[str, int]] = {
    "__default__": {
        "topview": 0,
        "gripperPOV": 6,
        "corner": 1,
        "corner2": 2,
        "corner3": 3,
        "corner4": 4,
        "behindGripper": 5
    }
}


# ================== Utils ==================
def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Return HWC uint8; robust to float [0,1]/[0,255] inputs; copy-safe."""
    x = np.asarray(img)
    if np.issubdtype(x.dtype, np.floating):
        m = float(x.max()) if x.size else 1.0
        if m <= 1.0 + 1e-6:
            x = (x * 255.0).round()
        x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)
    return x


def sanitize_obs(obs: np.ndarray, space: gym.Space) -> np.ndarray:
    """Cast dtype; clip only when bounds are finite AND non-degenerate."""
    x = np.asarray(obs)
    if hasattr(space, "dtype") and x.dtype != space.dtype:
        x = x.astype(space.dtype, copy=False)
    if hasattr(space, "low") and hasattr(space, "high"):
        low, high = space.low, space.high
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            if np.any(np.abs(high - low) > 1e-6):
                x = np.clip(x, low, high)
    return x


def get_images_by_id(env: gym.Env, env_name: str, camera_names: List[str]) -> Dict[str, np.ndarray]:
    """Render by explicit camera_id; return dict[name] = HWC uint8 (independent)."""
    out = {}
    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "mujoco_renderer", None)
    if renderer is None or not hasattr(renderer, "camera_id"):
        return out
    specific = CAMERA_ID_MAP_PER_ENV.get(env_name, CAMERA_ID_MAP_PER_ENV["__default__"])
    original_id = renderer.camera_id
    for name in camera_names:
        cid = specific.get(name)
        if cid is None:
            continue
        renderer.camera_id = cid
        img = env.render()
        img = to_uint8_rgb(img)
        img = img[::-1].copy() if CAMERA_FIP else img.copy()  # break negative stride & alias
        out[name] = img
    renderer.camera_id = original_id
    return out


def resolve_model_data(env: gym.Env) -> Tuple[mj.MjModel, mj.MjData]:
    """Locate active (model, data)."""
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
    assert model is not None and data is not None, "Cannot locate MuJoCo model/data."
    return model, data


def precompute_arm_indices(model: mj.MjModel) -> np.ndarray:
    """Pick 7-DoF arm joints (skip fingers/gripper)."""
    nj = model.njnt
    adr = model.jnt_qposadr
    jtype = model.jnt_type
    idx = []
    for j in range(nj):
        if int(jtype[j]) != mj.mjtJoint.mjJNT_HINGE:
            continue
        name = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or "").lower()
        if (name.startswith(("right_j", "sawyer", "arm")) and 
            ("finger" not in name) and 
            ("grip" not in name)):
            idx.append(int(adr[j]))
    return np.asarray(idx, dtype=np.int64)


def maybe_reset_policy(policy) -> bool:
    """Call policy.reset() only if exists and no-arg."""
    if not hasattr(policy, "reset"):
        return False
    fn = getattr(policy, "reset")
    if not callable(fn):
        return False
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if (getattr(fn, "__self__", None) is not None and 
        len(params) > 0 and 
        params[0].name in ("self", "cls")):
        params = params[1:]
    req = [p for p in params 
           if p.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                        inspect.Parameter.POSITIONAL_OR_KEYWORD)
           and p.default is inspect._empty]
    if len(req) == 0:
        fn()
        return True
    return False


# ================== Core: one env -> one zarr ==================
def collect_one_env(env_name: str, descriptions: List[str]):
    """Collect 1 successful episode for a single env and append to <env>.zarr."""
    task_description = descriptions[np.random.randint(len(descriptions))]

    # 1) Make env + scripted policy
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode=RENDER_MODE,
        camera_name=CAMERA_NAMES[0],
        width=224,
        height=224,
        disable_env_checker=True,  # silence passive checker warnings
    )
    policy = ENV_POLICY_MAP[env_name]()  # persistent instance

    # 2) Probe once
    obs, _ = env.reset(seed=SEED)
    maybe_reset_policy(policy)
    obs = sanitize_obs(obs, env.observation_space)
    multiview = get_images_by_id(env, env_name, CAMERA_NAMES)
    active_cams = [n for n in CAMERA_NAMES if n in multiview]
    assert len(active_cams) > 0, f"No active cameras for {env_name}"
    H, W, _ = list(multiview.values())[0].shape
    action_dim = int(env.action_space.shape[0])

    model, data = resolve_model_data(env)
    qpos_dim = int(data.qpos.shape[0])
    arm_idx = precompute_arm_indices(model)
    arm_dim = int(arm_idx.size)

    # 3) Create/Load ReplayBuffer (let it init episode_ends)
    zarr_path = os.path.join(SAVE_ROOT, f"{env_name}.zarr")
    if os.path.isdir(zarr_path):
        shutil.rmtree(zarr_path) 
    rb = ReplayBuffer.create_from_path(zarr_path, mode='a')   # A 方案
    root = rb.root  # zarr.Group

    # Attrs metadata (optional but handy)
    root.attrs["env_name"] = env_name
    root.attrs["description"] = task_description
    root.attrs["dataset_version"] = "mw_mt50_v3_rb_chw_uint8_disk"
    root.attrs["flip_vertical"] = bool(CAMERA_FIP)
    root.attrs["render_mode"] = RENDER_MODE
    root.attrs["seed_base"] = int(SEED)
    root.attrs["image_layout"] = "NCHW_uint8"
    root.attrs["cameras"] = {
        k: {
            "dataset": k, 
            "index": i, 
            "flip_vertical": bool(CAMERA_FIP), 
            "color": "rgb"
        }
        for i, k in enumerate(active_cams)
    }

    # 4) Episode loop
    successes = 0
    attempts = 0

    pbar = tqdm(total=EPISODES_NUMBER, desc=f"{env_name} (episodes)", ncols=100)
    while successes < EPISODES_NUMBER:
        ep_seed = SEED + attempts
        attempts += 1

        obs, _ = env.reset(seed=ep_seed)
        maybe_reset_policy(policy)
        obs = sanitize_obs(obs, env.observation_space)

        # Buffers for this episode (lists for speed; stack at the end)
        ep_states: List[np.ndarray] = []
        ep_qpos: List[np.ndarray] = []
        ep_prop: List[np.ndarray] = [] if arm_dim > 0 else None
        ep_action: List[np.ndarray] = []
        ep_imgs: Dict[str, List[np.ndarray]] = {k: [] for k in active_cams}

        for _ in range(MAX_EPISODE_STEPS):
            # Render images
            multiview = get_images_by_id(env, env_name, active_cams)
            for cam in active_cams:
                ep_imgs[cam].append(multiview[cam])  # (H,W,3) uint8

            # Low-dim signals
            model, data = resolve_model_data(env)
            qpos_now = data.qpos.astype(np.float32).copy()
            arm_now = qpos_now[arm_idx].copy() if arm_dim > 0 else None

            # Scripted policy action (reuse the same instance!)
            act = policy.get_action(obs).astype(np.float32)
            low, high = env.action_space.low, env.action_space.high
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                act = np.clip(act, low, high)

            # Stash
            ep_states.append(obs.astype(np.float32).reshape(-1))
            ep_qpos.append(qpos_now.reshape(-1))
            if arm_dim > 0:
                ep_prop.append(arm_now.reshape(-1))
            ep_action.append(act.reshape(-1))

            # Step
            obs, rew, terminated, truncated, info = env.step(act)
            obs = sanitize_obs(obs, env.observation_space)

            done = bool(info.get("success", 0)) or bool(terminated) or bool(truncated)
            if done:
                # Assemble episode dict
                ep_data = {
                    "state": np.stack(ep_states, axis=0).astype(np.float32),  # (T,D)
                    "qpos": np.stack(ep_qpos, axis=0).astype(np.float32),     # (T,J)
                    "action": np.stack(ep_action, axis=0).astype(np.float32), # (T,A)
                }
                if arm_dim > 0:
                    ep_data["proprio"] = np.stack(ep_prop, axis=0).astype(np.float32)  # (T,P)
                
                for cam in active_cams:
                    chw = np.stack([f.transpose(2, 0, 1) for f in ep_imgs[cam]], axis=0)  # (T,3,H,W)
                    ep_data[cam] = chw.astype(np.uint8)

                # Time-only chunking
                chunks: Dict[str, tuple] = {}
                for key in ("state", "qpos", "action", "proprio"):
                    if key in ep_data:
                        shape = ep_data[key].shape
                        chunks[key] = (CHUNK_T_LOWDIM,) + shape[1:]
                for cam in active_cams:
                    shape = ep_data[cam].shape
                    chunks[cam] = (CHUNK_T_IMAGE,) + shape[1:]

                # Keep disk compression (zstd + bitshuffle as defined in ReplayBuffer.resolve_compressor)
                rb.add_episode(ep_data, chunks=chunks, compressors='disk')

                if info.get("success", 0):
                    successes += 1
                    pbar.update(1)
                break

    pbar.close()
    env.close()
    del env  # reduce EGL destructor noise

    # Quick on-disk verify (reopen)
    root2 = zarr.open(zarr_path, mode="r")
    data_grp = root2["data"]
    meta_grp = root2["meta"]
    ends = meta_grp["episode_ends"][:]
    start = 0 if ends.size == 1 else int(ends[-2])
    end = int(ends[-1])
    
    for key in ("action", "state", "qpos", "proprio"):
        if key in data_grp:
            arr = data_grp[key][start:end]
            if arr.size:
                print(f"[verify] {env_name} /data/{key}: mean0={float(arr[0].mean()):.2f}  meanL={float(arr[-1].mean()):.2f}")
    
    for cam in active_cams:
        A = data_grp[cam][start:end]
        if A.shape[0]:
            print(f"[verify] {env_name} /data/{cam}: mean0={float(A[0].mean()):.2f}  meanL={float(A[-1].mean()):.2f}")

    print(f"[done] {env_name}: {end-start} steps saved -> {zarr_path}")


# ================== Task list helpers ==================
def get_task_descriptions(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def to_env_desc_map(task_list: list) -> Dict[str, List[str]]:
    out = {}
    for item in task_list:
        env_name = item["env"]
        assert env_name in MT50_V3, f"{env_name} not in MT50_V3"
        out.setdefault(env_name, [])
        out[env_name].append(item["description"])
    return out


# ================== Entry ==================
if __name__ == "__main__":
    tasks_json = "/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json"
    env_map = to_env_desc_map(get_task_descriptions(tasks_json))

    for env_name, descs in tqdm(list(env_map.items())[:50], desc="All tasks", ncols=100):
        collect_one_env(env_name, descs)