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
from numcodecs import Blosc
from zarr.storage import NestedDirectoryStore
import warnings
from tqdm import tqdm

# ==============================================================================
# 全局配置
# ==============================================================================
warnings.filterwarnings("ignore", message=".*Overriding environment.*already in registry.*")
warnings.filterwarnings("ignore", message=".*Constant\\(s\\) may be too high.*")

os.environ.setdefault("MUJOCO_GL", "egl")     # 离屏渲染
os.environ.setdefault("BLOSC_NTHREADS", "8")  # Blosc 多线程压缩
np.random.seed(42)

DEBUG = False
RANDOM_DIS = True
RENDER_MODE = "rgb_array"

# 你要采的相机顺序（名字与数据集键保持一致）
CAMERA_NAMES = ["corner4", "gripperPOV"]
# CAMERA_NAMES = ["corner", "corner2", "corner3", "corner4", "gripperPOV", "topview", "behindGripper"]

CAMERA_FIP = True              # 是否垂直翻转
SEED = 42
EPISODES_NUMBER = 1          # 目标成功 episode 数
MAX_EPISODE_STEPS = 500

# SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3_zarr"
SAVE_ROOT = "/data/robot_dataset/metaworld/debug"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 压缩器：lz4（更快）或 zstd（clevel=1~3 也很快）
COMPRESSOR = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)

# 批写入缓冲长度（时间维）
BUFFER_T_IMAGE = 64
BUFFER_T_LOWDIM = 4096  # state/qpos/proprio/action 的时间维 chunk

# 你的相机 ID 映射（保持“无缺 id 回退”的语义）
CAMERA_ID_MAP_PER_ENV: Dict[str, Dict[str, int]] = {
    "__default__": {
        "topview": 0, "gripperPOV": 6,
        "corner": 1, "corner2": 2, "corner3": 3, "corner4": 4, "behindGripper": 5
    }
}

# ==============================================================================
# 工具函数
# ==============================================================================
def get_task_discriptions() -> dict:
    with open("/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json", "r") as f:
        return json.load(f)

def get_task_name_and_desc(task_list: list) -> dict:
    result = {}
    for item in task_list:
        env_name = item["env"]
        assert env_name in MT50_V3, f"Task {env_name} not found in MT50_V3"
        result[env_name] = item["description"]
    return result

def sanitize_obs(obs: np.ndarray, space: gym.Space) -> np.ndarray:
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

# ========== 按相机 ID 渲染（保持原语义，不做缺 id 回退） ==========
def get_images_by_id(
    env: gym.Env,
    env_name: str,
    camera_names: List[str],
    camera_id_map: Dict[str, Dict[str, int]],
) -> Dict[str, np.ndarray]:
    """使用显式 camera_id 渲染；缺 id 仅告警并跳过，不回退。"""
    multiview_images = {}
    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "mujoco_renderer", None)

    if renderer is None or not hasattr(renderer, "camera_id"):
        print("Warning: mujoco_renderer with camera_id not available.")
        return multiview_images

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
            except Exception as e:
                print(f"Error rendering camera '{name}' (ID: {camera_id}): {e}")
    finally:
        renderer.camera_id = original_id
    return multiview_images
# ================================================================

def resolve_model_data(env: gym.Env) -> Tuple[mj.MjModel, mj.MjData]:
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
    try:
        return int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name))
    except Exception:
        return None

def precompute_arm_indices(model: mj.MjModel) -> np.ndarray:
    """预计算 7DoF 手臂的 qpos 索引，避免每步做字符串匹配。"""
    nj = model.njnt
    adr = model.jnt_qposadr
    jtype = model.jnt_type
    idx = []
    for j in range(nj):
        jt = int(jtype[j])
        if jt != mj.mjtJoint.mjJNT_HINGE:
            continue
        try:
            name = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or "").lower()
        except Exception:
            name = ""
        if name.startswith(("right_j", "sawyer", "arm")) and ("finger" not in name) and ("grip" not in name):
            qpos_i = int(adr[j])  # hinge 关节占 1 个 qpos
            idx.append(qpos_i)
    return np.asarray(idx, dtype=np.int64)

def create_or_open_zarr(path: str) -> Tuple[zarr.Group, zarr.Group, zarr.Group]:
    # NestedDirectoryStore 减少单目录文件数量
    store = NestedDirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")
    return root, data, meta

def zarr_create_1d(meta_grp: zarr.Group, name: str, dtype, compressor=None) -> zarr.core.Array:
    return meta_grp.create(
        name, shape=(0,), chunks=(1024,), dtype=dtype, compressor=compressor, overwrite=True
    )

def zarr_create_timeseries(
    data_grp: zarr.Group,
    name: str,
    per_step_shape: Tuple[int, ...],
    dtype,
    chunks_t: int,
    compressor=None,
) -> zarr.core.Array:
    shape = (0,) + tuple(per_step_shape)
    chunks = (chunks_t,) + tuple(per_step_shape)
    return data_grp.create(
        name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor, overwrite=True
    )

def zarr_append(arr: zarr.core.Array, data_np: np.ndarray):
    assert arr.ndim == data_np.ndim, f"ndim mismatch: {arr.ndim} vs {data_np.ndim}"
    old_len = arr.shape[0]
    new_len = old_len + data_np.shape[0]
    arr.resize((new_len,) + arr.shape[1:])
    arr[old_len:new_len] = data_np

class ZarrBatchAppender:
    """把逐步写入改为“攒满再写”，显著减少 I/O 与压缩调用。"""
    def __init__(self, buffer_t: int = 64):
        self.buffer_t = int(buffer_t)
        self.buffers: Dict[str, List[np.ndarray]] = {}
        self.targets: Dict[str, zarr.core.Array] = {}

    def register(self, name: str, zarr_arr: zarr.core.Array):
        self.targets[name] = zarr_arr
        self.buffers[name] = []

    def append(self, name: str, sample: np.ndarray):
        self.buffers[name].append(sample)
        if len(self.buffers[name]) >= self.buffer_t:
            self.flush_one(name)

    def flush_one(self, name: str):
        if self.buffers[name]:
            block = np.ascontiguousarray(np.stack(self.buffers[name], axis=0))
            zarr_append(self.targets[name], block)
            self.buffers[name].clear()

    def flush_all(self):
        for name in list(self.buffers.keys()):
            self.flush_one(name)

# ==============================================================================
# 采集主流程
# ==============================================================================
def collect_data_to_zarr(task_env_dis: Dict[str, List[str]], save_root: str = SAVE_ROOT):
    os.makedirs(save_root, exist_ok=True)

    tasks = list(task_env_dis.items())
    outer_bar = tqdm(tasks, desc="All tasks", ncols=100)

    for env_name, desc_list in outer_bar:
        outer_bar.set_postfix_str(env_name)
        task_description = (desc_list[np.random.randint(len(desc_list))] if RANDOM_DIS else desc_list[0])

        # 1) 创建 env
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode=RENDER_MODE,
            camera_name=CAMERA_NAMES[0],
            width=256,
            height=256,
        )
        policy = ENV_POLICY_MAP[env_name]()

        # 2) reset 后探测尺寸
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
        arm_idx = precompute_arm_indices(model)
        arm_dim = int(arm_idx.size)

        # 3) 创建 zarr
        zarr_path = os.path.join(save_root, f"{env_name}.zarr")
        if os.path.exists(zarr_path):
            print(f"  -> Removing existing directory: {zarr_path}")
            shutil.rmtree(zarr_path)

        root, data_grp, meta_grp = create_or_open_zarr(zarr_path)
        root.attrs["env_name"] = env_name
        root.attrs["description"] = task_description
        root.attrs["dataset_version"] = "mw_mt50_v3_multi_cam_chw_uint8_v3"
        specific_map = CAMERA_ID_MAP_PER_ENV.get(env_name, CAMERA_ID_MAP_PER_ENV["__default__"])
        root.attrs["camera_id_map_used"] = {k: specific_map.get(k, None) for k in active_cams}
        root.attrs["flip_vertical"] = bool(CAMERA_FIP)
        root.attrs["render_mode"] = RENDER_MODE
        root.attrs["seed_base"] = int(SEED)
        root.attrs["image_layout"] = "NCHW_uint8"

        # 相机元信息（记录 name->mj_id，真正渲染走 id）
        camera_map = {}
        for i, cam_name in enumerate(active_cams):
            mj_id = mj_camera_name_to_id(model, cam_name)
            camera_map[cam_name] = {
                "dataset": cam_name, "index": i, "mj_id": mj_id,
                "flip_vertical": bool(CAMERA_FIP), "color": "rgb",
            }
        root.attrs["cameras"] = camera_map

        # 低维数组（大 chunk，压缩）
        arr_action = zarr_create_timeseries(
            data_grp, "action", (action_dim,), np.float32, chunks_t=BUFFER_T_LOWDIM, compressor=COMPRESSOR
        )
        arr_state = zarr_create_timeseries(
            data_grp, "state", (state_dim,), np.float32, chunks_t=BUFFER_T_LOWDIM, compressor=COMPRESSOR
        )
        arr_qpos = zarr_create_timeseries(
            data_grp, "qpos", (qpos_dim,), np.float32, chunks_t=BUFFER_T_LOWDIM, compressor=COMPRESSOR
        )
        arr_proprio = None
        if arm_dim > 0:
            arr_proprio = zarr_create_timeseries(
                data_grp, "proprio", (arm_dim,), np.float32, chunks_t=BUFFER_T_LOWDIM, compressor=COMPRESSOR
            )

        # 图像数组（时间维 chunk=BUFFER_T_IMAGE）
        cam_arrays: Dict[str, zarr.core.Array] = {}
        for cam_name in active_cams:
            cam_arrays[cam_name] = zarr_create_timeseries(
                data_grp, cam_name, (3, H, W), np.uint8, chunks_t=BUFFER_T_IMAGE, compressor=COMPRESSOR
            )
            cam_arrays[cam_name].attrs["source_camera_name"] = cam_name
            cam_arrays[cam_name].attrs["flip_vertical"] = bool(CAMERA_FIP)
            cam_arrays[cam_name].attrs["color"] = "rgb"
            cam_arrays[cam_name].attrs["layout"] = "CHW"

        arr_ep_ends = zarr_create_1d(meta_grp, "episode_ends", np.int64, compressor=COMPRESSOR)

        # 注册批量缓冲
        app = ZarrBatchAppender(buffer_t=BUFFER_T_IMAGE)   # 图像/低维分别 flush，但这边统一用 64 也 OK
        app.register("action", arr_action)
        app.register("state",  arr_state)
        app.register("qpos",   arr_qpos)
        if arr_proprio is not None:
            app.register("proprio", arr_proprio)
        for cam_name, arr in cam_arrays.items():
            app.register(f"cam:{cam_name}", arr)

        # 4) 采集循环（成功计数到 EPISODES_NUMBER 为止）
        total_steps = 0
        successes = 0
        attempts = 0

        pbar = tqdm(total=EPISODES_NUMBER, desc=f"{env_name} (episodes)", ncols=100)
        while successes < EPISODES_NUMBER:
            ep_seed = SEED + attempts
            attempts += 1

            obs, _ = env.reset(seed=ep_seed)
            obs = sanitize_obs(obs, env.observation_space)
            step_count = 0

            # 每个 episode 开头解析一次 data（避免每步重复 get）
            model, data = resolve_model_data(env)

            for _ in range(MAX_EPISODE_STEPS):
                # 渲染（按 ID；已包含翻转）
                multiview = get_images_by_id(env, env_name, active_cams, CAMERA_ID_MAP_PER_ENV)

                # qpos & proprio（用预计算索引）
                qpos_now = data.qpos.astype(np.float32).copy()
                arm_now = qpos_now[arm_idx].copy() if arm_idx.size > 0 else None

                # policy
                act = ENV_POLICY_MAP[env_name]().get_action(obs).astype(np.float32)

                # ---- 缓冲写 ----
                app.append("state",  obs.astype(np.float32).reshape(-1))
                app.append("qpos",   qpos_now.reshape(-1))
                if arr_proprio is not None:
                    app.append("proprio", arm_now.reshape(-1))
                app.append("action", act.reshape(-1))
                print("Buffered data for step:", step_count)

                for cam_name in active_cams:
                    img_nhwc = multiview[cam_name]            # (H,W,3) uint8（已翻转）
                    img_chw = img_nhwc.transpose(2, 0, 1)     # (3,H,W)
                    app.append(f"cam:{cam_name}", img_chw)

                # env step
                obs, rew, terminated, truncated, info = env.step(act)
                obs = sanitize_obs(obs, env.observation_space)
                step_count += 1
                total_steps += 1

                done = bool(info.get("success", 0)) or bool(terminated) or bool(truncated)
                if done:
                    zarr_append(arr_ep_ends, np.asarray([total_steps], dtype=np.int64))
                    if info.get("success", 0):
                        successes += 1
                        pbar.update(1)
                    break

        # flush 剩余
        app.flush_all()
        pbar.close()

        # 清理
        try:
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "mujoco_renderer"):
                renderer = env.unwrapped.mujoco_renderer
                if hasattr(renderer, "close"):
                    renderer.close()
        except Exception:
            pass
        env.close()

        # 简要输出
        print(f"Saved {successes} episodes ({total_steps} steps) to: {zarr_path}")
        print("Keys under /data:", list(data_grp.array_keys()))
        print("Episode ends:", arr_ep_ends[:])

    print("\nAll tasks done.")

# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    task_descriptions = get_task_discriptions()
    task_env_map = get_task_name_and_desc(task_descriptions)
    collect_data_to_zarr(task_env_map, save_root=SAVE_ROOT)
