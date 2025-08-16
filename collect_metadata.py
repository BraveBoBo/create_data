# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*already in registry.*")
warnings.filterwarnings("ignore", message=".*Constant\(s\) may be too high.*")
import os
import json
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco as mj
import numpy as np
import zarr

from Metaworld.metaworld.env_dict import MT50_V3
from Metaworld.metaworld.policies import ENV_POLICY_MAP

# ---------------------
# 全局配置
# ---------------------
os.environ["MUJOCO_GL"] = "egl"   # 无头渲染
np.random.seed(42)

DEBUG = False                      # 不写视频
RANDOM_DIS = True                  # 任务描述随机
RENDER_MODE = "rgb_array"
CAMERA_NAMES = ["topview", "gripperPOV",]
# CAMERA_NAMES = ["corner", "topview", "behindGripper", "gripperPOV", "corner2", "corner3", "corner4"]
CAMERA_FIP = False                 # 是否垂直翻转（flip vertical）
SEED = 42
EPISODES_NUMBER = 2
MAX_EPISODE_STEPS = 500

SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3_zarr"                # 可改成你的数据根目录
os.makedirs(SAVE_ROOT, exist_ok=True)


# ---------------------
# 工具函数
# ---------------------
def get_task_discriptions():
    """读取任务描述文件。"""
    with open(
        "/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json",
        "r",
    ) as f:
        task_descriptions = json.load(f)
    return task_descriptions


def get_task_name_and_desc(task_list) -> dict:
    """将 env_name 映射到描述列表。"""
    result = {}
    for item in task_list:
        env_name = item["env"]
        assert env_name in MT50_V3, f"Task {env_name} not found in MT50_V3"
        result[env_name] = item["description"]
    return result


def sanitize_obs(obs, space):
    """将 obs 转成 space.dtype、去 NaN/Inf，并在 Box 有界时裁剪到 [low, high]。"""
    x = np.asarray(obs)
    # dtype 对齐
    if hasattr(space, "dtype") and x.dtype != space.dtype:
        x = x.astype(space.dtype, copy=False)
    # 去 NaN/Inf
    if np.isnan(x).any() or np.isinf(x).any():
        x = np.nan_to_num(x, copy=False)
    # Box 裁剪
    if hasattr(space, "low") and hasattr(space, "high"):
        low, high = space.low, space.high
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            x = np.clip(x, low, high)
    return x


def get_multiview_images(env, camera_names) -> Dict[str, np.ndarray]:
    """按相机名优先渲染，失败时回退按 camera_id。返回 {name: (H,W,3) uint8}"""
    multiview_images = {}
    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "mujoco_renderer", None)

    # 1) 优先按名字渲染
    if renderer is not None and hasattr(renderer, "render"):
        for name in camera_names:
            try:
                img = renderer.render(camera_name=name)  # (H,W,3) uint8
                if CAMERA_FIP:
                    img = img[::-1]                      # 先在 NHWC 下翻转
                multiview_images[name] = img
            except Exception:
                pass

    # 2) 对缺失相机，按 camera_id 顺序回退
    missing = [n for n in camera_names if n not in multiview_images]
    if missing:
        if renderer is not None and hasattr(renderer, "camera_id"):
            original_id = renderer.camera_id
            for i, name in enumerate(camera_names):
                if name in multiview_images:
                    continue
                try:
                    renderer.camera_id = i
                    img = env.render()
                    if CAMERA_FIP:
                        img = img[::-1]
                    multiview_images[name] = img
                except Exception:
                    pass
            try:
                renderer.camera_id = original_id
            except Exception:
                pass
        else:
            # 无 renderer.camera_id，则尝试默认 env.render()
            try:
                img = env.render()
                if CAMERA_FIP:
                    img = img[::-1]
                if missing:
                    multiview_images[missing[0]] = img
            except Exception:
                pass
    return multiview_images


def resolve_model_data(env) -> Tuple[mj.MjModel, mj.MjData]:
    """解析出 MuJoCo 的 model/data 句柄（优先 mujoco_renderer）。"""
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
    raise AttributeError("Cannot locate MuJoCo model/data (renderer/model/data not found).")


def mj_camera_name_to_id(model: mj.MjModel, name: str) -> Optional[int]:
    """返回 MuJoCo 的相机 id，失败返回 None。"""
    try:
        return int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name))
    except Exception:
        return None


def split_arm_qpos(model: mj.MjModel, qpos: np.ndarray) -> np.ndarray:
    """从 qpos 中按名称启发式抽取手臂 HINGE 关节（Sawyer 常见 7 个）。"""
    nj = model.njnt
    adr = model.jnt_qposadr
    jtype = model.jnt_type
    sizes = [int((adr[j + 1] - adr[j]) if j < nj - 1 else (model.nq - adr[j])) for j in range(nj)]

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
            if name.startswith(("right_j", "sawyer", "arm")) and ("finger" not in name) and ("grip" not in name):
                arm.append(float(qpos[sl][0]))
    return np.asarray(arm, dtype=np.float32)


def create_or_open_zarr(path: str):
    """创建 zarr 根组并返回 (root, data_group, meta_group)。"""
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")
    return root, data, meta


def zarr_create_1d(meta_grp: zarr.Group, name: str, dtype, compressor=None):
    """在 meta 组里创建一维可扩展数组（初始 0 长度）。"""
    return meta_grp.create(
        name,
        shape=(0,),
        chunks=(1024,),
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )


def zarr_create_timeseries(data_grp: zarr.Group, name: str, per_step_shape: Tuple[int, ...], dtype, chunks_t: int,
                           compressor=None):
    """创建按时间拼接的一维时间序列数组（初始 0），第 0 维为时间。"""
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
    """把 data_np 追加到 zarr 数组末尾。"""
    assert arr.ndim == data_np.ndim, f"ndim mismatch: {arr.ndim} vs {data_np.ndim}"
    old = arr.shape[0]
    new = old + data_np.shape[0]
    arr.resize((new,) + arr.shape[1:])
    arr[old:new] = data_np


# ---------------------
# 主流程：写 Zarr（兼容 Diffusion Policy ReplayBuffer）
# ---------------------
def collect_data_to_zarr(task_env_dis: Dict[str, List[str]], save_root: str = SAVE_ROOT):
    """为每个 env 采集数据，并写入 Zarr：
    /<env_name>.zarr/
      data/
        action  (N, Da) float32
        state   (N, Ds) float32
        qpos    (N, nq) float32
        robot0_arm_qpos (N, n_arm) float32
        proprio (N, n_arm) float32         # = robot0_arm_qpos
        <camera_name> (N, 3, H, W) uint8   # 每个可用相机，以相机名直接命名
      meta/
        episode_ends (E,) int64            # 每个 episode 的 end（exclusive，下标累计）
      attrs:
        cameras: {name: {"dataset": str, "index": int, "mj_id": int|None, "flip_vertical": bool, "color":"rgb"}}
        image_layout: "NCHW_uint8"
    """
    os.makedirs(save_root, exist_ok=True)

    for env_name, desc_list in task_env_dis.items():
        task_description = desc_list[np.random.randint(len(desc_list))] if RANDOM_DIS else desc_list[0]
        print(f"\n--- Processing {env_name} ---")

        # 1) 创建环境
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode=RENDER_MODE,
            camera_name=CAMERA_NAMES[0],
            width=256, height=256,
        )
        policy = ENV_POLICY_MAP[env_name]()

        # 2) reset 后探测观测维与相机尺寸
        obs, _ = env.reset(seed=SEED)
        obs = sanitize_obs(obs, env.observation_space)
        multiview = get_multiview_images(env, CAMERA_NAMES)

        # 只保留能成功渲染的相机，并固定顺序
        active_cams = [name for name in CAMERA_NAMES if name in multiview]
        if len(active_cams) == 0:
            # 至少保证一个视角（用 env.render 的默认）
            img = env.render()
            if CAMERA_FIP:
                img = img[::-1]
            multiview = {CAMERA_NAMES[0]: img}
            active_cams = [CAMERA_NAMES[0]]

        H, W, C = list(multiview.values())[0].shape
        assert C == 3, "期望 RGB 图像"
        state_dim = int(np.asarray(obs).shape[0])
        action_dim = int(env.action_space.shape[0])

        # MuJoCo 句柄（用于 qpos）
        model, data = resolve_model_data(env)
        qpos_dim = int(data.qpos.shape[0])
        arm_probe = split_arm_qpos(model, data.qpos)
        arm_dim = int(arm_probe.shape[0])  # 可能为 0

        # 3) 为该 env 创建 zarr 目录与数组
        zarr_path = os.path.join(save_root, f"{env_name}.zarr")
        if os.path.exists(zarr_path):
            print(f"  -> Remove existing: {zarr_path}")
            import shutil
            shutil.rmtree(zarr_path)

        root, data_grp, meta_grp = create_or_open_zarr(zarr_path)
        root.attrs["env_name"] = env_name
        root.attrs["description"] = task_description
        root.attrs["dataset_version"] = "mw_mt50_v3_multi_cam_chw_uint8_v2" # version bump

        # === 相机映射与元数据 ===
        camera_map = {}  # { "<camera_name>": {"dataset": "<camera_name>", "index": i, ...} }
        for i, cam_name in enumerate(active_cams):
            # --------------------------------------------------------------------
            # 修改点: 直接使用相机名作为数据集名称，并移除 "_chw" 后缀
            # --------------------------------------------------------------------
            ds_name = cam_name
            mj_id = mj_camera_name_to_id(model, cam_name)
            camera_map[cam_name] = {
                "dataset": ds_name,
                "index": i,
                "mj_id": mj_id,
                "flip_vertical": bool(CAMERA_FIP),
                "color": "rgb"
            }
        root.attrs["cameras"] = camera_map
        root.attrs["image_layout"] = "NCHW_uint8"

        # === 标量与低维 ===
        arr_action = zarr_create_timeseries(data_grp, "action", (action_dim,), np.float32, chunks_t=1024)
        arr_state  = zarr_create_timeseries(data_grp, "state",  (state_dim,),  np.float32, chunks_t=1024)
        arr_qpos   = zarr_create_timeseries(data_grp, "qpos",   (qpos_dim,),   np.float32, chunks_t=1024)

        arr_arm = None
        arr_proprio = None
        if arm_dim > 0:
            # arr_arm     = zarr_create_timeseries(data_grp, "robot0_arm_qpos", (arm_dim,), np.float32, chunks_t=1024)
            arr_proprio = zarr_create_timeseries(data_grp, "proprio",         (arm_dim,), np.float32, chunks_t=1024)

        # === 相机数组（CHW uint8） ===
        cam_arrays: Dict[str, zarr.core.Array] = {}
        for cam_name, meta in camera_map.items():
            ds_name = meta["dataset"]  # e.g., "corner"
            cam_arrays[cam_name] = zarr_create_timeseries(
                data_grp, ds_name, (3, H, W), np.uint8, chunks_t=1
            )
            cam_arrays[cam_name].attrs["source_camera_name"] = cam_name
            cam_arrays[cam_name].attrs["flip_vertical"] = bool(CAMERA_FIP)
            cam_arrays[cam_name].attrs["color"] = "rgb"
            cam_arrays[cam_name].attrs["layout"] = "CHW"

        # episode 结束索引
        arr_ep_ends = zarr_create_1d(meta_grp, "episode_ends", np.int64)

        # 4) 采集循环：把每步 append 到各数组，并在 episode 末尾追加 end 索引
        total_steps = 0
        successes = 0

        while successes < EPISODES_NUMBER:
            obs, _ = env.reset(seed=SEED)
            obs = sanitize_obs(obs, env.observation_space)
            step_count = 0

            while step_count < MAX_EPISODE_STEPS:
                # 渲染所有相机 (NHWC uint8)，此时已按 CAMERA_FIP 处理
                multiview = get_multiview_images(env, active_cams)

                # 关节
                model, data = resolve_model_data(env)
                qpos_now = data.qpos.copy().astype(np.float32)
                arm_now = split_arm_qpos(model, qpos_now) if arm_dim > 0 else None

                # 动作（expert policy）
                act = policy.get_action(obs).astype(np.float32)

                # ---- 写状态/关节/动作 ----
                zarr_append(arr_state,  obs.reshape(1, -1).astype(np.float32))
                zarr_append(arr_qpos,   qpos_now.reshape(1, -1))
                if arr_arm is not None:
                    zarr_append(arr_arm,     arm_now.reshape(1, -1))
                if arr_proprio is not None:
                    zarr_append(arr_proprio, arm_now.reshape(1, -1))

                # ---- 写相机帧（转 CHW uint8）----
                for cam_name in active_cams:
                    img_nhwc = multiview[cam_name]               # (H,W,3) uint8
                    assert img_nhwc.dtype == np.uint8 and img_nhwc.ndim == 3
                    img_chw = np.ascontiguousarray(img_nhwc.transpose(2, 0, 1))  # -> (3,H,W) uint8
                    zarr_append(cam_arrays[cam_name], img_chw[None, ...])

                # ---- 动作 ----
                zarr_append(arr_action, act.reshape(1, -1))

                # 环境前进一步
                obs, rew, terminated, truncated, info = env.step(act)
                obs = sanitize_obs(obs, env.observation_space)
                step_count += 1
                total_steps += 1

                done = bool(info.get("success", 0)) or bool(terminated) or bool(truncated)
                if done:
                    # 把 episode 的结束位置（exclusive 累积长度）写到 meta/episode_ends
                    zarr_append(arr_ep_ends, np.asarray([total_steps], dtype=np.int64))
                    if info.get("success", 0):
                        successes += 1
                    break

        # 清理渲染器（更干净地释放 EGL/OSMesa）
        try:
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "mujoco_renderer"):
                renderer = env.unwrapped.mujoco_renderer
                if hasattr(renderer, "close"):
                    renderer.close()
        except Exception:
            pass
        env.close()

        print(f"Saved {successes} episodes, {total_steps} steps -> {zarr_path}")
        print("Keys under /data:", list(data_grp.array_keys()))
        print("Episode ends:", arr_ep_ends[:])
    print("\nAll tasks done.")


if __name__ == "__main__":
    task_ = get_task_discriptions()
    task_env_dis = get_task_name_and_desc(task_)
    collect_data_to_zarr(task_env_dis, save_root=SAVE_ROOT)