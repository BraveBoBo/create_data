# import os
# import pickle
# import json
# from typing import Union

# import cv2
# import gymnasium as gym
# import numpy as np
# from sympy import true
# import numpy as np
# import mujoco as mj


# from Metaworld.metaworld.env_dict import (
#     ALL_V3_ENVIRONMENTS,
#     MT50_V3,
# )
# from Metaworld.metaworld.policies import ENV_POLICY_MAP

# # Set random seeds for reproducibility
# np.random.seed(42)

# os.environ['MUJOCO_GL'] = 'egl'

# # Configurations
# debug = False
# RANDOM_DIS = True
# RENDER_MODE = "rgb_array"
# CAMERA_NAMES = [
#     "corner", "topview", "behindGripper", "gripperPOV",
#     "corner2", "corner3", "corner4"
# ]
# CAMERA_FIP = False
# SEED = 42
# EPISODES_NUMBER = 100
# MAX_EPISODE_STEPS = 500

# SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3"
# DEBUG_DIR = "/home/libo/project/cm/trunck-consistency-policy/create_data/datademo"
# os.makedirs(DEBUG_DIR, exist_ok=True)


# def store_rendered_image(images, img):
#     """Append rendered image to list, flipping if required."""
#     if CAMERA_FIP:
#         img = img[::-1]
#     images.append(img)
#     return images


# def get_multiview_images(env, camera_names):
#     """Get images from multiple camera angles using camera_id manipulation."""
#     multiview_images = {}
#     for i, camera_name in enumerate(camera_names):
#         if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'mujoco_renderer'):
#             renderer = env.unwrapped.mujoco_renderer
#             if hasattr(renderer, 'camera_id'):
#                 original_camera_id = renderer.camera_id
#                 renderer.camera_id = i
#                 img = env.render()
#                 renderer.camera_id = original_camera_id
#                 if CAMERA_FIP:
#                     img = img[::-1]
#                 multiview_images[camera_name] = img
#     return multiview_images


# def get_video_writer(path, frame_shape):
#     """Return an OpenCV VideoWriter given output path and first frame shape."""
#     VIDEO_FPS = 30
#     height, width = frame_shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     return cv2.VideoWriter(path, fourcc, VIDEO_FPS, (width, height))


# def get_task_discriptions():
#     """Returns a dictionary mapping environment names to their task descriptions."""
#     with open(
#         "/home/libo/project/cm/trunck-consistency-policy/create_data/metaworld_tasks_50_v2.json",
#         "r",
#     ) as f:
#         task_descriptions = json.load(f)
#     return task_descriptions


# def get_task_name_and_desc(task_list) -> dict:
#     """Get mapping from environment name to description list."""
#     result = {}
#     for item in task_list:
#         env_name = item["env"]
#         assert env_name in MT50_V3, f"Task {env_name} not found in MT50_V3"
#         result[env_name] = item["description"]
#     return result


# def get_task_expert_policy(env_names):
#     """Return mapping from environment name to expert policy class name."""
#     policy_dict = {}
#     for env_name in env_names:
#         base = "Sawyer"
#         res = env_name.split("-")
#         for substr in res:
#             base += substr.capitalize()
#         policy_name = base + "Policy"
#         if policy_name == "SawyerPegInsertSideV3Policy":
#             policy_name = "SawyerPegInsertionSideV3Policy"
#         policy_dict[env_name] = policy_name
#     return policy_dict


# def _get_task_names(envs: Union[gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv]) -> list[str]:
#     """Get task names for a vectorized Meta-World environment."""
#     metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
#     return [
#         metaworld_cls_to_task_name[task_name]
#         for task_name in envs.get_attr("task_name")
#     ]


# def get_joint_angles(env):
#     """优先通过 SawyerMocapBase.get_env_state() 拿关节与 mocap 状态。
#     回退：从 MuJoCo sim 中读取 qpos / mocap body 的 pos + quat。
#     返回:
#         joint_state: np.ndarray
#         mocap_pos: np.ndarray shape (3,)
#         mocap_quat: np.ndarray shape (4,)
#     """
#     # 1) 优先：直接调用 get_env_state（SawyerMocapBase）
#     for cand in [env, getattr(env, "unwrapped", None)]:
#         if cand is None:
#             continue
#         if hasattr(cand, "get_env_state") and callable(cand.get_env_state):
#             state = cand.get_env_state()
#             if isinstance(state, tuple) and len(state) >= 2:
#                 joint_state, mocap_state = state[0], state[1]
#                 print(joint_state)   
#                 return np.asarray(joint_state), np.asarray(mocap_state)
#     raise AttributeError("Could not locate joint & mocap state; neither get_env_state() nor MuJoCo sim found.")


# def get_joint_info(env, split=True):
#     """获取当前关节信息（qpos/qvel），并可选按语义拆分。
    
#     返回:
#       若 split=True:
#         {
#           "qpos": np.ndarray [nq],
#           "qvel": np.ndarray [nv],
#           "arm_qpos": np.ndarray [n_arm],          # Sawyer 7 个
#           "gripper_qpos": np.ndarray [n_grip],     # 指爪关节
#           "free_qpos7": np.ndarray [k, 7],         # 每个自由体 7 维: xyz(3)+quat(4)
#           "meta": {...},                           # 关节切片与类型信息
#         }
#       若 split=False:
#         {"qpos": ..., "qvel": ...}
#     """
#     base = getattr(env, "unwrapped", env)

#     # 1) 解析到 MuJoCo 的 model / data（优先从 Gymnasium 的 MujocoRenderer）
#     renderer = getattr(base, "mujoco_renderer", None)
#     model = data = None
#     if renderer is not None:
#         # gymnasium>=1.x: renderer 直接暴露 model/data；有的版本是 renderer.sim.model/data
#         model = getattr(renderer, "model", None)
#         data  = getattr(renderer, "data",  None)

#     qpos = data.qpos.copy()
#     qvel = data.qvel.copy()
#     if not split:
#         return {"qpos": qpos, "qvel": qvel}

#     # 2) 依据 mjModel 的关节地址与类型，推断每个关节在 qpos 的切片
#     nj = model.njnt
#     adr = model.jnt_qposadr           # 每个关节在 qpos 的起始下标
#     jtype = model.jnt_type            # 关节类型枚举 (HINGE/SLIDE/BALL/FREE)
#     sizes = [int((adr[j+1] - adr[j]) if j < nj-1 else (model.nq - adr[j])) for j in range(nj)]

#     def jname(j):
#         try:
#             return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
#         except Exception:
#             return ""

#     arm, grip, free_blocks = [], [], []
#     other_hinge, ball_blocks, slide_vals = [], [], []

#     for j in range(nj):
#         sl = slice(int(adr[j]), int(adr[j]) + sizes[j])
#         jt = int(jtype[j])
#         name = jname(j).lower()

#         if jt == mj.mjtJoint.mjJNT_HINGE:
#             val = float(qpos[sl][0])
#             # 名称启发式：Sawyer 的臂关节 vs 指爪
#             if ("finger" in name) or ("grip" in name):
#                 grip.append(val)
#             elif name.startswith(("right_j", "sawyer", "arm")):
#                 arm.append(val)
#             else:
#                 other_hinge.append(val)
#         elif jt == mj.mjtJoint.mjJNT_FREE:
#             block = np.asarray(qpos[sl]).copy()   # 期望长度 7：xyz + quat(wxyz)
#             if block.shape[0] == 7:
#                 free_blocks.append(block)
#         elif jt == mj.mjtJoint.mjJNT_BALL:
#             ball_blocks.append(np.asarray(qpos[sl]).copy())   # 4: quat
#         elif jt == mj.mjtJoint.mjJNT_SLIDE:
#             slide_vals.append(float(qpos[sl][0]))

#     out = {
#         "qpos": qpos,
#         "qvel": qvel,
#         "arm_qpos": np.asarray(arm, dtype=np.float32),
#         "gripper_qpos": np.asarray(grip, dtype=np.float32),
#         "free_qpos7": np.stack(free_blocks, axis=0).astype(np.float32) if free_blocks else np.zeros((0, 7), np.float32),
#         "meta": {
#             "other_hinge": np.asarray(other_hinge, dtype=np.float32),
#             "ball_qpos4": np.stack(ball_blocks, axis=0).astype(np.float32) if ball_blocks else np.zeros((0, 4), np.float32),
#             "slide_qpos": np.asarray(slide_vals, dtype=np.float32),
#         },
#     }
#     return out


# task_ = get_task_discriptions()
# task_env_dis = get_task_name_and_desc(task_)


# def collect_data(task_env_dis, save_path_root=SAVE_ROOT):
#     """Collect data for each environment in task_env_dis and save to path."""
#     if not os.path.exists(save_path_root):
#         os.makedirs(save_path_root)

#     for env_name, task_descrps in task_env_dis.items():
#         if RANDOM_DIS:
#             task_description = task_descrps[np.random.randint(len(task_descrps))]

#         env = gym.make(
#             'Meta-World/MT1',
#             env_name=env_name,
#             render_mode=RENDER_MODE,
#             camera_name=CAMERA_NAMES[0]
#         )

#         policy = ENV_POLICY_MAP[env_name]()
#         print(f"Environment: {env_name}")

#         task_path = os.path.join(save_path_root, env_name)
#         if not os.path.exists(task_path):
#             os.makedirs(task_path)

#         suc_episode = 0

#         while suc_episode < EPISODES_NUMBER:
#             done = False
#             step = 0

#             obss = []
#             acts = []
#             rews = []
#             dones = []
#             images = []
#             multiview_images = []
#             joint_angles = []  # NEW: store joint angles

#             if debug:
#                 video_writers = {}
#                 for camera_name in CAMERA_NAMES:
#                     video_path = os.path.join(
#                         DEBUG_DIR,
#                         f"{env_name}_episode_{suc_episode:04d}_{camera_name}.mp4"
#                     )
#                     video_writers[camera_name] = None

#             obs, info = env.reset(seed=SEED)

#             while not done and step < MAX_EPISODE_STEPS:
#                 obss.append(obs.copy())

#                 multiview_imgs = get_multiview_images(env, CAMERA_NAMES)
#                 multiview_images.append(multiview_imgs)

#                 main_img = list(multiview_imgs.values())[0] if multiview_imgs else env.render()
#                 images = store_rendered_image(images, main_img)
                
#                 joint_angles.append(get_joint_info(env, split=True)["arm_qpos"])   # 手臂 7 关节
                
#                 # if debug:
#                 #     for camera_name, img in multiview_imgs.items():
#                 #         if video_writers[camera_name] is None:
#                 #             video_path = os.path.join(
#                 #                 DEBUG_DIR,
#                 #                 f"{env_name}_episode_{suc_episode:04d}_{camera_name}.mp4"
#                 #             )
#                 #             video_writers[camera_name] = get_video_writer(video_path, img.shape)
#                 #         video_writers[camera_name].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#                 a = policy.get_action(obs)
#                 acts.append(a.copy())

#                 obs, rew, _, _, info = env.step(a)

#                 rews.append(rew)
#                 done = int(info["success"]) == 1
#                 dones.append(done)
#                 step += 1

#                 if done:
#                     obss.append(obs.copy())
#                     joint_angles.append(get_joint_angles(env))  # final step qpos
#                     multiview_imgs = get_multiview_images(env, CAMERA_NAMES)
#                     multiview_images.append(multiview_imgs)

#                     main_img = list(multiview_imgs.values())[0] if multiview_imgs else env.render()
#                     images = store_rendered_image(images, main_img)

#                     if debug:
#                         for camera_name, img in multiview_imgs.items():
#                             if video_writers[camera_name] is not None:
#                                 video_writers[camera_name].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#                     suc_episode += 1

#             obss = np.array(obss)
#             acts = np.array(acts)
#             rews = np.array(rews)
#             images = np.array(images)
#             dones = np.array(dones)
#             joint_angles.append(get_joint_info(env, split=True)["arm_qpos"])   # 手臂 7 关节

#             episode_data = {
#                 "observations": obss,
#                 "actions": acts,
#                 "rewards": rews,
#                 "dones": dones,
#                 "images": images,
#                 "multiview_images": multiview_images,
#                 "joint_angles": joint_angles,  # NEW
#                 "description": task_description,
#             }
#             episode_file = os.path.join(task_path, f"episode_{suc_episode:04d}.pkl")
#             with open(episode_file, "wb") as pf:
#                 pickle.dump(episode_data, pf)

#             if debug:
#                 for camera_name, writer in video_writers.items():
#                     if writer is not None and writer.isOpened():
#                         writer.release()

#         env.close()


# if __name__ == "__main__":
#     collect_data(task_env_dis, save_path_root=SAVE_ROOT)


import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import gymnasium as gym
import mujoco as mj
import numpy as np
import zarr

from Metaworld.metaworld.env_dict import MT50_V3
from Metaworld.metaworld.policies import ENV_POLICY_MAP

# ---------------------
# 全局配置
# ---------------------
os.environ["MUJOCO_GL"] = "egl"
np.random.seed(42)

DEBUG = False                      # 不再写视频，保留开关但不使用
RANDOM_DIS = True
RENDER_MODE = "rgb_array"
CAMERA_NAMES = ["corner", "topview", "behindGripper", "gripperPOV", "corner2", "corner3", "corner4"]
CAMERA_FIP = False
SEED = 42
EPISODES_NUMBER = 1
MAX_EPISODE_STEPS = 500

SAVE_ROOT = "/data/robot_dataset/metaworld/mt50_v3_zarr"
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
    """将 obs 转成 float32、去 NaN/Inf，并在 Box 有界时裁剪到 [low, high]。"""
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
    """按相机名优先渲染，失败时回退按 camera_id。"""
    multiview_images = {}
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "mujoco_renderer"):
        renderer = env.unwrapped.mujoco_renderer

        # 1) 优先按名字渲染
        if hasattr(renderer, "render"):
            for name in camera_names:
                try:
                    img = renderer.render(camera_name=name)
                    if CAMERA_FIP:
                        img = img[::-1]
                    multiview_images[name] = img
                except Exception:
                    pass

        # 2) 缺失的回退到 camera_id
        missing = [n for n in camera_names if n not in multiview_images]
        if missing and hasattr(renderer, "camera_id"):
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


def split_arm_qpos(model: mj.MjModel, qpos: np.ndarray) -> np.ndarray:
    """从 qpos 中按名称启发式抽取手臂 HINGE 关节（Sawyer 常见 7 个）。
    如果名称不匹配，将返回空数组（不会影响写盘）。
    """
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
    assert arr.ndim == data_np.ndim
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
        action (N, Da) float32
        state  (N, Ds) float32              # 直接存 env obs
        qpos   (N, nq) float32              # MuJoCo 全量 qpos
        robot0_arm_qpos (N, n_arm) float32  # 可选：名称启发式
        camera{i}_rgb (N, H, W, 3) uint8    # 有多少可用相机就写多少
      meta/
        episode_ends (E,) int64             # 每个 episode 的 end（exclusive，下标累计）
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
            camera_name=CAMERA_NAMES[0]
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
            multiview = {CAMERA_NAMES[0]: img}
            active_cams = [CAMERA_NAMES[0]]

        H, W, C = list(multiview.values())[0].shape
        state_dim = int(np.asarray(obs).shape[0])
        action_dim = int(env.action_space.shape[0])

        # MuJoCo 句柄（用于 qpos）
        model, data = resolve_model_data(env)
        qpos_dim = int(data.qpos.shape[0])
        arm_dim = int(split_arm_qpos(model, data.qpos).shape[0])  # 可能为 0

        # 3) 为该 env 创建 zarr 目录与数组
        zarr_path = os.path.join(save_root, f"{env_name}.zarr")
        if os.path.exists(zarr_path):
            print(f"  -> Remove existing: {zarr_path}")
            import shutil
            shutil.rmtree(zarr_path)

        root, data_grp, meta_grp = create_or_open_zarr(zarr_path)
        root.attrs["env_name"] = env_name
        root.attrs["description"] = task_description

        arr_action = zarr_create_timeseries(data_grp, "action", (action_dim,), np.float32, chunks_t=1024)
        arr_state = zarr_create_timeseries(data_grp, "state", (state_dim,), np.float32, chunks_t=1024)
        arr_qpos = zarr_create_timeseries(data_grp, "qpos", (qpos_dim,), np.float32, chunks_t=1024)
        arr_arm = None
        if arm_dim > 0:
            arr_arm = zarr_create_timeseries(data_grp, "robot0_arm_qpos", (arm_dim,), np.float32, chunks_t=1024)

        # 每个有效相机创建一个数组，命名 camera{i}_rgb，按 active_cams 的索引
        cam_arrays: Dict[str, zarr.core.Array] = {}
        for i, cam_name in enumerate(active_cams):
            ds_name = f"camera{i}_rgb"
            cam_arrays[cam_name] = zarr_create_timeseries(
                data_grp, ds_name, (H, W, C), np.uint8, chunks_t=1
            )
            cam_arrays[cam_name].attrs["source_camera_name"] = cam_name

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
                # 渲染所有相机
                multiview = get_multiview_images(env, active_cams)

                # 关节
                model, data = resolve_model_data(env)
                qpos_now = data.qpos.copy().astype(np.float32)
                arm_now = split_arm_qpos(model, qpos_now) if arm_dim > 0 else None

                # 动作（用 expert policy）
                act = policy.get_action(obs).astype(np.float32)

                # 先写当前步的观测（state/qpos/arm/img），动作
                zarr_append(arr_state, obs.reshape(1, -1).astype(np.float32))
                zarr_append(arr_qpos, qpos_now.reshape(1, -1))
                if arr_arm is not None:
                    zarr_append(arr_arm, arm_now.reshape(1, -1))

                for cam_name in active_cams:
                    img = multiview[cam_name]
                    zarr_append(cam_arrays[cam_name], img[None, ...])

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
