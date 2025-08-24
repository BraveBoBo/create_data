#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze a Meta-World-style Zarr dataset:
- Slice by episode using /meta/episode_ends (cumulative ends)
- Plot action curves (and optionally qpos/proprio)
- Visualize RGB from selected camera(s): GIF/MP4 and a sampled grid PNG

Assumptions (matching your writer):
- Images stored in /data/<camera>: (T, 3, H, W) uint8 (CHW)
- Low-dim states in /data/{action,state,qpos,proprio}
- Episode cut points in /meta/episode_ends (1D cumulative)

Dependencies:
  pip install zarr numpy matplotlib imageio pillow imageio-ffmpeg
"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import zarr
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Optional text overlay
try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# ----------------------------- helpers -----------------------------
def chw_to_hwc(frames_chw: np.ndarray) -> np.ndarray:
    """(T, 3, H, W) or (3, H, W) -> (T, H, W, 3)"""
    assert frames_chw.ndim in (3, 4), f"expect 3D/4D, got {frames_chw.shape}"
    if frames_chw.ndim == 3:
        frames_chw = frames_chw[None, ...]
    assert frames_chw.shape[1] in (1, 3, 4), f"unexpected channel dim: {frames_chw.shape}"
    return np.transpose(frames_chw, (0, 2, 3, 1))


def get_episode_bounds(ep_ends: np.ndarray, ep_idx: int) -> Tuple[int, int]:
    """Return [start, end) for episode ep_idx given cumulative episode_ends."""
    n = int(ep_ends.size)
    ep = int(np.clip(ep_idx, 0, max(0, n - 1)))
    end = int(ep_ends[ep])
    start = 0 if ep == 0 else int(ep_ends[ep - 1])
    assert end > start, f"invalid episode range: start={start}, end={end}"
    return start, end


def pick_cameras(root: zarr.hierarchy.Group, data_grp: zarr.hierarchy.Group, cams_arg: str) -> List[str]:
    """Pick camera names to visualize."""
    keys = set(data_grp.array_keys())
    if cams_arg.strip():
        cams = [c.strip() for c in cams_arg.split(",") if c.strip() in keys]
        assert len(cams) > 0, f"no valid cameras in: {cams_arg}; available: {sorted(list(keys))}"
        return cams
    cams_attr = root.attrs.get("cameras", {})
    if isinstance(cams_attr, dict) and len(cams_attr) > 0:
        cams = [k for k in cams_attr.keys() if k in keys]
        if len(cams) > 0:
            return cams
    common = ["corner4", "gripperPOV", "corner", "topview", "behindGripper"]
    cams = [k for k in common if k in keys]
    assert len(cams) > 0, f"no camera arrays found under /data; got: {sorted(list(keys))}"
    return cams


def even_indices(total: int, count: int) -> np.ndarray:
    """Evenly sample 'count' indices from range(total)."""
    count = max(1, min(count, total))
    if count == 1:
        return np.array([total // 2], dtype=np.int64)
    return np.linspace(0, total - 1, num=count, dtype=np.int64)


def overlay_text(img_hwc: np.ndarray, text: str) -> np.ndarray:
    """Draw a small text box in the top-left; requires PIL."""
    if not _HAS_PIL:
        return img_hwc
    im = Image.fromarray(img_hwc)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 260, 26], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


def save_gif(frames_hwc: np.ndarray, out_path: str, fps: int):
    duration = 1.0 / float(fps)
    imageio.mimsave(out_path, list(frames_hwc), duration=duration, loop=0)


def save_mp4(frames_hwc: np.ndarray, out_path: str, fps: int):
    imageio.mimsave(out_path, frames_hwc, fps=fps, codec="libx264", quality=8)


def make_grid(frames: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Make a simple rows x cols grid from (N,H,W,3) frames (N >= rows*cols)."""
    n, h, w, c = frames.shape
    k = rows * cols
    assert n >= k, f"need at least {k} frames, got {n}"
    tiles = frames[:k]
    grid = []
    for r in range(rows):
        row = tiles[r*cols:(r+1)*cols]
        grid.append(np.concatenate(list(row), axis=1))
    return np.concatenate(grid, axis=0)


# ----------------------------- plotting -----------------------------
def plot_curves(ts: np.ndarray, Y: np.ndarray, title: str, y_label: str, out_path: str, legend_prefix: str):
    """Plot multi-dim time series. Y: (T, D)"""
    assert Y.ndim == 2 and ts.ndim == 1 and Y.shape[0] == ts.size, "shape mismatch for plotting"
    T, D = Y.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    for d in range(D):
        ax.plot(ts, Y[:, d], linewidth=1.3, label=f"{legend_prefix}{d}")
    ax.set_title(title)
    ax.set_xlabel("t (step)")
    ax.set_ylabel(y_label)
    if D <= 10:
        ax.legend(ncol=min(D, 5), frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze Zarr: plot actions & visualize RGB.")
    # Make positional zarr_path optional with a default
    ap.add_argument("zarr_path", type=str, nargs="?",
                    default="/data/robot_dataset/metaworld/debug_rb/basketball-v3.zarr",
                    help="Path to <env>.zarr (optional, default set)")
    ap.add_argument("--ep", type=int, default=0, help="Episode index (default 0)")
    ap.add_argument("--cams", type=str, default="", help="Camera names, comma-separated; auto if empty")
    ap.add_argument("--outdir", type=str, default="analysis_out", help="Output directory")
    ap.add_argument("--fps", type=int, default=20, help="FPS for GIF/MP4")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
    # Booleans default TRUE as requested
    ap.add_argument("--gif", action="store_true", default=True, help="Export GIF (default: True)")
    ap.add_argument("--mp4", action="store_true", default=True, help="Export MP4 (default: True)")
    ap.add_argument("--grid", type=str, default="3x3", help="Sampled grid layout, e.g., 3x3 or 4x4")
    ap.add_argument("--txt", action="store_true", default=True, help="Overlay step index text (default: True; needs PIL)")
    ap.add_argument("--plot-qpos", action="store_true", default=True, help="Also plot qpos curves (default: True)")
    ap.add_argument("--plot-proprio", action="store_true", default=True, help="Also plot proprio curves if available (default: True)")
    args = ap.parse_args()

    # open zarr & basic groups
    root = zarr.open(args.zarr_path, mode="r")
    assert "data" in root and "meta" in root, "expect /data and /meta groups"
    data_grp = root["data"]
    meta_grp = root["meta"]

    # episode slicing
    ep_ends = meta_grp["episode_ends"][:]
    assert ep_ends.ndim == 1 and ep_ends.size > 0, "meta/episode_ends is invalid"
    start, end = get_episode_bounds(ep_ends, args.ep)
    T = end - start

    # choose cameras
    cams = pick_cameras(root, data_grp, args.cams)

    # load low-dim signals (slice by episode)
    has_action = "action" in data_grp
    has_qpos = "qpos" in data_grp
    has_proprio = "proprio" in data_grp

    if has_action:
        action = data_grp["action"][start:end]           # (T, A)
        assert action.ndim == 2 and action.shape[0] == T
    if has_qpos and args.plot_qpos:
        qpos = data_grp["qpos"][start:end]               # (T, J)
        assert qpos.ndim == 2 and qpos.shape[0] == T
    if has_proprio and args.plot_proprio:
        proprio = data_grp["proprio"][start:end]         # (T, P)
        assert proprio.ndim == 2 and proprio.shape[0] == T

    # ensure output dir
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.zarr_path.rstrip("/")))[0]
    tag = f"ep{int(args.ep):04d}"

    # --------- 1) plot action (and optional qpos/proprio) ---------
    ts = np.arange(T, dtype=np.int64)

    if has_action:
        out_png = os.path.join(args.outdir, f"{base}__{tag}__action.png")
        plot_curves(ts, action, title=f"{base} | episode {args.ep} | action",
                    y_label="action value", out_path=out_png, legend_prefix="a")
        print(f"[OK] saved action plot -> {out_png}")
    else:
        print("[Info] /data/action not found; skip action plot.")

    if has_qpos and args.plot_qpos:
        out_png = os.path.join(args.outdir, f"{base}__{tag}__qpos.png")
        plot_curves(ts, qpos, title=f"{base} | episode {args.ep} | qpos",
                    y_label="qpos", out_path=out_png, legend_prefix="q")
        print(f"[OK] saved qpos plot -> {out_png}")

    if has_proprio and args.plot_proprio:
        out_png = os.path.join(args.outdir, f"{base}__{tag}__proprio.png")
        plot_curves(ts, proprio, title=f"{base} | episode {args.ep} | proprio",
                    y_label="proprio", out_path=out_png, legend_prefix="p")
        print(f"[OK] saved proprio plot -> {out_png}")

    # --------- 2) RGB visualization ---------
    main_cam = cams[0]
    frames_chw = data_grp[main_cam][start:end]           # (T, 3, H, W)
    frames = chw_to_hwc(frames_chw)                      # (T, H, W, 3)

    if args.max_frames and args.max_frames > 0:
        frames = frames[:args.max_frames]
        T = frames.shape[0]
        ts = np.arange(T, dtype=np.int64)

    if args.txt and _HAS_PIL:
        for t in range(T):
            frames[t] = overlay_text(frames[t], f"{main_cam} | t={t:04d}")

    # Export both if both flags are True
    if args.mp4:
        mp4_path = os.path.join(args.outdir, f"{base}__{tag}__{main_cam}.mp4")
        save_mp4(frames, mp4_path, args.fps)
        print(f"[OK] saved MP4 -> {mp4_path}")
    if args.gif:
        gif_path = os.path.join(args.outdir, f"{base}__{tag}__{main_cam}.gif")
        save_gif(frames, gif_path, args.fps)
        print(f"[OK] saved GIF -> {gif_path}")
    if not (args.gif or args.mp4):
        print("[Info] neither GIF nor MP4 requested; skipping video export.")

    # 2b) Sampled grid for each selected camera
    grid_rc = args.grid.lower().split("x")
    assert len(grid_rc) == 2 and grid_rc[0].isdigit() and grid_rc[1].isdigit(), f"invalid --grid: {args.grid}"
    R, C = int(grid_rc[0]), int(grid_rc[1])
    K = R * C

    for cam in cams:
        arr = data_grp[cam][start:end]                  # (T, 3, H, W)
        f = chw_to_hwc(arr)
        if args.max_frames and args.max_frames > 0:
            f = f[:args.max_frames]
        T_cam = f.shape[0]
        idx = even_indices(T_cam, K)
        samples = f[idx]
        if args.txt and _HAS_PIL:
            for i, t in enumerate(idx.tolist()):
                samples[i] = overlay_text(samples[i], f"{cam} | t={t:04d}")
        grid_img = make_grid(samples, rows=R, cols=C)
        out_grid = os.path.join(args.outdir, f"{base}__{tag}__{cam}__grid_{R}x{C}.png")
        imageio.imwrite(out_grid, grid_img)
        print(f"[OK] saved grid -> {out_grid}")

    # summary
    print(f"[Info] Zarr: {args.zarr_path}")
    print(f"[Info] episode[{args.ep}]: start={start}, end={end}, len={T}")
    print(f"[Info] cameras: {cams}")


if __name__ == "__main__":
    main()
