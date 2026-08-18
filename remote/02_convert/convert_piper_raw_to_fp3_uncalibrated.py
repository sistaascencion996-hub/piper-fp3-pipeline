#!/usr/bin/env python3
"""
Convert one Piper RAW episode into an FP3-style trajectory HDF5.

IMPORTANT:
- Experimental UNCALIBRATED mode.
- D405 and D455 XYZ remain in their own camera coordinate frames.
- No fake camera extrinsics are written.
- Uses 8000 XYZRGB points per camera per timestep, matching the released FP3 checkpoint shape.
- Action[t] is constructed from follower feedback at t+1 so the training sequence represents future executed motion.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import robomimic.utils.torch_utils as TorchUtils
from openpoints.models.layers import furthest_point_sample


HAND_KEY = "hand_camera_left_pcd_4000"
VARIED_KEY = "varied_camera_2_left_pcd_4000"


def make_xyzrgb(depth_raw, rgb, K, depth_scale, npoints, candidate_cap, seed):
    h, w = depth_raw.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    chosen = None
    candidate_count = 0

    for stride in (4, 2, 1):
        z = depth_raw[::stride, ::stride].astype(np.float32) * np.float32(depth_scale)
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        uu = uu.astype(np.float32)
        vv = vv.astype(np.float32)

        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        valid = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
            (z > 0.0) &
            (x >= -1.0) & (x <= 1.0) &
            (y >= -1.0) & (y <= 1.0) &
            (z >= -1.0) & (z <= 1.0)
        )

        xyz = np.stack([x[valid], y[valid], z[valid]], axis=1)
        colors = rgb[::stride, ::stride][valid].astype(np.float32) / 255.0
        candidate_count = len(xyz)

        if candidate_count >= npoints:
            chosen = np.concatenate([xyz, colors], axis=1)
            break

    if chosen is None:
        raise RuntimeError(
            f"Only {candidate_count} valid camera-frame points remain inside [-1,1]; "
            f"need at least {npoints}."
        )

    rng = np.random.default_rng(seed)
    if len(chosen) > candidate_cap:
        ids = rng.choice(len(chosen), candidate_cap, replace=False)
        chosen = chosen[ids]

    if len(chosen) < npoints:
        raise RuntimeError(f"candidate_cap / crop left only {len(chosen)} points")

    pts = torch.from_numpy(chosen).to(device="cuda", dtype=torch.float32).unsqueeze(0)
    idx = furthest_point_sample(pts[..., :3].contiguous(), npoints)
    sampled = torch.gather(
        pts, 1, idx.unsqueeze(-1).long().expand(-1, -1, pts.shape[-1])
    )[0]
    return sampled.cpu().numpy().astype(np.float16), candidate_count


def create_dataset(group, name, data, dtype=None):
    if dtype is not None:
        data = data.astype(dtype)
    group.create_dataset(name, data=data, compression="gzip", compression_opts=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path.home() / "fp3_piper_data/fp3_raw_uncalibrated")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "fp3_piper_data/fp3_h5_uncalibrated")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--npoints", type=int, default=8000)
    ap.add_argument("--candidate-cap", type=int, default=12000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; activate the FP3 environment first.")

    ep_name = f"episode_{args.episode:06d}"
    src = args.raw_root / ep_name / "frames.h5"
    out_dir = args.out_root / ep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "trajectory_pcd.h5"

    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {dst}")

    print("SOURCE:", src)
    print("OUTPUT:", dst)
    print("MODE: EXPERIMENTAL_UNCALIBRATED_CAMERA_FRAMES")
    print("POINTS PER CAMERA PER FRAME:", args.npoints)

    with h5py.File(src, "r") as f:
        ee = f["robot/ee_pose_xyz_rpy"][:].astype(np.float64)
        grip = f["robot/gripper_position_m"][:].astype(np.float64).reshape(-1, 1)

        T = len(ee)
        if T < 17:
            raise RuntimeError(f"Episode is too short: T={T}")

        target_pose = np.concatenate([ee[1:], ee[-1:]], axis=0)
        target_grip = np.concatenate([grip[1:], grip[-1:]], axis=0)

        abs_pos = target_pose[:, :3]
        euler = torch.from_numpy(target_pose[:, 3:6])
        abs_rot_6d = (
            TorchUtils.euler_angles_to_rot_6d(euler, convention="XYZ")
            .cpu().numpy().astype(np.float64)
        )

        K405 = f["cameras/D405/intrinsics"][:]
        K455 = f["cameras/D455/intrinsics"][:]
        s405 = float(f["cameras/D405"].attrs["depth_scale_m_per_unit"])
        s455 = float(f["cameras/D455"].attrs["depth_scale_m_per_unit"])

        p405 = np.empty((T, args.npoints, 6), dtype=np.float16)
        p455 = np.empty((T, args.npoints, 6), dtype=np.float16)

        min405 = 10**9
        min455 = 10**9

        for i in tqdm(range(T), desc=f"Converting {ep_name}"):
            rgb405 = f["cameras/D405/color_rgb"][i]
            dep405 = f["cameras/D405/depth_raw"][i]
            rgb455 = f["cameras/D455/color_rgb"][i]
            dep455 = f["cameras/D455/depth_raw"][i]

            p405[i], c405 = make_xyzrgb(
                dep405, rgb405, K405, s405,
                args.npoints, args.candidate_cap,
                seed=args.episode * 100000 + i
            )
            p455[i], c455 = make_xyzrgb(
                dep455, rgb455, K455, s455,
                args.npoints, args.candidate_cap,
                seed=args.episode * 200000 + i
            )
            min405 = min(min405, c405)
            min455 = min(min455, c455)

    with h5py.File(dst, "w") as g:
        g.attrs["source_episode"] = ep_name
        g.attrs["experimental_uncalibrated"] = True
        g.attrs["camera_coordinate_warning"] = (
            "D405 and D455 point clouds are in their own camera frames; "
            "this is not the calibrated world-frame preprocessing used by released FP3."
        )
        g.attrs["action_source"] = "follower_feedback_t_plus_1"

        obs = g.create_group("observation")
        robot = obs.create_group("robot_state")
        create_dataset(robot, "cartesian_position", ee, np.float64)
        create_dataset(robot, "gripper_position", grip, np.float64)

        cam = obs.create_group("camera")
        pcd = cam.create_group("pointcloud")
        create_dataset(pcd, HAND_KEY, p405, np.float16)
        create_dataset(pcd, VARIED_KEY, p455, np.float16)

        action = g.create_group("action")
        create_dataset(action, "abs_pos", abs_pos, np.float64)
        create_dataset(action, "abs_rot_6d", abs_rot_6d, np.float64)
        create_dataset(action, "gripper_position", target_grip, np.float64)

    print("\nCONVERSION PASS")
    print("T =", T)
    print("D405 =", p405.shape, p405.dtype, "min pre-FPS candidates =", min405)
    print("D455 =", p455.shape, p455.dtype, "min pre-FPS candidates =", min455)
    print("state/cartesian =", ee.shape)
    print("state/gripper =", grip.shape)
    print("action/abs_pos =", abs_pos.shape)
    print("action/abs_rot_6d =", abs_rot_6d.shape)
    print("action/gripper =", target_grip.shape)
    print("saved =", dst)


if __name__ == "__main__":
    main()
