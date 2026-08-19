#!/usr/bin/env python3
"""
Canonical D405 + D455 + Piper follower RAW recorder for this FP3 pipeline.

Purpose:
- Record D405 RGB+Depth
- Record D455 RGB+Depth
- Record Piper follower joint / EE / gripper trajectory
- Preserve camera intrinsics, distortion, depth scale and timestamps
- DO NOT require camera extrinsic calibration yet
- DO NOT create fake FP3 actions
- DO NOT command or enable the robot

Later workflow:
raw RGB+Depth + follower trajectory
 -> camera extrinsic calibration
 -> base-frame XYZRGB point clouds
 -> future follower trajectory -> 10D FP3 actions
 -> FP3 HDF5
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


DEFAULT_D405_SERIAL = os.environ.get("PIPER_FP3_D405_SERIAL", "")
DEFAULT_D455_SERIAL = os.environ.get("PIPER_FP3_D455_SERIAL", "")

def pose_xyz_rpy_to_matrix(pose):
    """pyAgxArm flange pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    x, y, z, roll, pitch, yaw = [float(v) for v in pose]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


class AgxCandoPiperReader:
    """Read-only Piper follower state through the working Windows agx_cando backend.

    This class never calls enable(), move_j(), move_p(), move_l(), move_gripper_m(),
    reset(), or any other robot motion method.
    """

    def __init__(self, channel="0", bitrate=1_000_000):
        self.channel = str(channel)
        self.bitrate = int(bitrate)
        self.robot = None
        self.effector = None
        self.last_joint_timestamp_s = None
        self.last_flange_timestamp_s = None

    def start(self):
        cfg = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=PiperFW.V189,
            interface="agx_cando",
            channel=self.channel,
            bitrate=self.bitrate,
            auto_connect=False,
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        self.effector = self.robot.init_effector(
            self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER
        )
        self.robot.connect()
        time.sleep(2.0)

        if not self.robot.is_ok():
            raise RuntimeError(
                "Piper agx_cando connected, but robot.is_ok() is False. "
                "Close ArmRobotUI and check channel 0 / 1 Mbps."
            )
        if self.robot.has_comm_error():
            raise RuntimeError(
                f"Piper communication error: {self.robot.get_comm_error()}"
            )
        return self

    @staticmethod
    def _timestamp(obj, fallback):
        value = getattr(obj, "timestamp", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return value if np.isfinite(value) else float(fallback)

    def read(self):
        if self.robot is None:
            raise RuntimeError("Piper reader is not started")

        joints_ret = self.robot.get_joint_angles()
        flange_ret = self.robot.get_flange_pose()
        gripper_ret = (
            None if self.effector is None else self.effector.get_gripper_status()
        )

        if joints_ret is None or getattr(joints_ret, "msg", None) is None:
            raise RuntimeError("No fresh Piper joint feedback")
        if flange_ret is None or getattr(flange_ret, "msg", None) is None:
            raise RuntimeError("No fresh Piper flange pose feedback")
        if gripper_ret is None or getattr(gripper_ret, "msg", None) is None:
            raise RuntimeError("No fresh Piper gripper feedback")

        q = np.asarray(joints_ret.msg, dtype=np.float64).reshape(-1)
        ee = np.asarray(flange_ret.msg, dtype=np.float64).reshape(-1)
        if q.size < 6 or not np.isfinite(q[:6]).all():
            raise RuntimeError("Invalid Piper joint feedback")
        if ee.size < 6 or not np.isfinite(ee[:6]).all():
            raise RuntimeError("Invalid Piper flange pose feedback")

        grip_value = getattr(gripper_ret.msg, "value", None)
        try:
            grip = float(grip_value)
        except (TypeError, ValueError):
            raise RuntimeError("Invalid Piper gripper feedback")
        if not np.isfinite(grip):
            raise RuntimeError("Invalid Piper gripper feedback")

        host_now = time.time()
        joint_ts = self._timestamp(joints_ret, host_now)
        flange_ts = self._timestamp(flange_ret, host_now)
        gripper_ts = self._timestamp(gripper_ret, host_now)

        # If SDK timestamps are available, require them to advance. If the SDK object
        # lacks timestamps, _timestamp falls back to host time, which is monotonic enough
        # for this read-only calibration capture loop.
        if (
            self.last_joint_timestamp_s is not None
            and joint_ts <= self.last_joint_timestamp_s
        ):
            raise RuntimeError("No fresh Piper joint feedback")
        if (
            self.last_flange_timestamp_s is not None
            and flange_ts <= self.last_flange_timestamp_s
        ):
            raise RuntimeError("No fresh Piper flange pose feedback")

        self.last_joint_timestamp_s = joint_ts
        self.last_flange_timestamp_s = flange_ts

        ee6 = ee[:6].astype(np.float64)
        return {
            "joint_position_rad": q[:6].astype(np.float64),
            "ee_pose_xyz_rpy": ee6,
            "T_base_from_EE": pose_xyz_rpy_to_matrix(ee6),
            "gripper_position_m": grip,
            "joint_timestamp_s": joint_ts,
            "gripper_timestamp_s": gripper_ts,
        }

    def close(self):
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
                self.effector = None


class RealSenseCamera:
    def __init__(self, serial: str, role: str, width=640, height=480, fps=30):
        self.serial = serial
        self.role = role
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        dev = self.profile.get_device()
        found_serial = dev.get_info(rs.camera_info.serial_number)
        if found_serial != self.serial:
            raise RuntimeError(f"{self.role}: expected serial {self.serial}, got {found_serial}")
        self.depth_scale = float(dev.first_depth_sensor().get_depth_scale())
        for _ in range(30):
            self.pipeline.wait_for_frames()
        return self

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            raise RuntimeError(f"{self.role}: missing aligned depth/color frame")

        color_np = np.asanyarray(color.get_data()).copy()
        depth_np = np.asanyarray(depth.get_data()).copy()

        color_intr = color.profile.as_video_stream_profile().intrinsics
        depth_intr = depth.profile.as_video_stream_profile().intrinsics

        return {
            "color_bgr": color_np,
            "depth_raw": depth_np,
            "color_timestamp_ms": float(color.get_timestamp()),
            "depth_timestamp_ms": float(depth.get_timestamp()),
            "color_intrinsics": color_intr,
            "depth_intrinsics": depth_intr,
        }

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def intrinsics_matrix(intr):
    return np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def distortion_coeffs(intr):
    return np.asarray(list(intr.coeffs), dtype=np.float64)


def append(ds, value):
    n = ds.shape[0]
    ds.resize((n + 1,) + ds.shape[1:])
    ds[n] = value


def create_dataset(group, name, tail_shape, dtype, compression=None):
    kwargs = {}
    if compression:
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = 1
    return group.create_dataset(
        name,
        shape=(0,) + tuple(tail_shape),
        maxshape=(None,) + tuple(tail_shape),
        chunks=(1,) + tuple(tail_shape),
        dtype=dtype,
        **kwargs,
    )


def state_value(state, key, fallback=None):
    if key in state:
        return state[key]
    return fallback



def read_piper_with_retry(piper, timeout_s=1.0):
    """Return one fresh Piper sample, tolerating short startup/transient gaps."""
    deadline = time.monotonic() + timeout_s
    last_exc = None
    while time.monotonic() < deadline:
        try:
            return piper.read()
        except RuntimeError as exc:
            last_exc = exc
            if "No fresh Piper" not in str(exc):
                raise
            time.sleep(0.02)
    raise RuntimeError(
        f"Piper feedback was not fresh for {timeout_s:.1f}s; "
        f"last error: {last_exc}"
    )


def warmup_piper(piper, timeout_s=10.0, required_consecutive=10):
    """Wait until follower feedback is stably fresh before recording starts."""
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    total_good = 0
    last_exc = None

    print("Waiting for fresh Piper follower feedback...")
    while time.monotonic() < deadline:
        try:
            s = piper.read()
            total_good += 1
            consecutive += 1
            if consecutive >= required_consecutive:
                print(
                    f"PIPER PRECHECK PASS: {consecutive} consecutive fresh samples "
                    f"(total_good={total_good})"
                )
                return s
        except RuntimeError as exc:
            last_exc = exc
            consecutive = 0
            if "No fresh Piper" not in str(exc):
                raise
        time.sleep(0.02)

    raise RuntimeError(
        "Piper follower feedback did not become stably fresh within "
        f"{timeout_s:.1f}s. Last error: {last_exc}. "
        "Run the existing recorder --test-piper --action-source feedback first."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="e.g. episode_000001")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "fp3_raw_uncalibrated",
    )
    ap.add_argument("--duration", type=float, default=30.0, help="seconds; 0 = until Ctrl+C")
    ap.add_argument("--record-fps", type=float, default=10.0)
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--instruction", default="Fold the towel from right to left.")
    ap.add_argument("--d405-serial", default=DEFAULT_D405_SERIAL)
    ap.add_argument("--d455-serial", default=DEFAULT_D455_SERIAL)
    args = ap.parse_args()

    if args.record_fps <= 0:
        raise SystemExit("--record-fps must be > 0")
    if not args.d405_serial:
        raise SystemExit("--d405-serial is required")
    if not args.d455_serial:
        raise SystemExit("--d455-serial is required")
    episode_dir = args.output / args.episode
    if episode_dir.exists():
        raise SystemExit(f"Episode already exists: {episode_dir}")
    episode_dir.mkdir(parents=True, exist_ok=False)
    h5_path = episode_dir / "frames.h5"
    metadata_path = episode_dir / "metadata.json"

    d405 = RealSenseCamera(args.d405_serial, "D405_wrist").start()
    d455 = RealSenseCamera(args.d455_serial, "D455_external").start()

    # Read-only Piper follower feedback through the verified Windows agx_cando backend.
    piper = AgxCandoPiperReader(channel="0", bitrate=1_000_000).start()
    warmup_piper(piper, timeout_s=10.0, required_consecutive=10)

    first405 = d405.read()
    first455 = d455.read()

    print("RAW-ONLY FP3 SOURCE CAPTURE")
    print("No calibration is written. No fake action is written.")
    print("Robot path is read-only follower feedback via agx_cando.")
    print("D405:", args.d405_serial, "depth_scale=", d405.depth_scale)
    print("D455:", args.d455_serial, "depth_scale=", d455.depth_scale)
    print("Output:", episode_dir)
    print(f"Recording starts in {args.countdown} s...")
    for i in range(args.countdown, 0, -1):
        print(i)
        time.sleep(1)

    start_unix_ns = time.time_ns()
    start_mono = time.monotonic()
    sample_count = 0
    complete = False

    try:
        with h5py.File(h5_path, "w") as f:
            f.attrs["format"] = "piper_fp3_raw_uncalibrated_v1"
            f.attrs["complete"] = False
            f.attrs["instruction"] = args.instruction
            f.attrs["calibration_complete"] = False
            f.attrs["action_complete"] = False
            f.attrs["action_source"] = "follower_future_trajectory_pending"
            f.attrs["observation_source"] = "follower_feedback_agx_cando"

            cams = f.create_group("cameras")
            robot = f.create_group("robot")
            ts = f.create_group("timestamps")

            cam_datasets = {}
            for role, serial, cam, first in [
                ("D405", args.d405_serial, d405, first405),
                ("D455", args.d455_serial, d455, first455),
            ]:
                g = cams.create_group(role)
                g.attrs["serial"] = serial
                g.attrs["role"] = "wrist" if role == "D405" else "external"
                g.attrs["depth_scale_m_per_unit"] = float(cam.depth_scale)
                g.attrs["distortion_model"] = str(first["color_intrinsics"].model)
                g.create_dataset(
                    "intrinsics",
                    data=intrinsics_matrix(first["color_intrinsics"]),
                    dtype=np.float64,
                )
                g.create_dataset(
                    "distortion",
                    data=distortion_coeffs(first["color_intrinsics"]),
                    dtype=np.float64,
                )
                cam_datasets[role] = {
                    "color_rgb": create_dataset(
                        g, "color_rgb", (cam.height, cam.width, 3), np.uint8, "gzip"
                    ),
                    "depth_raw": create_dataset(
                        g, "depth_raw", (cam.height, cam.width), np.uint16, "gzip"
                    ),
                    "color_timestamp_ms": create_dataset(
                        g, "color_timestamp_ms", (), np.float64
                    ),
                    "depth_timestamp_ms": create_dataset(
                        g, "depth_timestamp_ms", (), np.float64
                    ),
                    "depth_valid_ratio": create_dataset(
                        g, "depth_valid_ratio", (), np.float32
                    ),
                }

            rds = {
                "joint_position_rad": create_dataset(robot, "joint_position_rad", (6,), np.float64),
                "ee_pose_xyz_rpy": create_dataset(robot, "ee_pose_xyz_rpy", (6,), np.float64),
                "T_base_from_EE": create_dataset(robot, "T_base_from_EE", (4, 4), np.float64),
                "gripper_position_m": create_dataset(robot, "gripper_position_m", (), np.float64),
            }
            tds = {
                "host_monotonic_ns": create_dataset(ts, "host_monotonic_ns", (), np.int64),
                "host_unix_ns": create_dataset(ts, "host_unix_ns", (), np.int64),
                "joint_timestamp_s": create_dataset(ts, "joint_timestamp_s", (), np.float64),
                "gripper_timestamp_s": create_dataset(ts, "gripper_timestamp_s", (), np.float64),
            }

            period = 1.0 / args.record_fps
            # Reset capture timer only after HDF5 datasets are ready.
            start_mono = time.monotonic()
            next_tick = start_mono
            while True:
                now = time.monotonic()
                if args.duration > 0 and now - start_mono >= args.duration:
                    break
                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.002))
                    continue

                s = read_piper_with_retry(piper, timeout_s=1.0)
                a = d405.read()
                b = d455.read()

                q = np.asarray(s["joint_position_rad"], dtype=np.float64)
                ee = np.asarray(s["ee_pose_xyz_rpy"], dtype=np.float64)
                T = np.asarray(s["T_base_from_EE"], dtype=np.float64)
                grip = float(s["gripper_position_m"])

                if q.shape != (6,) or ee.shape != (6,) or T.shape != (4, 4):
                    raise RuntimeError("Unexpected Piper follower state shape")

                host_mono_ns = time.monotonic_ns()
                host_unix_ns = time.time_ns()

                append(rds["joint_position_rad"], q)
                append(rds["ee_pose_xyz_rpy"], ee)
                append(rds["T_base_from_EE"], T)
                append(rds["gripper_position_m"], grip)
                append(tds["host_monotonic_ns"], host_mono_ns)
                append(tds["host_unix_ns"], host_unix_ns)
                append(tds["joint_timestamp_s"], float(state_value(s, "joint_timestamp_s", np.nan)))
                append(tds["gripper_timestamp_s"], float(state_value(s, "gripper_timestamp_s", np.nan)))

                for role, frame in [("D405", a), ("D455", b)]:
                    rgb = cv2.cvtColor(frame["color_bgr"], cv2.COLOR_BGR2RGB)
                    depth = frame["depth_raw"]
                    append(cam_datasets[role]["color_rgb"], rgb)
                    append(cam_datasets[role]["depth_raw"], depth)
                    append(
                        cam_datasets[role]["color_timestamp_ms"],
                        frame["color_timestamp_ms"],
                    )
                    append(
                        cam_datasets[role]["depth_timestamp_ms"],
                        frame["depth_timestamp_ms"],
                    )
                    append(
                        cam_datasets[role]["depth_valid_ratio"],
                        np.float32((depth > 0).mean()),
                    )

                sample_count += 1
                if sample_count % max(1, int(args.record_fps)) == 0:
                    elapsed = time.monotonic() - start_mono
                    print(
                        f"{elapsed:6.1f}s  frames={sample_count:4d}  "
                        f"D405 depth={float((a['depth_raw']>0).mean())*100:5.1f}%  "
                        f"D455 depth={float((b['depth_raw']>0).mean())*100:5.1f}%"
                    )

                next_tick += period

            f.attrs["num_frames"] = sample_count
            f.attrs["complete"] = sample_count >= 2
            complete = sample_count >= 2

    except KeyboardInterrupt:
        print("\nStopped by user. Closing files safely...")
        complete = sample_count >= 2
        # HDF5 context has already closed while unwinding.
        try:
            with h5py.File(h5_path, "a") as f:
                f.attrs["num_frames"] = sample_count
                f.attrs["complete"] = complete
        except Exception:
            pass
    finally:
        try:
            piper.close()
        except Exception:
            pass
        d405.stop()
        d455.stop()
        cv2.destroyAllWindows()

    metadata = {
        "format": "piper_fp3_raw_uncalibrated_v1",
        "complete": bool(complete),
        "num_frames": int(sample_count),
        "instruction": args.instruction,
        "record_fps": float(args.record_fps),
        "camera_roles": {"D405": "wrist", "D455": "external"},
        "camera_serials": {"D405": args.d405_serial, "D455": args.d455_serial},
        "calibration_complete": False,
        "action_complete": False,
        "action_source": "follower_future_trajectory_pending",
        "observation_source": "follower_feedback_agx_cando",
        "start_unix_ns": start_unix_ns,
        "notes": (
            "RAW source episode only. Keep D405 mount, D455 pose and Piper base fixed. "
            "Before FP3 fine-tuning: solve camera extrinsics, generate base-frame XYZRGB "
            "point clouds, construct future follower trajectory 10D actions, then convert "
            "to validated FP3 HDF5."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nRAW EPISODE COMPLETE")
    print("frames:", sample_count)
    print("folder:", episode_dir)
    print("IMPORTANT: this is not yet final FP3 HDF5; do not train on it before calibration/action conversion.")


if __name__ == "__main__":
    main()
