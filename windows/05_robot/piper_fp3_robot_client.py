#!/usr/bin/env python
"""
Windows Piper + mixed RGB-D camera client for FP3 real-robot execution.

Camera mapping used by this file:
- hand camera: Orbbec DaBai DC1
- varied/external camera: Intel RealSense D455

Important:
- Close ArmRobotUI before running.
- Keep the physical emergency stop reachable.
- Q / Esc: stop sending commands and exit.
- E: invoke Piper electronic emergency stop and exit.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any

import msvcrt
import numpy as np
import pyrealsense2 as rs

try:
    from pyorbbecsdk import (
        Config as OBConfig,
        Context as OBContext,
        OBAlignMode,
        OBError,
        OBSensorType,
        Pipeline as OBPipeline,
    )
    ORBBEC_IMPORT_ERROR = None
except Exception as exc:
    OBConfig = None
    OBContext = None
    OBAlignMode = None
    OBError = Exception
    OBSensorType = None
    OBPipeline = None
    ORBBEC_IMPORT_ERROR = exc
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

HAND_KEY = "camera/pointcloud/hand_camera_left_pcd_4000"
VARIED_KEY = "camera/pointcloud/varied_camera_2_left_pcd_4000"
CART_KEY = "robot_state/cartesian_position"
GRIPPER_STATE_KEY = "robot_state/gripper_position"


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("model server disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(sock: socket.socket) -> bytes:
    (size,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid packet size: {size}")
    return recv_exact(sock, size)


def send_packet(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"packet too large: {len(payload)}")
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    value = json.loads(recv_packet(sock).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("server response is not an object")
    return value


def list_rgbd_devices() -> None:
    print("===== Intel RealSense =====")
    devices = rs.context().query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
    else:
        for index, dev in enumerate(devices):
            print(
                f"[{index}] name={dev.get_info(rs.camera_info.name)} "
                f"serial={dev.get_info(rs.camera_info.serial_number)}"
            )

    print("\n===== Orbbec =====")
    if ORBBEC_IMPORT_ERROR is not None:
        print(
            "Orbbec Python SDK is not installed: "
            f"{type(ORBBEC_IMPORT_ERROR).__name__}: {ORBBEC_IMPORT_ERROR}"
        )
        return

    context = OBContext()
    device_list = context.query_devices()
    count = int(device_list.get_count())
    if count == 0:
        print("No Orbbec devices found.")
        return

    for index in range(count):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        try:
            name = info.get_name()
        except Exception:
            name = str(info)
        try:
            serial = info.get_serial_number()
        except Exception:
            serial = "unknown"
        try:
            pid = info.get_pid()
        except Exception:
            pid = "unknown"
        print(f"[{index}] name={name} serial={serial} pid={pid}")


def load_transform(path: str | None, label: str) -> np.ndarray:
    if not path:
        print(f"WARNING: {label} camera-to-base transform not supplied; using identity.")
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(np.load(Path(path)), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} transform must be a finite 4x4 .npy matrix")
    return matrix


class RealSensePointCloud:
    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        self.pipeline.stop()

    def capture_raw(self) -> tuple[np.ndarray, np.ndarray, Any]:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"missing RealSense frame from {self.serial}")
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[..., ::-1].copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        return depth, color_rgb, intrinsics


    def capture_pointcloud(
        self,
        transform: np.ndarray,
        point_count: int,
        rgb_mode: str,
        min_depth: float,
        max_depth: float,
        rng: np.random.Generator,
        workspace_min: np.ndarray | None = None,
        workspace_max: np.ndarray | None = None,
    ) -> np.ndarray:
        depth, color, intrinsics = self.capture_raw()
        return raw_to_pointcloud(
            depth,
            color,
            intrinsics,
            self.depth_scale,
            transform,
            point_count,
            rgb_mode,
            min_depth,
            max_depth,
            rng,
            workspace_min,
            workspace_max,
        )


class OrbbecPointCloud:
    """DaBai DC1 colored point cloud using Orbbec SDK v1.x bindings."""

    def __init__(self):
        if ORBBEC_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Orbbec Python SDK is unavailable. Install "
                "'pyorbbecsdk-community' in the lerobot environment first. "
                f"Original import error: {ORBBEC_IMPORT_ERROR}"
            )

        self.pipeline = OBPipeline()
        self.device = self.pipeline.get_device()
        info = self.device.get_device_info()
        try:
            self.name = str(info.get_name())
        except Exception:
            self.name = "Orbbec"
        try:
            self.serial = str(info.get_serial_number())
        except Exception:
            self.serial = "unknown"

        config = OBConfig()

        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        if depth_profiles is None:
            raise RuntimeError("DaBai DC1 has no usable depth profile")
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        if color_profiles is None:
            raise RuntimeError("DaBai DC1 has no usable color profile")
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # Official Orbbec point-cloud samples use D2C alignment before
        # generating a colored cloud. Prefer hardware alignment, then software.
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.start(config)
        except Exception:
            try:
                config = OBConfig()
                config.enable_stream(depth_profile)
                config.enable_stream(color_profile)
                config.set_align_mode(OBAlignMode.SW_MODE)
                self.pipeline.start(config)
            except Exception as exc:
                raise RuntimeError(
                    "Could not start DaBai DC1 with depth-to-color alignment"
                ) from exc

        self.pipeline.enable_frame_sync()
        for _ in range(15):
            self.pipeline.wait_for_frames(1000)

        print(
            f"Opened Orbbec hand camera: name={self.name} serial={self.serial}"
        )

    def stop(self) -> None:
        self.pipeline.stop()

    @staticmethod
    def _points_to_array(points: Any) -> np.ndarray:
        arr = np.asarray([tuple(point) for point in points], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise RuntimeError(
                f"unexpected Orbbec colored point-cloud shape: {arr.shape}"
            )
        return arr[:, :6]

    def capture_pointcloud(
        self,
        transform: np.ndarray,
        point_count: int,
        rgb_mode: str,
        min_depth: float,
        max_depth: float,
        rng: np.random.Generator,
        workspace_min: np.ndarray | None = None,
        workspace_max: np.ndarray | None = None,
    ) -> np.ndarray:
        frames = None
        for _ in range(10):
            frames = self.pipeline.wait_for_frames(1000)
            if frames is not None and frames.get_depth_frame() is not None:
                break
        if frames is None or frames.get_depth_frame() is None:
            raise RuntimeError("missing DaBai DC1 depth/color frames")

        camera_param = self.pipeline.get_camera_param()
        points = frames.get_color_point_cloud(camera_param)
        if points is None or len(points) == 0:
            raise RuntimeError("DaBai DC1 returned an empty colored point cloud")

        cloud = self._points_to_array(points)
        xyz_camera = cloud[:, :3].astype(np.float64)
        rgb = cloud[:, 3:6].astype(np.float32)

        # Legacy Orbbec point clouds are commonly expressed in millimetres.
        # Auto-detect by scene scale so the FP3 input remains in metres.
        finite_xyz = xyz_camera[np.isfinite(xyz_camera).all(axis=1)]
        if finite_xyz.size == 0:
            raise RuntimeError("DaBai DC1 point cloud contains no finite XYZ")
        median_abs_z = float(np.median(np.abs(finite_xyz[:, 2])))
        if median_abs_z > 10.0:
            xyz_camera *= 0.001

        valid = (
            np.isfinite(xyz_camera).all(axis=1)
            & np.isfinite(rgb).all(axis=1)
            & (xyz_camera[:, 2] >= min_depth)
            & (xyz_camera[:, 2] <= max_depth)
        )

        xyz_camera = xyz_camera[valid]
        rgb = rgb[valid]
        if xyz_camera.shape[0] == 0:
            raise RuntimeError("all DaBai DC1 points were removed by depth filtering")

        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        xyz_base = xyz_camera @ rotation.T + translation

        if rgb_mode == "minus1_1":
            rgb = rgb / 127.5 - 1.0
        elif rgb_mode == "zero1":
            rgb = rgb / 255.0
        elif rgb_mode == "zero255":
            pass
        else:
            raise ValueError(f"unsupported RGB mode: {rgb_mode}")

        valid = np.isfinite(xyz_base).all(axis=1) & np.isfinite(rgb).all(axis=1)
        if workspace_min is not None and workspace_max is not None:
            valid &= np.all(xyz_base >= workspace_min, axis=1)
            valid &= np.all(xyz_base <= workspace_max, axis=1)

        xyz_base = xyz_base[valid]
        rgb = rgb[valid]
        if xyz_base.shape[0] == 0:
            raise RuntimeError(
                "all DaBai DC1 points were removed by workspace filtering"
            )

        replace = xyz_base.shape[0] < point_count
        indices = rng.choice(
            xyz_base.shape[0],
            size=point_count,
            replace=replace,
        )
        return np.concatenate(
            (xyz_base[indices], rgb[indices]),
            axis=1,
        ).astype(np.float32)


def raw_to_pointcloud(
    depth: np.ndarray,
    color_rgb: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    transform: np.ndarray,
    point_count: int,
    rgb_mode: str,
    min_depth: float,
    max_depth: float,
    rng: np.random.Generator,
    workspace_min: np.ndarray | None = None,
    workspace_max: np.ndarray | None = None,
) -> np.ndarray:
    z = depth * float(depth_scale)
    height, width = z.shape
    v, u = np.indices((height, width), dtype=np.float32)

    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        raise RuntimeError("no valid depth points")

    z_valid = z[valid]
    x = (u[valid] - float(intrinsics.ppx)) / float(intrinsics.fx) * z_valid
    y = (v[valid] - float(intrinsics.ppy)) / float(intrinsics.fy) * z_valid
    xyz_camera = np.column_stack((x, y, z_valid)).astype(np.float64)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    xyz_base = xyz_camera @ rotation.T + translation

    rgb = color_rgb[valid].astype(np.float32)
    if rgb_mode == "minus1_1":
        rgb = rgb / 127.5 - 1.0
    elif rgb_mode == "zero1":
        rgb = rgb / 255.0
    elif rgb_mode == "zero255":
        pass
    else:
        raise ValueError(f"unsupported RGB mode: {rgb_mode}")

    finite = np.isfinite(xyz_base).all(axis=1) & np.isfinite(rgb).all(axis=1)
    if workspace_min is not None and workspace_max is not None:
        finite &= np.all(xyz_base >= workspace_min, axis=1)
        finite &= np.all(xyz_base <= workspace_max, axis=1)

    xyz_base = xyz_base[finite]
    rgb = rgb[finite]
    if xyz_base.shape[0] == 0:
        raise RuntimeError("all point-cloud points were filtered out")

    replace = xyz_base.shape[0] < point_count
    indices = rng.choice(xyz_base.shape[0], size=point_count, replace=replace)
    result = np.concatenate((xyz_base[indices], rgb[indices]), axis=1)
    return result.astype(np.float32)


def enum_value(value: Any) -> int | None:
    for candidate in (value, getattr(value, "value", None)):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass
    text = str(value)
    if "(0x" in text:
        try:
            return int(text.split("(0x", 1)[1].split(")", 1)[0], 16)
        except Exception:
            pass
    return None


def rot6d_to_rotation(rot6d: np.ndarray) -> Rotation:
    """
    Inverse of the FP3 training converter used for Piper.

    The training converter stores:
        matrix[..., :2, :].reshape(..., 6)
    i.e. the first TWO ROWS of the 3x3 rotation matrix.

    Therefore the inverse must reconstruct rows:
        row0 = b1
        row1 = b2
        row2 = cross(row0, row1)

    Do NOT column_stack these vectors.
    """
    vector = np.asarray(rot6d, dtype=np.float64).reshape(6)

    row0_raw = vector[:3]
    row1_raw = vector[3:]

    n0 = float(np.linalg.norm(row0_raw))
    if n0 < 1e-8:
        raise ValueError("invalid FP3 rotation 6D: first row is zero")
    row0 = row0_raw / n0

    row1_orth = row1_raw - np.dot(row0, row1_raw) * row0
    n1 = float(np.linalg.norm(row1_orth))
    if n1 < 1e-8:
        raise ValueError("invalid FP3 rotation 6D: rows are collinear")
    row1 = row1_orth / n1

    row2 = np.cross(row0, row1)

    matrix = np.stack((row0, row1, row2), axis=0)
    return Rotation.from_matrix(matrix)

def map_gripper_value(
    raw_value: float,
    training_min: float | None,
    training_max: float | None,
    max_width: float,
    invert: bool,
) -> float:
    value = float(raw_value)
    # FP3 RolloutPolicy normally returns de-normalized training units.
    # A range no larger than about 0.11 is treated as metres.
    if (
        training_min is not None
        and training_max is not None
        and training_min >= -0.02
        and training_max <= 0.11
    ):
        width = value
    elif -0.1 <= value <= 1.1:
        normalized = float(np.clip(value, 0.0, 1.0))
        if invert:
            normalized = 1.0 - normalized
        width = normalized * max_width
    else:
        raise ValueError(f"unsupported gripper output value: {value}")
    return float(np.clip(width, 0.0, max_width))


def keyboard_command() -> str | None:
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch().lower()
    if key in ("\x1b", "q"):
        return "quit"
    if key == "e":
        return "estop"
    return None




# Official Piper joint limits (radians), matching AgileX Piper SDK / pyAgxArm conventions.
# A small inner margin is applied by this deployment client so repeated model commands
# do not continuously push directly against the mechanical / firmware limits.
PIPER_JOINT_MIN = np.array([
    -2.6179,   # J1 -150 deg
     0.0,      # J2    0 deg
    -2.9670,   # J3 -170 deg
    -1.7450,   # J4 -100 deg
    -1.2200,   # J5  -70 deg
    -2.09439,  # J6 -120 deg
], dtype=np.float64)
PIPER_JOINT_MAX = np.array([
     2.6179,   # J1  150 deg
     3.1400,   # J2  180 deg
     0.0,      # J3    0 deg
     1.7450,   # J4  100 deg
     1.2200,   # J5   70 deg
     2.09439,  # J6  120 deg
], dtype=np.float64)


def joint_limit_pressure(
    current: np.ndarray,
    requested: np.ndarray,
    margin_rad: float,
) -> tuple[bool, list[int], np.ndarray]:
    """Detect commands that are at / beyond a safe inner joint boundary.

    Returns (active, one-based joint indices, safely-clamped target).
    The timer is based on this condition, not on parsing SDK console text.
    """
    current = np.asarray(current, dtype=np.float64).reshape(6)
    requested = np.asarray(requested, dtype=np.float64).reshape(6)
    safe_min = PIPER_JOINT_MIN + margin_rad
    safe_max = PIPER_JOINT_MAX - margin_rad
    # Only hard-clamp commands that truly exceed the documented joint range.
    # The inner margin is used for detection only, so a valid starting pose near
    # a limit is never pushed away from it merely by this safety monitor.
    sent_target = np.clip(requested, PIPER_JOINT_MIN, PIPER_JOINT_MAX)

    clipped = np.abs(sent_target - requested) > 1e-8
    near_low = current <= safe_min
    near_high = current >= safe_max
    pushing_low = requested < current - 1e-8
    pushing_high = requested > current + 1e-8
    pressure = clipped | (near_low & pushing_low) | (near_high & pushing_high)
    joints = (np.nonzero(pressure)[0] + 1).tolist()
    return bool(np.any(pressure)), joints, sent_target


def write_diag_row(writer, *, step, current_pose, predicted_position, target_position,
                   current_joints, requested_joints, sent_joints, limit_joints,
                   limit_consecutive_steps, predicted_gripper, target_gripper, pos_error, rot_error):
    if writer is None:
        return
    row = {
        'wall_time': time.time(),
        'step': step,
        'current_x': current_pose[0], 'current_y': current_pose[1], 'current_z': current_pose[2],
        'pred_x': predicted_position[0], 'pred_y': predicted_position[1], 'pred_z': predicted_position[2],
        'target_x': target_position[0], 'target_y': target_position[1], 'target_z': target_position[2],
        'requested_dz': predicted_position[2] - current_pose[2],
        'limit_joints': '|'.join(map(str, limit_joints)),
        'limit_consecutive_steps': limit_consecutive_steps,
        'predicted_gripper': predicted_gripper,
        'target_gripper': target_gripper,
        'ik_pos_error_m': pos_error,
        'ik_rot_error_rad': rot_error,
    }
    for i in range(6):
        row[f'j{i+1}_current_deg'] = math.degrees(float(current_joints[i]))
        row[f'j{i+1}_requested_deg'] = math.degrees(float(requested_joints[i]))
        row[f'j{i+1}_sent_deg'] = math.degrees(float(sent_joints[i]))
    writer.writerow(row)


def return_to_saved_joint_pose(
    robot: Any,
    home_joints: np.ndarray,
    max_joint_step: float,
    hz: float,
    tolerance: float = math.radians(0.2),
    max_iterations: int = 600,
) -> str:
    """Return to the joint pose captured before FP3 rollout.

    The return is feedback-driven and incremental. Q / Esc cancels the return.
    E invokes the electronic emergency stop.
    """
    period = 1.0 / hz
    home = np.asarray(home_joints, dtype=np.float64).reshape(-1)

    print("\nReturning to saved start joint pose...")
    print("Q / Esc = stop return; E = electronic emergency stop")

    for index in range(max_iterations):
        command = keyboard_command()
        if command == "quit":
            print("Return-home cancelled by user.")
            return "quit"
        if command == "estop":
            print("Electronic emergency stop requested during return-home.")
            robot.electronic_emergency_stop()
            return "estop"

        arm_status = robot.get_arm_status()
        joints_ret = robot.get_joint_angles()
        if arm_status is None or joints_ret is None:
            raise RuntimeError("missing Piper feedback during return-home")
        arm_status_code = enum_value(arm_status.msg.arm_status)
        if arm_status_code not in (None, 0):
            raise RuntimeError(
                f"Piper arm_status is not NORMAL during return-home: {arm_status.msg.arm_status}"
            )

        current = np.asarray(joints_ret.msg, dtype=np.float64)
        delta = home - current
        max_error = float(np.max(np.abs(delta)))
        if max_error <= tolerance:
            print(
                f"HOME_REACHED max_joint_error={math.degrees(max_error):.3f} deg"
            )
            return "ok"

        scale = min(1.0, max_joint_step / max_error)
        target = current + delta * scale
        robot.move_j(target.tolist())

        if index % 5 == 0:
            print(
                f"return_step={index:03d} "
                f"remaining={math.degrees(max_error):.2f} deg"
            )
        time.sleep(period)

    raise RuntimeError("return-home exceeded maximum iterations")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Direct FP3 -> Piper client: no fixed-Hz scheduler, no Cartesian/joint/"
            "gripper step smoothing, no local IK, no automatic return-home."
        )
    )
    parser.add_argument("--server-ip", default=os.environ.get("PIPER_FP3_SERVER_IP", ""))
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--hand-serial", default=os.environ.get("PIPER_FP3_D405_SERIAL", ""))
    parser.add_argument("--varied-serial", default=os.environ.get("PIPER_FP3_D455_SERIAL", ""))
    parser.add_argument("--speed-percent", type=int, default=20)
    parser.add_argument("--gripper-width", type=float, default=0.08)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 = run until Ctrl+C",
    )
    parser.add_argument("--return-speed-percent", type=int, default=10)
    parser.add_argument("--return-timeout", type=float, default=15.0)
    args = parser.parse_args()

    if not args.server_ip:
        parser.error("--server-ip is required")
    if not args.hand_serial:
        parser.error("--hand-serial is required")
    if not args.varied_serial:
        parser.error("--varied-serial is required")

    if not (0 <= args.speed_percent <= 100):
        parser.error("--speed-percent must be in pyAgxArm SDK range 0..100")
    if not (0 <= args.return_speed_percent <= 100):
        parser.error("--return-speed-percent must be in SDK range 0..100")
    if args.return_timeout <= 0:
        parser.error("--return-timeout must be positive")

    print("=" * 78)
    print("FP3 DIRECT PIPER CLIENT / PiperFW.V189")
    print("=" * 78)
    print("NO fixed-Hz sleep")
    print("NO max-position-step")
    print("NO max-rotation-step")
    print("NO local least_squares IK")
    print("NO max-joint-step")
    print("NO max-gripper-step")
    print("Ctrl+C = stop FP3 commands, then return to startup joint state")
    print("Second Ctrl+C during return = disconnect immediately")
    print("=" * 78)

    rng = np.random.default_rng()

    # This model was trained on camera-local / uncalibrated point clouds.
    identity = np.eye(4, dtype=np.float64)

    hand_camera = RealSensePointCloud(args.hand_serial)
    varied_camera = RealSensePointCloud(args.varied_serial)

    sock = socket.create_connection((args.server_ip, args.port), timeout=30)
    sock.settimeout(180)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    metadata = recv_json(sock)
    if metadata.get("type") != "metadata":
        raise RuntimeError(f"unexpected server greeting: {metadata}")

    points_per_frame = int(metadata["points_per_frame"])
    rgb_mode = str(metadata.get("rgb_mode", "zero1"))

    print("FP3 checkpoint =", metadata.get("checkpoint"))
    print("points_per_frame =", points_per_frame)
    print("rgb_mode =", rgb_mode)

    # User's Piper firmware is S-V1.9-0 -> use V189.
    if not hasattr(PiperFW, "V189"):
        raise RuntimeError(
            "Installed pyAgxArm does not expose PiperFW.V189. "
            "Upgrade pyAgxArm before running."
        )

    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.V189,
        interface="agx_cando",
        channel="0",
        bitrate=1_000_000,
        auto_connect=False,
    )
    robot = AgxArmFactory.create_arm(cfg)
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    last_gripper = args.gripper_width
    step = 0
    initial_joints = None
    initial_pose = None
    ctrl_c_requested = False

    try:
        robot.connect()
        time.sleep(1.0)

        print("Piper firmware feedback =", robot.get_firmware())
        print("Piper CAN channel =", robot.get_channel())

        if not robot.is_ok():
            raise RuntimeError("Piper CAN communication is not OK")

        deadline = time.monotonic() + 5.0
        while not robot.get_joint_enable_status(255):
            robot.enable()
            if time.monotonic() >= deadline:
                raise RuntimeError("could not enable all Piper joints")
            time.sleep(0.02)

        joints_ret = robot.get_joint_angles()
        pose_ret = robot.get_flange_pose()
        if joints_ret is None:
            raise RuntimeError("cannot capture initial Piper joint state")

        initial_joints = np.asarray(
            joints_ret.msg, dtype=np.float64
        ).reshape(6)

        if pose_ret is not None:
            initial_pose = np.asarray(
                pose_ret.msg, dtype=np.float64
            ).reshape(6)

        print(
            "Captured initial joints(rad) =",
            np.round(initial_joints, 5).tolist(),
        )
        if initial_pose is not None:
            print(
                "Captured initial pose =",
                np.round(initial_pose, 5).tolist(),
            )

        # Keep only SDK/firmware-native limits. No client-side motion smoothing.
        robot.set_joint_limits_enabled(True)

        # Keep pyAgxArm's official per-command automatic mode switching enabled.
        # move_p() will set P mode before sending each Cartesian command.
        robot.set_auto_set_motion_mode_enabled(True)
        robot.set_speed_percent(args.speed_percent)

        print(
            f"Piper automatic motion-mode switching ENABLED, "
            f"speed={args.speed_percent}%"
        )
        print("Starting rollout. Press Ctrl+C to stop.\n")

        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                print("max-steps reached")
                break

            t_loop = time.perf_counter()

            flange_ret = robot.get_flange_pose()
            if flange_ret is None:
                raise RuntimeError("missing Piper flange feedback")
            current_pose = np.asarray(flange_ret.msg, dtype=np.float64).reshape(6)

            status_ret = robot.get_arm_status()
            if status_ret is None:
                arm_status_code = None
                ctrl_mode = None
                mode_feedback = None
                motion_status = None
            else:
                s = status_ret.msg
                arm_status_code = getattr(s, "arm_status", None)
                ctrl_mode = getattr(s, "ctrl_mode", None)
                mode_feedback = getattr(s, "mode_feedback", None)
                motion_status = getattr(s, "motion_status", None)

            gripper_ret = effector.get_gripper_status()
            if gripper_ret is not None and str(gripper_ret.msg.mode) == "width":
                current_gripper = float(gripper_ret.msg.value)
            else:
                current_gripper = last_gripper

            t_cam = time.perf_counter()

            hand_pcd = hand_camera.capture_pointcloud(
                identity,
                points_per_frame,
                rgb_mode,
                args.min_depth,
                args.max_depth,
                rng,
            )
            varied_pcd = varied_camera.capture_pointcloud(
                identity,
                points_per_frame,
                rgb_mode,
                args.min_depth,
                args.max_depth,
                rng,
            )

            camera_ms = (time.perf_counter() - t_cam) * 1000.0

            request_buffer = io.BytesIO()

            # Uncompressed NPZ reduces CPU packing latency on the Windows client.
            np.savez(
                request_buffer,
                **{
                    HAND_KEY: hand_pcd,
                    VARIED_KEY: varied_pcd,
                    CART_KEY: current_pose.astype(np.float32),
                    GRIPPER_STATE_KEY: np.array(
                        [current_gripper],
                        dtype=np.float32,
                    ),
                },
            )

            t_infer = time.perf_counter()
            send_packet(sock, request_buffer.getvalue())
            response = recv_json(sock)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            if response.get("type") == "error":
                raise RuntimeError(
                    f"FP3 server error: {response.get('message')}"
                )
            if response.get("type") != "action":
                raise RuntimeError(f"unexpected FP3 response: {response}")

            named = response["named"]

            predicted_position = np.asarray(
                named["action/abs_pos"],
                dtype=np.float64,
            ).reshape(3)

            predicted_rotation = rot6d_to_rotation(
                np.asarray(
                    named["action/abs_rot_6d"],
                    dtype=np.float64,
                )
            )

            predicted_rpy = predicted_rotation.as_euler(
                "XYZ",
                degrees=False,
            )

            target_pose = np.concatenate(
                (predicted_position, predicted_rpy)
            ).astype(np.float64)

            predicted_gripper = float(
                named["action/gripper_position"][0]
            )

            target_gripper = map_gripper_value(
                predicted_gripper,
                metadata.get("training_gripper_min"),
                metadata.get("training_gripper_max"),
                args.gripper_width,
                False,
            )

            t_cmd = time.perf_counter()

            # DIRECT FP3 absolute Cartesian command.
            # No safe_target_pose(), no local IK, no per-step joint clamp.
            robot.move_p(target_pose.tolist())

            # DIRECT gripper target.
            # No max-gripper-step smoothing.
            effector.move_gripper_m(
                value=target_gripper,
                force=args.gripper_force,
            )

            command_ms = (time.perf_counter() - t_cmd) * 1000.0
            last_gripper = target_gripper

            loop_ms = (time.perf_counter() - t_loop) * 1000.0
            effective_hz = 1000.0 / loop_ms if loop_ms > 0 else float("inf")

            dz_mm = (
                predicted_position[2] - current_pose[2]
            ) * 1000.0

            print(
                f"step={step:04d} "
                f"xyz_now={np.round(current_pose[:3], 4).tolist()} "
                f"xyz_pred={np.round(predicted_position, 4).tolist()} "
                f"rpy_pred={np.round(predicted_rpy, 3).tolist()} "
                f"dz={dz_mm:+.1f}mm "
                f"gripper={target_gripper:.4f}m "
                f"status={arm_status_code} "
                f"ctrl={ctrl_mode} "
                f"mode={mode_feedback} "
                f"motion={motion_status} "
                f"camera={camera_ms:.1f}ms "
                f"infer={infer_ms:.1f}ms "
                f"cmd={command_ms:.1f}ms "
                f"loop={loop_ms:.1f}ms "
                f"effective_hz={effective_hz:.2f}"
            )

            step += 1

    except KeyboardInterrupt:
        ctrl_c_requested = True
        print("\nCtrl+C received. Stop sending FP3 commands.")

    finally:
        # Stop perception and FP3 traffic first.
        try:
            varied_camera.stop()
        except Exception:
            pass
        try:
            hand_camera.stop()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

        # Ctrl+C returns to the exact joint state captured at startup.
        if ctrl_c_requested and initial_joints is not None:
            try:
                print("\nReturning to captured initial joint state...")
                robot.set_auto_set_motion_mode_enabled(True)
                robot.set_speed_percent(args.return_speed_percent)
                robot.move_j(initial_joints.tolist())

                deadline = time.monotonic() + args.return_timeout
                while time.monotonic() < deadline:
                    joints_now = robot.get_joint_angles()
                    if joints_now is not None:
                        q_now = np.asarray(
                            joints_now.msg, dtype=np.float64
                        ).reshape(6)
                        max_err_deg = float(
                            np.max(
                                np.rad2deg(
                                    np.abs(q_now - initial_joints)
                                )
                            )
                        )
                    else:
                        max_err_deg = float("inf")

                    print(
                        f"return max_joint_error={max_err_deg:.3f} deg",
                        end="\r",
                        flush=True,
                    )

                    if max_err_deg < 0.5:
                        print()
                        print(
                            "Returned to captured initial joint state."
                        )
                        break

                    time.sleep(0.05)
                else:
                    print()
                    print(
                        "Return timeout reached; disconnecting."
                    )

            except KeyboardInterrupt:
                print()
                print(
                    "Second Ctrl+C received during return; "
                    "disconnecting immediately."
                )
            except Exception as exc:
                print()
                print(f"Return-to-initial failed: {exc}")

        try:
            robot.disconnect()
        except Exception:
            pass

        print("Disconnected.")


if __name__ == "__main__":
    main()
