#!/usr/bin/env python3
"""
FP3 policy server for Piper deployment.

Run this file on the Ubuntu / RTX 4090D computer.
It loads the fine-tuned checkpoint once, receives live Piper observations
from the Windows robot computer, and returns one named action per request.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import traceback
from collections import deque
from typing import Any

import h5py
import numpy as np
import torch


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

HAND_KEY = "camera/pointcloud/hand_camera_left_pcd_4000"
VARIED_KEY = "camera/pointcloud/varied_camera_2_left_pcd_4000"
LANG_KEY = "lang_fixed/language_distilbert"
CART_KEY = "robot_state/cartesian_position"
GRIPPER_STATE_KEY = "robot_state/gripper_position"

DEFAULT_ACTION_DIMS = {
    "action/abs_pos": 3,
    "action/abs_rot_6d": 6,
    "action/gripper_position": 1,
}


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer disconnected")
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


def send_json(sock: socket.socket, value: dict[str, Any]) -> None:
    send_packet(sock, json.dumps(value, ensure_ascii=False).encode("utf-8"))


def find_dataset(group: h5py.Group, suffix: str) -> h5py.Dataset | None:
    suffix = suffix.strip("/")
    found: list[h5py.Dataset] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and (
            name.strip("/") == suffix or name.strip("/").endswith("/" + suffix)
        ):
            found.append(obj)

    group.visititems(visitor)
    return found[0] if found else None


def load_fixed_language(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = find_dataset(f, LANG_KEY)
        if ds is None:
            raise KeyError(f"cannot find H5 dataset ending with {LANG_KEY!r}")
        arr = np.asarray(ds, dtype=np.float32)
    while arr.ndim > 1:
        arr = arr[0]
    if arr.shape != (1024,):
        raise ValueError(f"language embedding must be (1024,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("language embedding contains NaN/inf")
    return arr


def inspect_training_h5(h5_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        xyz_samples: list[np.ndarray] = []
        rgb_samples: list[np.ndarray] = []
        point_counts: list[int] = []

        for key in (HAND_KEY, VARIED_KEY):
            ds = find_dataset(f, key)
            if ds is None:
                raise KeyError(f"cannot find H5 dataset ending with {key!r}")
            shape = tuple(ds.shape)
            if len(shape) < 2 or shape[-1] != 6:
                raise ValueError(f"{key} must end in [N,6], got {shape}")
            point_counts.append(int(shape[-2]))
            sample = np.asarray(ds[0] if len(shape) >= 3 else ds[...], dtype=np.float32)
            sample = sample.reshape(-1, 6)
            if sample.size:
                xyz_samples.append(sample[:, :3])
                rgb_samples.append(sample[:, 3:6])

        if len(set(point_counts)) != 1:
            raise ValueError(f"two camera point counts differ: {point_counts}")
        meta["points_per_frame"] = point_counts[0]

        xyz = np.concatenate(xyz_samples, axis=0)
        rgb = np.concatenate(rgb_samples, axis=0)
        finite_xyz = xyz[np.isfinite(xyz).all(axis=1)]
        finite_rgb = rgb[np.isfinite(rgb).all(axis=1)]
        if finite_xyz.size:
            meta["training_xyz_min"] = np.percentile(finite_xyz, 0.5, axis=0).tolist()
            meta["training_xyz_max"] = np.percentile(finite_xyz, 99.5, axis=0).tolist()
        if finite_rgb.size:
            rgb_min = float(np.min(finite_rgb))
            rgb_max = float(np.max(finite_rgb))
            meta["training_rgb_min"] = rgb_min
            meta["training_rgb_max"] = rgb_max
            if rgb_min < -0.05 and rgb_max <= 1.1:
                meta["rgb_mode"] = "minus1_1"
            elif rgb_max <= 1.1:
                meta["rgb_mode"] = "zero1"
            else:
                meta["rgb_mode"] = "zero255"

        gripper_ds = find_dataset(f, "action/gripper_position")
        if gripper_ds is not None:
            g = np.asarray(gripper_ds, dtype=np.float32)
            g = g[np.isfinite(g)]
            if g.size:
                meta["training_gripper_min"] = float(np.min(g))
                meta["training_gripper_max"] = float(np.max(g))

    return meta


class FrameStack:
    def __init__(self, frame_stack: int):
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        self.frame_stack = frame_stack
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=frame_stack)

    def reset(self) -> None:
        self.history.clear()

    def add(self, obs: dict[str, np.ndarray]) -> None:
        clean = {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()}
        if not self.history:
            for _ in range(self.frame_stack):
                self.history.append({key: value.copy() for key, value in clean.items()})
        else:
            self.history.append(clean)

    def get(self) -> dict[str, np.ndarray]:
        if len(self.history) != self.frame_stack:
            raise RuntimeError("frame history is not initialized")
        return {
            key: np.stack([frame[key] for frame in self.history], axis=0)
            for key in self.history[0]
        }


def normalize_config(raw_config: Any) -> dict[str, Any]:
    if isinstance(raw_config, str):
        return json.loads(raw_config)
    if isinstance(raw_config, dict):
        return raw_config
    raise TypeError(f"unsupported checkpoint config type: {type(raw_config)!r}")


def get_action_layout(config: dict[str, Any], ac_dim: int) -> list[tuple[str, int]]:
    action_cfg = config.get("train", {}).get("action_config", {})
    names = list(action_cfg.keys()) if isinstance(action_cfg, dict) else []
    layout: list[tuple[str, int]] = []
    for name in names:
        if name in DEFAULT_ACTION_DIMS:
            layout.append((name, DEFAULT_ACTION_DIMS[name]))

    if sum(dim for _, dim in layout) == ac_dim:
        return layout

    fallback = [
        ("action/abs_pos", 3),
        ("action/abs_rot_6d", 6),
        ("action/gripper_position", 1),
    ]
    if sum(dim for _, dim in fallback) != ac_dim:
        raise ValueError(
            f"checkpoint action dimension is {ac_dim}; expected the Piper FP3 10-D action"
        )
    return fallback


def validate_observation(
    obs: dict[str, np.ndarray], points_per_frame: int
) -> dict[str, np.ndarray]:
    expected = {
        HAND_KEY: (points_per_frame, 6),
        VARIED_KEY: (points_per_frame, 6),
        CART_KEY: (6,),
        GRIPPER_STATE_KEY: (1,),
    }
    clean: dict[str, np.ndarray] = {}
    for key, shape in expected.items():
        if key not in obs:
            raise KeyError(f"missing observation key: {key}")
        arr = np.asarray(obs[key], dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains NaN/inf")
        clean[key] = arr
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=os.environ.get(
            "FP3_REPO",
            str(Path.home() / "3d-foundation-policy"),
        ),
        help="3d-foundation-policy repository",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FP3_CHECKPOINT", ""),
        help="fine-tuned FP3 checkpoint; may also be set with FP3_CHECKPOINT",
    )
    parser.add_argument(
        "--language-h5",
        default=os.environ.get("FP3_LANGUAGE_H5", ""),
        help="training H5 used for the fixed language embedding and input metadata",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--task", default="Fold the towel from right to left.")
    parser.add_argument(
        "--replan-every-step",
        action="store_true",
        help="discard the remaining MDT action queue before each request so every control step replans from the latest observation",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required (or set FP3_CHECKPOINT)")
    if not args.language_h5:
        parser.error("--language-h5 is required (or set FP3_LANGUAGE_H5)")

    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    language_h5 = Path(args.language_h5).expanduser().resolve()

    for path, label in (
        (repo, "repository"),
        (checkpoint, "checkpoint"),
        (language_h5, "language H5"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "droid_policy_learning"))

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.tensor_utils as TensorUtils
    from robomimic.algo import algo_factory
    from accelerate import Accelerator

    accelerator = Accelerator()
    device = accelerator.device
    print(f"[FP3] device={device}")
    print(f"[FP3] loading checkpoint: {checkpoint}")

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint))
    config = normalize_config(ckpt_dict["config"])

    # The FP3 fork uses algo_factory(..., accelerator=...), while the older
    # robomimic policy_from_checkpoint helper still calls algo_factory(..., device=...).
    # Rebuild the policy directly with the current FP3 API.
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config_obj, _ = FileUtils.config_from_checkpoint(
        algo_name=algo_name,
        ckpt_dict=ckpt_dict,
        verbose=True,
    )
    ObsUtils.initialize_obs_utils_with_config(config_obj)

    shape_meta = ckpt_dict["shape_metadata"]
    model = algo_factory(
        algo_name=algo_name,
        config=config_obj,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        accelerator=accelerator,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()

    action_stats = ckpt_dict.get("action_normalization_stats", None)
    if action_stats is not None:
        for key in action_stats:
            for stat_name in action_stats[key]:
                action_stats[key][stat_name] = np.asarray(action_stats[key][stat_name])

    class FP3DeploymentPolicy:
        """Minimal rollout wrapper that preserves the checkpoint's 10-D rot6d action."""

        def __init__(self, model, stats, replan_every_step=False):
            self.model = model
            self.stats = stats
            self.replan_every_step = bool(replan_every_step)

        def start_episode(self):
            self.model.set_eval()
            self.model.reset()

        def __call__(self, ob):
            obs = TensorUtils.to_tensor(ob)
            obs = TensorUtils.to_batch(obs)
            obs = TensorUtils.to_device(obs, self.model.device)
            obs = TensorUtils.to_float(obs)

            if self.replan_every_step and hasattr(self.model, "action_queue"):
                self.model.action_queue.clear()

            with torch.inference_mode():
                normalized = self.model.get_action(obs_dict=obs)

            action = TensorUtils.to_numpy(normalized[0]).reshape(-1).astype(np.float32)

            # Keep the original 10-D layout:
            # abs_pos(3) + abs_rot_6d(6) + gripper(1).
            if self.stats is not None:
                layout = [
                    ("action/abs_pos", 3),
                    ("action/abs_rot_6d", 6),
                    ("action/gripper_position", 1),
                ]
                cursor = 0
                raw_parts = []
                for key, dim in layout:
                    part = action[cursor:cursor + dim]
                    cursor += dim
                    stats = self.stats.get(key)
                    if stats is None:
                        raw_parts.append(part)
                        continue

                    if "scale" in stats and "offset" in stats:
                        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
                        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
                        part = part * scale[:dim] + offset[:dim]
                    elif "std" in stats and "mean" in stats:
                        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
                        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
                        part = part * std[:dim] + mean[:dim]

                    raw_parts.append(np.asarray(part, dtype=np.float32).reshape(-1))

                action = np.concatenate(raw_parts, axis=0).astype(np.float32)

            if action.shape != (10,):
                raise ValueError(f"deployment policy produced {action.shape}, expected (10,)")
            if not np.isfinite(action).all():
                raise ValueError("deployment policy produced NaN/inf")
            return action

    policy = FP3DeploymentPolicy(model, action_stats, replan_every_step=args.replan_every_step)
    print(f"[FP3] replan_every_step={args.replan_every_step}")

    if hasattr(policy, "goal_mode"):
        policy.goal_mode = config.get("train", {}).get("goal_mode")
    if hasattr(policy, "eval_mode"):
        policy.eval_mode = True
    if hasattr(policy, "start_episode"):
        policy.start_episode()

    frame_stack = int(
        config.get("train", {}).get(
            "frame_stack",
            config.get("algo", {}).get("horizon", {}).get("observation_horizon", 2),
        )
    )

    shape_metadata = ckpt_dict.get("shape_metadata", {})
    ac_dim = int(shape_metadata.get("ac_dim", 10))
    action_layout = get_action_layout(config, ac_dim)

    language = load_fixed_language(language_h5)
    h5_meta = inspect_training_h5(language_h5)
    points_per_frame = int(h5_meta["points_per_frame"])

    metadata: dict[str, Any] = {
        "protocol": 1,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "frame_stack": frame_stack,
        "points_per_frame": points_per_frame,
        "action_dim": ac_dim,
        "action_layout": [{"name": name, "dim": dim} for name, dim in action_layout],
        **h5_meta,
    }

    print("[FP3] deployment metadata:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[FP3] listening on {args.host}:{args.port}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    try:
        while True:
            conn, address = server.accept()
            print(f"[FP3] client connected: {address}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stack = FrameStack(frame_stack)
            if hasattr(policy, "start_episode"):
                policy.start_episode()
            send_json(conn, {"type": "metadata", **metadata})

            try:
                while True:
                    payload = recv_packet(conn)
                    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
                        obs = {key: np.asarray(npz[key]) for key in npz.files}

                    obs = validate_observation(obs, points_per_frame)
                    obs[LANG_KEY] = language
                    stack.add(obs)
                    policy_obs = stack.get()

                    with torch.inference_mode():
                        action = policy(policy_obs)

                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    if action_arr.size != ac_dim:
                        raise ValueError(
                            f"policy returned {action_arr.size} values, expected {ac_dim}"
                        )
                    if not np.isfinite(action_arr).all():
                        raise ValueError("policy returned NaN/inf")

                    named: dict[str, list[float]] = {}
                    cursor = 0
                    for name, dim in action_layout:
                        named[name] = action_arr[cursor : cursor + dim].tolist()
                        cursor += dim

                    send_json(
                        conn,
                        {
                            "type": "action",
                            "action": action_arr.tolist(),
                            "named": named,
                        },
                    )
            except (ConnectionError, BrokenPipeError):
                print(f"[FP3] client disconnected: {address}")
            except Exception as exc:
                print(f"[FP3] request failed: {exc}")
                traceback.print_exc()
                try:
                    send_json(
                        conn,
                        {
                            "type": "error",
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=4),
                        },
                    )
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
