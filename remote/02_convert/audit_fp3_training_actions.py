#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np

def all_datasets(h5):
    out = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    h5.visititems(visitor)
    return out

def pick_dataset(dsets, suffixes):
    for suffix in suffixes:
        for name, ds in dsets.items():
            if name == suffix or name.endswith("/" + suffix):
                return name, np.asarray(ds)
    return None, None

def flatten_last1(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.astype(np.float64)
    return a.reshape(a.shape[0], -1)[:, 0].astype(np.float64)

def first_window_median(a, n=10):
    n = min(len(a), n)
    return float(np.median(a[:n]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path.home() / "fp3_piper_data/fp3_h5_uncalibrated"),
        help="root containing episode_xxxxxx/trajectory_pcd.h5",
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "fp3_piper_data/fp3_training_action_audit.csv"),
    )
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("episode_*/trajectory_pcd.h5"))
    if not files:
        raise SystemExit(f"No trajectory_pcd.h5 found under {root}")

    rows = []
    print(f"Found {len(files)} episodes under {root}")

    for path in files:
        with h5py.File(path, "r") as h5:
            dsets = all_datasets(h5)

            cart_name, cart = pick_dataset(
                dsets,
                [
                    "observation/robot_state/cartesian_position",
                    "robot_state/cartesian_position",
                    "cartesian_position",
                ],
            )
            obs_grip_name, obs_grip = pick_dataset(
                dsets,
                [
                    "observation/robot_state/gripper_position",
                    "robot_state/gripper_position",
                ],
            )
            apos_name, apos = pick_dataset(
                dsets,
                ["action/abs_pos", "abs_pos"],
            )
            agrip_name, agrip = pick_dataset(
                dsets,
                ["action/gripper_position", "gripper_position"],
            )

            if apos is None:
                print(f"\n[{path.parent.name}] ERROR: action/abs_pos not found")
                print("Available datasets:")
                for k in sorted(dsets):
                    print(" ", k)
                continue

            apos = np.asarray(apos, dtype=np.float64)
            if apos.ndim != 2 or apos.shape[1] < 3:
                raise RuntimeError(f"{path}: unexpected abs_pos shape {apos.shape}")

            x, y, z = apos[:, 0], apos[:, 1], apos[:, 2]
            dz = np.diff(z)
            z0 = first_window_median(z, 10)
            z_min = float(np.min(z))
            z_max = float(np.max(z))
            descent_mm = (z0 - z_min) * 1000.0

            if agrip is not None:
                g = flatten_last1(agrip)
                dg = np.diff(g)
                grip_min = float(np.min(g))
                grip_max = float(np.max(g))
                grip_range_mm = (grip_max - grip_min) * 1000.0
                max_close_step_mm = float(np.min(dg) * 1000.0) if len(dg) else 0.0
            else:
                g = None
                grip_min = grip_max = grip_range_mm = max_close_step_mm = np.nan

            row = {
                "episode": path.parent.name,
                "T": len(z),
                "action_pos_key": apos_name,
                "action_grip_key": agrip_name or "",
                "obs_cart_key": cart_name or "",
                "obs_grip_key": obs_grip_name or "",
                "z_start10_median_m": z0,
                "z_min_m": z_min,
                "z_max_m": z_max,
                "descent_from_start_mm": descent_mm,
                "largest_single_down_step_mm": float(np.min(dz) * 1000.0) if len(dz) else 0.0,
                "largest_single_up_step_mm": float(np.max(dz) * 1000.0) if len(dz) else 0.0,
                "down_steps_gt2mm": int(np.sum(dz < -0.002)),
                "up_steps_gt2mm": int(np.sum(dz > 0.002)),
                "x_range_mm": float(np.ptp(x) * 1000.0),
                "y_range_mm": float(np.ptp(y) * 1000.0),
                "x_net_mm": float((x[-1] - x[0]) * 1000.0),
                "y_net_mm": float((y[-1] - y[0]) * 1000.0),
                "grip_min_m": grip_min,
                "grip_max_m": grip_max,
                "grip_range_mm": grip_range_mm,
                "largest_grip_decrease_step_mm": max_close_step_mm,
                "has_clear_descent_10mm": bool(descent_mm >= 10.0),
                "has_clear_gripper_change_5mm": bool(
                    np.isfinite(grip_range_mm) and grip_range_mm >= 5.0
                ),
            }
            rows.append(row)

            print(f"\n[{row['episode']}] T={row['T']}")
            print(
                f"  Z: start≈{z0:.4f} m  min={z_min:.4f} m  max={z_max:.4f} m  "
                f"descent={descent_mm:.1f} mm"
            )
            print(
                f"  dZ: biggest_down={row['largest_single_down_step_mm']:.1f} mm  "
                f"biggest_up={row['largest_single_up_step_mm']:.1f} mm  "
                f"down_steps>2mm={row['down_steps_gt2mm']}"
            )
            print(
                f"  XY: x_range={row['x_range_mm']:.1f} mm  "
                f"y_range={row['y_range_mm']:.1f} mm  "
                f"x_net={row['x_net_mm']:.1f} mm  y_net={row['y_net_mm']:.1f} mm"
            )
            if g is not None:
                print(
                    f"  Gripper: min={grip_min:.4f} m  max={grip_max:.4f} m  "
                    f"range={grip_range_mm:.1f} mm"
                )
            print(
                "  FLAGS:",
                f"descent10mm={row['has_clear_descent_10mm']}",
                f"gripper5mm={row['has_clear_gripper_change_5mm']}",
            )

    if not rows:
        raise SystemExit("No valid episodes were audited.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    descent_ok = sum(r["has_clear_descent_10mm"] for r in rows)
    grip_ok = sum(r["has_clear_gripper_change_5mm"] for r in rows)

    print("\n========== SUMMARY ==========")
    print(f"episodes audited: {len(rows)}")
    print(f"clear descent >=10 mm: {descent_ok}/{len(rows)}")
    print(f"clear gripper range >=5 mm: {grip_ok}/{len(rows)}")
    print(f"CSV saved to: {out_path}")
    print("\nInterpretation:")
    print("- If most episodes do NOT have a clear Z descent, inspect conversion/action labels first.")
    print("- If Z descent exists but gripper barely changes, inspect gripper labels/mapping.")
    print("- If both are clear in demonstrations but rollout still hovers, focus next on model/generalization and camera-frame geometry.")

if __name__ == "__main__":
    main()
