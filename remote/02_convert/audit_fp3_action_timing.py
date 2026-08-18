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


def pick(dsets, suffixes):
    for suffix in suffixes:
        for name, ds in dsets.items():
            if name == suffix or name.endswith("/" + suffix):
                return name, np.asarray(ds)
    return None, None


def one_col(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.astype(np.float64)
    return a.reshape(a.shape[0], -1)[:, 0].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path.home() / "fp3_piper_data/fp3_h5_uncalibrated"),
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "fp3_piper_data/fp3_action_timing_audit.csv"),
    )
    args = ap.parse_args()

    files = sorted(Path(args.root).glob("episode_*/trajectory_pcd.h5"))
    if not files:
        raise SystemExit(f"No episodes found under {args.root}")

    rows = []

    for path in files:
        with h5py.File(path, "r") as h5:
            d = all_datasets(h5)

            _, state = pick(d, [
                "observation/robot_state/cartesian_position",
                "robot_state/cartesian_position",
            ])
            _, state_g = pick(d, [
                "observation/robot_state/gripper_position",
                "robot_state/gripper_position",
            ])
            _, act = pick(d, ["action/abs_pos"])
            _, act_g = pick(d, ["action/gripper_position"])

            if state is None or state_g is None or act is None or act_g is None:
                print(f"{path.parent.name}: missing required dataset")
                continue

            state = np.asarray(state, dtype=np.float64)
            act = np.asarray(act, dtype=np.float64)
            state_g = one_col(state_g)
            act_g = one_col(act_g)

            T = min(len(state), len(act), len(state_g), len(act_g))
            state = state[:T]
            act = act[:T]
            state_g = state_g[:T]
            act_g = act_g[:T]

            # Converter was intended to make action[t] = feedback[t+1].
            n = max(T - 1, 0)
            if n:
                pos_align = np.linalg.norm(act[:n, :3] - state[1:n+1, :3], axis=1)
                grip_align = np.abs(act_g[:n] - state_g[1:n+1])
                pos_mae_mm = float(np.mean(pos_align) * 1000.0)
                pos_p95_mm = float(np.percentile(pos_align, 95) * 1000.0)
                grip_mae_mm = float(np.mean(grip_align) * 1000.0)
            else:
                pos_mae_mm = pos_p95_mm = grip_mae_mm = np.nan

            gmin, gmax = float(np.min(act_g)), float(np.max(act_g))
            grange = gmax - gmin
            open_thr = gmin + 0.60 * grange
            close_thr = gmin + 0.20 * grange

            open_idxs = np.flatnonzero(act_g >= open_thr)
            first_open = int(open_idxs[0]) if len(open_idxs) else -1

            first_close_after_open = -1
            if first_open >= 0:
                close_idxs = np.flatnonzero(
                    (np.arange(T) > first_open) & (act_g <= close_thr)
                )
                if len(close_idxs):
                    first_close_after_open = int(close_idxs[0])

            z = act[:, 2]
            min_z_idx = int(np.argmin(z))
            max_z_idx = int(np.argmax(z))

            if first_close_after_open >= 0:
                base_xy = act[first_close_after_open, :2]
                later_xy = act[first_close_after_open:, :2]
                dist = np.linalg.norm(later_xy - base_xy, axis=1)
                k = int(np.argmax(dist))
                fold_peak_idx = first_close_after_open + k
                fold_dx_mm = float(
                    (act[fold_peak_idx, 0] - act[first_close_after_open, 0]) * 1000.0
                )
                fold_dy_mm = float(
                    (act[fold_peak_idx, 1] - act[first_close_after_open, 1]) * 1000.0
                )
                fold_xy_mm = float(dist[k] * 1000.0)
                z_at_close = float(z[first_close_after_open])
            else:
                fold_peak_idx = -1
                fold_dx_mm = fold_dy_mm = fold_xy_mm = np.nan
                z_at_close = np.nan

            row = {
                "episode": path.parent.name,
                "T": T,
                "pos_action_vs_next_state_mae_mm": pos_mae_mm,
                "pos_action_vs_next_state_p95_mm": pos_p95_mm,
                "gripper_action_vs_next_state_mae_mm": grip_mae_mm,
                "gripper_min_m": gmin,
                "gripper_max_m": gmax,
                "first_open_step": first_open,
                "first_close_after_open_step": first_close_after_open,
                "min_z_step": min_z_idx,
                "min_z_m": float(z[min_z_idx]),
                "max_z_step": max_z_idx,
                "max_z_m": float(z[max_z_idx]),
                "z_at_close_m": z_at_close,
                "fold_peak_step": fold_peak_idx,
                "fold_dx_after_close_mm": fold_dx_mm,
                "fold_dy_after_close_mm": fold_dy_mm,
                "fold_xy_after_close_mm": fold_xy_mm,
            }
            rows.append(row)

            print(f"\n[{row['episode']}]")
            print(
                f"  action[t] vs state[t+1]: pos MAE={pos_mae_mm:.3f} mm, "
                f"p95={pos_p95_mm:.3f} mm, gripper MAE={grip_mae_mm:.3f} mm"
            )
            print(
                f"  gripper: open_step={first_open}, close_after_open={first_close_after_open}, "
                f"range={(grange*1000):.1f} mm"
            )
            print(
                f"  Z: min={row['min_z_m']:.4f} m @ {min_z_idx}, "
                f"max={row['max_z_m']:.4f} m @ {max_z_idx}, "
                f"z_at_close={z_at_close:.4f} m" if np.isfinite(z_at_close)
                else f"  Z: min={row['min_z_m']:.4f} m @ {min_z_idx}, max={row['max_z_m']:.4f} m @ {max_z_idx}"
            )
            if first_close_after_open >= 0:
                print(
                    f"  motion after close: peak XY={fold_xy_mm:.1f} mm "
                    f"(dx={fold_dx_mm:+.1f}, dy={fold_dy_mm:+.1f}) @ step {fold_peak_idx}"
                )

    if not rows:
        raise SystemExit("No valid episodes.")

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos_ok = sum(r["pos_action_vs_next_state_mae_mm"] < 1.0 for r in rows)
    grip_ok = sum(r["gripper_action_vs_next_state_mae_mm"] < 1.0 for r in rows)
    phase_ok = sum(r["first_close_after_open_step"] >= 0 for r in rows)

    print("\n========== SUMMARY ==========")
    print(f"episodes audited: {len(rows)}")
    print(f"action[t]≈state[t+1] position MAE <1 mm: {pos_ok}/{len(rows)}")
    print(f"action[t]≈state[t+1] gripper MAE <1 mm: {grip_ok}/{len(rows)}")
    print(f"clear open -> close sequence detected: {phase_ok}/{len(rows)}")
    print(f"CSV saved to: {out_path}")
    print()
    print("Interpretation:")
    print("- If action-vs-next-state error is near zero, one-step action labels are aligned.")
    print("- If open->close exists in most episodes, the gripper phase is present in labels.")
    print("- If both pass but real rollout still hovers, the next priority is model/generalization and camera-frame geometry, not recollecting data blindly.")


if __name__ == "__main__":
    main()
