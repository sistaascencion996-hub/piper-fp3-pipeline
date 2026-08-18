#!/usr/bin/env python3
from pathlib import Path
import os

REPO = Path(os.environ.get("FP3_REPO", Path.home() / "3d-foundation-policy"))
src = REPO / "droid_policy_learning/robomimic/scripts/train.py"
dst = REPO / "droid_policy_learning/robomimic/scripts/train_best_loss.py"

if not src.exists():
    raise SystemExit(f"Missing source train.py: {src}")

text = src.read_text(encoding="utf-8")

# Do not double patch.
if "BEST-TRAIN-LOSS PATCH" in text:
    raise SystemExit(
        "train.py itself already contains BEST-TRAIN-LOSS PATCH; "
        "leave it unchanged and inspect manually."
    )

old = "    best_valid_loss = None\n"
new = """    # BEST-TRAIN-LOSS PATCH
    # Keep only the checkpoint with the globally lowest epoch training Loss.
    best_train_loss = float("inf")
    best_train_epoch = None
    best_valid_loss = None
"""
if old not in text:
    raise SystemExit("Cannot find best_valid_loss initialization in train.py")
text = text.replace(old, new, 1)

marker = (
    "        # Save model checkpoints based on conditions "
    "(success rate, validation loss, etc)\n"
)
insert = """        # ============================================================
        # BEST-TRAIN-LOSS PATCH
        # Save only when this epoch beats the GLOBAL minimum train Loss.
        # ============================================================
        current_train_loss = float(step_log["Loss"])

        if (
            np.isfinite(current_train_loss)
            and current_train_loss < best_train_loss
        ):
            previous_best = best_train_loss
            best_train_loss = current_train_loss
            best_train_epoch = epoch

            should_save_ckpt = True
            epoch_ckpt_name = "model_best_train_loss"
            ckpt_reason = "best_train_loss"

            print(
                "[BEST] NEW GLOBAL BEST | "
                f"epoch={epoch} | "
                f"optimizer_steps={epoch * train_num_steps} | "
                f"loss={current_train_loss:.10f} | "
                f"previous={previous_best:.10f}"
            )
        else:
            should_save_ckpt = False
            print(
                "[BEST] KEEP CURRENT BEST | "
                f"epoch={epoch} | "
                f"loss={current_train_loss:.10f} | "
                f"best_epoch={best_train_epoch} | "
                f"best_loss={best_train_loss:.10f}"
            )

"""
if marker not in text:
    raise SystemExit("Cannot find checkpoint save marker in train.py")
text = text.replace(marker, insert + marker, 1)

old_path = (
    '                    ckpt_path=os.path.join('
    'ckpt_dir, epoch_ckpt_name + ".pth"),\n'
)
new_path = """                    ckpt_path=os.path.join(
                        ckpt_dir,
                        "model_best_train_loss.tmp.pth",
                    ),
"""
if old_path not in text:
    raise SystemExit("Cannot find checkpoint path in train.py")
text = text.replace(old_path, new_path, 1)

old_end = """                    action_normalization_stats=action_normalization_stats,
                )
        accelerator.wait_for_everyone()
"""
new_end = """                    action_normalization_stats=action_normalization_stats,
                )

                tmp_ckpt = os.path.join(
                    ckpt_dir,
                    "model_best_train_loss.tmp.pth",
                )
                final_ckpt = os.path.join(
                    ckpt_dir,
                    "model_best_train_loss.pth",
                )
                os.replace(tmp_ckpt, final_ckpt)

                best_info = {
                    "epoch": int(best_train_epoch),
                    "optimizer_steps": int(
                        best_train_epoch * train_num_steps
                    ),
                    "loss": float(best_train_loss),
                    "checkpoint": final_ckpt,
                }

                info_tmp = os.path.join(
                    ckpt_dir,
                    "best_train_loss.json.tmp",
                )
                info_final = os.path.join(
                    ckpt_dir,
                    "best_train_loss.json",
                )
                with open(info_tmp, "w") as f:
                    json.dump(best_info, f, indent=4)
                os.replace(info_tmp, info_final)

                print("[BEST] SAVED -> " + final_ckpt)

        accelerator.wait_for_everyone()
"""
if old_end not in text:
    raise SystemExit("Cannot find save_model closing block in train.py")
text = text.replace(old_end, new_end, 1)

dst.write_text(text, encoding="utf-8")
print("CREATED =", dst)
print("BEST_TRAIN_LOSS_PATCH = YES")
