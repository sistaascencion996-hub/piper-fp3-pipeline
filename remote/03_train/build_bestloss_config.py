#!/usr/bin/env python3
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("PIPER_FP3_DATA_ROOT", Path.home() / "fp3_piper_data"))
FP3_REPO = Path(os.environ.get("FP3_REPO", Path.home() / "3d-foundation-policy"))
TEMPLATE = Path(os.environ.get(
    "PIPER_FP3_TEMPLATE",
    DATA_ROOT / "piper_fp3_train_10000steps_FIXEDOPT_20260818.json",
))
H5_ROOT = Path(os.environ.get(
    "PIPER_FP3_H5_ROOT",
    DATA_ROOT / "fp3_h5_uncalibrated",
))
OUT = Path(os.environ.get(
    "PIPER_FP3_TRAIN_CONFIG",
    DATA_ROOT / "piper_fp3_train_PIPELINE_BESTLOSS.json",
))
PRETRAINED = Path(os.environ.get(
    "FP3_PRETRAINED",
    FP3_REPO / "checkpoints/fp3_pretrained_weight.pth",
))
TASK = "Fold the towel from right to left."

if not TEMPLATE.exists():
    raise SystemExit(f"Missing validated template config: {TEMPLATE}")
if not PRETRAINED.exists():
    raise SystemExit(f"Missing FP3 pretrained checkpoint: {PRETRAINED}")

h5_files = sorted(H5_ROOT.glob("episode_*/trajectory_pcd.h5"))
if not h5_files:
    raise SystemExit(f"No trajectory_pcd.h5 files under {H5_ROOT}")

cfg = json.loads(TEMPLATE.read_text(encoding="utf-8"))

cfg["experiment"]["name"] = "piper_fold_fp3_lora_10000steps_BESTLOSS_PIPELINE"
cfg["experiment"]["ckpt_path"] = str(PRETRAINED)
cfg["train"]["num_epochs"] = 2000
cfg["experiment"]["epoch_every_n_steps"] = 5
cfg["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = 0.0001
cfg["algo"]["ema"]["enabled"] = True
cfg["experiment"]["logging"]["terminal_output_to_txt"] = True

# Auto-regenerate dataset list from every converted episode.
cfg["train"]["data"] = [
    {
        "path": str(path),
        "lang": TASK,
        "train": True,
        "weight": 1.0,
        "label": "piper_fold",
    }
    for path in h5_files
]

# Disable old periodic checkpoint schedule. train_best_loss.py owns saving.
save = cfg["experiment"]["save"]
save["enabled"] = True
save["every_n_epochs"] = None
save["every_n_seconds"] = None
save["epochs"] = []

OUT.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

print("CONFIG =", OUT)
print("DATASET_COUNT =", len(h5_files))
print("FIRST =", h5_files[0])
print("LAST =", h5_files[-1])
print("TOTAL_OPTIMIZER_STEPS =", 2000 * 5)
print("CHECKPOINT_POLICY = global minimum epoch train Loss")
