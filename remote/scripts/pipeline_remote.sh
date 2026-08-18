#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${FP3_REPO:-$HOME/3d-foundation-policy}"
CONDA="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
ENV="${FP3_CONDA_ENV:-fp3}"
DATA_ROOT="${PIPER_FP3_DATA_ROOT:-$HOME/fp3_piper_data}"
OPENPOINTS_ROOT="${OPENPOINTS_ROOT:-$HOME/openpoints}"
RAW_ROOT="${PIPER_FP3_RAW_ROOT:-$DATA_ROOT/fp3_raw_uncalibrated}"
H5_ROOT="${PIPER_FP3_H5_ROOT:-$DATA_ROOT/fp3_h5_uncalibrated}"
TRAIN_CFG="${PIPER_FP3_TRAIN_CONFIG:-$DATA_ROOT/piper_fp3_train_PIPELINE_BESTLOSS.json}"
TRAIN_OUT="${PIPER_FP3_TRAIN_OUT:-$DATA_ROOT/training_outputs}"

export FP3_REPO="$REPO"
export PIPER_FP3_DATA_ROOT="$DATA_ROOT"
export PIPER_FP3_H5_ROOT="$H5_ROOT"
export PIPER_FP3_TRAIN_CONFIG="$TRAIN_CFG"
export PYTHONPATH="$HOME:${OPENPOINTS_ROOT}:${REPO}/droid_policy_learning:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage:
  ./pipeline_remote.sh convert <episode_number>
  ./pipeline_remote.sh convert-all
  ./pipeline_remote.sh audit
  ./pipeline_remote.sh prepare
  ./pipeline_remote.sh train
  ./pipeline_remote.sh serve
  ./pipeline_remote.sh status
EOF
}

convert_one() {
  local n="$1"
  "$CONDA" run --no-capture-output -n "$ENV" \
    python "$ROOT/02_convert/convert_piper_raw_to_fp3_uncalibrated.py" \
    --raw-root "$RAW_ROOT" \
    --out-root "$H5_ROOT" \
    --episode "$n" \
    --npoints 8000 \
    --candidate-cap 12000
}

convert_all() {
  shopt -s nullglob
  local found=0
  for d in "$RAW_ROOT"/episode_*; do
    [[ -d "$d" ]] || continue
    found=1
    local base="${d##*/}"
    local digits="${base#episode_}"
    local n=$((10#$digits))
    local dst="$H5_ROOT/$base/trajectory_pcd.h5"
    if [[ -f "$dst" ]]; then
      echo "[SKIP] already converted: $dst"
    else
      echo "[CONVERT] $base"
      convert_one "$n"
    fi
  done
  [[ "$found" -eq 1 ]] || { echo "No raw episodes under $RAW_ROOT"; exit 1; }
}

audit_all() {
  "$CONDA" run --no-capture-output -n "$ENV" \
    python "$ROOT/02_convert/audit_fp3_action_timing.py" \
    --root "$H5_ROOT" \
    --out "$DATA_ROOT/fp3_action_timing_audit.csv"

  "$CONDA" run --no-capture-output -n "$ENV" \
    python "$ROOT/02_convert/audit_fp3_training_actions.py" \
    --root "$H5_ROOT" \
    --out "$DATA_ROOT/fp3_training_action_audit.csv"
}

train_model() {
  python "$ROOT/03_train/install_bestloss_train.py" || {
    # It is fine if train_best_loss.py already exists from a previous install.
    if [[ -f "$REPO/droid_policy_learning/robomimic/scripts/train_best_loss.py" ]]; then
      echo "[INFO] Existing train_best_loss.py found; using it."
    else
      exit 1
    fi
  }

  python "$ROOT/03_train/build_bestloss_config.py"

  cd "$REPO"
  "$CONDA" run --no-capture-output -n "$ENV" \
    python "$REPO/droid_policy_learning/robomimic/scripts/train_best_loss.py" \
    --config "$TRAIN_CFG"
}

latest_best_ckpt() {
  find "$TRAIN_OUT" -type f -path '*/models/model_best_train_loss.pth' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

serve_model() {
  local ckpt
  ckpt="$(latest_best_ckpt)"
  [[ -n "$ckpt" ]] || { echo "No model_best_train_loss.pth found."; exit 1; }

  local lang_h5
  lang_h5="$(find "$H5_ROOT" -type f -name trajectory_pcd.h5 | sort | head -n 1)"
  [[ -n "$lang_h5" ]] || { echo "No language H5 found."; exit 1; }

  echo "[SERVE] checkpoint=$ckpt"
  echo "[SERVE] language_h5=$lang_h5"
  echo "[SERVE] IMPORTANT: replan_every_step is OFF"

  cd "$REPO"
  "$CONDA" run --no-capture-output -n "$ENV" \
    python "$ROOT/04_inference/server_replan.py" \
    --repo "$REPO" \
    --checkpoint "$ckpt" \
    --language-h5 "$lang_h5" \
    --host 0.0.0.0 \
    --port 5555
}

status() {
  echo "Disk:"
  df -h /
  echo
  echo "H5 count:"
  find "$H5_ROOT" -type f -name trajectory_pcd.h5 | wc -l
  echo
  echo "Latest BEST checkpoint:"
  latest_best_ckpt || true
  echo
  local ckpt
  ckpt="$(latest_best_ckpt || true)"
  if [[ -n "$ckpt" ]]; then
    local info="$(dirname "$ckpt")/best_train_loss.json"
    [[ -f "$info" ]] && cat "$info"
  fi
}

cmd="${1:-}"
case "$cmd" in
  convert)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    convert_one "$2"
    ;;
  convert-all)
    convert_all
    ;;
  audit)
    audit_all
    ;;
  prepare)
    convert_all
    audit_all
    ;;
  train)
    train_model
    ;;
  serve)
    serve_model
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
