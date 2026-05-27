#!/bin/bash
# Run the full experiment pipeline using the shared YAML config.

set -euo pipefail

CONFIG_PATH="${1:-configs/baseline.yaml}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

config_value() {
    python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for part in dotted_key.split("."):
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

config_list() {
    python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for part in dotted_key.split("."):
    value = value[part]
for item in value:
    print(item)
PY
}

contains_value() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

TRAIN_DIR="$(config_value paths.clic_train_dir)"
VAL_DIR="$(config_value paths.clic_val_dir)"
KODAK_DIR="$(config_value paths.kodak_dir)"
CKPT_DIR="$(config_value paths.checkpoint_dir)"
RESULTS_DIR="$(config_value paths.results_dir)"
PLOTS_DIR="$(config_value paths.plots_dir)"
RECON_DIR="$(config_value paths.recon_dir)"
EPOCHS="$(config_value training.epochs)"
BATCH_SIZE="$(config_value training.batch_size)"
DISTORTION="$(config_value training.distortion)"

mapfile -t LAMBDAS < <(config_list lambdas)
mapfile -t QUALITIES < <(config_list qualities)
mapfile -t TRAINED_VARIANTS < <(config_list pipeline.trained_variants)
mapfile -t FREEZE_MODES < <(config_list pipeline.freeze_modes)
mapfile -t ATTENTION_ONLY_VARIANTS < <(config_list pipeline.attention_only_variants)
mapfile -t REFERENCE_VARIANTS < <(config_list pipeline.reference_variants)

pointer_file() {
    local variant="$1"
    local lmbda="$2"
    local quality="$3"
    local freeze_mode="$4"
    echo "$CKPT_DIR/latest_variant_${variant}_lmbda_${lmbda}_quality_${quality}_${DISTORTION}_freeze_${freeze_mode}.txt"
}

run_dir_for() {
    local variant="$1"
    local lmbda="$2"
    local quality="$3"
    local freeze_mode="$4"
    echo "$CKPT_DIR/variant_${variant}_lmbda_${lmbda}_quality_${quality}_${DISTORTION}_freeze_${freeze_mode}_${RUN_TIMESTAMP}"
}

train_with_resume() {
    local variant="$1"
    local lmbda="$2"
    local quality="$3"
    local freeze_mode="$4"
    local run_dir
    run_dir="$(run_dir_for "$variant" "$lmbda" "$quality" "$freeze_mode")"
    local resume_args=()

    if [ -f "$run_dir/checkpoint_last.pth.tar" ]; then
        echo "Resuming variant ${variant}, freeze=${freeze_mode}, lambda=${lmbda}, quality=${quality}"
        resume_args=(--resume "$run_dir/checkpoint_last.pth.tar")
    fi

    python train.py --config "$CONFIG_PATH" --variant "$variant" --lmbda "$lmbda" --quality "$quality" \
        --train-dir "$TRAIN_DIR" --val-dir "$VAL_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --freeze-mode "$freeze_mode" --save-dir "$CKPT_DIR" --timestamp "$RUN_TIMESTAMP" "${resume_args[@]}"
}

evaluate_reference_variant() {
    local variant="$1"
    local lmbda="$2"
    local quality="$3"
    python evaluate.py --config "$CONFIG_PATH" --variant "$variant" --quality "$quality" --lmbda "$lmbda" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR" --recon-dir "$RECON_DIR" \
        --run-name "variant_${variant}_lmbda_${lmbda}_quality_${quality}_${DISTORTION}_reference_${RUN_TIMESTAMP}"
}

evaluate_trained_variant() {
    local variant="$1"
    local lmbda="$2"
    local quality="$3"
    local freeze_mode="$4"
    local checkpoint
    checkpoint="$(cat "$(pointer_file "$variant" "$lmbda" "$quality" "$freeze_mode")")"
    python evaluate.py --config "$CONFIG_PATH" --variant "$variant" --quality "$quality" --lmbda "$lmbda" \
        --checkpoint "$checkpoint" --data-dir "$KODAK_DIR" --output "$RESULTS_DIR" --recon-dir "$RECON_DIR"
}

echo "=== Step 1: Check datasets ==="
test -d "$TRAIN_DIR" || { echo "Missing train split at $TRAIN_DIR"; exit 1; }
test -d "$VAL_DIR" || { echo "Missing validation split at $VAL_DIR"; exit 1; }
test -d "$KODAK_DIR" || { echo "Missing Kodak images at $KODAK_DIR"; exit 1; }

echo ""
echo "=== Step 2: Train learned variants across freeze modes ==="
for freeze_mode in "${FREEZE_MODES[@]}"; do
    echo "--- Freeze mode: ${freeze_mode} ---"
    for variant in "${TRAINED_VARIANTS[@]}"; do
        if [ "$freeze_mode" = "attention_only" ] && ! contains_value "$variant" "${ATTENTION_ONLY_VARIANTS[@]}"; then
            echo "Skipping invalid combination: variant=${variant}, freeze_mode=${freeze_mode}"
            continue
        fi
        for idx in "${!LAMBDAS[@]}"; do
            lmbda="${LAMBDAS[$idx]}"
            quality="${QUALITIES[$idx]}"
            echo "Training variant=${variant}, freeze=${freeze_mode}, lambda=${lmbda}, quality=${quality}"
            train_with_resume "$variant" "$lmbda" "$quality" "$freeze_mode"
        done
    done
done

echo ""
echo "=== Step 3: Evaluate on Kodak ==="
for variant in "${REFERENCE_VARIANTS[@]}"; do
    for idx in "${!LAMBDAS[@]}"; do
        lmbda="${LAMBDAS[$idx]}"
        quality="${QUALITIES[$idx]}"
        echo "Evaluating reference variant=${variant}, lambda=${lmbda}, quality=${quality}"
        evaluate_reference_variant "$variant" "$lmbda" "$quality"
    done
done

for freeze_mode in "${FREEZE_MODES[@]}"; do
    echo "--- Evaluating freeze mode: ${freeze_mode} ---"
    for variant in "${TRAINED_VARIANTS[@]}"; do
        if [ "$freeze_mode" = "attention_only" ] && ! contains_value "$variant" "${ATTENTION_ONLY_VARIANTS[@]}"; then
            continue
        fi
        for idx in "${!LAMBDAS[@]}"; do
            lmbda="${LAMBDAS[$idx]}"
            quality="${QUALITIES[$idx]}"
            echo "Evaluating variant=${variant}, freeze=${freeze_mode}, lambda=${lmbda}, quality=${quality}"
            evaluate_trained_variant "$variant" "$lmbda" "$quality" "$freeze_mode"
        done
    done
done

echo ""
echo "=== Step 4: Plot RD curves ==="
python plot.py --config "$CONFIG_PATH" --results-dir "$RESULTS_DIR" --output "$PLOTS_DIR"

echo ""
echo "=== Done! ==="
echo "Checkpoints:      $CKPT_DIR"
echo "Results:          $RESULTS_DIR"
echo "Reconstructions:  $RECON_DIR"
echo "Plots:            $PLOTS_DIR"
