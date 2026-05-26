#!/bin/bash
# Run the full experiment pipeline:
# 1. Check datasets
# 2. Train variants A, B, C at all lambda values
# 3. Evaluate on Kodak
# 4. Plot RD curves
#
# Usage: bash run_all.sh
# Assumes conda env "learned-image-compression" is activated.

set -e

CLIC_DIR="datasets/clic_images"
KODAK_DIR="datasets/kodak"
CKPT_DIR="checkpoints"
RESULTS_DIR="experiments/results"
PLOTS_DIR="experiments/plots"

LAMBDAS=(0.0018 0.0035 0.0067 0.013)
QUALITIES=(1 2 3 4)
EPOCHS=100
BATCH_SIZE=32
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PRETRAINED_COMPARE_MODELS=(
    "bmshj2018_factorized"
    "mbt2018_mean"
    "mbt2018"
    "cheng2020_anchor"
)

train_with_resume() {
    local VARIANT="$1"
    local LMBDA="$2"
    local QUALITY="$3"
    local RUN_DIR="$CKPT_DIR/variant_${VARIANT}_lmbda_${LMBDA}_quality_${QUALITY}_mse_${RUN_TIMESTAMP}"
    local RESUME_ARGS=()

    if [ -f "$RUN_DIR/checkpoint_last.pth.tar" ]; then
        echo "Resuming variant ${VARIANT}, lambda=${LMBDA}, quality=${QUALITY} from checkpoint_last.pth.tar"
        RESUME_ARGS=(--resume "$RUN_DIR/checkpoint_last.pth.tar")
    fi

    python train.py --variant "$VARIANT" --lmbda "$LMBDA" --quality "$QUALITY" \
        --data-dir "$CLIC_DIR" --epochs $EPOCHS --batch-size $BATCH_SIZE \
        --save-dir "$CKPT_DIR" --timestamp "$RUN_TIMESTAMP" "${RESUME_ARGS[@]}"
}

echo "=== Step 1: Check datasets ==="
test -d "$CLIC_DIR" || { echo "Missing CLIC images at $CLIC_DIR"; exit 1; }
test -d "$KODAK_DIR" || { echo "Missing Kodak images at $KODAK_DIR"; exit 1; }

echo ""
echo "=== Step 2: Train all variants ==="

# Variant A: save pretrained baseline (no training needed)
for IDX in "${!LAMBDAS[@]}"; do
    LMBDA="${LAMBDAS[$IDX]}"
    QUALITY="${QUALITIES[$IDX]}"
    echo "--- Variant A, lambda=$LMBDA ---"
    python train.py --variant A --lmbda "$LMBDA" --quality $QUALITY \
        --data-dir "$CLIC_DIR" --save-dir "$CKPT_DIR" --timestamp "$RUN_TIMESTAMP"
done

# Variant B: fine-tune encoder-attention model
for IDX in "${!LAMBDAS[@]}"; do
    LMBDA="${LAMBDAS[$IDX]}"
    QUALITY="${QUALITIES[$IDX]}"
    echo "--- Variant B, lambda=$LMBDA ---"
    train_with_resume B "$LMBDA" "$QUALITY"
done

# Variant C: fine-tune encoder+decoder-attention model
for IDX in "${!LAMBDAS[@]}"; do
    LMBDA="${LAMBDAS[$IDX]}"
    QUALITY="${QUALITIES[$IDX]}"
    echo "--- Variant C, lambda=$LMBDA ---"
    train_with_resume C "$LMBDA" "$QUALITY"
done

echo ""
echo "=== Step 3: Evaluate on Kodak ==="

for IDX in "${!LAMBDAS[@]}"; do
    LMBDA="${LAMBDAS[$IDX]}"
    QUALITY="${QUALITIES[$IDX]}"
    echo "--- Evaluating Variant A, lambda=$LMBDA ---"
    python evaluate.py --variant A --quality $QUALITY --lmbda "$LMBDA" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR" \
        --run-name "variant_A_lmbda_${LMBDA}_quality_${QUALITY}_mse_${RUN_TIMESTAMP}"

    echo "--- Evaluating Variant B, lambda=$LMBDA ---"
    B_CKPT="$(cat "$CKPT_DIR/latest_variant_B_lmbda_${LMBDA}_quality_${QUALITY}_mse.txt")"
    python evaluate.py --variant B --quality $QUALITY --lmbda "$LMBDA" \
        --checkpoint "$B_CKPT" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR"

    echo "--- Evaluating Variant C, lambda=$LMBDA ---"
    C_CKPT="$(cat "$CKPT_DIR/latest_variant_C_lmbda_${LMBDA}_quality_${QUALITY}_mse.txt")"
    python evaluate.py --variant C --quality $QUALITY --lmbda "$LMBDA" \
        --checkpoint "$C_CKPT" \
        --data-dir "$KODAK_DIR" --output "$RESULTS_DIR"
done

for MODEL_NAME in "${PRETRAINED_COMPARE_MODELS[@]}"; do
    for IDX in "${!LAMBDAS[@]}"; do
        LMBDA="${LAMBDAS[$IDX]}"
        QUALITY="${QUALITIES[$IDX]}"
        echo "--- Evaluating ${MODEL_NAME}, lambda=$LMBDA ---"
        python evaluate.py --variant "$MODEL_NAME" --quality $QUALITY --lmbda "$LMBDA" \
            --data-dir "$KODAK_DIR" --output "$RESULTS_DIR" \
            --run-name "variant_${MODEL_NAME}_lmbda_${LMBDA}_quality_${QUALITY}_mse_${RUN_TIMESTAMP}"
    done
done

echo ""
echo "=== Step 4: Plot RD curves ==="
python plot.py --results-dir "$RESULTS_DIR" --output "$PLOTS_DIR"

echo ""
echo "=== Done! ==="
echo "Checkpoints: $CKPT_DIR"
echo "Results:     $RESULTS_DIR"
echo "Plots:       $PLOTS_DIR"
