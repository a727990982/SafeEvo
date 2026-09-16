#!/usr/bin/env bash
# Run the free/restricted checkpoint chain with MLP-only extraction.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
: "${BASE_MODEL:?Set BASE_MODEL to the starting model}"
: "${SFT_DIR:?Set SFT_DIR to the directory containing checkpoint-1 through checkpoint-30}"
: "${PAIR_DATA:?Set PAIR_DATA to the ordered contrastive JSON}"
: "${C1_MASK_DIR:?Set C1_MASK_DIR to the per-checkpoint first-pass masks}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
MODE="${MODE:-free}"
case "$MODE" in free|warm) ;; *) echo 'MODE must be free or warm' >&2; exit 2 ;; esac
mkdir -p "$OUTPUT_DIR/logs"

# Both variants share the restricted cold extraction at checkpoint 1.
if [[ ! -f "$OUTPUT_DIR/full_ckpt1/final_safety_masks.pt" ]]; then
  "$PYTHON" "$HERE/extract_checkpoint.py" \
    --base_model_path "$BASE_MODEL" --checkpoint_path "$SFT_DIR/checkpoint-1" \
    --data_path "$PAIR_DATA" \
    --restrict_mask "$C1_MASK_DIR/checkpoint-1/s0.5/best_safety_masks.pt" \
    --output_base "$OUTPUT_DIR" --checkpoint_name full_ckpt1 \
    --sparsity_weight_mlp 0.05 --seed 1 > "$OUTPUT_DIR/logs/full_ckpt1.log" 2>&1
fi
PREV="$OUTPUT_DIR/full_ckpt1/best_safety_masks.pt"
[[ -f "$PREV" ]]
for STEP in $(seq 2 30); do
  TAG="full_ckpt${STEP}_${MODE}"
  if [[ ! -f "$OUTPUT_DIR/$TAG/final_safety_masks.pt" ]]; then
    RESTRICT=()
    if [[ "$MODE" == warm ]]; then
      RESTRICT=(--restrict_mask "$C1_MASK_DIR/checkpoint-$STEP/s0.5/best_safety_masks.pt")
    fi
    "$PYTHON" "$HERE/extract_checkpoint.py" \
      --base_model_path "$BASE_MODEL" --checkpoint_path "$SFT_DIR/checkpoint-$STEP" \
      --data_path "$PAIR_DATA" "${RESTRICT[@]}" --init_masks_path "$PREV" \
      --output_base "$OUTPUT_DIR" --checkpoint_name "$TAG" \
      --sparsity_weight_mlp 0.05 --seed 1 > "$OUTPUT_DIR/logs/$TAG.log" 2>&1
  fi
  PREV="$OUTPUT_DIR/$TAG/best_safety_masks.pt"
  [[ -f "$PREV" ]]
done
echo "Completed $MODE chain: $OUTPUT_DIR"
