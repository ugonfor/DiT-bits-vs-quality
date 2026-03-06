#!/bin/bash
# V10: Rank-128 LoRA — break the capacity ceiling
# V9c showed rank-64 saturates at ~90% OOD CLIP. Rank-128 doubles LoRA capacity.
# Fresh SVD init (can't warm-start from rank-64 checkpoint — shape mismatch).
# Same dataset as V9b (2132 prompts) to isolate rank as the only variable.
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

DATASET=output/teacher_dataset_v9b_combined.pt
RANK=128

echo "=== V10: Rank-128 LoRA ==="
echo "  Dataset: $DATASET (2132 unique prompts)"
echo "  Rank: $RANK (2x V9b's rank-64)"
echo "  Init: Fresh SVD (no warm-start)"

echo ""
echo "=== Step 1: Train V10 (6000 steps offline FM, rank-128, SVD init) ==="
# 6000 steps × ~3.5s = ~5.8h (rank-128 is slower than rank-64)
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 6000 \
    --rank $RANK \
    --res 1024 \
    --lr-lora 1e-4 \
    --lr-scale 3e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 \
    --t-dist logit-normal \
    --dataset "$DATASET" \
    2>&1 | tee output/train_v10.log

echo ""
echo "=== Step 2: CLIP eval on 4 fixed prompts ==="
V10_CKPT=output/ternary_distilled_r${RANK}_res1024_s6000_fm_lpips1e-01.pt
echo "  Using checkpoint: $V10_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V10_CKPT" \
    --rank $RANK \
    --save-dir output/eval_fm_clip_v10 \
    2>&1 | tee output/eval_v10_clip.log

echo ""
echo "=== Step 3: Diverse prompt eval (20 unseen prompts) ==="
V9B_CKPT=output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V9b:${V9B_CKPT}:64" "V10:${V10_CKPT}:${RANK}" \
    --rank $RANK \
    2>&1 | tee output/eval_v10_diverse.log

echo ""
echo "=== V10 pipeline complete ==="
