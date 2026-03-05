#!/bin/bash
# V9c: Offline FM data scaling — 1875 new prompts → ~4007 total unique prompts
# Hypothesis: log2 scaling law predicts ~91.2% OOD CLIP at ~4007 prompts (vs 90.0% at 2132)
# Warm-start from V9b (90.0% OOD CLIP baseline)
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

V9B_CKPT=output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt
V9B_COMBINED=output/teacher_dataset_v9b_combined.pt
NEW_DATASET=output/teacher_dataset_v9c_new.pt
COMBINED_DATASET=output/teacher_dataset_v9c_combined.pt

echo "=== V9c: Offline FM data scaling ==="
echo "  Warm-start: $V9B_CKPT"
echo "  New prompts: prompts_v9c_new.txt (1875 prompts)"
echo "  Target total: ~4007 unique prompts"

echo ""
echo "=== Step 1: Generate teacher latents for 1875 new prompts ==="
# ~2.9h at ~5.5s/image (1875 images)
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON generate_teacher_dataset.py \
    --n-images 1875 \
    --steps 28 \
    --res 1024 \
    --seed 0 \
    --prompts-file prompts_v9c_new.txt \
    --out "$NEW_DATASET" \
    2>&1 | tee output/gen_v9c_new.log

echo ""
echo "=== Step 2: Merge V9b combined + new dataset ==="
PYTHONUNBUFFERED=1 $PYTHON merge_datasets.py \
    --datasets "$V9B_COMBINED" "$NEW_DATASET" \
    --out "$COMBINED_DATASET" \
    2>&1 | tee output/merge_v9c.log

echo ""
echo "=== Step 3: Train V9c (8000 steps offline FM, warm-start from V9b) ==="
# V9b was 6000 steps on 2132 prompts (2504 images). V9c: 8000 steps on ~4382 images
# 8000 steps × 2.7s = ~6h
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 8000 \
    --rank 64 \
    --res 1024 \
    --lr-lora 1e-4 \
    --lr-scale 3e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 \
    --t-dist logit-normal \
    --dataset "$COMBINED_DATASET" \
    --init-ckpt "$V9B_CKPT" \
    2>&1 | tee output/train_v9c.log

echo ""
echo "=== Step 4: CLIP eval on 4 fixed prompts ==="
V9C_CKPT=output/ternary_distilled_r64_res1024_s8000_fm_lpips1e-01.pt
echo "  Using checkpoint: $V9C_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V9C_CKPT" \
    --rank 64 \
    --save-dir output/eval_fm_clip_v9c \
    2>&1 | tee output/eval_v9c_clip.log

echo ""
echo "=== Step 5: Diverse prompt eval (20 unseen prompts) ==="
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V9b:${V9B_CKPT}" "V9c:${V9C_CKPT}" \
    --rank 64 \
    2>&1 | tee output/eval_v9c_diverse.log

echo ""
echo "=== V9c pipeline complete ==="
