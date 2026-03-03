#!/bin/bash
# V9b: Offline FM data scaling — 1130 new prompts → ~2130 total unique prompts
# Hypothesis: log2 scaling law predicts ~90.2% OOD CLIP at ~2130 prompts (vs 88.9% at 1002)
# Offline FM avoids the distribution mismatch that limits online FM (V8, V9a)
# Warm-start from V7 (88.9% OOD CLIP baseline)
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

V7_CKPT=output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt
V7_DATASET=output/teacher_dataset_v7.pt
NEW_DATASET=output/teacher_dataset_v9b_new.pt
COMBINED_DATASET=output/teacher_dataset_v9b_combined.pt

echo "=== V9b: Offline FM data scaling ==="
echo "  Warm-start: $V7_CKPT"
echo "  New prompts: prompts_v9b_new.txt (1130 prompts)"

echo ""
echo "=== Step 1: Generate teacher latents for 1130 new prompts ==="
# ~1.7h at ~5.5s/image; generates 1 image per prompt
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON generate_teacher_dataset.py \
    --n-images 1130 \
    --steps 28 \
    --res 1024 \
    --seed 0 \
    --prompts-file prompts_v9b_new.txt \
    --out "$NEW_DATASET" \
    2>&1 | tee output/gen_v9b_new.log

echo ""
echo "=== Step 2: Merge V7 dataset + new dataset ==="
PYTHONUNBUFFERED=1 $PYTHON merge_datasets.py \
    --datasets "$V7_DATASET" "$NEW_DATASET" \
    --out "$COMBINED_DATASET" \
    2>&1 | tee output/merge_v9b.log

echo ""
echo "=== Step 3: Train V9b (6000 steps offline FM, warm-start from V7) ==="
# V7 was 4000 steps on 1002 prompts (1374 images). V9b: 6000 steps on ~2504 images
# 6000 steps × 2.7s = ~4.5h
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 6000 \
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
    --init-ckpt "$V7_CKPT" \
    2>&1 | tee output/train_v9b.log

echo ""
echo "=== Step 4: CLIP eval on 4 fixed prompts ==="
V9B_CKPT=output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt
echo "  Using checkpoint: $V9B_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V9B_CKPT" \
    --rank 64 \
    --save-dir output/eval_fm_clip_v9b \
    2>&1 | tee output/eval_v9b_clip.log

echo ""
echo "=== Step 5: Diverse prompt eval (20 unseen prompts) ==="
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V7:${V7_CKPT}" "V9b:${V9B_CKPT}" \
    --rank 64 \
    2>&1 | tee output/eval_v9b_diverse.log

echo ""
echo "=== V9b pipeline complete ==="
