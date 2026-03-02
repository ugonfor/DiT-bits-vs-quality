#!/bin/bash
# V8 pipeline: online FM with multi-step denoising + LPIPS + 1000-prompt calib set
# Key hypothesis: infinite prompt diversity (no memorization) + clean pseudo-z_0 from 5-step denoising
# should close the remaining 11.1% OOD gap without needing more teacher latents.
# Warm-start from V7: output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

V7_CKPT=output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt
PROMPTS_FILE=prompts_1000.txt

echo "=== V8: Online FM, 5-step denoising, LPIPS, 1000-prompt calib ==="
echo "  Warm-start: $V7_CKPT"
echo "  Prompts:    $PROMPTS_FILE"
echo ""

echo "=== Step 1: Train V8 (warm-start from V7, 2000 steps online FM) ==="
# Key improvements vs V7 offline FM:
#  --loss-type online       : infinite prompt diversity, no dataset memorization
#  --online-steps 5         : 5-step Euler denoising for cleaner pseudo-z_0 (vs single-step V4)
#  --calib-prompts-file     : 1002 unique prompts from V7 dataset (vs 174 CALIB_PROMPTS)
#  --lpips-weight 0.1       : perceptual loss vs pseudo-z_0 reference (new: added to online mode)
#  --t-dist logit-normal    : keep intermediate-t focus from V7
#  --lr-lora 3e-5           : conservative LR to avoid V4 grid artifacts
#  --steps 2000             : shorter run to test online stability before committing
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 2000 \
    --rank 64 \
    --res 1024 \
    --lr-lora 3e-5 \
    --lr-scale 1e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type online \
    --online-steps 5 \
    --lpips-weight 0.1 \
    --t-dist logit-normal \
    --calib-prompts-file "$PROMPTS_FILE" \
    --init-ckpt "$V7_CKPT" \
    2>&1 | tee output/train_v8.log

echo ""
echo "=== Step 2: CLIP eval on 4 fixed prompts ==="
V8_CKPT=output/ternary_distilled_r64_res1024_s2000_online_lpips1e-01.pt
echo "  Using checkpoint: $V8_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V8_CKPT" \
    --rank 64 \
    --save-dir output/eval_fm_clip_v8 \
    2>&1 | tee output/eval_v8_clip.log

echo ""
echo "=== Step 3: Quality eval (LPIPS + aesthetic) ==="
PYTHONUNBUFFERED=1 $PYTHON eval_quality.py \
    --models BF16:output/bf16_reference V7:output/eval_fm_clip_v7 V8:output/eval_fm_clip_v8 \
    2>&1 | tee output/eval_v8_quality.log

echo ""
echo "=== Step 4: Diverse prompt eval (20 unseen prompts) ==="
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V7:${V7_CKPT}" "V8:${V8_CKPT}" \
    --rank 64 \
    2>&1 | tee output/eval_v8_diverse.log

echo ""
echo "=== V8 pipeline complete ==="
