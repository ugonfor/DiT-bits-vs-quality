#!/bin/bash
# V9a: Online FM at LR=1e-4 (same as V7 offline LR, but with 5-step denoising)
# Hypothesis: V4's grid artifacts were caused by single-step pseudo-z_0 noise,
# not the LR itself. 5-step pseudo-z_0 is cleaner → LR=1e-4 should be safe.
# If this works, online FM finally beats offline FM. If artifacts appear → cap is ~3e-5.
# Warm-start from V7 (best offline checkpoint).
set -e
PYTHON=/home/jovyan/conda/dit-bits-vs-quality/bin/python
WD=/home/ugonfor/DiT-bits-vs-quality/DiT-bits-vs-quality
cd "$WD"

V7_CKPT=output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt
PROMPTS_FILE=prompts_1000.txt

echo "=== V9a: Online FM, LR=1e-4, 5-step denoising ==="
echo "  Warm-start: $V7_CKPT"

echo ""
echo "=== Step 1: Train V9a (1000 steps — early stop to check for artifacts) ==="
# Conservative steps first (1000) to check if artifacts appear at LR=1e-4 with 5-step.
# If step 500 eval images look clean → extend to 2000 steps.
PYTHONUNBUFFERED=1 $PYTHON train_ternary.py \
    --steps 1000 \
    --rank 64 \
    --res 1024 \
    --lr-lora 1e-4 \
    --lr-scale 3e-4 \
    --grad-checkpointing \
    --grad-accum 4 \
    --loss-type online \
    --online-steps 5 \
    --lpips-weight 0.1 \
    --t-dist logit-normal \
    --calib-prompts-file "$PROMPTS_FILE" \
    --init-ckpt "$V7_CKPT" \
    2>&1 | tee output/train_v9a.log

echo ""
echo "=== Step 2: CLIP eval on 4 fixed prompts ==="
V9A_CKPT=output/ternary_distilled_r64_res1024_s1000_online_lpips1e-01.pt
echo "  Using checkpoint: $V9A_CKPT"
PYTHONUNBUFFERED=1 $PYTHON eval_ternary_clip.py \
    --ckpt "$V9A_CKPT" \
    --rank 64 \
    --save-dir output/eval_fm_clip_v9a \
    2>&1 | tee output/eval_v9a_clip.log

echo ""
echo "=== Step 3: Diverse prompt eval (20 unseen prompts) ==="
HF_HOME=/home/jovyan/.cache/huggingface \
PYTHONUNBUFFERED=1 $PYTHON eval_diverse.py \
    --models "V7:${V7_CKPT}" "V9a:${V9A_CKPT}" \
    --rank 64 \
    2>&1 | tee output/eval_v9a_diverse.log

echo ""
echo "=== V9a pipeline complete ==="
