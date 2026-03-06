# Post 013: V10 — Rank-128 LoRA to Break the Capacity Ceiling

**Date**: 2026-03-06
**Status**: Training in progress

---

## Motivation

V9c proved the ternary+rank-64 architecture hits a capacity ceiling at ~90% OOD CLIP:

| Version | Prompts | OOD CLIP | Rank |
|---|---|---|---|
| V7 | 1,002 | 88.9% | 64 |
| V9b | 2,132 | **90.0%** | 64 |
| V9c | 4,007 | 88.8% (↓) | 64 |

More data didn't help — it hurt. The model can't simultaneously maintain semantic precision and learn broader distributions with only 64-dimensional LoRA corrections. The remaining ~10% gap vs BF16 is a capacity limitation.

V10 doubles the LoRA rank from 64 to 128, giving **2× trainable parameters** (~350M → ~700M LoRA). This tests whether increased expressivity can break through the 90% ceiling.

---

## V10 Config

**Dataset**: V9b combined (`teacher_dataset_v9b_combined.pt`, 2,504 items, 2,132 unique prompts) — same dataset as V9b to isolate rank as the only variable.

**Init**: Fresh SVD initialization (cannot warm-start from rank-64 checkpoint — LoRA A/B matrix dimensions don't match).

**Training**:
```bash
python train_ternary.py \
    --steps 6000 \
    --rank 128 --res 1024 \
    --lr-lora 1e-4 --lr-scale 3e-4 \
    --grad-checkpointing --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 --t-dist logit-normal \
    --dataset output/teacher_dataset_v9b_combined.pt
```

Checkpoint: `output/ternary_distilled_r128_res1024_s6000_fm_lpips1e-01.pt`

**Key differences vs V9b**:
| | V9b (rank-64) | V10 (rank-128) |
|---|---|---|
| LoRA rank | 64 | 128 |
| Trainable params | ~350M | ~700M |
| Checkpoint size | ~668 MB | ~1,330 MB |
| Training VRAM | ~43 GB | ~48-52 GB |
| Speed | ~2.7 s/step | ~3.5 s/step |
| Dataset | 2,132 prompts | 2,132 prompts (same) |
| Init | Warm-start from V7 | Fresh SVD |

---

## Results

*(To be filled in after training completes — ~9h total)*

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | vs BF16 |
|---|---|---|
| BF16 | 0.322 | 100% |
| V9b (r64) | 0.3250 | 100.9% |
| V10 (r128) | — | — |

### Diverse-Prompt Eval (20 unseen prompts)

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.842 | ref | 100% |
| V9b (r64) | 0.3036 | 5.939 | 0.664 | 90.0% |
| **V10 (r128)** | — | — | — | — |

### Interpretation Guide

- **V10 > 90.0%**: Capacity was the bottleneck. Higher rank helps. Path: rank-128 + more data.
- **V10 ≈ 90.0%**: Rank-128 doesn't help on 2,132 prompts. May need more data + higher rank together.
- **V10 < 90.0%**: SVD cold-start penalty. May need more steps or warm-start strategy.

---

## Path Forward (dependent on results)

### If V10 > 90%:
1. V10b: rank-128 + V9c combined dataset (4,007 prompts) — test if higher rank also enables better data scaling
2. V10c: rank-128 + 7,232 prompts (paper scale) — full capacity + data

### If V10 ≈ 90%:
1. V10b: more steps (8000-10000) — cold-start may need longer convergence
2. V10c: rank-256 — even more capacity

### If V10 < 90%:
1. Investigate cold-start convergence — may need partial warm-start (copy scale params, SVD-init only extra rank dims)
2. Higher LR or different schedule for cold start

---

GPU: NVIDIA A100-SXM4-80GB
Estimated pipeline: ~9h (training ~5.8h + eval ~30min)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
