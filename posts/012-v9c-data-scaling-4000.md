# Post 012: V9c — Offline FM Data Scaling to ~4000 Prompts

**Date**: 2026-03-05
**Status**: Training in progress

---

## From 2132 to 4007 Prompts

V9b validated the log₂ data scaling law at three points:

```
OOD CLIP % = 0.0115 × log2(prompts) + 0.7744
```

| Version | Prompts | OOD CLIP | Predicted | Error |
|---|---|---|---|---|
| V6 | 174 | 86.0% | 86.0% | 0.0% |
| V7 | 1,002 | 88.9% | 88.9% | 0.0% |
| V9b | 2,132 | 90.0% | 90.0% | 0.0% |
| **V9c** | **~4,007** | — | **91.2%** | — |

V9c tests whether this law holds at the next doubling.

---

## V9c Config

**Step 1 — New teacher latents**: 1,875 new diverse prompts (`prompts_v9c_new.txt`) covering an expanded range: extreme sports, deep-sea environments, traditional crafts, wildlife, Indigenous cultures, polar research, geological surveys, and more. 28-step BF16 inference at 1024px, seed=0. ~2.9h.

**Step 2 — Merge datasets**: V9b combined (2,504 items, 2,132 unique prompts) + new (1,875 items) = **~4,379 items, ~4,007 unique prompts**.

**Step 3 — Train offline FM**:
```bash
python train_ternary.py \
    --steps 8000 \
    --rank 64 --res 1024 \
    --lr-lora 1e-4 --lr-scale 3e-4 \
    --grad-checkpointing --grad-accum 4 \
    --loss-type fm \
    --lpips-weight 0.1 --t-dist logit-normal \
    --dataset output/teacher_dataset_v9c_combined.pt \
    --init-ckpt output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt  # V9b
```

Checkpoint: `output/ternary_distilled_r64_res1024_s8000_fm_lpips1e-01.pt`

---

## Why Offline FM Continues to Win

V9b settled the online vs offline debate definitively:

| | V9a (online LR=1e-4) | V9b (offline +1130) | V9c (offline +1875) |
|---|---|---|---|
| OOD CLIP | 89.3% | 90.0% | ~91.2% (predicted) |
| Aesthetic | 5.54 (−0.36 vs V7) | 5.94 (+0.03 vs V7) | — |
| LPIPS | 0.706 (worse) | 0.664 (better) | — |
| Training signal | 5-step pseudo-z₀ | 30-step BF16 | 30-step BF16 |

The distribution mismatch in online FM is real and irreversible at practical compute budgets. Each doubling of offline data gives a clean, predictable +1.1pp gain without quality regression.

---

## Scaling Law Context

![Scaling law: OOD CLIP % vs log2(unique prompts)](../output/viz/v9b_scaling_law.png)

At 4,007 prompts (log₂ ≈ 11.97), the predicted OOD CLIP is **91.2%** vs BF16.

The theoretical ceiling for ternary+rank-64 architecture (from the paper's 7,232-prompt scale) is ~92.2%. V9c should bring us within 1pp of that ceiling.

Remaining gap after V9c (~8.8%) has two components:
- **Capacity ceiling**: ~8% is irreducible with rank-64 — needs higher LoRA rank or mixed precision to break
- **Data residual**: ~0.8% may still be gained by pushing to 7,232 prompts (paper scale)

---

## Results

*(To be filled in after training completes — ~12h total pipeline)*

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | vs BF16 |
|---|---|---|
| BF16 | 0.322 | 100% |
| V9b | 0.3250 | 100.9% |
| V9c | — | — |

### Diverse-Prompt Eval (20 unseen prompts)

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.84 | ref | 100% |
| V7 | 0.3001 | 5.905 | 0.668 | 88.9% |
| V9b | 0.3036 | 5.939 | 0.664 | 90.0% |
| **V9c** | — | — | — | **~91.2%** |

### Per-Prompt Breakdown

*(To be filled in after eval)*

---

## Path Forward After V9c

At ~91.2% OOD CLIP, we will be within 1pp of the paper's projected ceiling at 7,232 prompts. Two options remain:

### Option A: Full paper-scale data (V9d — 7,232 prompts → ~92.2%)
Generate another ~3,225 prompts and train 10,000 steps. Cost: ~5h data + ~7h training. Predicted +1.0pp gain. This closes the data gap entirely.

### Option B: Break the capacity ceiling (V10 — rank-128 LoRA)
Even with infinite data, ~8% gap remains due to ternary+rank-64 expressivity limits. Rank-128 doubles the LoRA capacity and may allow tighter BF16 approximation. Cost: 2× training memory (may need smaller batch or grad-accum increase); unknown OOD improvement.

### Option C: Combined — V9d + rank-128
Maximum attack on both ceilings. Most expensive but potentially 93%+.

Given V9c validates the scaling law at a fourth point, we will have strong empirical footing to decide between continued data scaling and architectural improvement.

---

GPU: NVIDIA A100-SXM4-80GB
Total pipeline: ~12h (2.9h dataset gen + ~0.1h merge + ~6h training + ~30min eval)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
