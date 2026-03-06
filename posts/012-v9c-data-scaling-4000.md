# Post 012: V9c — Scaling Law Breaks at ~4000 Prompts

**Date**: 2026-03-06
**Status**: Complete

---

## The Hypothesis

V9b validated the log₂ data scaling law at three consecutive points with zero error:

```
OOD CLIP % = 0.0115 × log2(prompts) + 0.7744
```

| Version | Prompts | Predicted | Actual |
|---|---|---|---|
| V6 | 174 | 86.0% | 86.0% ✓ |
| V7 | 1,002 | 88.9% | 88.9% ✓ |
| V9b | 2,132 | 90.0% | 90.0% ✓ |
| **V9c** | **4,007** | **91.2%** | **88.8% ✗** |

V9c was expected to gain +1.2pp. Instead it **lost 1.2pp** — a 2.4pp miss from the prediction.

---

## V9c Config

**Step 1 — New teacher latents**: 1,875 new diverse prompts (`prompts_v9c_new.txt`) covering extreme sports, deep-sea environments, traditional crafts, wildlife, Indigenous cultures, polar research, geological surveys, and more. 28-step BF16 inference at 1024px, seed=0. ~7.2h.

**Step 2 — Merge datasets**: V9b combined (2,504 items, 2,132 prompts) + new (1,875 items) = **4,379 items, 4,007 unique prompts**.

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

## Results

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | vs BF16 |
|---|---|---|
| BF16 | 0.322 | 100% |
| V9b | 0.3250 | 100.9% |
| **V9c** | **0.3321** | **103.1%** |

Fixed-prompt CLIP improved significantly (+2.2pp vs V9b) — the model generalizes better to the training prompts with more data.

### Diverse-Prompt Eval (20 unseen prompts)

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.842 | ref | 100% |
| V7 | 0.3001 | 5.905 | 0.668 | 88.9% |
| V9b | 0.3036 | 5.939 | 0.664 | 90.0% |
| **V9c** | **0.2998** | **5.998** | **0.683** | **88.8%** |

**V9c is worse than V9b on CLIP and LPIPS, but better on aesthetics.**

### Per-Prompt Breakdown

| Prompt | BF16 | V9b | V9c | V9c% | vs V9b |
|---|---|---|---|---|---|
| p00 Lion | 0.3276 | 0.3128 | 0.3190 | 97.4% | +6.2 |
| p01 Parrot | 0.3220 | 0.3232 | 0.3133 | 97.3% | −9.9 |
| p02 Wolf | 0.3417 | 0.3340 | **0.3466** | **101.4%** | +12.6 |
| p03 Cathedral | 0.3134 | 0.2941 | 0.2643 | 84.3% | −29.8 |
| p04 Skyscraper | 0.3246 | 0.3000 | 0.3000 | 92.4% | 0.0 |
| p05 Pagoda | 0.3510 | 0.3507 | 0.3424 | 97.5% | −8.3 |
| p06 N. Lights | 0.3006 | 0.2873 | 0.2201 | 73.2% | **−67.2** |
| p07 Volcano | 0.3456 | 0.3355 | **0.3445** | 99.7% | +9.0 |
| p08 Lavender | 0.3384 | 0.3262 | 0.3117 | 92.1% | −14.5 |
| p09 Fisherman | 0.3845 | 0.3263 | 0.3259 | 84.8% | −0.4 |
| p10 Ballet | 0.3433 | 0.2616 | 0.2707 | 78.8% | +9.1 |
| p11 Musician | 0.3677 | 0.3133 | 0.3064 | 83.3% | −6.9 |
| p12 Sushi | 0.3122 | 0.2181 | 0.1785 | 57.2% | **−39.6** |
| p13 Bread | 0.3433 | 0.2672 | **0.2974** | 86.6% | **+30.2** |
| p14 Dragon | 0.3668 | 0.2976 | **0.3139** | 85.6% | +16.3 |
| p15 Astronaut | 0.3365 | 0.3071 | **0.3191** | 94.8% | +12.0 |
| p16 Magic forest | 0.3684 | 0.3404 | 0.3238 | 87.9% | −16.6 |
| p17 Venice | 0.3179 | 0.2850 | 0.2912 | 91.6% | +6.2 |
| p18 Renaissance | 0.2902 | 0.2918 | 0.2915 | 100.4% | −0.3 |
| p19 Tokyo neon | 0.3534 | 0.3006 | **0.3159** | 89.4% | +15.3 |

**Wins vs V9b (≥+10)**: p13 bread (+30.2), p14 dragon (+16.3), p19 Tokyo (+15.3), p02 wolf (+12.6), p15 astronaut (+12.0)

**Losses vs V9b (≤−10)**: p06 N.lights (**−67.2**), p12 sushi (−39.6), p03 cathedral (−29.8), p16 forest (−16.6), p08 lavender (−14.5)

---

## Why the Scaling Law Broke

Three hypotheses for V9c's regression:

### 1. Capacity saturation at rank-64
The ternary+LoRA-64 architecture may have reached its expressivity limit around 2,000 prompts. Adding more data doesn't help because the model lacks the capacity to learn more diverse velocity fields. The 3-point log₂ fit was coincidental — the true capacity ceiling is ~90%, not ~92%.

### 2. Too many training steps (overtraining)
V9c ran 8,000 steps with cosine LR decay on 4,379 images. That's ~1.83 effective epochs (8000 × 1 / 4379). But the optimal checkpoint may have been at step 4000–5000. With cosine decay, the final LR was 1.0e-6 — the model may have settled into a slightly worse local minimum. V9b trained for 6,000 steps on 2,504 images (2.4 effective epochs), which may have been closer to optimal.

### 3. Distribution dilution
The 1,875 new prompts cover very niche domains (polar research, deep-sea biology, traditional crafts). These may pull the model's limited capacity away from common domains (landscapes, architecture, food) without improving the test set. The eval prompts are relatively mainstream — they don't test deep-sea or polar imagery.

The most likely cause is a combination of **1 (capacity saturation)** and **3 (distribution dilution)**. The model trades semantic precision on mainstream prompts for breadth on niche domains. The p06 northern lights regression (95.6% → 73.2%) and p12 sushi (69.9% → 57.2%) suggest the model is "forgetting" some domains as it learns new ones.

---

## Updated Scaling Law

The log₂ scaling law no longer holds:

| Prompts | Predicted | Actual | Error |
|---|---|---|---|
| 174 (V6) | 86.0% | 86.0% | 0.0pp |
| 1,002 (V7) | 88.9% | 88.9% | 0.0pp |
| 2,132 (V9b) | 90.0% | 90.0% | 0.0pp |
| **4,007 (V9c)** | **91.2%** | **88.8%** | **−2.4pp** |

The model hit a wall at ~90% OOD CLIP. Further data scaling provides diminishing (or negative) returns without increasing model capacity.

---

## Key Insight: Aesthetic vs Semantic Trade-off

V9c improved aesthetics (5.939 → 5.998) while degrading CLIP alignment (90.0% → 88.8%). The model produces more visually appealing images but they're less semantically matched to prompts. This is the signature of **capacity-limited training**: the model can't simultaneously maintain semantic precision and improve visual quality, so it optimizes the easier objective (aesthetics/LPIPS) at the expense of the harder one (prompt-image alignment).

---

## Path Forward

### Option A: Fewer steps (V9c-retry at 4000–5000 steps)
If overtraining is the primary cause, a shorter training run may recover V9b's 90.0% while still benefiting from the larger dataset. Low cost (~3h training).

### Option B: Higher LoRA rank (V10 — rank-128)
If capacity saturation is the primary cause (most likely), doubling the LoRA rank from 64 to 128 is the right fix. This gives the model 2× more trainable parameters to learn diverse velocity fields. Cost: ~2× training memory; may need smaller batch or higher grad-accum.

### Option C: Dataset curation
Restrict the training set to prompts that are distributionally similar to the eval set (mainstream imagery: animals, landscapes, architecture, food, portraits). Remove niche domains (polar research, deep-sea, traditional crafts). This tests hypothesis 3 directly.

**Recommendation**: Option B (rank-128). The 3-point scaling law working perfectly up to 90% and then breaking suggests the model hit a capacity ceiling, not a data quality issue.

---

GPU: NVIDIA A100-SXM4-80GB
Total pipeline: ~13.6h (7.2h dataset gen + ~0.1h merge + ~7.2h training + ~30min eval)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
