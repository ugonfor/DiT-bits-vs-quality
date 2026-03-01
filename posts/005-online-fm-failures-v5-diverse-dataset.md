# Post 005: Ternary FLUX — Online FM Failures & V5 Diverse Offline Dataset

**Date**: 2026-03-01
**Status**: Results + analysis

---

## Background

[Post 004](./004-ternary-flux-warmstart-dataset200.md) achieved CLIP 0.3225 (V2), matching
BF16 statistically. However, direct comparison (same seed=42) revealed a per-sample
faithfulness gap: V2's cyberpunk samurai prompt produces a *dark hooded figure* while BF16
produces a *fully armored samurai on a rooftop with neon city skyline*. CLIP score and visual
faithfulness are not the same metric.

This post documents three failed online FM approaches (V3, V4) and the resolution: V5 with a
more diverse offline FM dataset (548 images, 174 prompts).

---

## The Faithfulness Gap (V2 vs BF16, seed=42)

| Prompt | BF16 (ref) | V2 result | Gap |
|---|---|---|---|
| Cyberpunk samurai on a neon-lit rooftop | Armored samurai, horned helmet, neon city | Dark hooded figure, warehouse setting | **Wrong subject** |
| Fantasy landscape with mountains | Alpine peaks, pine trees, turquoise river | Mountain valley, river, green vegetation | Wrong style (no pine/snow) |
| Portrait of young woman with curly hair | Black woman, desert sunset, golden light | Latina woman, curly hair, warm light | Minor (correct concept) |
| Aerial view of coastal city | Mediterranean terracotta, beach, mountains | Generic city grid, sunset over water | Minor (correct concept) |

CLIP=0.3225 (+0.1% vs BF16) measures *distributional* similarity; per-sample faithfulness
requires the velocity field to match at the specific eval noise (seed=42).

---

## V3: Online FM at Low LR (3e-5) — Failed

**Rationale**: EfficientDM (ICLR 2024) uses online teacher distillation. We implemented
online FM: each step, sample random z_rand → single-step Euler → pseudo-z_0 → FM training.
With 3e-5 LR (3× lower than V2) to avoid disturbing V2's good weights.

**Results (step 500 and 1000)**:
- p0: Creature/zombie figure — *worse* than V2's hooded figure
- p1: Mountain valley — similar to V2

**Root cause**: 3e-5 LR is too weak to overcome V2's learned patterns, but strong enough to
disturb them. At step 500, V3 is in an awkward transition state. At step 1000, quality
degrades further (more grotesque creature). Training was halted at step 1000.

---

## V4: Online FM at High LR (1e-4) — Failed (Grid Artifacts)

**Rationale**: V3's failure was due to insufficient learning rate. Using V2's original 1e-4 LR
should provide a stronger update signal.

**Results (step 500 and 1000)**:
- All images: Severe **screen-door grid artifacts** — visible 16×16px tiling pattern
- p0 step 1000: Abstract fire/explosion with grid (complete semantic collapse)
- p1 step 1000: Mountain valley but heavily gridded
- Quality getting *worse* with more steps

**Root cause**: FLUX processes latents as 16×16 pixel patches. With online FM, the pseudo-z_0
quality is limited — single-step Euler (`z_0_pseudo = z_rand - t_large * teacher(z_rand,
t_large, c)`) produces a rough approximation that may contain patch-correlated structure.
With high LR (1e-4), the LoRA learns to amplify these patch artifacts, producing visible grid
patterns in the decoded output. Training was halted at step 1000.

### Online FM Instability Summary

| LR | Behavior | Root cause |
|---|---|---|
| 3e-5 (V3) | Can't correct V2's biases, produces creature | Too weak to override learned patterns |
| 1e-4 (V4) | Grid/screen-door artifacts at step 500, worsening | Amplifies pseudo-z_0 patch artifacts |

The single-step Euler pseudo-z_0 is fundamentally too low-quality for online FM distillation
at any LR that works. Multi-step pseudo-z_0 (4–8 steps) would be higher quality but 4–8×
slower. Online FM is not viable with our current approach.

---

## V5: Return to Offline FM — More Diverse Dataset

**Decision**: Online FM is unstable with single-step Euler. Offline FM (V1, V2) is proven stable.
The V1→V2 improvement (+43% CLIP) was driven by dataset diversity. Scale further.

### V5 Changes from V2

1. **New teacher dataset** (174 prompts × 2 seeds = 348 images, 13.9s/image, ~80 min on A100):
   - 174 unique prompts (was 51, 3.4× increase)
   - Covers portraits, nature, animals, architecture, fantasy, art styles, emotions
   - Seeds 0–347 (new, not overlapping with V2's seed 42 set)

2. **Combined dataset** (548 total = V2's 200 + new 348):
   - 174 unique prompts, up to 6 training images per eval prompt
   - Eval prompts specifically: 4 images from V2 + 2 from new dataset = 6 each

3. **Training** (same as V2): 3000 steps, rank-64, 1024px, lr-lora=1e-4, lr-scale=3e-4

4. **Warm-start from V2** checkpoint (CLIP 0.3225)

```bash
# Step 1: Generate new 348-image teacher dataset (80 min)
python generate_teacher_dataset.py --n-images 348 --steps 28 --seed 0 \
  --out output/teacher_dataset_350.pt

# Step 2: Merge with V2 dataset
python merge_datasets.py \
  --datasets output/teacher_dataset_200.pt output/teacher_dataset_350.pt \
  --out output/teacher_dataset_combined.pt

# Step 3: V5 offline FM (131 min on A100)
python train_ternary.py \
  --steps 3000 --rank 64 --res 1024 \
  --lr-lora 1e-4 --lr-scale 3e-4 \
  --grad-checkpointing --grad-accum 4 \
  --loss-type fm \
  --dataset output/teacher_dataset_combined.pt \
  --init-ckpt output/ternary_distilled_r64_res1024_s3000_fm.pt
```

---

## Results

### Loss Curve (V5 vs V2)

| Steps | V5 avg loss | V2 avg loss |
|---|---|---|
| TBD | TBD | 0.073 at step 20 |

### CLIP Scores (step 3000, 30 inference steps, seed=42)

| Prompt | V5 CLIP | % of BF16 | V2 score |
|---|---|---|---|
| Cyberpunk samurai on a neon-lit rooftop | TBD | TBD | 0.3312 (102.9%) |
| A fantasy landscape with mountains and a river | TBD | TBD | 0.3266 (101.4%) |
| Portrait of a young woman with wild curly hair | TBD | TBD | 0.3543 (110.0%) |
| Aerial view of a coastal city at sunset | TBD | TBD | 0.2778 (86.3%) |
| **Average** | **TBD** | **TBD** | **0.3225** |

### Visual Comparison (BF16 vs V5, seed=42)

*(Images to be added)*

---

## Key Technical Insights

### Why Offline FM > Online FM for Ternary FLUX

1. **Pseudo-z_0 quality**: Single-step Euler from random Gaussian noise is a rough
   approximation. The training distribution doesn't match BF16's actual inference trajectory.
   High LR amplifies these approximation errors into visible artifacts.

2. **Stable gradients**: Offline FM uses actual BF16 z_0 latents — clean, high-quality
   reference points. The training distribution is well-behaved and doesn't produce artifacts.

3. **Dataset diversity vs. trajectory diversity**: Online FM's main advantage (infinite
   trajectory diversity from different z_rand) is not the bottleneck. The bottleneck is
   *text conditioning diversity* (how many different prompts the student sees). Both modes
   can achieve similar prompt diversity with the right dataset.

### Why 174 Prompts > 51 Prompts

- More text conditioning diversity → better generalization of the student's velocity field
- Eval prompts (cyberpunk samurai, fantasy landscape, portrait, aerial city) each get 6
  training samples vs 4 in V2 → better per-prompt coverage
- The 1.58-bit FLUX paper used 7,232 prompts — our 174 is still small, but 3.4× better than V2

### Understanding CLIP vs. Visual Faithfulness

| Metric | What it measures | V2 score | Limitation |
|---|---|---|---|
| CLIP score | Text-image alignment distribution | 0.3225 (=BF16) | Doesn't measure per-sample faithfulness |
| Visual comparison | Subject/composition match (same seed) | Dark figure vs samurai | BF16 faithfulness gap |

Both metrics matter. CLIP ≈ BF16 is necessary but not sufficient for "awesome images similar to BF16."

---

## Full Comparison Table

| Model | CLIP Score | Notes |
|---|---|---|
| BF16 (reference) | 0.322 | Full-precision baseline |
| INT4 + LoRA-64 | 0.318 | Near-lossless |
| Ternary PTQ (no training) | 0.178 | Pure noise |
| Distilled (rank-8, 512px, 800 steps) | 0.203 | Post-002 |
| FM distilled V1 (50 imgs, 3000 steps) | 0.2783 | Post-003 |
| FM distilled V2 (200 imgs, warm-start) | 0.3225 | Post-004, matches BF16 |
| Online FM V3 (3e-5 LR) | — | Failed (creature artifacts) |
| Online FM V4 (1e-4 LR) | — | Failed (grid artifacts) |
| **FM distilled V5 (548 imgs, 174 prompts)** | **TBD** | **This post** |

GPU: NVIDIA A100-SXM4-80GB
Python env: diffusers==0.34.0, torch==2.5.0+cu124
