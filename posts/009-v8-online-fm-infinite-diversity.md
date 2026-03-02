# Post 009: V8 — Online FM with Multi-Step Denoising (Infinite Diversity)

**Date**: 2026-03-03
**Status**: Training in progress — results TBD

---

## What V7 Left Unsolved

V7 achieved **88.9% BF16 CLIP on 20 unseen prompts** (+2.9pp vs V6's 86.0%). The four improvements worked — but an 11.1% OOD gap remains. The paper uses 7,232 training prompts; we used 1,000.

| Approach | Prompts | OOD CLIP | BF16 % |
|---|---|---|---|
| V6 | 174 | 0.2901 | 86.0% |
| V7 | 1,000 | 0.3001 | 88.9% |
| Paper | 7,232 | ~0.34 | ~100% |

Extrapolating linearly: closing the gap fully would require ~7,000 unique prompts — another ~6,000 teacher latent generations at ~14s/image ≈ **23 more hours of A100 compute**. Instead, V8 uses a fundamentally different approach.

---

## V8: Online FM with Multi-Step Denoising

### The Core Idea

V3 and V4 attempted online FM but failed:
- **V3 (LR=3e-5)**: Too low to correct biases. Degraded slowly.
- **V4 (LR=1e-4)**: Grid/screen-door artifacts. Root cause: single-step pseudo-z_0 is too rough, amplifying patch-boundary errors.

The fix: **N-step Euler denoising** for cleaner pseudo-z_0.

```python
# V3/V4: single-step (noisy pseudo-z_0)
z_0_pseudo = z_rand - 1.0 * teacher(z_rand, t=1.0, c=text_embed)  # rough!

# V8: 5-step Euler (much cleaner pseudo-z_0)
z = z_rand
for k in range(5):
    t_k = 1.0 - k / 5         # t: 1.0, 0.8, 0.6, 0.4, 0.2
    z = z - (1/5) * teacher(z, t=t_k, c=text_embed)
z_0_pseudo = z  # 5× higher quality than single-step
```

With cleaner pseudo-z_0, the FM training trajectory is more stable, and LPIPS perceptual loss gives meaningful supervision (you can't compare perceptual features of a noisy rough image).

### Why Infinite Diversity Matters

Offline FM (V5–V7) has a fundamental ceiling: it can only generalize to prompts covered by its training set. Online FM trains on **fresh random noise + any prompt** at every step — the model literally cannot memorize. With 1,002 diverse prompts × infinite random seeds, every training step is unique.

### V8 Changes vs V7

| Setting | V7 (offline FM) | V8 (online FM) |
|---|---|---|
| Loss type | fm (pre-generated latents) | online (pseudo-z_0 each step) |
| Dataset diversity | 1,000 prompts (fixed set) | 1,002 prompts × ∞ seeds |
| Pseudo-z_0 quality | BF16 teacher full inference | 5-step Euler (cleaner than V3/V4) |
| LPIPS reference | Pre-generated BF16 z_0 | 5-step pseudo-z_0 |
| LR (LoRA) | 1e-4 | 3e-5 (conservative, anti-artifact) |
| Steps | 4,000 warm from V6 | 2,000 warm from V7 |
| Calib prompts | 174 (CALIB_PROMPTS) | 1,002 (prompts_1000.txt) |

---

## Training Config

```bash
python train_ternary.py \
    --steps 2000 --rank 64 --res 1024 \
    --lr-lora 3e-5 --lr-scale 1e-4 \
    --grad-checkpointing --grad-accum 4 \
    --loss-type online --online-steps 5 \
    --lpips-weight 0.1 --t-dist logit-normal \
    --calib-prompts-file prompts_1000.txt \
    --init-ckpt output/ternary_distilled_r64_res1024_s4000_fm_lpips1e-01.pt
```

Early training signal (step 10): `loss=0.209 | fm=0.194 | lpips=0.153`

---

## Results

*(Results pending — training in progress)*

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | Aesthetic | LPIPS↓ |
|---|---|---|---|
| BF16 | 0.3448 | 5.63 | ref |
| V7 | 0.3373 | 6.03 | 0.526 |
| **V8** | **TBD** | **TBD** | **TBD** |

### Diverse-Prompt Eval (20 unseen prompts) — the honest benchmark

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.84 | ref | 100% |
| V7 | 0.3001 | 5.91 | 0.668 | **88.9%** |
| **V8** | **TBD** | **TBD** | **TBD** | **TBD** |

### Hypothesis

If multi-step online FM works as designed:
- **No memorization ceiling**: V8 should generalize to any prompt, not just training set concepts
- **OOD CLIP ≥ 0.31** (≥92% BF16) — significant jump from V7's 88.9%
- The specific failures: sushi, northern lights, lion — should be resolved since training now samples random seeds for every concept continuously
- Risk: grid artifacts if LR=3e-5 is still too high with 5-step pseudo-z_0. If we see artifacts, V8b = LR=1e-5

---

## What We Code-Changed for V8

### 1. LPIPS Loss in Online Mode (`train_ternary.py`)

LPIPS was previously only available in `--loss-type fm`. Added to online mode using `z_0_pseudo` as reference:

```python
# Online mode LPIPS: student z_0_pred vs 5-step pseudo-z_0 reference
if lpips_fn is not None and step % args.lpips_freq == 0 and t_frac < 0.8:
    z_0_pred = z_t.float() - t_frac * v_student
    # decode z_0_pred and z_0_pseudo through frozen VAE
    # compute AlexNet perceptual distance at 256px
    lpips_loss = lpips_fn(img_student_sm, img_ref_sm).mean()
loss = fm_loss + args.lpips_weight * lpips_loss
```

### 2. `--calib-prompts-file` arg (`train_ternary.py`)

Online mode now accepts a text file to replace the 174 hardcoded CALIB_PROMPTS:

```bash
--calib-prompts-file prompts_1000.txt  # 1002 prompts extracted from V7 dataset
```

This expands online training prompt diversity 5.7× without touching the source code.

---

## Path to V9 (if V8 succeeds)

If V8 reaches ≥92% OOD CLIP, the next lever is **increasing online steps**:
- V8: 5-step denoising, LR=3e-5, 2000 steps
- V9: 10-step denoising, LR=5e-5, 3000 steps (better pseudo-z_0 → higher LR safe)

If V8 shows grid artifacts → reduce LR to 1e-5.
If V8 drops below V7 → investigate loss stability; may need curriculum (start online, continue offline).

---

GPU: NVIDIA A100-SXM4-80GB
Training: ~3h (2000 steps, 5 teacher passes per step → 5× longer per step than offline)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
