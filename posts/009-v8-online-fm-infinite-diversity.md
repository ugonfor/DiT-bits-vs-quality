# Post 009: V8 — Online FM with Multi-Step Denoising

**Date**: 2026-03-03
**Status**: Complete

---

## What V7 Left Unsolved

V7 achieved **88.9% BF16 CLIP on 20 unseen prompts** (+2.9pp vs V6's 86.0%). The four improvements worked — but an 11.1% OOD gap remains. The paper uses 7,232 training prompts; we used 1,000.

| Approach | Prompts | OOD CLIP | BF16 % |
|---|---|---|---|
| V6 | 174 | 0.2901 | 86.0% |
| V7 | 1,000 | 0.3001 | 88.9% |
| Paper | 7,232 | ~0.34 | ~100% |

Generating 6,000 more teacher latents ≈ 23 hours of A100 compute. Instead, V8 tests whether **infinite-diversity online FM** can close the gap without more data.

---

## V8 Design: Online FM with 5-Step Denoising

V3 and V4 attempted online FM but failed:
- **V3 (LR=3e-5)**: Too conservative — couldn't correct biases.
- **V4 (LR=1e-4)**: Grid/screen-door artifacts. Root cause: single-step pseudo-z_0 was too rough, amplifying patch-boundary errors at high LR.

V8's theory: **5 Euler steps** produce a cleaner pseudo-z_0, making higher-LR training safe.

```python
# V8: 5-step Euler (much cleaner pseudo-z_0 than V3/V4 single-step)
z = z_rand
for k in range(5):
    t_k = 1.0 - k / 5         # t: 1.0, 0.8, 0.6, 0.4, 0.2
    z = z - (1/5) * teacher(z, t=t_k, c=text_embed)
z_0_pseudo = z
```

### V8 Changes vs V7

| Setting | V7 (offline FM) | V8 (online FM) |
|---|---|---|
| Loss type | fm (pre-generated latents) | online (pseudo-z_0 each step) |
| Dataset diversity | 1,000 prompts (fixed set) | 1,002 prompts × ∞ random seeds |
| Pseudo-z_0 quality | 30-step BF16 teacher | 5-step Euler teacher |
| LPIPS reference | Pre-generated BF16 z_0 | 5-step pseudo-z_0 |
| LR (LoRA) | 1e-4 | **3e-5** (conservative, anti-artifact) |
| Steps | 4,000 | 2,000 (warm from V7) |
| Calib prompts | 174 (CALIB_PROMPTS) | 1,002 (prompts_1000.txt) |

New code: LPIPS loss added to online mode (using pseudo-z_0 as reference); `--calib-prompts-file` arg for external prompt lists.

---

## Results

### Fixed-Prompt Eval (4 training prompts)

| Model | CLIP | Aesthetic | LPIPS↓ |
|---|---|---|---|
| BF16 | 0.3448 | 5.63 | ref |
| V7 | 0.3373 | 6.03 | 0.526 |
| **V8** | **0.3372** | **5.84** | **0.529** |

V8 CLIP: identical to V7 (0.3372 vs 0.3373). Aesthetic dropped 0.19 points vs V7.

### Diverse-Prompt Eval (20 unseen prompts) — the honest benchmark

| Model | CLIP | Aesthetic | LPIPS↓ | vs BF16 |
|---|---|---|---|---|
| BF16 | 0.3375 | 5.84 | ref | 100% |
| V7 | 0.3001 | 5.91 | 0.668 | **88.9%** |
| **V8** | **0.2994** | **5.72** | **0.669** | **88.7%** |

**V8 = V7 within noise.** The online FM approach produced no measurable improvement over V7's offline FM.

---

## Why V8 Didn't Improve

### Reason 1: LR was too conservative

At LR=3e-5 with cosine decay, the final effective LR was ~1e-6. Over 2000 steps, the model barely moved from V7's weights. The near-identical CLIP scores (0.3372 vs 0.3373) confirm this — V8 essentially **froze V7's checkpoint** with minimal updates.

This was a deliberate safety choice (avoiding V4's grid artifacts at LR=1e-4), but it meant no benefit from the online signal.

### Reason 2: Distribution mismatch in the training signal

The key theoretical problem with online FM:

- **V7 (offline FM)**: Trains on BF16 teacher z_0 from **30-step inference** → FM trajectories are from the correct distribution
- **V8 (online FM)**: Trains on pseudo-z_0 from **5-step Euler** → trajectories are from a lower-quality distribution

Even if the student learns to perfectly predict velocity for 5-step trajectories, those trajectories are not what happens during 30-step inference. The distribution shift is fundamental.

The fix would be 30-step online denoising — but at 5.5s/step overhead × 6× more steps = 18h training for a 3h run. Not practical.

### Reason 3: The 88.9% level may be real capacity

At ternary + rank-64 LoRA, the model may simply not have enough expressivity to capture the full BF16 velocity field for arbitrary unseen prompts. The offline FM V5→V6→V7 progression (more data → more OOD generalization) suggests the ceiling is data-limited, not capacity-limited. But V8 suggests we can't escape this with online tricks at the current LR.

---

## What V8 Did Confirm

1. **No artifacts**: 5-step online FM with LR=3e-5 is stable — no grid/screen-door patterns
2. **Warm-start works**: Starting from V7 converges immediately (step 10 loss=0.209 → step 20 loss=0.047 → step 50 loss=0.035)
3. **Online LPIPS is viable**: The LPIPS loss against pseudo-z_0 reference is stable (0.04–0.15 range throughout)
4. **LR was the bottleneck** (hypothesis): V8 froze at V7 quality because LR was too low to explore

---

## Path Forward: V9 Options

### Option A: Higher LR Online FM (V9a — 3h, test tonight)

Test LR=1e-4 with 5-step online FM. The grid artifacts in V4 were caused by **single-step** pseudo-z_0 quality, not the LR itself. 5-step pseudo-z_0 is 5× cleaner — hypothesis: LR=1e-4 is now safe.

If V9a succeeds → first online FM model that genuinely improves on offline.
If V9a shows artifacts → LR ceiling for 5-step online is between 3e-5 and 1e-4.

### Option B: Scale Offline Data (V9b — 24h, more reliable)

Generate 2,000 more teacher latents (total 3,000 unique prompts) and train offline FM for 6,000 steps. This follows V5→V6→V7's consistently successful pattern. Expected: ~91-93% OOD CLIP.

### Option C: Higher LoRA Rank (V9c — break capacity ceiling)

If the 88.9% ceiling is capacity-limited: switch to rank-128 LoRA (2× more LoRA params). Doubles training compute but may break through the expressivity floor.

**Plan**: Launch V9a (higher LR online, 3h) now. If no improvement or artifacts → escalate to V9b (more data).

---

GPU: NVIDIA A100-SXM4-80GB
Training: 188 min (2000 steps, 5 teacher passes per step = 5× teacher overhead vs offline)
Python env: diffusers==0.34.0, torch==2.5.0+cu124
