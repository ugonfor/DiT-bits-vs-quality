# Post 001: Reproducing 1.58-bit FLUX — PoC Findings

**Date:** 2026-02-28  
**Reference:** [1.58-bit FLUX](https://chenglin-yang.github.io/1.58bit.flux.github.io/) — Yang et al., arXiv 2412.18653

---

## What We Set Out To Do

Reproduce the PoC of the 1.58-bit FLUX paper, which claims to quantize the 11.9B-parameter FLUX.1-dev transformer to ternary weights {−1, 0, +1} while maintaining near-BF16 generation quality — with 7.7× storage reduction and 5.1× inference memory reduction.

---

## Method Overview

**Ternary quantization (absmean / BitNet b1.58 style):**

```python
scale[i] = mean(|W[i,:]|)                       # per output-channel
W_q[i,:] = clamp(round(W[i,:] / scale[i]), -1, 1)   # ternary: {-1, 0, +1}
W_eff     = W_q * scale                          # dequantize for forward pass
```

Applied to all 504 linear layers in the FLUX transformer (covering 100% of the 11.9B parameters). Activations remain in BF16. No training — pure post-training quantization (PTQ).

**Note on naming:** The existing `generate_images.py` experiments (w1–w8) use **unsigned min-max affine quantization** (`utils_quant.py`). For w_bits=1 this gives binary {0, 1}, not ternary. Our ternary experiments use a new `models/ternary.py` implementing the absmean {−1, 0, +1} scheme.

---

## Infrastructure Built

| File | Purpose |
|---|---|
| `models/ternary.py` | `TernaryLinear` class + `quantize_to_ternary()` (new) |
| `generate_ternary.py` | Load FLUX, apply ternary PTQ, generate images |
| `generate_images.py` | Sweep INT1–INT8 with optional SVD low-rank compensation |
| `visualize.py` | Build comparison grids from all outputs |

---

## Key Results

### 1. Quantization SNR across bit widths

Measured on 20 randomly sampled transformer layers (Frobenius-norm SNR, 20·log₁₀(‖W‖/‖W−W_q‖)):

| Format | Avg. weight SNR | Visual quality |
|---|---|---|
| BF16 | ∞ | Perfect |
| INT8 (min-max) | +40.6 dB | Near-identical to BF16 |
| INT4 (min-max) | +16.0 dB | Good; slight stylistic drift |
| INT3 (min-max) | +9.5 dB | Recognizable, detail loss |
| **Ternary (absmean)** | **+5.5 dB** | **Complete noise** |
| **INT2 (min-max)** | **+2.9 dB** | **Complete noise** |

The model collapses somewhere in the **5.5–9.5 dB range** — both ternary and INT2 are below this threshold. Notably, ternary outperforms INT2 in weight SNR (+5.5 vs +2.9 dB) because per-channel absmean scaling suits the near-Gaussian weight distribution better than min-max range (which is dominated by outliers). Yet both are insufficient without fine-tuning.

> **Caveat:** Weight-domain SNR is a proxy for image quality. Actual quality also depends on error propagation through 28 denoising steps and activation distributions, so the threshold should not be taken as a precise number.

### 2. Memory savings (transformer component only)

| Format | Memory | Reduction |
|---|---|---|
| BF16 (baseline) | 22,700 MB | 1.0× |
| Ternary (stored as int8) | 11,359 MB | **2.0×** |
| Ternary (2-bit packed, theoretical) | ~2,838 MB | **~8.0×** |

We stored ternary weights as int8 for simplicity. True 2-bit packing (as in the paper's custom kernel) would yield ~8× reduction. The paper reports 5.1× inference memory reduction because VAE and text encoders are not quantized.

504 layers quantized in **0.7 seconds** on an A100.

### 3. SVD low-rank compensation (single-step)

We tested a single-step SVD-of-residual initialization (1 iteration of the LoftQ alternating procedure):

- **INT4 + rank64**: Near-BF16 quality — the rank-64 adapter captures most of the quantization residual
- **INT2 + rank64**: Still noise — quantization error at 2 bits is too large for a rank-64 adapter

Note: full iterative LoftQ (multiple alternating rounds) is implemented in `utils_quant.py` but not used in the sweep — the images use single-step SVD init.

---

## The Gap: Why Naive PTQ Fails at 1.58-bit

The paper's core contribution is **self-supervised fine-tuning** that recovers quality after ternary PTQ.

Their approach (inferred from the paper):
1. Initialize transformer weights with ternary PTQ (as we did)
2. Run text prompts (no images) through the full-precision FLUX → collect transformer block outputs
3. Fine-tune the ternary model to minimize output mismatch against the full-precision teacher
4. The full-precision model supervises itself → no labeled data needed

Without this step, the +5.5 dB weight SNR is far below what FLUX's iterative denoising can tolerate. The fine-tuning recovers enough SNR to restore visual coherence.

Since the authors have not released weights or code yet, we cannot replicate the fine-tuned result. Our PoC confirms the infrastructure is correct and the failure mode of naive PTQ is the expected baseline.

---

## Visual Summary

See `output/viz/` for all grids. Key observations:

- `bits_vs_quality.png`: quality maintained through INT3, collapses at INT2 and below (including ternary)
- `ternary_vs_bf16.png`: sharp contrast between BF16 coherence and ternary noise
- `w4_lowrank_comparison.png`: rank=64 restores near-BF16 quality at INT4
- `w2_lowrank_comparison.png`: even rank=64 cannot rescue INT2

---

## Conclusions

1. **INT8 is lossless for FLUX.** No quality tradeoff at 40.6 dB SNR.
2. **INT4 + rank-64 single-step SVD ≈ BF16.** 4× weight compression, near-zero quality loss.
3. **The SNR collapse threshold is in the 5.5–9.5 dB range.** Both INT2 and ternary fall below it.
4. **Naive ternary PTQ fails completely.** Confirms that the paper's self-supervised fine-tuning is the essential and novel contribution.
5. **Ternary > INT2 in weight SNR** despite being lower bits — because absmean suits Gaussian weights better than min-max.
6. **Theoretical 8× memory** from 2-bit packing; paper achieves 7.7× storage reduction in practice.

---

## Reproducibility

```bash
# FLUX ternary PTQ + generation
/home/jovyan/conda/dit-bits-vs-quality/bin/python generate_ternary.py

# INT1–INT8 sweep + SVD low-rank
/home/jovyan/conda/dit-bits-vs-quality/bin/python generate_images.py

# All visualization grids
/home/jovyan/conda/dit-bits-vs-quality/bin/python visualize.py
```

**GPU:** NVIDIA A100-SXM4-80GB  
**Model cache:** `/home/jovyan/.cache/huggingface/`  
**Python env:** `/home/jovyan/conda/dit-bits-vs-quality/`

