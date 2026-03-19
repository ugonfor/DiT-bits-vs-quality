"""
Measure peak VRAM for BF16 vs Ternary+LoRA at various ranks during inference.
Runs a single 1024px generation for each config and reports peak allocated VRAM.
"""
import os, gc, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from diffusers import FluxPipeline
from models.ternary import quantize_to_ternary

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
CKPTS = {
    "ternary_r64":  "output/ternary_distilled_r64_res1024_s6000_fm_lpips1e-01.pt",  # V9b
    "ternary_r128": "output/ternary_distilled_r128_res1024_s12000_fm_lpips1e-01.pt",  # V10b
}
PROMPT = "A majestic lion resting on a savanna at golden hour"
RES    = 1024
STEPS  = 30


def mem_stats(label):
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak  = torch.cuda.max_memory_allocated() / 1024**3
    resv  = torch.cuda.memory_reserved() / 1024**3
    print(f"  [{label}]  allocated={alloc:.2f} GB  peak={peak:.2f} GB  reserved={resv:.2f} GB")
    return peak


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── BF16 baseline ──────────────────────────────────────────────────────────
print("=" * 60)
print("CONFIG: BF16 (no quantization)")
print("=" * 60)
reset()

pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                    local_files_only=True).to("cuda")
mem_stats("after load")

with torch.no_grad():
    _ = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
             guidance_scale=3.5, output_type="pil").images[0]

peak_bf16 = mem_stats("after inference")

del pipe
reset()
print()

# ── Ternary + LoRA configs ─────────────────────────────────────────────────
for name, ckpt_path in CKPTS.items():
    rank = int(name.split("_r")[1])
    print("=" * 60)
    print(f"CONFIG: Ternary + LoRA-r{rank}")
    print("=" * 60)
    reset()

    pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                        local_files_only=True).to("cuda")
    mem_stats("BF16 loaded")

    quantize_to_ternary(pipe.transformer, lora_rank=rank, svd_init=False)
    mem_stats("after quantize_to_ternary (weights still BF16 until ckpt)")

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    state = {k: v for k, v in pipe.transformer.named_parameters()}
    for name, tensor in ckpt.items():
        if name in state:
            state[name].data.copy_(tensor.to(torch.bfloat16))
    del ckpt
    torch.cuda.empty_cache()
    mem_stats("after ckpt load")

    # Count parameters
    total = sum(p.numel() for p in pipe.transformer.parameters())
    trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    print(f"  transformer params: total={total/1e9:.3f}B  trainable={trainable/1e6:.1f}M")

    # Reset peak so we measure only the inference pass, not the quantization overhead
    torch.cuda.reset_peak_memory_stats()
    baseline_ternary = torch.cuda.memory_allocated() / 1024**3
    print(f"  [baseline before inference]  allocated={baseline_ternary:.2f} GB  (peak reset)")

    with torch.no_grad():
        _ = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS,
                 guidance_scale=3.5, output_type="pil").images[0]

    peak_ternary = mem_stats("after inference")

    print(f"\n  >> Peak reduction vs BF16: {peak_bf16:.2f} → {peak_ternary:.2f} GB  "
          f"({(1 - peak_ternary/peak_bf16)*100:.1f}% less)")

    del pipe
    reset()
    print()

print("=" * 60)
print(f"Summary:")
print(f"  BF16 inference peak:          {peak_bf16:.2f} GB")
for name in CKPTS:
    print(f"  Ternary+LoRA-{name.split('_r')[1]} inference peak: {peak_ternary:.2f} GB")
print("=" * 60)
