"""
Generate images with 1.58-bit ternary-quantized FLUX.
Reproduces the PoC of: https://chenglin-yang.github.io/1.58bit.flux.github.io/
"""
import os, time, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from diffusers import DiffusionPipeline
from pathlib import Path
from models.ternary import quantize_to_ternary, memory_stats

PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
]
MODEL_NAME  = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR  = Path("output/samples")
SEED        = 42
STEPS       = 28
GUIDANCE    = 3.5


def generate(pipe, prompt: str, out_dir: Path, seed: int = SEED):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob("*.png")))
    gen = torch.Generator("cuda").manual_seed(seed)
    img = pipe(prompt, generator=gen, num_inference_steps=STEPS, guidance_scale=GUIDANCE).images[0]
    path = out_dir / f"{idx}.png"
    img.save(path)
    print(f"  Saved: {path}")
    return path


def main():
    print("=== 1.58-bit Ternary FLUX PoC ===\n")

    print("[1] Loading FLUX.1-dev (BF16)...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")

    bf16_mb = memory_stats(pipe.transformer)["total_mb"]
    print(f"    BF16 transformer: {bf16_mb:.0f} MB")

    # BF16 baseline (skip if exists)
    bf16_dir = OUTPUT_DIR / "bf16"
    if len(list(bf16_dir.glob("*.png"))) >= len(PROMPTS):
        print("    BF16 baseline already exists, skipping.")
    else:
        print("\n[1b] Generating BF16 baseline...")
        for prompt in PROMPTS:
            generate(pipe, prompt, bf16_dir)

    # Apply ternary quantization
    print("\n[2] Applying 1.58-bit ternary quantization (absmean, per-channel)...")
    t0 = time.time()
    quantize_to_ternary(pipe.transformer, per_channel=True, verbose=False)
    print(f"    Done in {time.time()-t0:.1f}s")

    ternary_mb = memory_stats(pipe.transformer)["total_mb"]
    print(f"    Ternary transformer: {ternary_mb:.0f} MB  ({bf16_mb/ternary_mb:.1f}x smaller)")

    # Generate ternary images
    print("\n[3] Generating ternary images...")
    ternary_dir = OUTPUT_DIR / "ternary"
    for prompt in PROMPTS:
        generate(pipe, prompt, ternary_dir)

    print(f"\n=== Done ===")
    print(f"BF16 mem:    {bf16_mb:.0f} MB")
    print(f"Ternary mem: {ternary_mb:.0f} MB")
    print(f"Ratio:       {bf16_mb/ternary_mb:.2f}x")


if __name__ == "__main__":
    main()
