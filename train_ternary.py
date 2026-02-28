"""
Self-supervised distillation: BF16 FLUX → Ternary FLUX.
Layer-wise MSE matching between teacher (BF16) and student (ternary + scale + LoRA).

Usage:
  python train_ternary.py [--steps 800] [--rank 8] [--lr 3e-4] [--eval-every 200]
"""
import os, sys, time, random, argparse, json
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import FluxPipeline, FluxTransformer2DModel

from models.ternary import quantize_to_ternary, memory_stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = Path("output")
CKPT_PATH  = OUTPUT_DIR / "ternary_distilled.pt"

CALIB_PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting",
    "A fantasy landscape with mountains and a river",
    "Portrait of a young woman with wild curly hair in golden light",
    "Aerial view of a coastal city at sunset",
    "Abstract geometric art in vivid colors, oil on canvas",
    "A cozy wooden cabin in a snowy forest at night, warm interior glow",
    "A futuristic humanoid robot in a busy marketplace",
    "Close-up of a red rose with water droplets, macro photography",
    "Ancient temple ruins covered in vines in a tropical jungle",
    "A chef preparing food in a modern open kitchen",
    "Underwater scene with colorful tropical fish and coral reef",
    "A steam locomotive crossing a mountain bridge, dramatic clouds",
    "Street art mural on a building wall, vibrant graffiti style",
    "A dragon flying over a medieval castle at dusk",
    "Minimalist black and white architectural photograph, symmetry",
    "Astronaut floating in space with Earth in background",
    "Oil painting of a harbor with sailing boats at golden hour",
    "Macro photography of a butterfly on a purple flower",
    "Post-apocalyptic overgrown city with nature reclaiming streets",
    "Young musician playing guitar on a stage with dramatic spotlights",
]

EVAL_PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
]

# ---------------------------------------------------------------------------
# Hook utilities for layer-wise distillation
# ---------------------------------------------------------------------------
class ActivationCache:
    """Stores intermediate activations for later loss computation."""
    def __init__(self):
        self.cache = {}
        self._hooks = []

    def register(self, model, double_every=1, single_every=4):
        """Register hooks on FluxTransformerBlocks."""
        for i, block in enumerate(model.transformer_blocks):
            if i % double_every == 0:
                key = f"d{i}"
                h = block.register_forward_hook(self._make_hook(key))
                self._hooks.append(h)

        for i, block in enumerate(model.single_transformer_blocks):
            if i % single_every == 0:
                key = f"s{i}"
                h = block.register_forward_hook(self._make_hook(key))
                self._hooks.append(h)

        return self

    def _make_hook(self, key):
        def hook(module, inp, output):
            # double-stream blocks return (enc_hs, hs); single return hs tensor
            if isinstance(output, (tuple, list)):
                # concatenate image and text hidden states
                self.cache[key] = torch.cat([o for o in output if isinstance(o, torch.Tensor)], dim=1)
            else:
                self.cache[key] = output
        return hook

    def clear(self):
        self.cache.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def layer_wise_loss(teacher_cache: dict, student_cache: dict) -> torch.Tensor:
    """Mean MSE across all matched layers."""
    assert teacher_cache.keys() == student_cache.keys(), \
        f"Cache key mismatch: {teacher_cache.keys()} vs {student_cache.keys()}"
    losses = []
    for key in teacher_cache:
        t = teacher_cache[key].float().detach()
        s = student_cache[key].float()
        losses.append(F.mse_loss(s, t))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def prepare_inputs(pipe, prompt: str, height: int, width: int,
                   timestep_frac: float, device: str, dtype):
    """
    Prepare one (latent, timestep, text_embeds, img_ids, txt_ids) tuple
    for a single transformer forward pass, without running the denoising loop.
    """
    # 1. Encode prompt
    (prompt_embeds, pooled_embeds,
     text_ids) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=256,
    )

    # 2. Prepare latents — pipe.prepare_latents handles vae_scale_factor
    #    rounding and 2×2 spatial packing internally.
    #    in_channels=64 (packed); pre-pack channels = 64 // 4 = 16
    num_channels_latents = pipe.transformer.config.in_channels // 4
    gen = torch.Generator(device=device).manual_seed(random.randint(0, 2**31))
    latents, img_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=gen,
    )

    # 3. Timestep as fraction in [0, 1] (FLUX flow-matching convention)
    t = torch.tensor([timestep_frac], device=device, dtype=dtype)
    guidance = torch.tensor([3.5], device=device, dtype=dtype)

    return {
        "hidden_states":         latents,
        "timestep":              t,
        "guidance":              guidance,
        "encoder_hidden_states": prompt_embeds,
        "pooled_projections":    pooled_embeds,
        "txt_ids":               text_ids,
        "img_ids":               img_ids,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_images(pipe, step: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        gen = torch.Generator("cuda").manual_seed(42)
        img = pipe(prompt, generator=gen,
                   num_inference_steps=28, guidance_scale=3.5).images[0]
        p = out_dir / f"step{step:04d}_p{i}.png"
        img.save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int,   default=800)
    p.add_argument("--rank",       type=int,   default=8)
    p.add_argument("--lr-lora",    type=float, default=3e-4)
    p.add_argument("--lr-scale",   type=float, default=1e-3)
    p.add_argument("--eval-every", type=int,   default=200)
    p.add_argument("--res",        type=int,   default=512,
                   help="Training resolution (512 is faster than 1024)")
    p.add_argument("--no-svd",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    print(f"=== Ternary FLUX Distillation ===")
    print(f"  steps={args.steps}, rank={args.rank}, "
          f"lr_scale={args.lr_scale}, lr_lora={args.lr_lora}, res={args.res}")

    # ------------------------------------------------------------------ #
    # 1. Load student pipeline + quantize
    # ------------------------------------------------------------------ #
    print("\n[1] Loading student pipeline (BF16 → ternary)...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, local_files_only=True,
    ).to(device)

    bf16_mem = memory_stats(pipe.transformer)["total_mb"]
    print(f"    BF16 transformer: {bf16_mem:.0f} MB")

    svd_init = not args.no_svd
    quantize_to_ternary(pipe.transformer, per_channel=True,
                        lora_rank=args.rank, svd_init=svd_init)
    print(f"    Ternary transformer: {memory_stats(pipe.transformer)['total_mb']:.0f} MB")

    # ------------------------------------------------------------------ #
    # 2. Load teacher transformer (frozen BF16 copy)
    # ------------------------------------------------------------------ #
    print("\n[2] Loading teacher transformer (frozen BF16)...")
    teacher = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer",
        torch_dtype=dtype, local_files_only=True,
    ).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    print(f"    Teacher loaded: {memory_stats(teacher)['total_mb']:.0f} MB")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    used_vram  = torch.cuda.memory_allocated() / 1024**3
    print(f"    VRAM used: {used_vram:.1f} / {total_vram:.0f} GB")

    # ------------------------------------------------------------------ #
    # 2b. Pre-encode calibration prompts, then offload text encoders + VAE
    # ------------------------------------------------------------------ #
    print("\n[2b] Pre-encoding calibration prompts...")
    calib_embeds = []
    with torch.no_grad():
        for prompt in CALIB_PROMPTS:
            pe, poe, ti = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device,
                num_images_per_prompt=1, max_sequence_length=256,
            )
            calib_embeds.append({
                "prompt_embeds": pe.cpu(),
                "pooled_embeds": poe.cpu(),
                "text_ids":      ti.cpu(),
            })
    print(f"    Encoded {len(calib_embeds)} prompts.")

    print("[2c] Offloading text encoders and VAE to CPU...")
    for attr in ("text_encoder", "text_encoder_2", "vae"):
        m = getattr(pipe, attr, None)
        if m is not None:
            m.to("cpu")
    torch.cuda.empty_cache()
    print(f"    VRAM after offload: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Pre-compute latent dimensions once (res → latent_h, latent_w after rounding)
    _lat_h = 2 * (int(args.res) // (pipe.vae_scale_factor * 2))
    _lat_w = _lat_h  # square
    _num_ch = pipe.transformer.config.in_channels // 4  # 64 // 4 = 16
    print(f"    Latent grid: {_lat_h}×{_lat_w}, channels: {_num_ch}")

    # ------------------------------------------------------------------ #
    # 3. Register activation hooks
    # ------------------------------------------------------------------ #
    teacher_cache = ActivationCache().register(teacher, double_every=1, single_every=4)
    student_cache = ActivationCache().register(pipe.transformer, double_every=1, single_every=4)
    print(f"\n[3] Matching points: {len(teacher_cache._hooks)} teacher "
          f"+ {len(student_cache._hooks)} student hooks")

    # ------------------------------------------------------------------ #
    # 4. Optimizer — separate LR for scale vs LoRA
    # ------------------------------------------------------------------ #
    scale_params = [p for n, p in pipe.transformer.named_parameters() if n.endswith(".scale")]
    lora_params  = [p for n, p in pipe.transformer.named_parameters()
                    if n.endswith(".lora_A") or n.endswith(".lora_B")]

    optimizer = torch.optim.AdamW([
        {"params": scale_params, "lr": args.lr_scale, "weight_decay": 0.0},
        {"params": lora_params,  "lr": args.lr_lora,  "weight_decay": 1e-4},
    ], betas=(0.9, 0.999), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.0
    )

    n_scale = sum(p.numel() for p in scale_params)
    n_lora  = sum(p.numel() for p in lora_params)
    print(f"    Trainable: {n_scale:,} scale + {n_lora:,} LoRA = {n_scale+n_lora:,} total")

    # ------------------------------------------------------------------ #
    # 5. Training loop (uses pre-encoded embeddings, no text encoder needed)
    # ------------------------------------------------------------------ #
    print(f"\n[4] Training for {args.steps} steps...")
    eval_dir = OUTPUT_DIR / "eval_ternary_distilled"
    log = []
    t0 = time.time()

    pipe.transformer.train()

    for step in range(1, args.steps + 1):
        # Sample random pre-encoded embedding and random timestep
        emb = random.choice(calib_embeds)
        t_frac = random.uniform(0.2, 0.95)

        # Build inputs from pre-encoded embeddings (no text encoder call)
        gen = torch.Generator(device=device).manual_seed(random.randint(0, 2**31))
        raw = torch.randn(1, _num_ch, _lat_h, _lat_w,
                          device=device, dtype=dtype, generator=gen)
        latents = FluxPipeline._pack_latents(raw, 1, _num_ch, _lat_h, _lat_w)
        img_ids = FluxPipeline._prepare_latent_image_ids(
            1, _lat_h // 2, _lat_w // 2, device, dtype)
        inputs = {
            "hidden_states":         latents,
            "timestep":              torch.tensor([t_frac], device=device, dtype=dtype),
            "guidance":              torch.tensor([3.5],   device=device, dtype=dtype),
            "encoder_hidden_states": emb["prompt_embeds"].to(device=device, dtype=dtype),
            "pooled_projections":    emb["pooled_embeds"].to(device=device, dtype=dtype),
            "txt_ids":               emb["text_ids"].to(device=device, dtype=dtype),
            "img_ids":               img_ids,
        }

        # Teacher forward (no grad, fills teacher_cache)
        teacher_cache.clear()
        with torch.no_grad():
            teacher(**inputs, return_dict=False)

        # Student forward (fills student_cache, computes grad)
        student_cache.clear()
        pipe.transformer(**inputs, return_dict=False)

        loss = layer_wise_loss(teacher_cache.cache, student_cache.cache)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scale_params + lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            elapsed = time.time() - t0
            lr_s = optimizer.param_groups[0]["lr"]
            lr_l = optimizer.param_groups[1]["lr"]
            print(f"  step {step:4d}/{args.steps} | loss={loss.item():.5f} "
                  f"| lr_scale={lr_s:.2e} lr_lora={lr_l:.2e} | {elapsed:.0f}s elapsed")
            log.append({"step": step, "loss": loss.item()})
            sys.stdout.flush()

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n  [eval] Generating images at step {step}...")
            pipe.transformer.eval()
            # Bring text encoders + VAE back to GPU for full pipeline inference
            for attr in ("text_encoder", "text_encoder_2", "vae"):
                m = getattr(pipe, attr, None)
                if m is not None:
                    m.to(device)
            eval_images(pipe, step, eval_dir)
            # Move them back to CPU to free VRAM for training
            for attr in ("text_encoder", "text_encoder_2", "vae"):
                m = getattr(pipe, attr, None)
                if m is not None:
                    m.to("cpu")
            torch.cuda.empty_cache()
            pipe.transformer.train()
            print(f"  [eval] Done → {eval_dir}/step{step:04d}_p*.png")
            sys.stdout.flush()

    # ------------------------------------------------------------------ #
    # 6. Save checkpoint
    # ------------------------------------------------------------------ #
    print(f"\n[5] Saving checkpoint → {CKPT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_dict = {
        name: param.data
        for name, param in pipe.transformer.named_parameters()
        if name.endswith((".scale", ".lora_A", ".lora_B"))
    }
    torch.save(save_dict, CKPT_PATH)
    print(f"    Saved {len(save_dict)} parameter tensors")

    # Save training log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n=== Done. Total time: {(time.time()-t0)/60:.1f} min ===")
    print(f"Eval images: {eval_dir}/")


if __name__ == "__main__":
    main()
