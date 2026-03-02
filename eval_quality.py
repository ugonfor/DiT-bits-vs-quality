"""
Compute LPIPS and aesthetic scores for ternary FLUX eval images.

Compares BF16 reference vs V5 vs V6 (and any other model dirs passed).
Outputs:
  - output/quality_scores.txt   — plain text table
  - output/quality_scores.json  — machine-readable
  - output/viz/quality_grid.png — visual comparison grid with scores

Metrics:
  LPIPS  : AlexNet perceptual distance vs BF16 reference (lower = closer to BF16)
  Aes    : LAION aesthetic score via CLIP ViT-L/14 + linear MLP (1-10 scale)
  CLIP   : text-image cosine similarity (existing metric, included for completeness)

Usage:
  python eval_quality.py
  python eval_quality.py --models BF16:output/bf16_reference V5:output/eval_fm_clip_v5 V6:output/eval_fm_clip_v6
"""
import os, sys, json, argparse, urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import lpips as lpips_lib
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

OUTPUT_DIR = Path("output")
VIZ_DIR    = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_LOCAL = ("/home/jovyan/.cache/huggingface/hub/"
              "models--openai--clip-vit-base-patch32/snapshots/"
              "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")

PROMPTS = [
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting, 32-bit render",
    "A fantasy landscape with mountains and a river",
    "Portrait of a young woman with wild curly hair in golden light",
    "Aerial view of a coastal city at sunset",
]
PROMPT_SHORT = ["Cyberpunk samurai", "Fantasy landscape", "Portrait", "Aerial city"]

# --------------------------------------------------------------------------- #
# Zero-shot aesthetic scoring via CLIP ViT-B/32
# Inspired by: LAION aesthetic predictor concept, adapted to available model.
# Score = sim(img, positive_prompts) - sim(img, negative_prompts), scaled 0-10.
# --------------------------------------------------------------------------- #
AES_POSITIVE = [
    "a high quality, beautiful, detailed, sharp photograph",
    "award-winning photo, stunning, vibrant, well-composed, 4K HDR",
    "masterpiece, professional photography, cinematic, rich colors",
]
AES_NEGATIVE = [
    "a blurry, low quality, noisy, distorted image",
    "ugly, amateur, flat, washed out, dull colors, artifact",
]


def aesthetic_score(clip_model, clip_processor, img_pil, device, _unused=None):
    """Zero-shot aesthetic score via CLIP text-image similarity (0-10 scale)."""
    inputs = clip_processor(
        text=AES_POSITIVE + AES_NEGATIVE,
        images=[img_pil] * (len(AES_POSITIVE) + len(AES_NEGATIVE)),
        return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    i_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t_feat = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    # Image embed is the same for all rows; we need the per-text similarity
    img_inputs  = clip_processor(images=[img_pil], return_tensors="pt").to(device)
    text_inputs = clip_processor(text=AES_POSITIVE + AES_NEGATIVE,
                                  return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        i_feat = clip_model.get_image_features(**img_inputs).float()
        t_feat = clip_model.get_text_features(**text_inputs).float()
    i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
    sims = (i_feat @ t_feat.T).squeeze(0)   # [n_texts]
    pos_score = sims[:len(AES_POSITIVE)].mean().item()
    neg_score = sims[len(AES_POSITIVE):].mean().item()
    # Rescale to 0-10
    raw   = pos_score - neg_score          # typically -0.15 to +0.15
    score = (raw + 0.15) / 0.30 * 10.0    # map [-0.15,+0.15] → [0,10]
    return float(torch.tensor(score).clamp(0, 10))


def clip_score_fn(clip_model, clip_processor, img_pil, prompt, device):
    """Return CLIP cosine similarity between image and text prompt."""
    inputs = clip_processor(text=[prompt], images=[img_pil],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    i_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t_feat = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (i_feat * t_feat).sum().item()


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
THUMB = 512
FONT_SZ = 20


def load_font(size=FONT_SZ):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate(img_pil, lines, bg=(20, 20, 20)):
    bar_h = (FONT_SZ + 6) * len(lines) + 8
    out   = Image.new("RGB", (img_pil.width, img_pil.height + bar_h), bg)
    out.paste(img_pil, (0, 0))
    draw  = ImageDraw.Draw(out)
    font  = load_font(FONT_SZ)
    y     = img_pil.height + 4
    for line in lines:
        draw.text((6, y), line, fill=(230, 230, 230), font=font)
        y += FONT_SZ + 6
    return out


def make_header(text, width, bg=(40, 40, 80)):
    h   = FONT_SZ + 14
    img = Image.new("RGB", (width, h), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(FONT_SZ)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 7), text, fill=(220, 220, 255), font=font)
    return img


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=[
                       "BF16:output/bf16_reference",
                       "V5:output/eval_fm_clip_v5",
                       "V6:output/eval_fm_clip_v6",
                   ])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    models = []
    for item in args.models:
        label, path = item.split(":", 1)
        models.append((label, Path(path)))

    print(f"=== Quality Evaluation: LPIPS + Aesthetic Score + CLIP ===")
    print(f"Models : {[m[0] for m in models]}")
    print(f"Device : {device}\n")

    # Load CLIP ViT-L/14
    print(f"[1] Loading CLIP ViT-B/32 (local) ...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_LOCAL, local_files_only=True)
    clip_model     = CLIPModel.from_pretrained(CLIP_LOCAL, local_files_only=True).to(device)
    clip_model.eval()
    print(f"    Loaded (zero-shot aesthetic scoring: no extra download needed).")

    # Load LPIPS
    print(f"[3] Loading LPIPS (AlexNet) ...")
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    print(f"    Loaded.\n")

    to_tensor = transforms.ToTensor()  # [0,1]

    # BF16 reference dir
    bf16_dir = None
    for label, path in models:
        if label.upper() == "BF16":
            bf16_dir = path
            break

    # ------------------------------------------------------------------ #
    # Score all models
    # ------------------------------------------------------------------ #
    print("[4] Scoring...\n")
    results = {}

    for label, model_dir in models:
        results[label] = []
        for pi, (prompt, pshort) in enumerate(zip(PROMPTS, PROMPT_SHORT)):
            img_path = model_dir / f"p{pi}.png"
            if not img_path.exists():
                print(f"  MISSING: {img_path}")
                results[label].append(None)
                continue

            img = Image.open(img_path).convert("RGB")

            aes = aesthetic_score(clip_model, clip_processor, img, device)
            cs  = clip_score_fn(clip_model, clip_processor, img, prompt, device)

            lp = None
            if bf16_dir is not None and label.upper() != "BF16":
                ref_path = bf16_dir / f"p{pi}.png"
                if ref_path.exists():
                    ref   = Image.open(ref_path).convert("RGB").resize((img.width, img.height))
                    img_t = to_tensor(img).unsqueeze(0).to(device) * 2 - 1
                    ref_t = to_tensor(ref).unsqueeze(0).to(device) * 2 - 1
                    with torch.no_grad():
                        lp = lpips_fn(img_t, ref_t).item()

            row = {"prompt": pshort, "aes": aes, "clip": cs, "lpips": lp}
            results[label].append(row)
            lp_str = f"{lp:.4f}" if lp is not None else "   ref"
            print(f"  [{label:>4}] p{pi} {pshort:<22} | aes={aes:.3f} | clip={cs:.4f} | lpips={lp_str}")

        valid   = [r for r in results[label] if r is not None]
        avg_aes = sum(r["aes"]  for r in valid) / len(valid)
        avg_cl  = sum(r["clip"] for r in valid) / len(valid)
        lp_vals = [r["lpips"] for r in valid if r["lpips"] is not None]
        avg_lp  = sum(lp_vals) / len(lp_vals) if lp_vals else None
        lp_str  = f"{avg_lp:.4f}" if avg_lp is not None else "   ref"
        print(f"  [{label:>4}] AVERAGE{'':22} | aes={avg_aes:.3f} | clip={avg_cl:.4f} | lpips={lp_str}")
        results[label].append({"prompt": "AVERAGE", "aes": avg_aes, "clip": avg_cl, "lpips": avg_lp})
        print()

    # ------------------------------------------------------------------ #
    # Print table
    # ------------------------------------------------------------------ #
    sep   = "-" * 80
    lines = ["\n=== QUALITY SCORES ===", sep]
    lines.append(f"{'Prompt':<24}" + "".join(f"  {lb:>4}_aes {lb:>4}_clip {lb:>4}_lpips" for lb, _ in models))
    lines.append(sep)
    n_rows = len(PROMPTS) + 1
    for i in range(n_rows):
        label0 = models[0][0]
        pname  = results[label0][i]["prompt"] if results[label0][i] else "?"
        row    = f"{pname:<24}"
        for lb, _ in models:
            r = results[lb][i]
            if r is None:
                row += "     N/A"
                continue
            lp_s = f"{r['lpips']:>8.4f}" if r["lpips"] is not None else "     ref"
            row += f"  {r['aes']:>7.3f} {r['clip']:>8.4f} {lp_s}"
        lines.append(row)
    lines.append(sep)
    table = "\n".join(lines)
    print(table)

    out_txt = OUTPUT_DIR / "quality_scores.txt"
    out_txt.write_text(table)
    print(f"\nSaved text: {out_txt}")

    out_json = OUTPUT_DIR / "quality_scores.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON: {out_json}")

    # ------------------------------------------------------------------ #
    # Comparison grid
    # ------------------------------------------------------------------ #
    print("\n[5] Building comparison grid...")
    n_cols = len(models)

    rows = []
    for pi in range(len(PROMPTS)):
        cells = []
        for label, model_dir in models:
            img_path = model_dir / f"p{pi}.png"
            img = (Image.open(img_path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
                   if img_path.exists() else Image.new("RGB", (THUMB, THUMB), (60,60,60)))
            r = results[label][pi]
            if r:
                lp_s = f"lpips={r['lpips']:.3f}" if r["lpips"] is not None else "lpips=ref"
                ann_lines = [f"aes={r['aes']:.2f}  clip={r['clip']:.4f}", lp_s]
            else:
                ann_lines = ["missing"]
            cells.append(annotate(img, ann_lines))

        row_img = Image.new("RGB", (THUMB * n_cols, cells[0].height), (15,15,15))
        for ci, c in enumerate(cells):
            row_img.paste(c, (ci * THUMB, 0))
        # Prompt label overlay
        draw = ImageDraw.Draw(row_img)
        draw.text((6, 4), PROMPT_SHORT[pi], fill=(255, 220, 80), font=load_font(18))
        rows.append(row_img)

    # Header row
    hrow = Image.new("RGB", (THUMB * n_cols, FONT_SZ + 14), (15,15,15))
    for ci, (lb, _) in enumerate(models):
        hrow.paste(make_header(lb, THUMB), (ci * THUMB, 0))

    all_rows = [hrow] + rows
    total_h  = sum(r.height for r in all_rows)
    grid     = Image.new("RGB", (THUMB * n_cols, total_h), (10,10,10))
    y = 0
    for r in all_rows:
        grid.paste(r, (0, y));  y += r.height

    out_grid = VIZ_DIR / "quality_grid.png"
    grid.save(out_grid)
    print(f"Saved grid: {out_grid}  ({grid.width}×{grid.height}px)")


if __name__ == "__main__":
    main()
