"""
Create comparison grids for DiT bits-vs-quality experiment.
Shows: BF16 vs ternary (1.58-bit), and all quantization levels.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SAMPLES = Path("output/samples")
OUTPUT = Path("output/viz")
OUTPUT.mkdir(parents=True, exist_ok=True)

PROMPT_NAMES = ["cyberpunk_samurai", "fantasy_landscape"]
FONT_SIZE = 28
THUMB = 512  # thumbnail size for grids


def load_img(path: Path, size: int = THUMB) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def labeled(img: Image.Image, text: str, font_size: int = FONT_SIZE) -> Image.Image:
    """Add a label bar below an image."""
    bar_h = font_size + 12
    out = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    tx = (img.width - tw) // 2
    draw.text((tx, img.height + 6), text, fill=(240, 240, 240), font=font)
    return out


def hstack(images: list) -> Image.Image:
    w = sum(i.width for i in images)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def vstack(images: list) -> Image.Image:
    w = max(i.width for i in images)
    h = sum(i.height for i in images)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height
    return out


def make_bits_vs_quality_grid():
    """Grid: rows = prompts, columns = bf16, w8, w4, w3, w2, w1, ternary (rank0)."""
    configs = [
        ("bf16",    SAMPLES / "bf16"),
        ("w8",      SAMPLES / "w8" / "rank0"),
        ("w4",      SAMPLES / "w4" / "rank0"),
        ("w3",      SAMPLES / "w3" / "rank0"),
        ("w2",      SAMPLES / "w2" / "rank0"),
        ("w1",      SAMPLES / "w1" / "rank0"),
        ("1.58-bit", SAMPLES / "ternary"),
    ]

    rows = []
    for pidx, pname in enumerate(PROMPT_NAMES):
        cols = []
        for label, dirpath in configs:
            imgs = sorted(dirpath.glob("*.png"))
            if pidx < len(imgs):
                img = load_img(imgs[pidx])
            else:
                img = Image.new("RGB", (THUMB, THUMB), (60, 60, 60))
            cols.append(labeled(img, label))
        rows.append(hstack(cols))

    grid = vstack(rows)
    out_path = OUTPUT / "bits_vs_quality.png"
    grid.save(out_path)
    print(f"Saved: {out_path} ({grid.width}x{grid.height})")
    return grid


def make_lowrank_grid(bits: int = 4):
    """Grid: rows = prompts, columns = rank0..64 for given bit-width."""
    ranks = [0, 4, 8, 16, 32, 64]
    rows = []
    for pidx, pname in enumerate(PROMPT_NAMES):
        cols = []
        for rank in ranks:
            dirpath = SAMPLES / f"w{bits}" / f"rank{rank}"
            imgs = sorted(dirpath.glob("*.png"))
            if pidx < len(imgs):
                img = load_img(imgs[pidx])
            else:
                img = Image.new("RGB", (THUMB, THUMB), (60, 60, 60))
            cols.append(labeled(img, f"rank={rank}"))
        rows.append(hstack(cols))

    grid = vstack(rows)
    out_path = OUTPUT / f"w{bits}_lowrank_comparison.png"
    grid.save(out_path)
    print(f"Saved: {out_path} ({grid.width}x{grid.height})")
    return grid


def make_ternary_vs_bf16():
    """Side-by-side: BF16 vs Ternary for both prompts."""
    bf16_dir = SAMPLES / "bf16"
    ternary_dir = SAMPLES / "ternary"

    rows = []
    for pidx, pname in enumerate(PROMPT_NAMES):
        bf16_imgs = sorted(bf16_dir.glob("*.png"))
        tern_imgs = sorted(ternary_dir.glob("*.png"))
        bf16_img = load_img(bf16_imgs[pidx]) if pidx < len(bf16_imgs) else Image.new("RGB", (THUMB, THUMB), (60, 60, 60))
        tern_img = load_img(tern_imgs[pidx]) if pidx < len(tern_imgs) else Image.new("RGB", (THUMB, THUMB), (60, 60, 60))
        row = hstack([labeled(bf16_img, "BF16 (full)"), labeled(tern_img, "1.58-bit Ternary")])
        rows.append(row)

    grid = vstack(rows)
    out_path = OUTPUT / "ternary_vs_bf16.png"
    grid.save(out_path)
    print(f"Saved: {out_path} ({grid.width}x{grid.height})")
    return grid


if __name__ == "__main__":
    print("=== Generating Visualization Grids ===")
    make_bits_vs_quality_grid()
    make_ternary_vs_bf16()
    for bits in [8, 4, 2]:
        make_lowrank_grid(bits)
    print("\nDone. All grids saved to output/viz/")
