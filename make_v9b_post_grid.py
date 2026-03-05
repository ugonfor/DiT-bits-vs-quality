"""
Generate comparison grids for post-011 (V9b).
Creates:
  1. output/viz/v9b_full_grid.png — all 20 prompts, BF16 | V7 | V9b
  2. output/viz/v9b_highlights.png — 8 selected prompts (wins + losses + persistent)
  3. output/viz/v9b_scaling_law.png — OOD CLIP % vs log2(prompts) plot
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("output")
VIZ_DIR = OUTPUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("Animals",      "Lion on savanna at golden hour"),
    ("Animals",      "Parrot on tropical branch"),
    ("Animals",      "Wolf howling at full moon"),
    ("Architecture", "Gothic cathedral interior"),
    ("Architecture", "Futuristic skyscraper"),
    ("Architecture", "Japanese pagoda + cherry blossoms"),
    ("Landscapes",   "Northern lights over frozen lake"),
    ("Landscapes",   "Volcanic eruption at night"),
    ("Landscapes",   "Lavender fields in Provence"),
    ("Portraits",    "Elderly fisherman portrait"),
    ("Portraits",    "Ballet dancer mid-leap"),
    ("Portraits",    "Street musician in rain"),
    ("Food",         "Sushi platter"),
    ("Food",         "Rustic bread on wooden table"),
    ("Fantasy",      "Dragon over medieval castle"),
    ("Sci-fi",       "Astronaut floating in space"),
    ("Fantasy",      "Magical forest with fireflies"),
    ("Art styles",   "Venice canals watercolor"),
    ("Art styles",   "Renaissance oil portrait"),
    ("Urban",        "Rainy Tokyo street at night"),
]

# CLIP scores
BF16_CLIPS  = [0.3276,0.3220,0.3417,0.3134,0.3246,0.3510,0.3006,0.3456,0.3384,0.3845,0.3433,0.3677,0.3122,0.3433,0.3668,0.3365,0.3684,0.3179,0.2902,0.3534]
V7_CLIPS    = [0.2817,0.3124,0.3386,0.3103,0.3116,0.3251,0.2610,0.3303,0.3205,0.3012,0.2776,0.2662,0.2044,0.3266,0.3179,0.3032,0.3237,0.2792,0.3055,0.3056]
V9B_CLIPS   = [0.3128,0.3232,0.3340,0.2941,0.3000,0.3507,0.2873,0.3355,0.3262,0.3263,0.2616,0.3133,0.2181,0.2672,0.2976,0.3071,0.3404,0.2850,0.2918,0.3006]

DIRS = {
    "BF16": OUTPUT_DIR / "eval_diverse_bf16",
    "V7":   OUTPUT_DIR / "eval_diverse_v7",
    "V9b":  OUTPUT_DIR / "eval_diverse_v9b",
}
MODELS = ["BF16", "V7", "V9b"]
CLIPS  = {"BF16": BF16_CLIPS, "V7": V7_CLIPS, "V9b": V9B_CLIPS}

THUMB = 256
LABEL_H = 36
HEADER_H = 50
PAD = 4


def load_thumb(path, size=THUMB):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def color_for_pct(pct):
    """Green if >95%, yellow if 85-95%, red if <85%."""
    if pct >= 95:
        return (40, 180, 40)
    elif pct >= 85:
        return (200, 160, 0)
    else:
        return (200, 50, 50)


def make_label(text, width, height, bg=(40, 40, 40), fg=(220, 220, 220), fontsize=13):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return img


def make_score_bar(clip, bf16_clip, width, height):
    pct = clip / bf16_clip * 100
    color = color_for_pct(pct)
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    bar_w = int(width * min(pct, 100) / 100)
    draw.rectangle([0, 0, bar_w, height], fill=color)
    text = f"{pct:.1f}%"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(255, 255, 255), font=font)
    return img


# ── 1. Full grid (all 20 prompts) ──────────────────────────────────────────
def make_full_grid():
    n_prompts = len(PROMPTS)
    n_models  = len(MODELS)
    col_w = THUMB + PAD
    row_h = THUMB + LABEL_H + PAD

    total_w = HEADER_H + n_models * col_w + PAD
    total_h = n_prompts * row_h + HEADER_H

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    # Column headers
    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        pfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        hfont = pfont = ImageFont.load_default()

    header_colors = {"BF16": (60, 100, 180), "V7": (80, 140, 80), "V9b": (160, 80, 160)}
    for mi, m in enumerate(MODELS):
        x = HEADER_H + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 12), m, fill=(255, 255, 255), font=hfont)

    for pi, (cat, label) in enumerate(PROMPTS):
        y_base = HEADER_H + pi * row_h

        # Row label (left)
        draw.rectangle([0, y_base, HEADER_H - 2, y_base + row_h - PAD], fill=(35, 35, 55))
        short = label[:12] if len(label) > 12 else label
        draw.text((2, y_base + THUMB // 2 - 6), f"p{pi:02d}", fill=(180, 180, 180), font=pfont)

        for mi, m in enumerate(MODELS):
            x = HEADER_H + mi * col_w
            img_path = DIRS[m] / f"p{pi:02d}.png"
            thumb = load_thumb(img_path)
            canvas.paste(thumb, (x, y_base))

            # Score bar below thumb
            clip = CLIPS[m][pi]
            bf16 = BF16_CLIPS[pi]
            bar = make_score_bar(clip, bf16, THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v9b_full_grid.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 2. Highlights grid (8 selected prompts) ────────────────────────────────
HIGHLIGHT_INDICES = [
    0,   # Lion — big win
    5,   # Pagoda — near-perfect
    11,  # Street musician — biggest win (+47.1)
    6,   # Northern lights — good win
    12,  # Sushi — persistent weakness
    13,  # Bread — big loss despite visual accuracy
    14,  # Dragon — regression
    10,  # Ballet — regression
]

def make_highlights_grid():
    n_sel = len(HIGHLIGHT_INDICES)
    col_w = THUMB + PAD
    row_h = THUMB + LABEL_H + PAD

    LABEL_W = 180
    total_w = LABEL_W + len(MODELS) * col_w + PAD
    total_h = n_sel * row_h + HEADER_H + 10

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    try:
        hfont  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        pfont  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        pfontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        hfont = pfont = pfontb = ImageFont.load_default()

    header_colors = {"BF16": (60, 100, 180), "V7": (80, 140, 80), "V9b": (160, 80, 160)}
    for mi, m in enumerate(MODELS):
        x = LABEL_W + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 14), m, fill=(255, 255, 255), font=hfont)

    for si, pi in enumerate(HIGHLIGHT_INDICES):
        cat, label = PROMPTS[pi]
        y_base = HEADER_H + 10 + si * row_h

        # Row label
        draw.rectangle([0, y_base, LABEL_W - 2, y_base + row_h - PAD], fill=(30, 30, 48))
        draw.text((6, y_base + 6), f"p{pi:02d} [{cat}]", fill=(160, 180, 220), font=pfontb)
        # Wrap label text
        words = label.split()
        line, lines = "", []
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > 20:
                if line:
                    lines.append(line)
                line = w
            else:
                line = test
        if line:
            lines.append(line)
        for li, l in enumerate(lines[:4]):
            draw.text((6, y_base + 22 + li * 14), l, fill=(200, 200, 200), font=pfont)

        # V9b vs V7 delta
        delta = V9B_CLIPS[pi] - V7_CLIPS[pi]
        dcol = (60, 200, 60) if delta >= 0 else (220, 60, 60)
        dsign = "+" if delta >= 0 else ""
        draw.text((6, y_base + THUMB - 20), f"V9b vs V7: {dsign}{delta*1000:.1f}", fill=dcol, font=pfontb)

        for mi, m in enumerate(MODELS):
            x = LABEL_W + mi * col_w
            img_path = DIRS[m] / f"p{pi:02d}.png"
            thumb = load_thumb(img_path)
            canvas.paste(thumb, (x, y_base))

            clip = CLIPS[m][pi]
            bf16 = BF16_CLIPS[pi]
            bar  = make_score_bar(clip, bf16, THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v9b_highlights.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 3. Scaling law plot ─────────────────────────────────────────────────────
def make_scaling_plot():
    prompts = [174, 1002, 2132]
    actual  = [86.0, 88.9, 90.0]
    proj_p  = [3000, 4000, 7232, 10000]
    proj_v  = [0.0115 * np.log2(p) + 0.7744 for p in proj_p]
    proj_v  = [v * 100 for v in proj_v]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12121f")

    x_all = prompts + proj_p
    y_fit = [0.0115 * np.log2(p) * 100 + 77.44 for p in x_all]
    ax.plot([np.log2(p) for p in x_all], y_fit, color="#8888cc", linewidth=1.5,
            linestyle="--", label="log₂ fit", zorder=1)

    ax.scatter([np.log2(p) for p in prompts], actual, color="#66ccff", s=80,
               zorder=3, label="Measured", edgecolors="white", linewidths=0.8)

    labels_m = ["V6\n(174)", "V7\n(1002)", "V9b\n(2132)"]
    for p, v, lbl in zip(prompts, actual, labels_m):
        ax.annotate(f"{lbl}\n{v:.1f}%", (np.log2(p), v),
                    textcoords="offset points", xytext=(8, 4),
                    color="#aaddff", fontsize=8.5)

    ax.scatter([np.log2(p) for p in proj_p], proj_v, color="#cc88ff", s=50,
               zorder=2, marker="D", label="Projected", alpha=0.8)

    proj_labels = {3000: "V9c?", 4000: "4k", 7232: "Paper\n(7232)", 10000: "10k"}
    for p, v in zip(proj_p, proj_v):
        lbl = proj_labels.get(p, str(p))
        ax.annotate(f"{lbl}\n{v:.1f}%", (np.log2(p), v),
                    textcoords="offset points", xytext=(6, -18),
                    color="#cc99ff", fontsize=8)

    ax.axhline(100, color="#ff6666", linewidth=1, linestyle=":", alpha=0.6, label="BF16 = 100%")

    ax.set_xlabel("log₂(unique prompts)", color="#aaaacc", fontsize=11)
    ax.set_ylabel("OOD CLIP  (% of BF16)", color="#aaaacc", fontsize=11)
    ax.set_title("Ternary FLUX: OOD Quality vs Training Prompts", color="#ddddff", fontsize=13, pad=10)
    ax.tick_params(colors="#aaaacc")
    ax.spines[:].set_color("#444466")
    ax.set_xlim(6.5, 14.5)
    ax.set_ylim(83, 102)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{2**v:.0f}"))
    ax.legend(framealpha=0.3, facecolor="#22223a", edgecolor="#666688", labelcolor="#ccccee", fontsize=9)
    plt.tight_layout()

    out = VIZ_DIR / "v9b_scaling_law.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Building V9b post grids...")
    make_full_grid()
    make_highlights_grid()
    make_scaling_plot()
    print("Done.")
