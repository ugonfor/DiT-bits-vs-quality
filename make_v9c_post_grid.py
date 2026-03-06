"""
Generate comparison grids for post-012 (V9c).
Creates:
  1. output/viz/v9c_full_grid.png — all 20 prompts, BF16 | V9b | V9c
  2. output/viz/v9c_highlights.png — 8 selected prompts (wins + regressions)
  3. output/viz/v9c_scaling_law.png — OOD CLIP % vs log2(prompts) with V9c miss
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
BF16_CLIPS = [0.3276,0.3220,0.3417,0.3134,0.3246,0.3510,0.3006,0.3456,0.3384,0.3845,0.3433,0.3677,0.3122,0.3433,0.3668,0.3365,0.3684,0.3179,0.2902,0.3534]
V9B_CLIPS  = [0.3128,0.3232,0.3340,0.2941,0.3000,0.3507,0.2873,0.3355,0.3262,0.3263,0.2616,0.3133,0.2181,0.2672,0.2976,0.3071,0.3404,0.2850,0.2918,0.3006]
V9C_CLIPS  = [0.3190,0.3133,0.3466,0.2643,0.3000,0.3424,0.2201,0.3445,0.3117,0.3259,0.2707,0.3064,0.1785,0.2974,0.3139,0.3191,0.3238,0.2912,0.2915,0.3159]

DIRS = {
    "BF16": OUTPUT_DIR / "eval_diverse_bf16",
    "V9b":  OUTPUT_DIR / "eval_diverse_v9b",
    "V9c":  OUTPUT_DIR / "eval_diverse_v9c",
}
MODELS = ["BF16", "V9b", "V9c"]
CLIPS  = {"BF16": BF16_CLIPS, "V9b": V9B_CLIPS, "V9c": V9C_CLIPS}

THUMB = 256
LABEL_H = 36
HEADER_H = 50
PAD = 4


def load_thumb(path, size=THUMB):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def color_for_pct(pct):
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

    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        pfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        hfont = pfont = ImageFont.load_default()

    header_colors = {"BF16": (60, 100, 180), "V9b": (80, 140, 80), "V9c": (180, 80, 60)}
    for mi, m in enumerate(MODELS):
        x = HEADER_H + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 12), m, fill=(255, 255, 255), font=hfont)

    for pi, (cat, label) in enumerate(PROMPTS):
        y_base = HEADER_H + pi * row_h
        draw.rectangle([0, y_base, HEADER_H - 2, y_base + row_h - PAD], fill=(35, 35, 55))
        draw.text((2, y_base + THUMB // 2 - 6), f"p{pi:02d}", fill=(180, 180, 180), font=pfont)

        for mi, m in enumerate(MODELS):
            x = HEADER_H + mi * col_w
            img_path = DIRS[m] / f"p{pi:02d}.png"
            thumb = load_thumb(img_path)
            canvas.paste(thumb, (x, y_base))

            clip = CLIPS[m][pi]
            bf16 = BF16_CLIPS[pi]
            bar = make_score_bar(clip, bf16, THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v9c_full_grid.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 2. Highlights grid (8 selected prompts) ────────────────────────────────
HIGHLIGHT_INDICES = [
    6,   # Northern lights — biggest regression (−67.2)
    12,  # Sushi — persistent worst + big regression (−39.6)
    3,   # Cathedral — big regression (−29.8)
    13,  # Bread — biggest win (+30.2)
    14,  # Dragon — good win (+16.3)
    2,   # Wolf — exceeded BF16 (101.4%)
    7,   # Volcano — near-perfect (99.7%)
    15,  # Astronaut — good win (+12.0)
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

    header_colors = {"BF16": (60, 100, 180), "V9b": (80, 140, 80), "V9c": (180, 80, 60)}
    for mi, m in enumerate(MODELS):
        x = LABEL_W + mi * col_w
        draw.rectangle([x, 0, x + THUMB, HEADER_H - 2], fill=header_colors[m])
        bbox = draw.textbbox((0, 0), m, font=hfont)
        tw = bbox[2] - bbox[0]
        draw.text((x + (THUMB - tw) // 2, 14), m, fill=(255, 255, 255), font=hfont)

    for si, pi in enumerate(HIGHLIGHT_INDICES):
        cat, label = PROMPTS[pi]
        y_base = HEADER_H + 10 + si * row_h

        draw.rectangle([0, y_base, LABEL_W - 2, y_base + row_h - PAD], fill=(30, 30, 48))
        draw.text((6, y_base + 6), f"p{pi:02d} [{cat}]", fill=(160, 180, 220), font=pfontb)
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

        # V9c vs V9b delta
        delta = V9C_CLIPS[pi] - V9B_CLIPS[pi]
        dcol = (60, 200, 60) if delta >= 0 else (220, 60, 60)
        dsign = "+" if delta >= 0 else ""
        draw.text((6, y_base + THUMB - 20), f"V9c vs V9b: {dsign}{delta*1000:.1f}", fill=dcol, font=pfontb)

        for mi, m in enumerate(MODELS):
            x = LABEL_W + mi * col_w
            img_path = DIRS[m] / f"p{pi:02d}.png"
            thumb = load_thumb(img_path)
            canvas.paste(thumb, (x, y_base))

            clip = CLIPS[m][pi]
            bf16 = BF16_CLIPS[pi]
            bar  = make_score_bar(clip, bf16, THUMB, LABEL_H - PAD)
            canvas.paste(bar, (x, y_base + THUMB))

    out = VIZ_DIR / "v9c_highlights.png"
    canvas.save(out)
    print(f"Saved: {out}  ({canvas.width}×{canvas.height}px)")


# ── 3. Scaling law plot (with V9c miss) ───────────────────────────────────
def make_scaling_plot():
    prompts_fit = [174, 1002, 2132]
    actual_fit  = [86.0, 88.9, 90.0]

    # V9c — the miss
    prompts_miss = [4007]
    actual_miss  = [88.8]
    predicted_miss = [0.0115 * np.log2(4007) * 100 + 77.44]  # 91.2%

    # Projected points
    proj_p  = [3000, 7232, 10000]
    proj_v  = [0.0115 * np.log2(p) * 100 + 77.44 for p in proj_p]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12121f")

    # Log2 fit line
    x_all = list(range(100, 12000, 50))
    y_fit = [0.0115 * np.log2(p) * 100 + 77.44 for p in x_all]
    ax.plot([np.log2(p) for p in x_all], y_fit, color="#8888cc", linewidth=1.5,
            linestyle="--", label="log₂ fit (V6–V9b)", zorder=1)

    # Measured points that fit
    ax.scatter([np.log2(p) for p in prompts_fit], actual_fit, color="#66ccff", s=80,
               zorder=3, label="Measured (fits)", edgecolors="white", linewidths=0.8)

    labels_m = ["V6\n(174)", "V7\n(1002)", "V9b\n(2132)"]
    for p, v, lbl in zip(prompts_fit, actual_fit, labels_m):
        ax.annotate(f"{lbl}\n{v:.1f}%", (np.log2(p), v),
                    textcoords="offset points", xytext=(8, 4),
                    color="#aaddff", fontsize=8.5)

    # V9c — the miss (red)
    ax.scatter([np.log2(4007)], [88.8], color="#ff5555", s=120,
               zorder=4, marker="X", label="V9c (MISS)", edgecolors="white", linewidths=1)
    ax.annotate(f"V9c\n(4007)\n88.8%\n(pred: 91.2%)", (np.log2(4007), 88.8),
                textcoords="offset points", xytext=(10, -30),
                color="#ff8888", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#ff8888", lw=1))

    # Predicted V9c point (hollow)
    ax.scatter([np.log2(4007)], predicted_miss, color="#cc88ff", s=60,
               zorder=2, marker="D", facecolors="none", edgecolors="#cc88ff", linewidths=1.5)
    ax.annotate(f"predicted\n91.2%", (np.log2(4007), predicted_miss[0]),
                textcoords="offset points", xytext=(10, 8),
                color="#cc99ff", fontsize=7.5)

    # Arrow showing the miss
    ax.annotate("", xy=(np.log2(4007), 88.8), xytext=(np.log2(4007), predicted_miss[0]),
                arrowprops=dict(arrowstyle="<->", color="#ff6666", lw=1.5, ls="--"))
    ax.text(np.log2(4007) - 0.3, 90.0, "−2.4pp", color="#ff6666", fontsize=9, fontweight="bold")

    # Projected (faded)
    ax.scatter([np.log2(p) for p in proj_p], proj_v, color="#cc88ff", s=30,
               zorder=2, marker="D", alpha=0.3)

    ax.axhline(100, color="#ff6666", linewidth=1, linestyle=":", alpha=0.6, label="BF16 = 100%")
    ax.axhline(90.0, color="#66ff66", linewidth=1, linestyle=":", alpha=0.3, label="V9b = 90.0%")

    ax.set_xlabel("log₂(unique prompts)", color="#aaaacc", fontsize=11)
    ax.set_ylabel("OOD CLIP  (% of BF16)", color="#aaaacc", fontsize=11)
    ax.set_title("Ternary FLUX: Scaling Law Breaks at ~4000 Prompts", color="#ffdddd", fontsize=13, pad=10)
    ax.tick_params(colors="#aaaacc")
    ax.spines[:].set_color("#444466")
    ax.set_xlim(6.5, 14.5)
    ax.set_ylim(83, 102)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{2**v:.0f}"))
    ax.legend(framealpha=0.3, facecolor="#22223a", edgecolor="#666688", labelcolor="#ccccee", fontsize=8, loc="upper left")
    plt.tight_layout()

    out = VIZ_DIR / "v9c_scaling_law.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Building V9c post grids...")
    make_full_grid()
    make_highlights_grid()
    make_scaling_plot()
    print("Done.")
