"""
Generate teacher (BF16) latents for flow-matching distillation.

The 1.58-bit FLUX paper uses self-supervised fine-tuning:
  1. Generate images from BF16 teacher using text prompts only.
  2. Encode to latent z_0.
  3. Train student with proper flow matching: z_t = (1-t)*z_0 + t*eps, velocity = eps - z_0.

This script handles step 1+2: generates z_0 latents from BF16 teacher.

Usage:
  python generate_teacher_dataset.py --n-images 50 --steps 28 --seed 42
  → saves output/teacher_dataset.pt
"""
import os, argparse, time, torch
os.environ.setdefault("HF_HOME", "/home/jovyan/.cache/huggingface")

from pathlib import Path
from diffusers import FluxPipeline
from models.ternary import memory_stats

MODEL_NAME = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = Path("output")

PROMPTS = [
    # People & portraits
    "Cyberpunk samurai on a neon-lit rooftop at dusk, dramatic rim lighting",
    "Portrait of a young woman with wild curly hair in golden light",
    "Elderly fisherman with weathered face and deep-set eyes, black and white portrait",
    "Young musician playing guitar on a stage with dramatic spotlights",
    "Astronaut floating in space with Earth in background",
    "A chef preparing food in a modern open kitchen",
    "Ballet dancer mid-leap in a sunlit studio, motion blur",
    "Street photographer in the rain, Tokyo, neon reflections on wet pavement",
    "A traditional Japanese geisha in full makeup, ornate kimono, cherry blossoms",
    "A medieval knight removing his helmet, battle-worn armor in dramatic lighting",
    "A Victorian gentleman with a top hat and monocle, foggy London street",
    "A female scientist in a futuristic laboratory, holographic interfaces",
    "A surfer riding a massive ocean wave at sunset, action photography",
    "A traditional Indian bride in red and gold attire, intricate jewelry",
    "A monk in saffron robes meditating at a mountaintop temple",
    "A pirate captain standing at the ship's helm, stormy seas",
    "A boxer mid-punch, dramatic arena lighting, sweat and intensity",
    "An elderly woman knitting by a fireplace, cozy winter scene",
    "A young girl in a sunlit flower field, reaching for butterflies",
    "A nomadic shepherd on a mountain pass, flock of sheep in mist",
    "A flamenco dancer in a red dress, mid-spin, dramatic stage light",
    "Portrait of an African tribal elder with ceremonial face paint",
    "A deep-sea diver exploring a sunken shipwreck, flashlight beams",
    # Nature & landscapes
    "A fantasy landscape with mountains and a river",
    "A cozy wooden cabin in a snowy forest at night, warm interior glow",
    "Aerial view of a coastal city at sunset",
    "Ancient temple ruins covered in vines in a tropical jungle",
    "Underwater scene with colorful tropical fish and coral reef",
    "Autumn forest path with golden leaves and soft fog",
    "Desert sand dunes at sunrise, long shadow patterns",
    "Volcanic eruption at night with lava flowing into the ocean",
    "Northern lights over a frozen tundra with a lone wolf silhouette",
    "Cherry blossom trees lining a river in Japan at golden hour",
    "A misty fjord at dawn, rocky cliffs and mirror-calm water",
    "Terraced rice paddies in Bali, green layers cascading down the hillside",
    "A frozen waterfall in winter, ice crystals in morning light",
    "A bamboo forest path in Kyoto, filtered green light",
    "Thunderstorm over the Grand Canyon, multiple lightning strikes",
    "A field of lavender at sunset, purple rows to the horizon",
    "Tropical beach at night, bioluminescent waves glowing blue",
    "Autumn reflections in a still mountain lake, mirror image",
    "Red rock formations in the American Southwest, dramatic golden hour",
    "A moonlit forest clearing with fireflies, ethereal atmosphere",
    "Arctic ice floes at midnight sun, polar bear silhouette",
    "A sandstorm approaching a Middle Eastern desert city",
    "Misty Scottish highlands at dawn, ancient castle ruins",
    "A secret garden with overgrown roses and iron gates",
    "The Amazon rainforest canopy from above, morning mist",
    "A massive wave crashing on a rocky coastline, spray in air",
    # Animals & macro
    "Close-up of a red rose with water droplets, macro photography",
    "Macro photography of a butterfly on a purple flower",
    "Majestic lion on a rocky outcrop at sunrise, African savanna",
    "Hummingbird hovering beside tropical flowers, ultra-sharp detail",
    "A pod of dolphins leaping through a turquoise wave",
    "A wolf howling at the full moon on a snowy mountain ridge",
    "An eagle in flight over snow-capped mountain peaks, majestic",
    "A jaguar stalking silently through rainforest undergrowth, intense eyes",
    "Colorful macaws perched in a tropical jungle tree canopy",
    "An octopus displaying iridescent colors on a coral reef",
    "A polar bear mother and cubs on arctic sea ice",
    "A school of fish swirling in a tornado formation in clear ocean",
    "Extreme macro of a chameleon's eye, scales and iris detail",
    "A herd of elephants at a waterhole, dramatic dusty sunset",
    "A snowy owl perched on a snow-covered pine branch, winter",
    "A sea turtle gliding through crystal clear Caribbean water",
    "Two red foxes playing in autumn leaves in a forest",
    "A peacock displaying its full plumage, iridescent feathers",
    "A great white shark breaching the ocean surface, dramatic spray",
    "Macro shot of a dewdrop on a spider web at sunrise",
    # Architecture & urban
    "Minimalist black and white architectural photograph, symmetry",
    "A steam locomotive crossing a mountain bridge, dramatic clouds",
    "Oil painting of a harbor with sailing boats at golden hour",
    "Street art mural on a building wall, vibrant graffiti style",
    "Post-apocalyptic overgrown city with nature reclaiming streets",
    "Spiral staircase in an old European library, warm amber light",
    "Brutalist concrete skyscraper at night, rain-slicked plaza",
    "Gothic cathedral interior with colorful stained glass flooding light",
    "Tokyo street intersection at night, crowds and neon signs",
    "The interior of the Alhambra palace, intricate Moorish tile patterns",
    "A Moroccan medina at night, lanterns and ornate archways",
    "A traditional Japanese wooden pagoda in autumn foliage",
    "An ancient Roman Colosseum at golden hour, warm stone light",
    "A luxury rooftop terrace overlooking a modern city at night",
    "A Victorian greenhouse filled with exotic tropical plants",
    "Colorful favela houses cascading down a hillside in Brazil",
    "An art nouveau metro station with ornate iron and glass",
    "A shipping container port at night, cranes and colored containers",
    "A medieval cobblestone market town on a misty morning",
    "A futuristic underwater research station with glowing corridors",
    "An old wooden fishing pier at golden hour, reflections",
    "Inside a massive cave cathedral, natural light shafts",
    "A cyberpunk megacity at night, flying vehicles and holographic ads",
    "The Great Wall of China winding over mountain peaks in mist",
    "A floating village on stilts in Southeast Asia, golden light",
    # Fantasy & sci-fi
    "A dragon flying over a medieval castle at dusk",
    "A futuristic humanoid robot in a busy marketplace",
    "Abstract geometric art in vivid colors, oil on canvas",
    "Glowing bioluminescent forest at night, ethereal blue and green",
    "Steampunk airship fleet above Victorian city, dramatic storm clouds",
    "Alien planet with two moons, exotic bioluminescent flora",
    "Crystal cave with prismatic light refractions, underground lake",
    "A wizard casting a spell in a dark enchanted forest",
    "A giant ancient tree housing an entire elven city in its canopy",
    "A time traveler emerging from a glowing portal in Victorian London",
    "An underwater mermaid kingdom with coral spires and sea glass",
    "A portal to another dimension opening in a dark forest",
    "A spaceship graveyard drifting through an asteroid belt",
    "A phoenix rising from the ashes, golden fire and feathers",
    "An ice palace in an arctic wasteland, blue crystalline spires",
    "A giant mecha robot standing in a rain-soaked ruined city",
    "A fairytale cottage in an enchanted mushroom forest",
    "A cosmic deity silhouetted against a swirling galaxy",
    "An ancient sea kraken rising from a storm-tossed ocean",
    "A temple built inside an active volcano crater, lava moat",
    "A magical library with floating books and glowing orbs",
    "Alien spacecraft wreckage in a terrestrial desert, old and overgrown",
    "A necromancer summoning spirits in a graveyard at midnight",
    "A sky city of floating islands connected by vine bridges",
    # Still life & food
    "A freshly baked sourdough loaf on a rustic wooden table, warm tones",
    "Colorful Indian spices arranged in small bowls, overhead view",
    "Rainy window with a coffee cup and a book, cozy atmosphere",
    "A Japanese tea ceremony setup on a bamboo mat, zen simplicity",
    "Fresh sushi arrangement on black slate, artistic overhead composition",
    "Old leather-bound books stacked in a dusty antique library nook",
    "A collection of vintage glass bottles with colored liquids, sunlit",
    "Macro shot of a snowflake crystal on dark velvet, perfect geometry",
    "Freshly baked croissants with steam rising in a Parisian café",
    "A vibrant farmer's market stall with colorful vegetables and flowers",
    "A single white candle flame reflected in dark water, minimal",
    "A pile of autumn leaves in gold and red, texture shot from above",
    "A stack of colorful French macarons in a Parisian pastry shop",
    "Macro photography of honey dripping from a wooden spoon",
    "An elaborate charcuterie board on a marble table, overhead",
    # Vehicles & industry
    "Formula 1 racing car blurred at speed on a night circuit",
    "An old rusted ship on a beach, low tide, dramatic sky",
    "Inside a busy forge, molten metal pouring, sparks flying",
    "A vintage 1930s biplane banking in a summer sky, dramatic clouds",
    "A luxury yacht moored in a Mediterranean harbor at golden hour",
    "A classic 1960s American muscle car on a rain-slicked neon street",
    "A hot air balloon drifting over the Serengeti at sunrise",
    "A cargo ship battling massive storm waves, waves over the bow",
    "A nuclear submarine surfacing in the arctic, ice breaking",
    "An old steam locomotive in a mountain blizzard, headlight piercing",
    # Art styles
    "Impressionist painting of a Parisian boulevard in the rain",
    "Watercolor illustration of a magical treehouse village at dusk",
    "Charcoal sketch style portrait of a Renaissance nobleman",
    "Pop art comic style superhero action scene",
    "A ukiyo-e style woodblock print scene, Mount Fuji in the distance",
    "A baroque oil painting of a lavish royal feast, candlelight",
    "A cubist portrait of a woman, fragmented geometric planes",
    "A pointillist park scene, thousands of tiny colored dots",
    "A Chinese ink wash painting of misty mountain valleys",
    "An art deco travel poster for Paris, 1920s geometric style",
    "A psychedelic poster, swirling neon colors and fractals",
    "A vintage Soviet propaganda poster style illustration",
    "A detailed pen and ink scientific illustration of a nautilus",
    "A neon sign art installation in a dark gallery space",
    "A linocut print of an ocean scene, bold black and white lines",
    # Emotional & conceptual
    "A silhouette of a lone figure standing at a cliff edge at sunset",
    "Two children sharing a secret, sitting in a sunlit treehouse",
    "An elderly violinist playing in an empty concert hall, evening light",
    "A soldier returning home, tearful reunion at a train station",
    "A crowd of protesters with candles in the night, peaceful vigil",
    "A child reading under a tree, dappled light and wonder",
    "Two elderly friends playing chess in a sunny park, laughing",
    "A person standing in the rain at a crossroads, looking up",
    # Extra diversity
    "Neon-lit karaoke bar interior, Tokyo, late night festive crowd",
    "A traditional Moroccan hammam, steam and intricate tilework",
    "A carnival funhouse mirror, surreal distorted reflections",
    "A cave painting in Lascaux style, ancient ochre animals",
    "A professional esports tournament arena, screens and crowds",
    "A coral bleaching scene underwater, hauntingly beautiful",
    "A glass chess set on a transparent table, macro detail",
    "An origami swan in a field of rice paper, soft lighting",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images",  type=int, default=50)
    p.add_argument("--steps",     type=int, default=28)
    p.add_argument("--res",       type=int, default=1024)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--out",       type=str, default="output/teacher_dataset.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype  = torch.bfloat16

    print(f"=== Teacher Dataset Generation ===")
    print(f"  n_images={args.n_images}, steps={args.steps}, res={args.res}")

    print("\nLoading BF16 pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    dataset = []
    t0 = time.time()

    for i in range(args.n_images):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed   = args.seed + i
        gen = torch.Generator("cuda").manual_seed(seed)

        # Generate with output_type="latent" → packed z_0 [1, seq_len, 64]
        with torch.no_grad():
            result = pipe(
                prompt,
                generator=gen,
                num_inference_steps=args.steps,
                guidance_scale=3.5,
                height=args.res,
                width=args.res,
                output_type="latent",
            )
        latent = result.images.cpu()
        del result  # release inference activations

        # Encode prompt (text encoders on GPU)
        with torch.no_grad():
            pe, poe, ti = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=device,
                num_images_per_prompt=1, max_sequence_length=256,
            )

        dataset.append({
            "latent_z0":     latent,
            "prompt_embeds": pe.cpu(),
            "pooled_embeds": poe.cpu(),
            "text_ids":      ti.cpu(),
            "prompt":        prompt,
            "seed":          seed,
        })
        del pe, poe, ti
        torch.cuda.empty_cache()  # release caching allocator memory after each image

        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (args.n_images - i - 1)
        print(f"  [{i+1:3d}/{args.n_images}] '{prompt[:50]}...' | "
              f"elapsed={elapsed:.0f}s eta={remaining:.0f}s")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)
    print(f"\nSaved {len(dataset)} items → {out_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    # Verify latent shape
    sample = dataset[0]
    print(f"  Latent z_0 shape: {sample['latent_z0'].shape}")
    print(f"  Prompt embeds: {sample['prompt_embeds'].shape}")


if __name__ == "__main__":
    main()
