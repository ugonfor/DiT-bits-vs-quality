"""
Merge two teacher datasets into one for V5 offline FM training.
Usage:
  python merge_datasets.py \
    --datasets output/teacher_dataset_200.pt output/teacher_dataset_350.pt \
    --out output/teacher_dataset_combined.pt
"""
import argparse, torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Paths to teacher dataset .pt files to merge")
    p.add_argument("--out", required=True,
                   help="Output path for merged dataset")
    args = p.parse_args()

    combined = []
    for path in args.datasets:
        ds = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(ds)} items from {path}")
        combined.extend(ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, out_path)

    # Verify
    unique_prompts = len({item["prompt"] for item in combined})
    print(f"\nMerged: {len(combined)} items, {unique_prompts} unique prompts → {out_path}")
    print(f"  Latent shape: {combined[0]['latent_z0'].shape}")


if __name__ == "__main__":
    main()
