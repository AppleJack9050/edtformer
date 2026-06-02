# Diverse image-subset selection (EDTformer + Facility Location)

Selects maximally-diverse image subsets (e.g. 300 / 332 / 364 images) from a larger
dataset by removing redundancy. Pipeline:

1. **EDTformer** ([Tong-Jin01/EDTformer](https://github.com/Tong-Jin01/EDTformer),
   DINOv2 ViT-B/14 backbone + decoder transformer) extracts a 4096-d L2-normalized
   global descriptor per image.
2. Build a dense cosine-similarity matrix (for datasets up to a few thousand images).
3. Maximize the **Facility Location** submodular objective with lazy-greedy
   (`apricot-select`) to pick the subset that best "covers" the full set — dropping
   near-duplicates while keeping diversity. Each target size is optimized independently.

## Setup

EDTformer's source (needed for the model's backbone package) is **not** vendored here —
clone it into `repo/`:

```bash
git clone https://github.com/Tong-Jin01/EDTformer.git repo
pip install torch torchvision timm apricot-select pillow tqdm numpy
```

Model weights download automatically from the EDTformer v1.0.0 torch.hub release on
first run (~449 MB, cached under `~/.cache/torch/hub/`).

## Usage

```bash
python select_diverse_subset.py \
    --image-dir /path/to/your/images \
    --output-dir ./output \
    --sizes 300 332 364
```

Outputs (in `--output-dir`):
- `subset_300/`, `subset_332/`, `subset_364/` — copied selected images
- `selected_<k>.txt` — file lists
- `embeddings.npz` — cached features (re-runs with different `--sizes` skip extraction)
- `selection_summary.json` — FL objective + mean intra-subset similarity per size

Useful flags: `--batch-size`, `--num-workers`, `--device`, `--resize H W`,
`--no-copy` (write lists only), `--repo-dir` (path to the cloned EDTformer repo).
