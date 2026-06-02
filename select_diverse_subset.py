#!/usr/bin/env python3
"""
Select maximally-diverse image subsets via EDTformer features + Facility Location.

Pipeline
--------
1. EDTformer (DINOv2 ViT-B/14 backbone + decoder, https://github.com/Tong-Jin01/EDTformer)
   produces a 4096-d, L2-normalized global descriptor per image.
2. Cosine similarity matrix S = E @ E.T  (cosine == dot product because descriptors
   are unit-normalized). Shape (N, N), kept dense in memory (intended for N < ~5k).
3. For each target size we maximize the Facility Location submodular objective
       f(A) = sum_i  max_{j in A} S[i, j]
   with apricot's lazy-greedy optimizer. Maximizing FL spreads the chosen subset so
   that every image in the full set is well "covered" by a near neighbour in the
   subset -> redundant / near-duplicate images are dropped, diversity is kept.

Each requested size is optimized INDEPENDENTLY (a fresh greedy run per size), so the
subsets are individually optimal rather than nested prefixes of one another.

Usage
-----
    python select_diverse_subset.py --image-dir /path/to/images \
        --output-dir ./output --sizes 300 332 364
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# EDTformer weights (torch.hub release v1.0.0). Cached after first download.
WEIGHTS_URL = "https://github.com/Tong-Jin01/EDTformer/releases/download/v1.0.0/EDTformer.pth"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Same preprocessing the repo uses for evaluation (datasets_ws.py): ToTensor ->
# ImageNet normalize -> resize. Default resize is 322x322 (parser.py default).
_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    def __init__(self, paths, resize):
        self.paths = paths
        self.resize = list(resize)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = _base_transform(img)
        img = TF.resize(img, self.resize, antialias=True)
        return img


def find_images(image_dir):
    paths = sorted(
        p for p in Path(image_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        sys.exit(f"No images found under {image_dir}")
    return paths


def load_model(repo_dir, device):
    """Instantiate VPRNet from the local EDTformer repo and load release weights."""
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import network  # noqa: from the EDTformer repo

    model = network.VPRNet()
    model = torch.nn.DataParallel(model)
    state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return model.module.to(device).eval()


@torch.no_grad()
def extract_features(model, paths, resize, batch_size, num_workers, device):
    loader = DataLoader(
        ImageDataset(paths, resize),
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=(device.type == "cuda"),
    )
    feats = []
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16) \
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    for batch in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device, non_blocking=True)
        with autocast:
            out = model(batch)
        feats.append(out.float().cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    # Re-normalize (autocast/half can perturb the unit norm a hair).
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# Facility-Location subset selection
# --------------------------------------------------------------------------- #
def select_facility_location(similarity, k):
    """Lazy-greedy Facility Location on a precomputed similarity matrix.

    Returns the selected indices in the order greedy picked them (so prefixes are
    themselves good FL subsets).
    """
    from apricot import FacilityLocationSelection
    sel = FacilityLocationSelection(k, metric="precomputed", optimizer="lazy").fit(similarity)
    return np.asarray(sel.ranking, dtype=int)


def facility_location_value(similarity, idx):
    """f(A) = sum_i max_{j in A} S[i,j]  -- higher == better coverage/diversity."""
    return float(similarity[:, idx].max(axis=1).sum())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True, help="Folder of source images (searched recursively).")
    ap.add_argument("--output-dir", default="./output", help="Where subset folders / lists are written.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 332, 364], help="Subset sizes to select.")
    ap.add_argument("--repo-dir", default=str(Path(__file__).parent / "repo"), help="Path to cloned EDTformer repo.")
    ap.add_argument("--resize", type=int, nargs=2, default=[322, 322], help="HxW resize for the model.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--embeddings", default=None,
                    help="Path to cache embeddings .npz. Reused if it matches the image set; "
                         "defaults to <output-dir>/embeddings.npz.")
    ap.add_argument("--copy", dest="copy", action="store_true", default=True,
                    help="Copy selected images into subset_<k>/ folders (default).")
    ap.add_argument("--no-copy", dest="copy", action="store_false",
                    help="Only write file lists, don't copy images.")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = Path(args.embeddings) if args.embeddings else out_dir / "embeddings.npz"

    paths = find_images(args.image_dir)
    path_strs = [str(p) for p in paths]
    print(f"Found {len(paths)} images under {args.image_dir}")

    for k in args.sizes:
        if k > len(paths):
            sys.exit(f"Requested subset size {k} > number of images {len(paths)}")

    # --- features (load cache if it matches this exact image set) -----------
    feats = None
    if emb_path.exists():
        cache = np.load(emb_path, allow_pickle=True)
        if list(cache["paths"]) == path_strs:
            feats = cache["features"]
            print(f"Loaded cached embeddings from {emb_path}  shape={feats.shape}")
        else:
            print(f"Cache {emb_path} is stale (image set changed); recomputing.")
    if feats is None:
        model = load_model(args.repo_dir, device)
        feats = extract_features(model, paths, args.resize, args.batch_size, args.num_workers, device)
        np.savez(emb_path, features=feats, paths=np.array(path_strs))
        print(f"Saved embeddings -> {emb_path}  shape={feats.shape}")

    # --- cosine similarity matrix (dense) -----------------------------------
    print(f"Building {feats.shape[0]}x{feats.shape[0]} cosine-similarity matrix ...")
    similarity = (feats @ feats.T).astype(np.float32)

    # --- independent Facility-Location selection per size -------------------
    summary = {}
    for k in sorted(args.sizes):
        print(f"\n=== Facility Location: selecting {k} of {len(paths)} ===")
        idx = select_facility_location(similarity, k)
        fl_val = facility_location_value(similarity, idx)
        # mean pairwise similarity within the subset (lower == more diverse)
        sub = feats[idx]
        sub_sim = sub @ sub.T
        n = len(idx)
        mean_pair = float((sub_sim.sum() - n) / (n * (n - 1)))
        print(f"  FL objective f(A) = {fl_val:.2f}   (max possible = {len(paths)})")
        print(f"  mean intra-subset cosine similarity = {mean_pair:.4f} (lower is more diverse)")

        selected_paths = [path_strs[i] for i in idx]
        list_file = out_dir / f"selected_{k}.txt"
        list_file.write_text("\n".join(selected_paths) + "\n")
        print(f"  wrote {list_file}")

        if args.copy:
            dst_dir = out_dir / f"subset_{k}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in selected_paths:
                shutil.copy2(src, dst_dir / Path(src).name)
            print(f"  copied {len(selected_paths)} images -> {dst_dir}")

        summary[k] = {
            "fl_objective": fl_val,
            "max_objective": len(paths),
            "mean_intra_similarity": mean_pair,
            "indices": idx.tolist(),
            "list_file": str(list_file),
        }

    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary -> {out_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()
