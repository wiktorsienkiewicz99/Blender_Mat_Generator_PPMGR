#!/usr/bin/env python3
"""
BASE‑COLOR TEXTURES → TWO KINDS OF CLUSTERING (COLOR & PATTERN) → COPY
+ SAVE BASE‑COLOR PATH LIST

What it does:
  1) Scans a textures directory for images (non-recursive by default).
  2) Filters to base‑color/albedo-like images (skips normals, height, roughness, etc.).
  3) Takes up to --limit images AFTER filtering (set 0 for ALL; default 0).
  4) Runs TWO independent clustering passes:
     a) COLOR-BASED clustering  (HSV histogram features)
     b) PATTERN-BASED clustering (CLIP ViT-B/32 embeddings)
  5) Copies files to:
       out_dir/color_clusters/cluster_XX/filename
       out_dir/pattern_clusters/cluster_XX/filename
  6) Saves CSV mappings:
       color_cluster_mapping.csv
       pattern_cluster_mapping.csv
  7) Saves base‑color texture paths (TXT + CSV) to --basecolor-list-dir

Usage example (entire dataset):
python texture_cluster.py \
  --textures-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/textures" \
  --out-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset" \
  --limit 0 --k-color 18 --k-pattern 24
"""

import os
import csv
import math
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans

# Reduce HF tokenizers fork warning spam
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------
# Base-color filter helpers
# --------------------------

POSITIVE_KEYS = [
    "basecolor", "base_color", "albedo", "diffuse", "albd",
    "_col", "-col", " color", "colour", "base-col", "base col", "beauty"
]
NEGATIVE_KEYS = [
    "normal", "_nrm", "-nrm", "nrm", "nor", "_n", "-n",
    "height", "disp", "displ", "displace", "bump", "dnm",
    "rough", "roughness", "gloss", "spec", "specular", "metal", "metallic",
    "ao", "ambientocclusion", "cavity", "mask", "opacity", "alpha", "emiss", "emission",
    "deriv", "ddx", "ddy", "dx", "dy"
]

def _has_any(s: str, keys) -> bool:
    s = s.lower()
    return any(k in s for k in keys)

def _positive_score(name: str) -> int:
    n = name.lower()
    return sum(1 for k in POSITIVE_KEYS if k in n)

def _negative_hit(name: str) -> bool:
    return _has_any(name, NEGATIVE_KEYS)

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)

def _colorfulness_hs(img: Image.Image) -> float:
    """Hasler & Süsstrunk colorfulness metric."""
    arr = _pil_to_np(img)
    if arr.ndim == 2:
        return 0.0
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(np.abs(rg)), np.mean(np.abs(yb))
    return math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2)

def _mean_saturation(img: Image.Image) -> float:
    hsv = img.convert("HSV")
    s = np.asarray(hsv)[..., 1].astype(np.float32) / 255.0
    return float(np.mean(s))

def _is_probable_normal(img: Image.Image) -> bool:
    """Normal maps tend to be bluish (B >> R,G) with mid saturation."""
    arr = _pil_to_np(img)
    if arr.ndim == 2:
        return False
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    b, g, r = mean_rgb[2], mean_rgb[1], mean_rgb[0]
    sat = _mean_saturation(img)
    bluish = (b > r + 15) and (b > g + 15)
    return bluish and (0.1 <= sat <= 0.6)

def _is_probable_grayscale(img: Image.Image) -> bool:
    """Height/AO often near grayscale (very low saturation)."""
    return _mean_saturation(img) < 0.05

def looks_like_basecolor(path: Path, strong_filename_filter: bool = False) -> bool:
    """
    Decide if a file is likely a base-color (albedo) texture.
    1) If strong_filename_filter=True, reject negatives by filename outright.
    2) If filename indicates albedo (positive token) and not negative → accept.
    3) Else use content heuristics: reject normals/grayscale; require some saturation & colorfulness.
    """
    lname = path.name.lower()
    if strong_filename_filter and _negative_hit(lname):
        return False
    if _positive_score(lname) > 0 and not _negative_hit(lname):
        return True
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return False
    if _is_probable_normal(img) or _is_probable_grayscale(img):
        return False
    return (_mean_saturation(img) > 0.10) and (_colorfulness_hs(img) > 5.0)

# --------------------------
# Discovery
# --------------------------

def list_basecolor_images(root: Path, exts: Tuple[str, ...], recursive: bool, limit: int, strong_filename_filter: bool) -> List[Path]:
    """Collect base-color image paths from `root`. If limit<=0, return ALL after filtering."""
    if recursive:
        all_paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    else:
        all_paths = sorted([p for p in root.iterdir() if p.suffix.lower() in exts])

    if not all_paths:
        return []

    candidates = []
    for p in tqdm(all_paths, desc="Filtering base-color"):
        if looks_like_basecolor(p, strong_filename_filter=strong_filename_filter):
            candidates.append(p)

    if not candidates:
        return []

    return candidates if limit <= 0 else candidates[:limit]

# --------------------------
# Save base-color path list
# --------------------------

def save_basecolor_list(paths: List[Path], out_dir: Path) -> None:
    """Save base-color absolute paths to TXT and CSV in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "basecolor_paths.txt"
    csv_path = out_dir / "basecolor_paths.csv"

    with open(txt_path, "w") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "abs_path", "parent_dir"])
        for p in paths:
            w.writerow([p.name, str(p.resolve()), str(p.parent.resolve())])

    print(f"[Report] Base‑color list saved:\n  - {txt_path}\n  - {csv_path}")

# --------------------------
# COLOR features & clustering
# --------------------------

def extract_color_features(img: Image.Image, blur: bool = True) -> np.ndarray:
    """
    Return a color-only descriptor:
      - optional small blur to suppress pattern detail
      - HSV histograms: H(18 bins), S(8), V(8), L1-normalized
      - plus mean HSV (3 vals)
    Total dims: 18 + 8 + 8 + 3 = 37
    """
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8)
    H, S, V = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)

    h_hist, _ = np.histogram(H, bins=18, range=(0, 255), density=False)
    s_hist, _ = np.histogram(S, bins=8, range=(0, 255), density=False)
    v_hist, _ = np.histogram(V, bins=8, range=(0, 255), density=False)

    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    feat = feat / (np.sum(feat) + 1e-6)  # L1 normalize

    mean_h = float(np.mean(H) / 255.0)
    mean_s = float(np.mean(S) / 255.0)
    mean_v = float(np.mean(V) / 255.0)

    return np.concatenate([feat, np.array([mean_h, mean_s, mean_v], dtype=np.float32)])

def build_color_feature_matrix(paths: List[Path]) -> np.ndarray:
    feats = []
    for p in tqdm(paths, desc="Extracting color features"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        feats.append(extract_color_features(img, blur=True))
    return np.stack(feats, axis=0)

# --------------------------
# PATTERN features (CLIP) & clustering
# --------------------------

def load_clip(device: torch.device):
    """Load CLIP processor + model."""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    return processor, model

def embed_images_clip(paths: List[Path], processor, model, device: torch.device, batch_size: int = 32) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding images (CLIP)"):
        batch_paths = paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            batch_imgs.append(img)

        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy()
        feats.append(emb)
    return np.concatenate(feats, axis=0)

# --------------------------
# Generic helpers
# --------------------------

def choose_k(n: int, k_arg: int | None, fallback: int) -> int:
    """If --k provided use it, else heuristic sqrt(n) clamped to [8, 30] with fallback."""
    if k_arg is not None:
        return max(2, int(k_arg))
    k = int(round(math.sqrt(n)))
    k = max(8, min(30, k))
    return k if k > 0 else fallback

def copy_into_clusters(paths: List[Path], labels: np.ndarray, out_dir: Path, prefix: str) -> None:
    """Copy images into out_dir/<prefix>/cluster_XX/filename."""
    base = out_dir / prefix
    base.mkdir(parents=True, exist_ok=True)
    for p, cid in tqdm(list(zip(paths, labels)), desc=f"Copying ({prefix})"):
        cluster_folder = base / f"cluster_{cid:02d}"
        cluster_folder.mkdir(parents=True, exist_ok=True)
        dst = cluster_folder / p.name
        if not dst.exists():
            shutil.copy2(str(p), str(dst))

def save_mapping_csv(paths: List[Path], labels: np.ndarray, out_dir: Path, csv_name: str, prefix: str) -> None:
    """Save a CSV mapping image → cluster for later manual renaming."""
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "abs_path", "cluster_id", "cluster_folder"])
        for p, cid in zip(paths, labels):
            w.writerow([p.name, str(p.resolve()), int(cid), f"{prefix}/cluster_{cid:02d}"])
    print(f"[Report] Mapping saved to {csv_path}")

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Cluster BASE-COLOR textures by COLOR and by PATTERN, then copy into folders.")
    ap.add_argument("--textures-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max images to cluster after filtering (0 = ALL, default: 0)")
    ap.add_argument("--recursive", action="store_true", help="Include images in subfolders")
    ap.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png", help="Comma-separated extensions")
    ap.add_argument("--strong-filename-filter", action="store_true",
                    help="Reject non-albedo by filename only (faster, stricter).")

    ap.add_argument("--k-color", type=int, default=None, help="K for color-based clustering (default heuristic)")
    ap.add_argument("--k-pattern", type=int, default=None, help="K for pattern-based clustering (default heuristic)")

    ap.add_argument(
        "--basecolor-list-dir",
        type=Path,
        default=Path("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary"),
        help="Directory to save/load base-color path list (TXT + CSV)."
    )

    args = ap.parse_args()

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    if not args.textures_dir.exists():
        raise SystemExit(f"[!] textures-dir does not exist: {args.textures_dir}")

    basecolor_txt_path = args.basecolor_list_dir / "basecolor_paths.txt"

    # -------------------------------
    # 1) Get BASE-COLOR images
    # -------------------------------
    if basecolor_txt_path.exists():
        print(f"[Info] Found existing base-color list: {basecolor_txt_path}")
        with open(basecolor_txt_path, "r") as f:
            paths = [Path(line.strip()) for line in f if line.strip()]
        # Validate files still exist
        paths = [p for p in paths if p.exists()]
        if args.limit > 0:
            paths = paths[:args.limit]
        print(f"[Info] Loaded {len(paths)} base-color images from saved list.")
    else:
        # No list found → run filtering step
        paths = list_basecolor_images(
            args.textures_dir, exts, args.recursive, args.limit, args.strong_filename_filter
        )
        if not paths:
            raise SystemExit("[!] No base-color images found with given settings.")
        print(f"[Info] Using {len(paths)} base-color images from {args.textures_dir}")
        save_basecolor_list(paths, args.basecolor_list_dir)

    # -------------------------------
    # 2) Color-based clustering
    # -------------------------------
    color_feats = build_color_feature_matrix(paths)
    k_color = choose_k(len(paths), args.k_color, fallback=12)
    print(f"[Color clustering] K={k_color}")
    kmeans_color = KMeans(n_clusters=k_color, n_init=20, random_state=42)
    labels_color = kmeans_color.fit_predict(color_feats)
    copy_into_clusters(paths, labels_color, args.out_dir, prefix="color_clusters")
    save_mapping_csv(paths, labels_color, args.out_dir, csv_name="color_cluster_mapping.csv", prefix="color_clusters")

    # -------------------------------
    # 3) Pattern-based clustering
    # -------------------------------
    processor, model = load_clip(device)
    clip_feats = embed_images_clip(paths, processor, model, device, batch_size=32)
    k_pattern = choose_k(len(paths), args.k_pattern, fallback=24)
    print(f"[Pattern clustering] K={k_pattern}")
    kmeans_pattern = KMeans(n_clusters=k_pattern, n_init=20, random_state=42)
    labels_pattern = kmeans_pattern.fit_predict(clip_feats)
    copy_into_clusters(paths, labels_pattern, args.out_dir, prefix="pattern_clusters")
    save_mapping_csv(paths, labels_pattern, args.out_dir, csv_name="pattern_cluster_mapping.csv", prefix="pattern_clusters")

    print(f"[Done] Copied {len(paths)} files into:")
    print(f"       {args.out_dir/'color_clusters'} and {args.out_dir/'pattern_clusters'}")
    print(f"[Done] Base-color path list location: {args.basecolor_list_dir}")
    print("       You can now rename each 'cluster_XX' folder to your desired keywords (color-wise and pattern-wise).")



if __name__ == "__main__":
    main()
