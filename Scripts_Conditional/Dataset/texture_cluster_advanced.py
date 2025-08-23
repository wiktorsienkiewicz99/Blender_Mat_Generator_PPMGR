#!/usr/bin/env python3
"""
BASE‑COLOR TEXTURES → DUAL CLUSTERING (COLOR & PATTERN) WITH ADVANCED OPTIONS

Features:
  - Base‑color filtering (or reuse saved basecolor_paths.txt)
  - COLOR pass:
      * Default: LAB histogram + palette descriptor (+kmeans)
      * Option: HSV histogram only
      * Optional K sweep (silhouette) to auto-select K
  - PATTERN pass:
      * CLIP embeddings (ViT-B/32 by default, switchable)
      * Multi‑crop embedding (average crops)
      * Spherical k‑means (L2-normalize for cosine)
      * Or UMAP + HDBSCAN (auto K, marks outliers)
      * Optional K sweep (silhouette) for k-means path
  - Caching: saves computed features to --cache-dir
  - Output: copies images to color_clusters/ and pattern_clusters/
            + CSV mappings; basecolor path list to TXT/CSV

Example:
python texture_cluster_advanced.py \
  --textures-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/textures"" \
  --out-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset" \
  --limit 0 \
  --k-color 40 --k-pattern 60 \
  --pattern-multicrop 4 --pattern-spherical \
  --cache-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Cache" \
  --basecolor-list-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary"

  python texture_cluster_advanced.py \
  --textures-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/textures" \
  --out-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset_AutoK" \
  --limit 0 \
  --color-feature lab --color-palette-k 4 --k-color-sweep "30,40,50" \
  --pattern-multicrop 4 --pattern-spherical --k-pattern-sweep "40,50,60,70" \
  --cache-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Cache"


python texture_cluster_advanced.py \
  --textures-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/textures" \
  --out-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset_UMAP" \
  --limit 0 \
  --color-feature lab --k-color 40 \
  --pattern-umap-hdbscan --umap-n-neighbors 15 --umap-min-dist 0.05 \
  --hdbscan-min-cluster-size 30 --hdbscan-min-samples 5 \
  --cache-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Cache"

"""

import os
import csv
import math
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional deps
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

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
    Filename heuristics + content tests for albedo:
    - reject typical map names (normal/height/roughness/...)
    - accept if filename has positive token and not negative
    - else reject if bluish-normal or very grayscale
    - else require some saturation + colorfulness
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
# Discovery / Basecolor list
# --------------------------
def list_basecolor_images(root: Path, exts: Tuple[str, ...], recursive: bool, limit: int, strong_filename_filter: bool) -> List[Path]:
    """Collect base-color image paths. If limit<=0, return ALL after filtering."""
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

def save_basecolor_list(paths: List[Path], out_dir: Path) -> None:
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

def load_basecolor_list(list_dir: Path, limit: int) -> Optional[List[Path]]:
    txt_path = list_dir / "basecolor_paths.txt"
    if not txt_path.exists():
        return None
    print(f"[Info] Found existing base-color list: {txt_path}")
    with open(txt_path, "r") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]
    paths = [p for p in paths if p.exists()]
    return paths if limit <= 0 else paths[:limit]

# --------------------------
# Color features
# --------------------------
def extract_color_features_hsv(img: Image.Image, blur: bool = True) -> np.ndarray:
    """HSV histograms + mean HSV. (37 dims)"""
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8)
    H, S, V = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    h_hist, _ = np.histogram(H, bins=18, range=(0, 255), density=False)
    s_hist, _ = np.histogram(S, bins=8, range=(0, 255), density=False)
    v_hist, _ = np.histogram(V, bins=8, range=(0, 255), density=False)
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    feat = feat / (np.sum(feat) + 1e-6)
    mean_h = float(np.mean(H) / 255.0)
    mean_s = float(np.mean(S) / 255.0)
    mean_v = float(np.mean(V) / 255.0)
    return np.concatenate([feat, np.array([mean_h, mean_s, mean_v], dtype=np.float32)])

def extract_color_features_lab_palette(img: Image.Image, k_palette: int = 4) -> np.ndarray:
    """
    LAB histograms + dominant LAB palette (centers + weights).
    Requires OpenCV. Fallbacks to HSV if cv2 missing.
    """
    if not HAS_CV2:
        return extract_color_features_hsv(img, blur=True)
    img_np = np.array(img.convert("RGB"))
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    L, A, B = img_lab[..., 0], img_lab[..., 1], img_lab[..., 2]
    hL, _ = np.histogram(L, 8, (0, 255)); hA, _ = np.histogram(A, 12, (0, 255)); hB, _ = np.histogram(B, 12, (0, 255))
    hist = np.concatenate([hL, hA, hB]).astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    Z = img_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _compactness, labels, centers = cv2.kmeans(
        Z, k_palette, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    weights = np.bincount(labels.flatten(), minlength=k_palette).astype(np.float32)
    weights /= (weights.sum() + 1e-6)
    order = np.argsort(weights)[::-1]
    palette = centers[order].reshape(-1) / 255.0
    weights = weights[order]
    return np.concatenate([hist, palette, weights]).astype(np.float32)

def build_color_feature_matrix(paths: List[Path], kind: str = "lab", k_palette: int = 4) -> np.ndarray:
    feats = []
    for p in tqdm(paths, desc=f"Extracting color features ({kind})"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if kind == "lab":
            feats.append(extract_color_features_lab_palette(img, k_palette=k_palette))
        else:
            feats.append(extract_color_features_hsv(img, blur=True))
    return np.stack(feats, axis=0)

# --------------------------
# Pattern features (CLIP)
# --------------------------
def load_clip(device: torch.device, model_name: str = "openai/clip-vit-base-patch32", prefer_fast: bool = True):
    """
    Load CLIP with a safe fallback:
    - Try fast image processor first (prefer_fast=True).
    - If the current transformers version lacks `_valid_processor_keys`
      on the fast processor, fall back to use_fast=False.
    """
    from transformers import CLIPProcessor, CLIPModel

    def _load(use_fast: bool):
        try:
            processor = CLIPProcessor.from_pretrained(model_name, use_fast=use_fast)
        except TypeError:
            # Older transformers may not support use_fast kwarg
            processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        return processor, model

    # First try fast
    processor, model = _load(use_fast=prefer_fast)

    # Probe for the attribute your stack is missing on the fast processor
    try:
        _ = getattr(processor.image_processor, "_valid_processor_keys")
    except AttributeError:
        # Fall back to the safe slow processor
        if prefer_fast:
            print("[Warn] Fast CLIP image processor missing `_valid_processor_keys`. Falling back to use_fast=False.")
            processor, model = _load(use_fast=False)

    return processor, model


def embed_image_multicrop(path: Path, processor, model, device: torch.device, n_crops: int = 4, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(123)
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    W, H = img.size
    crops = []
    for _ in range(n_crops):
        s = rng.uniform(0.6, 1.0)
        w, h = int(W * s), int(H * s)
        x = rng.randint(0, max(1, W - w))
        y = rng.randint(0, max(1, H - h))
        crops.append(img.crop((x, y, x + w, y + h)))
    inputs = processor(images=crops, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    return emb.mean(axis=0)

def embed_images_clip(paths: List[Path], processor, model, device: torch.device, batch_size: int = 32, multicrop: int = 1) -> np.ndarray:
    if multicrop and multicrop > 1:
        # per-image loop with multi-crop
        feats = []
        rng = np.random.RandomState(2024)
        for p in tqdm(paths, desc=f"Embedding images (CLIP, multicrop={multicrop})"):
            feats.append(embed_image_multicrop(p, processor, model, device, n_crops=multicrop, rng=rng))
        return np.stack(feats, axis=0)

    # fast batch path
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding images (CLIP)"):
        batch = []
        for p in paths[i:i + batch_size]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            batch.append(img)
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy()
        feats.append(emb)
    return np.concatenate(feats, axis=0)

# --------------------------
# Utils: caching, K choice, copying, CSV
# --------------------------
def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def load_or_compute(cache_dir: Optional[Path], key: dict, fn_compute):
    if cache_dir is None:
        return fn_compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"{key['name']}_{_hash_dict(key)}.npy"
    path = cache_dir / name
    if path.exists():
        try:
            arr = np.load(path)
            print(f"[Cache] Loaded {name}")
            return arr
        except Exception:
            pass
    arr = fn_compute()
    try:
        np.save(path, arr)
        print(f"[Cache] Saved {name}")
    except Exception:
        pass
    return arr

def choose_k(n: int, k_arg: Optional[int], fallback: int) -> int:
    if k_arg is not None:
        return max(2, int(k_arg))
    k = int(round(math.sqrt(n)))
    k = max(8, min(30, k))
    return k if k > 0 else fallback

def sweep_k_and_pick_best(X: np.ndarray, ks: List[int], spherical: bool = False) -> int:
    if spherical:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    best_k, best_s = None, -1.0
    print(f"[K-sweep] ks={ks}")
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels, metric="euclidean")
        print(f"  k={k:>3}  silhouette={s:.4f}")
        if s > best_s:
            best_s, best_k = s, k
    print(f"[K-sweep] chosen k={best_k} (silhouette={best_s:.4f})")
    return best_k if best_k else ks[0]

def copy_into_clusters(paths: List[Path], labels: np.ndarray, out_dir: Path, prefix: str, treat_minus1_as_misc: bool = False) -> None:
    base = out_dir / prefix
    base.mkdir(parents=True, exist_ok=True)
    for p, cid in tqdm(list(zip(paths, labels)), desc=f"Copying ({prefix})"):
        folder_name = ("cluster_misc" if (treat_minus1_as_misc and cid == -1) else f"cluster_{int(cid):02d}")
        dest_dir = base / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / p.name
        if not dst.exists():
            shutil.copy2(str(p), str(dst))

def save_mapping_csv(paths: List[Path], labels: np.ndarray, out_dir: Path, csv_name: str, prefix: str) -> None:
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "abs_path", "cluster_id", "cluster_folder"])
        for p, cid in zip(paths, labels):
            folder_name = ("cluster_misc" if cid == -1 else f"cluster_{int(cid):02d}")
            w.writerow([p.name, str(p.resolve()), int(cid), f"{prefix}/{folder_name}"])
    print(f"[Report] Mapping saved to {csv_path}")

def summarize_output(out_dir: Path):
    def count_files(p: Path):
        return sum(1 for _ in p.rglob("*") if _.is_file())
    color_dir   = out_dir / "color_clusters"
    pattern_dir = out_dir / "pattern_clusters"
    print("[Summary]")
    if color_dir.exists():
        print(f"  color_clusters:   {len(list(color_dir.glob('cluster_*')))} folders, {count_files(color_dir)} files")
    else:
        print("  color_clusters:   (missing)")
    if pattern_dir.exists():
        print(f"  pattern_clusters: {len(list(pattern_dir.glob('cluster_*')))} folders, {count_files(pattern_dir)} files")
    else:
        print("  pattern_clusters: (missing)")

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Cluster BASE‑COLOR textures by COLOR and by PATTERN with advanced options.")
    ap.add_argument("--textures-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max images after filtering (0 = ALL)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png")
    ap.add_argument("--strong-filename-filter", action="store_true")

    # Base-color list
    ap.add_argument("--basecolor-list-dir", type=Path,
                    default=Path("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary"))

    # Cache
    ap.add_argument("--cache-dir", type=Path, default=None, help="Cache dir for features (optional)")

    # Color options
    ap.add_argument("--color-feature", choices=["lab", "hsv"], default="lab")
    ap.add_argument("--color-palette-k", type=int, default=4)
    ap.add_argument("--k-color", type=int, default=None)
    ap.add_argument("--k-color-sweep", type=str, default="", help="Comma-separated K list, e.g. '30,40,50'")

    # Pattern options
    ap.add_argument("--pattern-encoder", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--pattern-multicrop", type=int, default=4, help="#crops per image; 1 disables multi-crop")
    ap.add_argument("--pattern-spherical", action="store_true", help="L2-normalize before k-means (cosine)")
    ap.add_argument("--k-pattern", type=int, default=None)
    ap.add_argument("--k-pattern-sweep", type=str, default="", help="Comma-separated K list")
    ap.add_argument("--pattern-umap-hdbscan", action="store_true",
                    help="Use UMAP+HDBSCAN instead of k-means (auto-K, outliers=-1).")
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.05)
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    ap.add_argument("--hdbscan-min-samples", type=int, default=5)

    args = ap.parse_args()

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    if not args.textures_dir.exists():
        raise SystemExit(f"[!] textures-dir does not exist: {args.textures_dir}")

    # ----- Gather base-color paths (from list if available) -----
    paths = load_basecolor_list(args.basecolor_list_dir, args.limit)
    if paths is None:
        paths = list_basecolor_images(args.textures_dir, exts, args.recursive, args.limit, args.strong_filename_filter)
        if not paths:
            raise SystemExit("[!] No base‑color images found.")
        print(f"[Info] Using {len(paths)} base‑color images from {args.textures_dir}")
        save_basecolor_list(paths, args.basecolor_list_dir)
    else:
        print(f"[Info] Loaded {len(paths)} base‑color images from saved list.")

    # =========================
    # COLOR PASS
    # =========================
    color_key = {
        "name": "color_feats",
        "kind": args.color_feature,
        "palette_k": args.color_palette_k,
        "n": len(paths)
    }
    def _compute_color():
        return build_color_feature_matrix(paths, kind=args.color_feature, k_palette=args.color_palette_k)
    color_feats = load_or_compute(args.cache_dir, color_key, _compute_color)

    # Decide K for color
    if args.k_color_sweep:
        ks = [int(k.strip()) for k in args.k_color_sweep.split(",") if k.strip()]
        k_color = sweep_k_and_pick_best(color_feats, ks, spherical=False)
    else:
        k_color = choose_k(len(paths), args.k_color, fallback=12)
    print(f"[Color clustering] K={k_color}")
    kmeans_color = KMeans(n_clusters=k_color, n_init=20, random_state=42)
    labels_color = kmeans_color.fit_predict(color_feats)
    copy_into_clusters(paths, labels_color, args.out_dir, prefix="color_clusters")
    save_mapping_csv(paths, labels_color, args.out_dir, "color_cluster_mapping.csv", "color_clusters")

    # =========================
    # PATTERN PASS
    # =========================
    processor, model = load_clip(device, model_name=args.pattern_encoder, prefer_fast=True)
    clip_key = {
        "name": "clip_feats",
        "model": args.pattern_encoder,
        "multicrop": args.pattern_multicrop,
        "n": len(paths)
    }
    def _compute_clip():
        return embed_images_clip(paths, processor, model, device, batch_size=32, multicrop=args.pattern_multicrop)
    clip_feats = load_or_compute(args.cache_dir, clip_key, _compute_clip)

    if args.pattern_umap_hdbscan:
        if not (HAS_UMAP and HAS_HDBSCAN):
            raise SystemExit("[!] UMAP/HDBSCAN requested but packages not available. pip install umap-learn hdbscan")
        # L2-normalize first (cosine geometry)
        X = clip_feats / (np.linalg.norm(clip_feats, axis=1, keepdims=True) + 1e-9)
        reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
                            metric='cosine', random_state=42)
        X_umap = reducer.fit_transform(X)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.hdbscan_min_cluster_size,
                                    min_samples=args.hdbscan_min_samples,
                                    metric='euclidean')
        labels_pattern = clusterer.fit_predict(X_umap)  # -1 = outliers
        copy_into_clusters(paths, labels_pattern, args.out_dir, prefix="pattern_clusters", treat_minus1_as_misc=True)
        save_mapping_csv(paths, labels_pattern, args.out_dir, "pattern_cluster_mapping.csv", "pattern_clusters")
    else:
        # KMeans path (with spherical option + optional K sweep)
        X = clip_feats
        spherical = args.pattern_spherical
        if spherical:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        if args.k_pattern_sweep:
            ks = [int(k.strip()) for k in args.k_pattern_sweep.split(",") if k.strip()]
            k_pattern = sweep_k_and_pick_best(X, ks, spherical=False)  # euclidean on normalized vectors
        else:
            k_pattern = choose_k(len(paths), args.k_pattern, fallback=24)

        print(f"[Pattern clustering] K={k_pattern}  (spherical={spherical})")
        kmeans_pattern = KMeans(n_clusters=k_pattern, n_init=20, random_state=42)
        labels_pattern = kmeans_pattern.fit_predict(X)
        copy_into_clusters(paths, labels_pattern, args.out_dir, prefix="pattern_clusters")
        save_mapping_csv(paths, labels_pattern, args.out_dir, "pattern_cluster_mapping.csv", "pattern_clusters")

    # -------------------------
    print(f"[Done] Copied {len(paths)} files into:")
    print(f"       {args.out_dir/'color_clusters'} and {args.out_dir/'pattern_clusters'}")
    print(f"[Done] Base‑color path list location: {args.basecolor_list_dir}")
    summarize_output(args.out_dir)

if __name__ == "__main__":
    main()
