#!/usr/bin/env python3
"""
Text → PBR map generator (local)

Generates:
  - basecolor.png (albedo)
  - height.png    (16-bit height from monocular depth)
  - normal.png    (OpenGL-style normal map from height)
  - roughness.png (heuristic from albedo)
  - metallic.png  (heuristic from prompt; mostly 0 unless 'metal' in prompt)

Requirements (install once):
  pip install torch torchvision --upgrade
  pip install diffusers transformers accelerate safetensors
  pip install pillow numpy

Optional:
  - If running on Apple Silicon: torch with MPS.
  - For CUDA: install CUDA-enabled torch wheels.

Model notes:
  - Uses 'dream-textures/texture-diffusion' (SD 1.5 finetune) via diffusers.
  - Depth estimation replaced by simple luminance-based pseudo-height to keep it 100% offline-ready.
    (If you prefer real depth, plug any local depth model and swap `estimate_height_from_albedo`.)

Run:
  python new_texture_generator_100825.py \
    --prompt "wooden floor, oak, clean, medium detail" \
    --out ./out/wooden_floor \
    --size 1024 \
    --steps 30 \
    --guidance 7.5 \
    --seed 42 \
    --seamless 1 \
    --normal_strength 4.0
"""
#!/usr/bin/env python3
# new_texture_generator_single_tile.py
# Prompt → single PBR tile (base/height/normal/roughness/metallic)
# Seamless via gradient equalization (periodic) — no inner blurry square.

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import torch
from diffusers import DiffusionPipeline

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def to_uint8(img_float):
    img = np.clip(img_float, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype("uint8")

def to_uint16(img_float):
    img = np.clip(img_float, 0.0, 1.0)
    return (img * 65535.0 + 0.5).astype("uint16")

def pil_to_np(img: Image.Image):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

def np_to_pil(arr_float):
    return Image.fromarray(to_uint8(arr_float))

def rgb_to_luma(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def normalize(v, eps=1e-8):
    n = np.maximum(np.sqrt((v**2).sum(axis=-1, keepdims=True)), eps)
    return v / n

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Seamless via periodic gradient equalization (no blur, no collage)
# ──────────────────────────────────────────────────────────────────────────────

def make_seamless_periodic(img: Image.Image, edge_ratio=0.12) -> Image.Image:
    """
    Forces opposite edges to match by adding smooth, edge-only ramps.
    edge_ratio: fraction of width/height affected on each side (0.05–0.2 good).
    """
    a = pil_to_np(img)  # H×W×3 in [0,1]
    H, W, C = a.shape

    # 1D edge window: 1 at edges, 0 at center (piecewise linear with smooth center)
    def edge_window(n, ratio):
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dist = np.minimum(x, 1.0 - x)  # distance to nearest edge
        w = np.clip(1.0 - dist / max(ratio, 1e-6), 0.0, 1.0)
        # Slight smooth to avoid kinks
        return (3*w**2 - 2*w**3)  # smoothstep

    wx = edge_window(W, edge_ratio)              # (W,)
    wy = edge_window(H, edge_ratio)              # (H,)
    hx = (2.0 * np.linspace(0, 1, W, dtype=np.float32) - 1.0) * wx  # -1..+1, fade at center
    hy = (2.0 * np.linspace(0, 1, H, dtype=np.float32) - 1.0) * wy

    # Left/Right correction (per row/channel)
    L = a[:, 0, :]          # H×3
    R = a[:, -1, :]         # H×3
    delta_lr = (L - R) * 0.5  # H×3; want left-=delta, right+=delta
    corr_x = (-delta_lr)[:, None, :] * hx[None, :, None]  # H×W×3

    # Top/Bottom correction (per col/channel)
    T = a[0, :, :]          # W×3
    B = a[-1, :, :]         # W×3
    delta_tb = (T - B) * 0.5  # W×3; top-=delta, bottom+=delta
    corr_y = (-delta_tb)[None, :, :] * hy[:, None, None]  # H×W×3

    out = a + corr_x + corr_y
    out = np.clip(out, 0.0, 1.0)
    return np_to_pil(out)

# ──────────────────────────────────────────────────────────────────────────────
# PBR heuristics
# ──────────────────────────────────────────────────────────────────────────────

def estimate_height_from_albedo(albedo_np, blur_px=3, contrast=1.25):
    gray = rgb_to_luma(albedo_np)
    gray_img = Image.fromarray(to_uint8(gray))
    if blur_px > 0:
        gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=blur_px))
    gray_b = np.asarray(gray_img, dtype=np.uint8) / 255.0
    height = np.clip(0.5 + contrast * (gray - gray_b), 0.0, 1.0)
    hmin, hmax = float(height.min()), float(height.max())
    if hmax > hmin:
        height = (height - hmin) / (hmax - hmin)
    else:
        height = np.zeros_like(height) + 0.5
    return height

def height_to_normal(height, strength=3.0, texel_size=1.0):
    h = height.astype(np.float32)
    dx = np.gradient(h, axis=1)
    dy = np.gradient(h, axis=0)
    nx = -dx * strength / texel_size
    ny = -dy * strength / texel_size
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], axis=-1)
    n = normalize(n)
    n01 = 0.5 * (n + 1.0)
    return n01

def roughness_from_albedo(albedo_np, bias=0.55, detail=0.45):
    rgb = np.clip(albedo_np, 0, 1)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    base_r = np.clip(1.0 - 0.5 * v - 0.5 * s, 0.0, 1.0)

    luma = rgb_to_luma(rgb)
    L0 = Image.fromarray(to_uint8(luma)).filter(ImageFilter.GaussianBlur(radius=2))
    L1 = Image.fromarray(to_uint8(luma)).filter(ImageFilter.GaussianBlur(radius=4))
    L0 = np.asarray(L0) / 255.0
    L1 = np.asarray(L1) / 255.0
    local_var = np.clip((L0 - L1) ** 2 * 4.0, 0.0, 1.0)

    r = np.clip(bias * base_r + detail * local_var, 0.0, 1.0)
    return r

def metallic_from_prompt(prompt: str):
    metals = ["metal", "steel", "iron", "aluminum", "aluminium", "copper", "brass", "chrome", "gold", "silver"]
    is_metal = any(m in prompt.lower() for m in metals)
    return 1.0 if is_metal else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Diffusion basecolor generation (fp32 on MPS, single image)
# ──────────────────────────────────────────────────────────────────────────────

def pick_dtype_for_device(device: str):
    if device == "cuda":
        return torch.float16
    return torch.float32  # MPS/CPU → fp32 to avoid NaNs

def generate_albedo(prompt, size, steps, guidance, seed, device, seamless=False, edge_ratio=0.12, negative=None):
    model_id = "dream-textures/texture-diffusion"
    dtype = pick_dtype_for_device(device)

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    gen = torch.Generator(device=device)
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    gen = gen.manual_seed(int(seed))
    if negative is None:
        negative = "low detail, blurry, watermark, text"

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(size),
        width=int(size),
        generator=gen,
        num_images_per_prompt=1,
    )
    image = out.images[0].convert("RGB")

    if seamless:
        image = make_seamless_periodic(image, edge_ratio=edge_ratio)

    return image, seed

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seamless", type=int, default=1)
    ap.add_argument("--edge_ratio", type=float, default=0.12, help="edge band per side (0.05–0.20)")
    ap.add_argument("--normal_strength", type=float, default=3.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    print(f"[+] Device: {device}")
    print("[+] Generating basecolor…")
    base_pil, used_seed = generate_albedo(
        prompt=args.prompt,
        size=args.size,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        device=device,
        seamless=bool(args.seamless),
        edge_ratio=args.edge_ratio,
    )

    base_np = pil_to_np(base_pil)

    print("[+] Deriving height…")
    height = estimate_height_from_albedo(base_np, blur_px=3, contrast=1.25)

    print("[+] Deriving normal…")
    normal = height_to_normal(height, strength=args.normal_strength)

    print("[+] Deriving roughness…")
    roughness = roughness_from_albedo(base_np, bias=0.55, detail=0.45)

    print("[+] Deriving metallic (prompt keywords)…")
    mval = metallic_from_prompt(args.prompt)
    metallic = np.zeros_like(height) + mval

    # Save
    base_pil.save(out_dir / "basecolor.png")
    Image.fromarray(to_uint16(height)).save(out_dir / "height.png")
    Image.fromarray(to_uint8(normal)).save(out_dir / "normal.png")
    Image.fromarray(to_uint8(roughness)).save(out_dir / "roughness.png")
    Image.fromarray(to_uint8(metallic)).save(out_dir / "metallic.png")

    meta = {
        "prompt": args.prompt,
        "seed": used_seed,
        "size": args.size,
        "steps": args.steps,
        "guidance": args.guidance,
        "seamless": bool(args.seamless),
        "edge_ratio": args.edge_ratio,
        "normal_strength": args.normal_strength,
        "maps": ["basecolor.png", "height.png", "normal.png", "roughness.png", "metallic.png"],
        "notes": "Seamless by edge-only gradient equalization (periodic)."
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Done → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
