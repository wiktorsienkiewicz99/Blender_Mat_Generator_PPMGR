import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFilter
import os

# ─────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
prompt = "seamless texture of oak wooden planks, tiled, PBR"
output_dir = f"./generated_textures/{prompt.replace(' ', '_')}"
num_variants = 5
seed = 42

# ─────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────
print("[+] Loading models...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
pipe = pipe.to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")  # stay on CPU to save memory
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ─────────────────────────────────────
# GENERATE CANDIDATE IMAGES
# ─────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)
image_paths = []

print("[+] Generating texture variants...")
for i in range(num_variants):
    torch.manual_seed(seed + i)
    image = pipe(prompt, guidance_scale=7.5).images[0]
    path = os.path.join(output_dir, f"variant_{i+1}.png")
    image.save(path)
    image_paths.append(path)

# ─────────────────────────────────────
# CLIP RANKING
# ─────────────────────────────────────
print("[+] Ranking with CLIP...")
scores = []

for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True).to("cpu")
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = outputs.logits_per_image.softmax(dim=1)[0][0].item()
        scores.append((path, similarity))

# Sort by similarity score
best_path, best_score = sorted(scores, key=lambda x: x[1], reverse=True)[0]
best_image = Image.open(best_path)
print(f"[✓] Best match: {os.path.basename(best_path)} (score={best_score:.4f})")

# Save as base_color + other maps
base_name = os.path.join(output_dir, "base_color.png")
best_image.save(base_name)

roughness = best_image.convert("L")
metallic = best_image.convert("L")
normal = best_image.filter(ImageFilter.CONTOUR)

roughness.save(os.path.join(output_dir, "roughness.png"))
metallic.save(os.path.join(output_dir, "metallic.png"))
normal.save(os.path.join(output_dir, "normal.png"))

print(f"[✓] Saved all maps in: {output_dir}")