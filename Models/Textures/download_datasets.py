import os
import zipfile
import tarfile
from tqdm import tqdm

# ─── Archive Paths (manually downloaded) ─────────────────────────────
MINC_ARCHIVE = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/minc.zip"
FMD_ARCHIVE = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/fmd.zip"

# Target extraction directories
MINC_EXTRACT_DIR = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/minc"
FMD_EXTRACT_DIR = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/fmd"

# ─── Extraction Logic ────────────────────────────────
def extract_archive(archive_path, extract_to):
    print(f"[+] Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

# ─── Main ─────────────────────────────────────────────
def main():
    extract_archive(MINC_ARCHIVE, MINC_EXTRACT_DIR)
    extract_archive(FMD_ARCHIVE, FMD_EXTRACT_DIR)
    print("[✓] All datasets extracted.")

if __name__ == "__main__":
    main()
