"""Automatic weight downloader for WeatherVision backend.

Downloads trained model weights directly from Hugging Face Hub (H-Kansal/weathervision-weights)
before launching the server.
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "app" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face Repository details
HF_BASE_URL = "https://huggingface.co/H-Kansal/weathervision-weights/resolve/main"

WEIGHTS = [
    {
        "filename": "classifier.pth",
        "url": os.getenv("CLASSIFIER_WEIGHT_URL", f"{HF_BASE_URL}/classifier.pth"),
        "expected_mb": 16.4,
    },
    {
        "filename": "rain_enhancer.pth",
        "url": os.getenv("RAIN_WEIGHT_URL", f"{HF_BASE_URL}/rain_enhancer.pth"),
        "expected_mb": 217.6,
    },
    {
        "filename": "lowlight_enhancer.pth",
        "url": os.getenv("LOWLIGHT_WEIGHT_URL", f"{HF_BASE_URL}/lowlight_enhancer.pth"),
        "expected_mb": 217.6,
    },
    {
        "filename": "fog_enhancer.pth",
        # In H-Kansal/weathervision-weights it is named foggy_enhancer.pth
        "url": os.getenv("FOG_WEIGHT_URL", f"{HF_BASE_URL}/foggy_enhancer.pth"),
        "expected_mb": 217.6,
        "alias": "foggy_enhancer.pth",
    },
]

def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2MB chunks
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(f"\r  -> Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                sys.stdout.flush()

    print(f"\n✓ Successfully saved {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")

def main():
    print(f"=== Checking model weights in {WEIGHTS_DIR} ===")
    for item in WEIGHTS:
        fname = item["filename"]
        target_path = WEIGHTS_DIR / fname
        
        # Check if file or its alias already exists
        already_exists = target_path.exists() and target_path.stat().st_size > 1024 * 1024
        alias_name = item.get("alias")
        if not already_exists and alias_name:
            alias_path = WEIGHTS_DIR / alias_name
            if alias_path.exists() and alias_path.stat().st_size > 1024 * 1024:
                shutil.copyfile(alias_path, target_path)
                already_exists = True

        if already_exists:
            print(f"✓ Found {fname} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        else:
            try:
                download_file(item["url"], target_path)
                # If alias defined, create a copy as well
                if alias_name:
                    shutil.copyfile(target_path, WEIGHTS_DIR / alias_name)
            except Exception as e:
                print(f"\n[ERROR] Failed to download {fname}: {e}")
                print("Server may fail to start if ALLOW_RANDOM_WEIGHTS=0.")

    print("=== Model weights ready! ===\n")

if __name__ == "__main__":
    main()
