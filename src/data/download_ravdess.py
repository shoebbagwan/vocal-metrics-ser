import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# URL for RAVDESS Audio-only dataset
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
TARGET_DIR = Path("../../data/raw/RAVDESS")
ZIP_FILE_PATH = TARGET_DIR / "RAVDESS.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(url: str, target_dir: Path, zip_path: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        print(f"Downloading RAVDESS dataset from Zenodo...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        print("Download complete.")
    else:
        print(f"Zip file already exists at {zip_path}. Skipping download.")
        
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    download_and_extract(RAVDESS_URL, TARGET_DIR, ZIP_FILE_PATH)
