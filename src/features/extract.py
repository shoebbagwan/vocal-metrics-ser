import os
import sys
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Add src to the path so we can import our utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.utils import get_ravdess_files_and_labels

RAW_DATA_DIR = Path("../../data/raw/RAVDESS")
PROCESSED_DATA_DIR = Path("../../data/processed")

# DSP Parameters
SAMPLE_RATE = 22050 # Standard librosa SR
DURATION = 3 # Truncate/Pad audio to 3 seconds for consistency
SAMPLES = SAMPLE_RATE * DURATION

def extract_features(file_path: str):
    """
    Extracts MFCCs and Mel Spectrograms from an audio file.
    """
    try:
        # Load audio with standardized sample rate and duration
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to ensure all inputs to the network have the exact same shape
        if len(y) < SAMPLES:
            padding = SAMPLES - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            y = y[:SAMPLES]
            
        # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
        # Good for simpler 1D CNNs or MLPs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 2. Mel Spectrogram
        # Good for 2D CNNs (treating audio like an image)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return {
            "mfcc": mfccs,
            "mel": mel_spectrogram_db
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_and_save_dataset():
    """Main pipeline for processing all RAVDESS data."""
    if not RAW_DATA_DIR.exists():
        print(f"Raw data directory {RAW_DATA_DIR} not found. Please download RAVDESS first.")
        return
        
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    files, labels = get_ravdess_files_and_labels(str(RAW_DATA_DIR))
    print(f"Found {len(files)} audio files.")
    
    all_mfccs = []
    all_mels = []
    all_labels = []
    
    print("Extracting features (This may take a few minutes)...")
    for file, label in tqdm(zip(files, labels), total=len(files)):
        features = extract_features(file)
        if features is not None:
            all_mfccs.append(features["mfcc"])
            all_mels.append(features["mel"])
            all_labels.append(label)
            
    # Convert lists to numpy arrays
    X_mfcc = np.array(all_mfccs)
    X_mel = np.array(all_mels)
    y = np.array(all_labels)
    
    print(f"Shape of X_mfcc: {X_mfcc.shape}")
    print(f"Shape of X_mel: {X_mel.shape}")
    print(f"Shape of y: {y.shape}")
    
    # Save feature matrices to processed data directory
    np.save(PROCESSED_DATA_DIR / "X_mfcc.npy", X_mfcc)
    np.save(PROCESSED_DATA_DIR / "X_mel.npy", X_mel)
    np.save(PROCESSED_DATA_DIR / "y_labels.npy", y)
    
    print("Feature extraction complete! Data saved to data/processed/")

if __name__ == "__main__":
    process_and_save_dataset()
