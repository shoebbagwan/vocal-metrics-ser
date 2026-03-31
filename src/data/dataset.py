import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SERDataset(Dataset):
    """
    Speech Emotion Recognition Dataset.
    Loads pre-extracted features (MFCCs or Mel Spectrograms) and their labels.
    """
    def __init__(self, processed_data_dir: str, feature_type: str = "mfcc", label_mapping: dict = None):
        """
        Args:
            processed_data_dir (str): Directory containing the .npy files.
            feature_type (str): "mfcc" or "mel".
            label_mapping (dict): Dictionary mapping string labels to integers.
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.feature_type = feature_type
        
        # Load features
        if feature_type == "mfcc":
            self.X = np.load(self.processed_data_dir / "X_mfcc.npy")
        elif feature_type == "mel":
            self.X = np.load(self.processed_data_dir / "X_mel.npy")
        else:
            raise ValueError("feature_type must be 'mfcc' or 'mel'")
            
        # Load raw labels (strings)
        raw_labels = np.load(self.processed_data_dir / "y_labels.npy")
        
        # Determine internal active mapping
        if label_mapping is None:
            unique_labels = sorted(list(set(raw_labels)))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
            
        # Convert string labels to integer indices
        self.y = np.array([self.label_mapping[label] for label in raw_labels])
        
        # Expand dims if using a 2D CNN (adding channel dimension)
        if feature_type == "mel":
            self.X = np.expand_dims(self.X, axis=1) # Shape: (N, 1, H, W)
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Convert to float32 standard for PyTorch
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return features, label
        
    def get_num_classes(self):
        return len(self.label_mapping)
        
    def get_input_shape(self):
        return self.X[0].shape
