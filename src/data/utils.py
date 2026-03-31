import os
import glob
from pathlib import Path

# RAVDESS Emotion mapping
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def get_ravdess_files_and_labels(raw_data_dir: str):
    """
    Scans the RAVDESS directory and returns a list of file paths and their corresponding emotion labels.
    """
    path = Path(raw_data_dir)
    file_paths = []
    labels = []
    
    # Traverse all actor folders
    for wav_file in path.rglob("*.wav"):
        filename = wav_file.name
        parts = filename.split(".")[0].split("-")
        
        # Ensure it's a valid RAVDESS file format
        if len(parts) == 7:
            emotion_code = parts[2]
            emotion = EMOTION_MAP.get(emotion_code)
            
            if emotion:
                file_paths.append(str(wav_file))
                labels.append(emotion)
                
    return file_paths, labels

def get_label_mapping():
    """Returns a dictionary mapping emotion string to integer label."""
    emotions = list(EMOTION_MAP.values())
    return {emotion: idx for idx, emotion in enumerate(list(set(emotions)))}
