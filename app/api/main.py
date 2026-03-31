from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import librosa
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.baseline_cnn import BaselineCNN1D
from src.data.utils import EMOTION_MAP, get_label_mapping

app = FastAPI(title="Speech Emotion Recognition API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for inference
MODEL_PATH = Path("../../models/best_model.pth")
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

# Try to load the model (it might not exist yet if training hasn't run)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
NUM_CLASSES = 8

try:
    if MODEL_PATH.exists():
        model = BaselineCNN1D(num_classes=NUM_CLASSES, input_channels=40)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction will be disabled.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "SER API is running. Model loaded: " + str(model is not None)}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Load audio data from the uploaded file
        # Librosa can process file-like objects, but since UploadFile might not seek properly 
        # for soundfile, it's safer to save temporarily or use an in-memory buffer, or soundfile directly.
        # For simplicity, we write to a temp file
        temp_file = Path("temp_audio.wav")
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        # Extract features (same logic as in our extract.py script)
        y, sr = librosa.load(temp_file, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < SAMPLES:
            padding = SAMPLES - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            y = y[:SAMPLES]
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()
            
        # Prepare for model
        # The model expects shape: (Batch, Channels, TimeSteps)
        features = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            
        # Map integer prediction back to emotion string
        # Reversing the mapping
        # `get_label_mapping()` returns e.g. {'neutral': 0, 'calm': 1}
        idx_to_emotion = {v: k for k, v in get_label_mapping().items()}
        predicted_emotion = idx_to_emotion.get(predicted_idx, "unknown")
        
        return {
            "prediction": predicted_emotion,
            "confidence": f"{probabilities[predicted_idx].item() * 100:.2f}%",
            "probabilities": {idx_to_emotion[i]: float(probabilities[i]) for i in range(len(probabilities))}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
