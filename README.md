<div align="center">
  <h1>🎙️ Vocal Metrics: Advanced Speech Emotion Recognition</h1>
  <p>An end-to-end Deep Learning Artificial Intelligence platform for real-time acoustic sentiment analysis.</p>

  [![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-%2320232a.svg?logo=react&logoColor=%2361DAFB)](https://react.dev/)
  [![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
</div>

---

## ⚡ Project Overview
**Vocal Metrics** is an advanced Machine Learning web application engineered to automatically detect distinct human emotions directly from raw audio waveforms.

Unlike traditional Natural Language Processing (NLP) models that rely on transcribing *spoken words* into text to analyze sentiment, Vocal Metrics operates entirely on the physics of sound. By extracting complex acoustic features, it analyzes *how* someone is speaking rather than *what* they are saying—making it capable of detecting emotions like sarcasm or panic that text-transcription inherently misses.

**Supported Emotional Classes:**  
`Neutral` | `Calm` | `Happy` | `Sad` | `Angry` | `Fearful` | `Disgust` | `Surprised`

<br>

## 🚀 Key Features
- **Highly Accurate Convolutional Neural Network:** Achieved an exceptional **89% Training Accuracy** and **76.4% Validation Accuracy** across 50 Epochs using the industry-standard RAVDESS dataset.
- **Microsecond Inference Times:** Powered by an asynchronous **FastAPI** Python architecture utilizing Uvicorn for rapid handling of heavy `.wav` payloads.
- **Stunning User Interface:** A completely custom, dark-mode, glassmorphism web application built with **Vite**, **React**, **TailwindCSS (v4)**, and fluid animations via **Framer Motion**.
- **Interactive Audio Playback:** Seamlessly listen to the recorded audio files natively within the web application while the neural network analyzes the acoustic coefficients.

<br>

## 🧠 Deep Dive: How the AI Works

### 1. Digital Signal Processing (DSP) via Librosa
Human voices are incredibly complex 1D waveforms. Before feeding audio into the Neural Network, the backend strips the file of noise and standardizes it to a `22,050 Hz` sample rate. 

We utilize **Librosa** to extract two core matrices:
- **Mel-Frequency Cepstral Coefficients (MFCCs):** A 40-bin representation of the short-term power spectrum of the voice based on a linear cosine transform of a log power spectrum. It perfectly maps to how the human ear perceives frequency.
- **Mel Spectrograms:** A visual representation of the spectrum of acoustic frequencies as they vary over time.

### 2. PyTorch 1D CNN Architecture
Instead of using Recurrent Neural Networks (RNNs) which are computationally heavy, we feed our 2-Dimensional extracted audio features (Batch Size × Feature Channels × Time Steps) directly into a **1D Convolutional Neural Network (CNN)**.

The CNN uses multiple `Conv1d` layers separated by `BatchNorm1d` (to stabilize learning) and `ReLU` activation functions. By sliding 1D filters across the audio's time axis, the AI easily learns the "visual" shapes of acoustic energy—for example, the sharp, chaotic frequency spikes of *Anger* versus the flat, low-energy pitch tracks of *Sadness*.

<br>

## 🌐 API Documentation

The backend is completely decoupled, making it ready to deploy on AWS or Heroku.

**`POST /predict`**  
Expects a `multipart/form-data` payload containing an audio file (`.wav`, `.mp3`, or `.flac`).

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_clip.wav"
```
**JSON Response Response:**
```json
{
  "predicted_emotion": "Angry",
  "confidence": "94.21%",
  "probabilities": {
    "Angry": 0.9421,
    "Happy": 0.0340,
    "Fearful": 0.0125
  }
}
```

<br>

## 📁 Repository Structure
```text
📦 vocal-metrics-ser
 ┣ 📂 app
 ┃ ┣ 📂 api               # FastAPI backend & inference engine (`main.py`)
 ┃ ┗ 📂 ui                # Vite + React + TailwindCSS User Interface
 ┣ 📂 models              # Contained pre-trained PyTorch weights (`best_model.pth`)
 ┣ 📂 src
 ┃ ┣ 📂 data              # PyTorch Dataset Loaders and download scripts
 ┃ ┣ 📂 features          # Librosa MFCC/Spectrogram extraction pipeline
 ┃ ┣ 📂 models            # PyTorch Deep Neural Network architecture configurations
 ┃ ┗ 📂 training          # Core training loops, validation, and epoch checkpointing
 ┣ 📜 .gitignore          # Keeps Github clean of huge raw datasets and environments
 ┣ 📜 README.md           # You are here
 ┗ 📜 requirements.txt    # Python dependencies
```
*(Note: Datasets and logs are intentionally ignored via `.gitignore` to prevent uploading 400MB+ to GitHub).*

<br>

## 🛠️ Installation & Setup

### 1. Backend (FastAPI & PyTorch)
```bash
# Clone the repository
git clone https://github.com/your-username/vocal-metrics-ser.git
cd vocal-metrics-ser

# Set up a new Python Virtual Environment
python -m venv venv
venv\Scripts\activate   # Use `source venv/bin/activate` on Mac/Linux

# Install Machine Learning Libraries
pip install -r requirements.txt

# Start the Backend Server (Usually takes 5 seconds to load the AI into memory)
python -m uvicorn app.api.main:app --host localhost --port 8000
```

### 2. Frontend (React UI)
Open a **new terminal tab**, and execute the following:
```bash
cd vocal-metrics-ser/app/ui

# Install Node dependencies
npm install

# Start the interactive UI
npm run dev
```

The application is now accessible at **`http://localhost:5173`**! 🎉

<br>

## 🧑‍💻 How To Use
1. Visit `http://localhost:5173` in your browser.
2. Provide a `.wav` file containing recorded speech. *(A subset of the RAVDESS dataset is ideal).*
3. The UI will instantly communicate with `localhost:8000/predict`.
4. Our state-of-the-art CNN will analyze the frequency coefficients and display the resulting emotion!

<br>

## 🔮 Future Improvements
- **Data Augmentation:** Introducing dynamic background noise, time-stretching, and pitch-shifting to the training pipeline to produce a highly robust, production-ready model capable of analyzing bad cell phone audio.
- **Multilingual Datasets:** Blending RAVDESS with `Emo-DB` (German) and `SAVEE` datasets to expand the cultural generalizability of the emotion acoustic physics.
- **Transformers:** Upgrading the underlying architecture from 1D CNNs to Audio Spectrogram Transformers (AST).

---
*Built from scratch using deeply integrated Audio Signal Processing and Convolutional Neural Networks.*
