<div align="center">
  <h1>🎙️ Vocal Metrics: Speech Emotion Recognition</h1>
  <p>An advanced Deep Learning solution for real-time speech emotion classification.</p>

  [![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-%2320232a.svg?logo=react&logoColor=%2361DAFB)](https://react.dev/)
  [![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
</div>

---

## ⚡ Project Overview
Vocal Metrics is an end-to-end Machine Learning web application designed to automatically detect human emotions from raw audio recordings. It leverages **Digital Signal Processing (DSP)** via Librosa to extract complex audio features (MFCCs and Mel Spectrograms) and a **PyTorch 1D Convolutional Neural Network (CNN)** to classify those features into 8 distinct emotions.

<br>

## 🚀 Features
- **Highly Accurate CNN Model:** Achieved 89% Training Accuracy and 76.4% Validation Accuracy over 50 Epochs on the RAVDESS dataset.
- **Lightning-Fast Inference:** The PyTorch model is wrapped in an asynchronous **FastAPI** backend endpoint `/predict`.
- **Premium User Interface:** A stunning, dark-mode, glassmorphism UI built with **React**, **TailwindCSS (v4)**, and **Framer Motion**.
- **Interactive Audio Previews:** Users can seamlessly listen to the audio clips inside the app before initializing analysis.

<br>

## 📁 Repository Structure
```text
📦 vocal-metrics-ser
 ┣ 📂 app
 ┃ ┣ 📂 api               # FastAPI backend & inference engine (`main.py`)
 ┃ ┗ 📂 ui                # Vite + React + TailwindCSS User Interface
 ┣ 📂 models              # Contains the pre-trained weights (`best_model.pth`)
 ┣ 📂 src
 ┃ ┣ 📂 data              # PyTorch Dataset Loaders and download scripts
 ┃ ┣ 📂 features          # Librosa MFCC/Spectrogram extraction pipeline
 ┃ ┣ 📂 models            # PyTorch Deep Neural Network architecture configurations
 ┃ ┗ 📂 training          # Core training loops, validation, and epoch checkpointing
 ┣ 📜 .gitignore          # Keeps your Github clean of huge datasets!
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

---
*Built from scratch using deeply integrated Audio Signal Processing and Neural Networks.*
