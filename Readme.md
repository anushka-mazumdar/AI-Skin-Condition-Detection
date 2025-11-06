# 🧠 AI Skin Condition Detection System

### *An AI-powered multi-angle facial analysis system for detecting and visualizing common skin conditions.*

---

## 🌿 Introduction

This project aims to build an **AI-driven dermatological assistive system** that detects visible facial skin conditions using computer vision and deep learning.
The system takes **three images** of the user’s face — front, left profile, and right profile — and performs a multi-stage pipeline: verifying if the inputs are valid faces, enhancing image quality, analyzing facial regions, and detecting possible skin conditions such as acne, redness, or pigmentation.

It is designed for **non-clinical, educational, and personal wellness purposes**, not as a medical diagnostic tool.

The final system will be accessible through a **Flutter-based interface**, where users can capture images, visualize results, and receive clear explanations of the analysis.

---

## 🧩 Basic Pipeline

```
Capture (Front, Left, Right)
       ↓
Face Verification
       ↓
Image Enhancement (Super-resolution + Contrast adjustment)
       ↓
Facial Segmentation (Forehead, Cheeks, Nose, Chin)
       ↓
Skin Condition Analysis
       ↓
Condition Classification & Visualization (Heatmaps / Overlays)
```

Each stage is modular, enabling easy updates to individual models or algorithms.

---

## 🗂️ Project Structure

```
AI-Skin-Condition-Detection/
│
├── data/                      # Raw and processed images (front, left, right)
│   ├── raw/
│   └── enhanced/
│
├── models/
│   ├── face_detector/         # Existing face detection model
│   ├── enhancer/              # Super-resolution or ESRGAN model
│   └── skin_condition/        # CNN or transfer learning model for analysis
│
├── src/
│   ├── capture_module.py      # Handles image capture and storage
│   ├── face_verification.py   # Ensures the image contains a valid face
│   ├── enhancer.py            # Enhances input image quality
│   ├── region_segmentation.py # Segments face into zones
│   ├── condition_detector.py  # Skin condition classification logic
│   └── visualize.py           # Generates annotated output overlays
│
├── app/
│   ├── flutter_ui/            # Flutter frontend for user interaction
│   └── api/                   # Backend endpoint (Flask/FastAPI)
│
├── requirements.txt
├── run.py                     # Main entry point to trigger pipeline
├── README.md
└── utils/
    ├── preprocessing.py
    └── helpers.py
```

---

## 🧠 Model Overview

* **Face Detector:** Mediapipe or custom CNN-based model to detect and crop faces.
* **Enhancer:** Real-ESRGAN or OpenCV-based enhancement for image clarity and resolution.
* **Skin Condition Classifier:** CNN (MobileNetV2 or EfficientNetB0 backbone) trained on facial skin condition datasets.
* **Region Segmentation:** Dlib or Mediapipe landmarks to divide facial zones.

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/AI-Skin-Condition-Detection.git
   cd AI-Skin-Condition-Detection
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧾 requirements.txt

```
torch
torchvision
tensorflow
opencv-python
numpy
pandas
matplotlib
Pillow
mediapipe
scikit-learn
tqdm
flask
real-esrgan
```

---

## 🚀 How to Run

1. **Run the backend pipeline**

   ```bash
   python run.py
   ```

2. **Access the interface (if using Flutter app)**

   * Launch the Flutter UI in a separate terminal:

     ```bash
     cd app/flutter_ui
     flutter run
     ```

3. **Upload or capture three angles**
   The system will:

   * Detect and verify faces
   * Enhance images
   * Run condition analysis
   * Display visual results

---

## 🧬 Future Enhancements

* Multi-angle attention-based feature fusion
* Improved explainability with heatmaps
* Integration with mobile photo guidelines (lighting/angle calibration)
* Optional anonymized data logging for model improvement

---

## ⚠️ Disclaimer

This tool is **not a substitute for medical advice or diagnosis**.
It is intended for educational and research purposes only. Always consult a dermatologist for any medical concerns.

---

