# SAM-CLIP



# SAM + CLIP Open-Vocabulary Segmentation

This repository implements an open-vocabulary segmentation pipeline combining
Meta AI's Segment Anything Model (SAM) with OpenAI's CLIP.

The system generates dense candidate masks using SAM and filters them using
CLIP based on a natural language prompt.

---

## 🔍 Pipeline Overview

1. **SAM (TensorFlow)** generates multiple candidate masks using grid-based point prompting.
2. **Mask post-processing** resizes masks to original image resolution.
3. **CLIP (PyTorch)** evaluates each masked region against a text prompt.
4. Masks exceeding a similarity threshold are retained and visualized.

---

## 🧠 Model Stack

| Model | Framework |
|-----|----------|
| SAM (ViT-L) | TensorFlow |
| CLIP (ViT-B/32) | PyTorch |

---

Results example: prompt given = "small blocks or microstructures" (The segmented object will be masked in red colour)
<img width="542" height="395" alt="image" src="https://github.com/user-attachments/assets/f633ebd7-29ca-47c8-aa78-20e9db6ea62f" />

---

## 📁 Project Structure

```text
sam-clip-open-vocabulary-segmentation/
├── sam_clip/
│   ├── models.py
│   ├── sam_inference.py
│   ├── clip_filtering.py
│   └── utils.py
├── scripts/
│   └── run_single_image.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
