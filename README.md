<h1 align="center">🧠 CNN Projects — Deep Learning Portfolio</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-Transfer%20Learning-red?style=for-the-badge&logo=keras"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=for-the-badge&logo=googlecolab"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A professional multi-project repository demonstrating CNN-based deep learning across three real-world domains.
  Built as part of an ML internship portfolio.
</p>

---

##  Repository Structure

```
CNN-Projects/
├── 1_Face_Recognition/
│   ├── face_recognition.ipynb     ← Google Colab notebook
│   ├── face_recognition.py        ← Standalone Python script
│   └── dataset_setup.md           ← Kaggle download instructions
├── 2_Image_Classification/
│   ├── image_classification.ipynb
│   ├── image_classification.py
│   └── dataset_setup.md
├── 3_Cancer_Detection/
│   ├── cancer_detection.ipynb
│   ├── cancer_detection.py
│   └── dataset_setup.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  Projects Overview

### 1.  Face Recognition
| Field | Details |
|-------|---------|
| **Model** | VGG16 (Transfer Learning) |
| **Dataset** | [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) |
| **Task** | Multi-class face classification (17 celebrities) |
| **Key Features** | Data augmentation, fine-tuning, confusion matrix |

### 2.  Image Classification
| Field | Details |
|-------|---------|
| **Model** | ResNet50 (Transfer Learning) |
| **Dataset** | [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |
| **Task** | 6-class scene classification |
| **Key Features** | Generator-based pipeline, sample prediction visualization |

### 3. Cancer Detection
| Field | Details |
|-------|---------|
| **Model** | EfficientNetB0 (Transfer Learning) |
| **Dataset** | [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection) |
| **Task** | Binary classification — Cancerous vs Non-Cancerous |
| **Key Features** | Class weights, ROC curve, AUC metric, 20% subset option |

---

##  Tech Stack

- **Framework:** TensorFlow 2.x / Keras
- **Pretrained Models:** VGG16, ResNet50, EfficientNetB0 (ImageNet weights)
- **Libraries:** NumPy, Pandas, OpenCV, Scikit-learn, Matplotlib, Seaborn
- **Platform:** Google Colab (GPU) / Local

---

##  How to Run (Google Colab)

1. Open any `.ipynb` file in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run **Step 1** to install dependencies
4. Run **Step 2** — upload your `kaggle.json` API key when prompted
5. Run remaining cells in order

> **Get your Kaggle API key:** kaggle.com → Profile → Settings → API → Create New Token

---

##  How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/CNN-Projects.git
cd CNN-Projects

# Install dependencies
pip install -r requirements.txt

# Run any project
cd 1_Face_Recognition
python face_recognition.py
```

Follow `dataset_setup.md` inside each project folder to download the dataset via Kaggle CLI.

---

##  Results Summary

Each project outputs:
-  Training & validation **accuracy/loss curves**
-  **Confusion matrix** (heatmap)
-  **Classification report** (Precision, Recall, F1)
-  **ROC Curve + AUC** *(Cancer Detection only)*
-  **Sample prediction visualization** *(Image Classification)*

---

##  Notes

- Dataset files are **not included** in this repo (too large). Download via Kaggle — see each `dataset_setup.md`.
- Model `.h5` weight files are excluded via `.gitignore`.
- Cancer detection uses a **20% stratified subset** by default (`USE_SUBSET = True`). Set to `False` for full training.
- Never commit your `kaggle.json` — it's listed in `.gitignore`.

---

##  License

This project is licensed under the [MIT License](LICENSE).
