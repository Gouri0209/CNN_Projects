# Dataset Setup — Cancer Detection

## Dataset: Histopathologic Cancer Detection
- **Kaggle Link:** https://www.kaggle.com/competitions/histopathologic-cancer-detection
- **Size:** ~6 GB (full) | Use subset for quick experiments
- **Task:** Binary classification — cancerous vs non-cancerous tissue
- **Images:** 220,025 training images (96x96 px, .tif format)

---

## Download Instructions

```bash
cd 3_Cancer_Detection/
# Accept competition rules on Kaggle first, then:
kaggle competitions download -c histopathologic-cancer-detection
unzip histopathologic-cancer-detection.zip -d data/
```

### ⚠️ Note on Dataset Size
The full dataset is ~6 GB. For experimentation, the script uses a **stratified 20% subset** automatically.  
To use the full dataset, set `USE_SUBSET = False` in `cancer_detection.py`.

### Expected Folder Structure
```
3_Cancer_Detection/
├── data/
│   ├── train/          ← ~220,000 .tif images
│   ├── test/           ← ~57,000 .tif images
│   └── train_labels.csv
├── cancer_detection.py
└── cancer_detection.ipynb
```

---

## Class Distribution
- **Label 0 (Non-cancerous):** ~130,908 images (59.5%)
- **Label 1 (Cancerous):**     ~89,117 images (40.5%)
- Slight class imbalance handled via `class_weight` in training.
