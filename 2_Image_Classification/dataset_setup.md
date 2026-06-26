# Dataset Setup — Image Classification

## Dataset: Intel Image Classification
- **Kaggle Link:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- **Size:** ~350 MB
- **Classes:** 6 (buildings, forest, glacier, mountain, sea, street)
- **Images:** ~25,000 images (150x150 px)

---

## Download Instructions

```bash
cd 2_Image_Classification/
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d data/
```

### Expected Folder Structure
```
2_Image_Classification/
├── data/
│   ├── seg_train/seg_train/
│   │   ├── buildings/
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   ├── seg_test/seg_test/
│   └── seg_pred/seg_pred/
├── image_classification.py
└── image_classification.ipynb
```

---

## Notes
- Training set: ~14,000 images
- Test set: ~3,000 images
- Images resized to **150x150** (already that size in dataset)
