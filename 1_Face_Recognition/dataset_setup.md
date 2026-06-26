# Dataset Setup — Face Recognition

## Dataset: Celebrity Face Image Dataset
- **Kaggle Link:** https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
- **Size:** ~200 MB
- **Classes:** 17 celebrity classes (multi-class face recognition)
- **Images:** ~1,700 images (100 per celebrity)

---

## Download Instructions

### Step 1: Setup Kaggle API
```bash
pip install kaggle
# Download your API token from: https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download Dataset
```bash
cd 1_Face_Recognition/
kaggle datasets download -d vishesh1412/celebrity-face-image-dataset
unzip celebrity-face-image-dataset.zip -d data/
```

### Expected Folder Structure After Extraction
```
1_Face_Recognition/
├── data/
│   └── Celebrity Faces Dataset/
│       ├── Angelina Jolie/
│       │   ├── img1.jpg
│       │   └── ...
│       ├── Brad Pitt/
│       └── ... (17 celebrity folders)
├── face_recognition.py
└── face_recognition.ipynb
```

---

## Notes
- The script automatically reads all subfolders as class labels.
- Images are resized to **128x128** during preprocessing.
- No manual labeling needed — folder names become class names.
