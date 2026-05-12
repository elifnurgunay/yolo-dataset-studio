# YOLO DATASET STUDIO

 
A PyQt5 desktop application that provides video frame extraction, automatic labeling, and data augmentation operations within a single interface.

---
<img width="1919" height="374" alt="image" src="https://github.com/user-attachments/assets/5ac047c5-9331-4d5b-91b7-dda118f5ff4e" />

<img width="1915" height="677" alt="image" src="https://github.com/user-attachments/assets/aee76832-9f67-4bd2-8fa0-9b8ea021b77e" />

## 1.FEATURES

### 1. Frame Extractor 
Extracts frames from video files for model training.
* **Interval Sampling:** Saves every Nth frame.
* **Time-Based Extraction:** Saves a specified number of frames per second.
* **Motion Threshold:** Saves only distinct scenes and skips nearly identical frames.
* **Preview:** Quickly previews the video using 6 random frames.
* **Video Information:** Automatically displays duration, FPS, resolution, and estimated output count.

### 2. Auto Label

Automatically labels images using a trained YOLO model.

* Generates annotation files in `.txt` and `.json` formats.
* Default classes: `ballistic_missile`, `helicopter`, `f-16`, `drone`.
* Supports GPU acceleration with CUDA (RTX 4060).
* Provides statistical reports in HTML and PNG formats.

### 3. Augmentation

Artificially expands the dataset using the `albumentations` framework.  
All augmentation settings can be managed through the interface.



| Category | Processes |
|---|---|
| Geometric | HorizontalFlip, Affine |
| Photometric — Linear | BrightnessContrast |
| Photometric — Gamma | RandomGamma, CLAHE |
| Photometric — Color | HueSaturationValue, ColorJitter |
| Blur | MotionBlur, GaussianBlur, Defocus |
| Sensor Noise | GaussNoise, ImageCompression |

Each transformation can be enabled or disabled individually or by group, and all parameters can be configured through the UI.

---

## 2. USAGE

### Requirements
- Python 3.10+
- CUDA 12.x + NVIDIA driver (for GPU acceleration)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/USERNAME/yolo-dataset-studio.git
cd yolo-dataset-studio

# 2. Install the PyTorch CUDA build (skip this step if using CPU only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Create the folder structure
.\setup_dirs.ps1

# 6. Launch the application
python app.py
```

---

## 3.PROJECT STRUCTURE

```
iyhss_data_prepare/
│
├── app.py                  # Entry point — QApplication, dark theme
├── main_window.py          # Main window, tab management, GPU badge
│
├── tabs/
│   ├── extract_tab.py      # Frame Extractor tab
│   ├── label_tab.py        # Auto Label tab
│   └── augment_tab.py      # Augmentation tab
│
├── workers/
│   ├── extract_worker.py   # QThread — video frame extraction
│   ├── label_worker.py     # QThread — YOLO inference
│   └── augment_worker.py   # QThread — albumentations pipeline
│
├── core/
│   ├── extractor.py        # Frame extraction logic
│   ├── labeler.py          # Auto-label pipeline (callback-based)
│   └── augmentor.py        # Augmentation pipeline
│
├── config.yaml             # Model and directory configuration
├── auto_label.py           # Standalone CLI script (runs without GUI)
├── setup_dirs.ps1          # Windows folder setup script
└── requirements.txt
```

---

## 4.CONFIGURATION

Edit `config.yaml` to configure the model path and output directories:

```yaml
model:
  path: "C:/Users/USERNAME/Documents/iyhss/weights/yolo26s/best.pt"
  img_size: 640
  conf_threshold: 0.5
  iou_threshold: 0.45

input:
  images_dir: "C:/Users/USERNAME/datasets/to_label/images"

output:
  labels_txt_dir:  "C:/Users/USERNAME/results/auto_label/labels_txt"
  labels_json_dir: "C:/Users/USERNAME/results/auto_label/labels_json"
  annotated_dir:   "C:/Users/USERNAME/results/auto_label/annotated"
  report_dir:      "C:/Users/USERNAME/results/auto_label/report"
```

---

## 5.CLI Usage (Without GUI)

```bash
# After editing config.yaml
python auto_label.py
```

---

## 6.Building the .exe 

```bash
pip install pyinstaller

pyinstaller --noconfirm --onedir --windowed \
    --name "IYHSS_DataPrepare" \
    --add-data "config.yaml;." \
    --hidden-import=PyQt5.sip \
    --collect-all=ultralytics \
    --collect-all=albumentations \
    app.py
```

Output: `dist/IYHSS_DataPrepare/IYHSS_DataPrepare.exe`

---

## 7.Dependencies

| Package | Purpose |
|---|---|
| PyQt5 | Desktop GUI Framework |
| ultralytics | YOLO model inference |
| torch + torchvision | CUDA-accelerated deep learning operations |
| opencv-python | Image Processing and video handling |
| albumentations | Data Augmentation Transformation |
| matplotlib | Visualization and graphs |
| pyyaml | Configuration Files |

---

## 8.License

MIT
