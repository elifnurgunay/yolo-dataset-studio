# YOLO DATASET STUDIO

 
A PyQt5 desktop application that provides video frame extraction, automatic labeling, and data augmentation operations within a single interface.
It has 3 steps to create your spesific dataset.

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

## 2.GUI Reference

Tab 1 — Frame Extract
Extract optimal image frames from raw video sources using customizable sampling algorithms.

| Interface Element | Operational Description |
| :--- | :--- |
| **Video path** | Accepts standard container formats supported by OpenCV (`.mp4`, `.avi`, `.mkv`, `.mov`). |
| **Output folder** | Designated local target directory for written `.jpg` image sequence frames. |
| **Mode Selection** | Choose between `every_nth` (interval), `fps` (temporal target), or `motion` (dynamic change). |
| **Nth frame** | Specifies frame gap distance when mode is set to `every_nth`. |
| **Target FPS** | Downsamples video playback temporal speed to absolute frames per second. |
| **Motion threshold**| Absolute pixel delta differential limit filter for capturing scene activity. |

Tab 2 — Auto Label
Automate object bounding box annotation workflows across vast unlabelled directories using raw inference power.

| Interface Element | Operational Description |
| :--- | :--- |
| **Image Folder** | Directory path holding source frames/images awaiting machine detection. |
| **Model Path** | Local file system trajectory to selected YOLO weights file (`.pt`). |
| **Output Root Folder** | Base location where partitioned structural sub-folders are constructed automatically. |
| **Conf Threshold** | Detection filter threshold value rejecting untrusted background classifications. |
| **IOU Threshold** | Non-Maximum Suppression overlap threshold preventing duplicate overlapping predictions. |
| **Image Size** | Square pixel target dimension fed directly into network processing layers. |
| **Device Toggle** | Direct computational device target override (`cuda` vs `cpu`). |
| **Output Flags** | Checkbox toggles enabling concurrent output stream writes (TXT, JSON, Annotated). |
| **Skip Empty** | Skip writing outputs for images with zero valid detections. |

Tab 3 — Augmentation
Multiply dataset scale and network robustness through tailored synthetic transformations.

| Interface Element | Operational Description |
| :--- | :--- |
| **Image Folder** | Source directory containing ground-truth unaugmented base images. |
| **Label Folder (.txt)** | Folder housing matched standard YOLO label files. |
| **Output Folder** | Destination root path outputting parallel expanded `images/` and `labels/` structures. |
| **Multiplier** | Integer defining output count factor generation per singular input item. |

## API Documentation

Integrate studio backends programmatically into external scripting routines.

### `core.extractor`

```python
from core.extractor import get_video_info, get_preview_frames, run_extraction
```
```get_video_info(video_path: str | Path) -> dict | None```

Parses physical video container parameters directly via low-level stream reading. Resolves Windows absolute Unicode strings securely. Returns metadata dictionary containing total_frames, fps, width, height, and duration.

```get_preview_frames(video_path: str | Path, n: int = 6) -> list[np.ndarray]```

Extracts random, non-contiguous raw memory representations (BGR numpy multidimensional matrices) across video boundaries for UI rendering pipelines.

```run_extraction(cfg: dict, log_cb: Callable, progress_cb: Callable, stop_flag: threading.Event) -> int```

Executes configured sequential frame captures driven by requested algorithmic sampling modes. Safe thread execution allows unconstrained cross-thread interrupts. Returns final output written frame quantity.

### `core.labeler`

```python
from core.labeler import run_pipeline
```

```run_pipeline(cfg: dict, log_cb: Callable, progress_cb: Callable, stop_flag: threading.Event) -> dict | None```

Instantiates local torch memory architectures, ingests configuration maps, sweeps directories sequentially executing NMS threshold filtering, and handles parallel storage pipelines. Returns finalized dictionary summary structures.

### `core.augmentor`

```python
from core.augmentor import run_augmentation
```
```run_augmentation(cfg: dict, log_cb: Callable, progress_cb: Callable, stop_flag: threading.Event) -> None```

Deploys dynamically created complex composition pipelines utilizing Albumentations. Reads companion coordinate map annotations and executes precise bounding-box spatial recalculations across multiple multiplied output variants simultaneously.

### `workers`

All underlying processing modules utilize robust threading constructs extending `QThread` combined with standard `pyqtSignal` events for thread-safe cross-boundary communication.

Background Execution Controllers:

`LabelWorker(cfg):` Wraps `core.labeler.run_pipeline` execution. Dispatches log, progress, and finished signals asynchronously.

`ExtractWorker(cfg):` Offloads `core.extractor.run_extraction` video decoding loops. Prevents interactive UI frame-drops.

`PreviewWorker(video_path):` Concurrently retrieves video sample arrays populating preview layout placeholders dynamically.

`AugmentWorker(cfg):` Handles high-throughput synthetic operations across expansive disk files concurrently.

---

## 3. USAGE

### Requirements
- Python 3.10+
- CUDA 12.x + NVIDIA driver (for GPU acceleration)
- Operating System: Windows, Linux, or macOS

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

## 4.PROJECT STRUCTURE

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

## 5.CONFIGURATION

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

## 6.CLI Usage (Without GUI)

```bash
# After editing config.yaml
python auto_label.py
```

---

## 7.Building the .exe 

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

## 8.Dependencies

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

## 9.License

MIT
