🇬🇧 English | [🇨🇿 Česky](README_cs.md)
# LiDAR-Camera Sensor Fusion (KITTI Dataset)

**Bachelor's Thesis** - Analysis of LiDAR and Camera Sensor Fusion Methods for Scene Interpretation

**Author:** Viacheslav Zlobin
**Supervisor:** Ing. Tomáš Klein, Ph.D.
**VŠB-TUO, FEI** - Automotive Electronic Systems, 2026

---

## Objectives

1. Review current LiDAR-camera fusion methods aimed at scene interpretation, such as object detection, segmentation and environment understanding.
2. Describe and analyse feature-level and decision-level fusion approaches, including their advantages, drawbacks and suitable applications.
3. Demonstrate the principle of the selected fusion methods on the provided recordings and compare their results visually and analytically.
4. Evaluate the accuracy, robustness and suitability of each fusion method for scene interpretation.

---

## Requirements

### Hardware

- **CPU:** any 64-bit processor (Intel/AMD)
- **RAM:** 8 GB minimum (16 GB recommended for large point clouds)
- **Disk:** ~50 GB free space (KITTI dataset)
- **GPU:** NVIDIA with CUDA support (optional — significantly speeds up YOLOv8)

### Software

- Python 3.8 or newer
- pip (Python package manager)
- NVIDIA CUDA Toolkit 11.x or 12.x (only for GPU acceleration)

### Python dependencies

| Library | Version | Purpose |
|---|---|---|
| numpy | >= 1.21.0 | numerical computation, point cloud handling |
| opencv-python | >= 4.5.0 | image loading, bounding box drawing |
| matplotlib | >= 3.5.0 | visualisation, comparison plots |
| pandas | >= 1.3.0 | exporting statistics and metrics to CSV |
| torch | >= 1.9.0 | deep learning (YOLOv8 backend) |
| ultralytics | >= 8.0.0 | YOLOv8 object detection in images |
| hdbscan | >= 0.8.0 | hierarchical clustering of LiDAR points |

The complete list including second-level dependencies is in `requirements.txt`.

---

## Installation

### 1. Project setup

Extract the project source files into any folder, for example:

```
D:\Python\sensor_fusion_kitti\
```

This folder should contain `main.py`, `config.py`, `data_loader.py`, `fusion_methods.py`, `visualization.py`, `analysis.py`, `requirements.txt` and the remaining files.

### 2. Virtual environment (recommended)

Open a command prompt in the project folder:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 weights

The file `yolov8n.pt` is downloaded automatically on the first run (approx. 6 MB). It can also be fetched manually:

```bash
yolo download model=yolov8n.pt
```

### 5. Prepare the KITTI dataset

Download the KITTI Object Detection Dataset from the official site:
http://www.cvlibs.net/datasets/kitti/eval_object.php

Required archives:

| Archive | Size | Content |
|---|---|---|
| `data_object_image_2.zip` | ~12 GB | left camera images |
| `data_object_velodyne.zip` | ~29 GB | LiDAR point clouds |
| `data_object_calib.zip` | ~16 MB | calibration files |
| `data_object_label_2.zip` | ~5 MB | ground-truth annotations |

Extract them into a `DATASET/` folder in the project root so that the resulting structure matches:

```
DATASET/
├── data_object_image_2/training/image_2/*.png
├── data_object_velodyne/training/velodyne/*.bin
├── data_object_calib/training/calib/*.txt
└── data_object_label_2/training/label_2/*.txt
```

`config.py` is set to `../Sample_data/DATASET`, which points to the bundled mini-dataset of 10 frames used for demonstration. To use your own full KITTI copy, adjust `KITTI_DATA_PATH` and `KITTI_TRAINING_PATH`.

### 6. Verify the installation

```bash
python main.py
```

When prompted for the number of frames, enter `5` for a quick test. Once finished, the `results/` folder should contain:

- `result_frame_000000.png` … `result_frame_000004.png`
- `statistics.csv`, `metrics.csv`, `report.txt`

### Optional: GPU acceleration (CUDA)

An NVIDIA GPU with CUDA support significantly speeds up YOLOv8 detection. PyTorch with CUDA support is installed separately according to the driver version:

```bash
# example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

GPU detection is automatic — if CUDA is unavailable, the program falls back to CPU without any intervention.

### Troubleshooting

| Error | Solution |
|---|---|
| `ModuleNotFoundError: hdbscan` | `pip install hdbscan` |
| `ERROR: KITTI data not found` | check `KITTI_DATA_PATH` in `config.py` |
| `Annotation files (label_2) not found` | download `data_object_label_2.zip` — evaluation requires ground truth |
| `CUDA out of memory` | set `device = 'cpu'` or reduce the YOLO batch size |

---

## System architecture

Processing pipeline for each frame:

```
Load data (image + LiDAR + calibration)
        │
        ├── LiDAR → camera projection (Tr_velo_to_cam → R0_rect → P2)
        │
        ├── Camera detection — YOLOv8 on the full frame
        │
        ├── LiDAR detection — ground removal → HDBSCAN clustering → NMS
        │
        ├── Decision-level fusion — independent detections → matching via IoU + centres → merge
        │
        ├── Feature-level fusion — LiDAR clusters → ROI crop → YOLOv8 on crops
        │
        ├── Accuracy evaluation vs. ground truth (Precision / Recall / F1)
        │
        └── Visualisation (six-panel comparison plot)
```

---

## Detection methods

### 1. Camera (YOLOv8)

- Model: YOLOv8n (nano) — `yolov8n.pt` pretrained weights (COCO, ~6 MB)
- Confidence threshold: 0.55 (to minimise false positives)
- Classes: Car, Pedestrian, Cyclist (COCO → KITTI mapping)
- GPU (CUDA) support with automatic CPU fallback

### 2. LiDAR (HDBSCAN)

- Ground point removal by Z height (threshold: −1.4 m)
- Voxelisation for clouds above 50,000 points (voxel size: 0.1 m)
- HDBSCAN clustering with adaptive epsilon based on mean distance
- Geometric cluster filter: height 0.5–3.5 m, width 0.4–3.0 m, depth 0.5–5.0 m
- Point density filter (> 5 points/m³)
- Non-Maximum Suppression (NMS) to remove duplicates

### 3. Decision-level fusion

- Camera and LiDAR detect independently
- Greedy matching: camera detections (sorted by descending confidence) against LiDAR detections
- Matching score: `0.6 × IoU + 0.4 × (1 − dist_centres / 150)`
- Matching threshold: 0.20
- Result confidence: `0.7 × camera + 0.3 × LiDAR`
- Three output types: `fused` / `image_only` / `lidar_only`

### 4. Feature-level fusion

- LiDAR determines **where** to look (ROI from clusters + 60 px margin)
- YOLOv8 classifies **what** is in the region
- Confidence threshold: 0.25 (lower due to the reduced context of the crop)
- Small crops (< 128 px) are upscaled
- One cluster = one object (the best detection is taken)

---

## Project structure

```
sensor_fusion_kitti/
│
├── main.py                    # Main pipeline: loading → detection → fusion → evaluation → visualisation
├── config.py                  # Configuration: paths, parameters, method switches
├── data_loader.py             # KITTI data loader (images, LiDAR, calibration, ground truth)
├── fusion_methods.py          # Fusion method implementations (decision + feature)
├── visualization.py           # Visualisation: six-panel comparison plots
├── analysis.py                # Accuracy evaluation: TP/FP/FN, precision/recall/F1 by class and distance
├── sensitivity_analysis.py    # Parameter sensitivity analysis for decision fusion
├── analyze_gt_sizes.py        # Geometric filter validation against real ground-truth dimensions
├── yolov8n.pt                 # Pretrained YOLOv8 nano weights (COCO, ~6 MB)
├── requirements.txt           # Python dependencies
├── __init__.py                # Package version (1.0.0)
│
├── DATASET/                                          # KITTI dataset (standard structure)
│   ├── data_object_image_2/training/image_2/         # PNG camera images
│   ├── data_object_velodyne/training/velodyne/       # Binary LiDAR files (.bin)
│   ├── data_object_calib/training/calib/             # Calibration matrices (.txt)
│   └── data_object_label_2/training/label_2/         # Ground-truth annotations (.txt)
│
├── results/                       # Output files
│   ├── result_frame_XXXXXX.png    # Six-panel visualisations
│   ├── statistics.csv             # Per-frame statistics
│   ├── metrics.csv                # P/R/F1 metrics (method × class × distance)
│   └── report.txt                 # Text report
│
└── sensitivity_results.csv        # Sensitivity analysis results
```

---

## Data preparation

The project uses the KITTI Object Detection Dataset. The data must be downloaded and placed in the `DATASET/` folder with the following structure:

1. **data_object_image_2** — colour images from the left camera (`.png`)
2. **data_object_velodyne** — Velodyne LiDAR point clouds (`.bin`, float32 format: x, y, z, intensity)
3. **data_object_calib** — calibration files (projection matrices P0–P3, R0_rect, Tr_velo_to_cam)
4. **data_object_label_2** — ground-truth annotations (optional, required for accuracy evaluation)

Each file has a six-digit index (e.g. `000043.png`, `000043.bin`, `000043.txt`).

### LiDAR → camera projection

A three-step transformation of 3D LiDAR coordinates into pixel coordinates:

1. **Tr_velo_to_cam** — conversion into the camera coordinate system (3×4)
2. **R0_rect** — rectification (camera axis alignment, 3×3)
3. **P2** — perspective projection onto the image plane (3×4, division by Z)

Visibility mask: a point is considered visible if Z > 0 and its coordinates lie within the image boundaries (1242 × 375).

---

## Configuration

`config.py` contains all system parameters:

| Parameter | Value | Description |
|---|---|---|
| `KITTI_DATA_PATH` | `"DATASET"` | dataset root folder |
| `IMAGE_WIDTH/HEIGHT` | 1242 × 375 | KITTI image resolution |
| `VISUALIZE` | `True` | generate six-panel plots |
| `SAVE_RESULTS` | `True` | save results to disk |
| `USE_CAMERA_DETECTION` | `True` | enable camera detection |
| `USE_LIDAR_DETECTION` | `True` | enable LiDAR detection |
| `USE_DECISION_FUSION` | `True` | enable decision-level fusion |
| `USE_FEATURE_FUSION` | `True` | enable feature-level fusion |
| `EVALUATE_WITH_GT` | `True` | evaluate against ground truth |
| `GT_IOU_THRESHOLD` | 0.5 | IoU threshold for ground-truth matching |
| `GT_CLASSES` | Car, Pedestrian, Cyclist | tracked classes |
| `DISTANCE_BINS` | `[0, 20, 40, 80]` | distance ranges (m) |

---

## Running the project

### Main pipeline

```bash
python main.py
```

The program asks for the number of frames to process (from 1 up to the maximum available). The output is a console log with a progress indicator, accuracy tables and files in the `results/` folder.

### Sensitivity analysis

```bash
python sensitivity_analysis.py
```

Verifies whether the baseline constants in `fusion_methods.py` are optimal. It runs the pipeline on 1500 frames for each parameter variant and compares the F1 score against the baseline.

Parameter groups examined:

| Group | Parameter | Values examined |
|---|---|---|
| A | `match_score` weights (IoU / centres) | 0.5/0.5; 0.6/0.4; 0.7/0.3 |
| B | matching threshold | 0.10; 0.15; 0.20; 0.25; 0.30 |
| C | centre distance normalisation | 100, 150, 200 px |
| D | confidence weights (camera / LiDAR) | 0.5/0.5; 0.6/0.4; 0.7/0.3; 0.8/0.2 |
| E | ground removal threshold | −1.2 m; −1.4 m , −1.6 m |

Output:
- console log with F1/P/R and a summary table of deviations from the baseline
- `sensitivity_results.csv`

Runtime: approx. 15–30 min on GPU. For a quick test, lower `N_FRAMES` to 100 in the script header.

### Geometric filter validation

```bash
python analyze_gt_sizes.py
```

Measures the real 3D dimensions of KITTI ground-truth objects and verifies what proportion of objects falls within the limits of `filter_cluster()` (h: 0.5–3.5 m, w: 0.4–3.0 m, d: 0.5–5.0 m).

Output (console only):
- dimension statistics for each class (min/max/mean, p5/p95)
- percentage of objects satisfying all filter limits

Runtime: approx. 5–10 s (no GPU required).

---

## Output data

### Six-panel visualisation (`result_frame_XXXXXX.png`)

Each processed frame generates an image made of six panels:

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| **Row 1** | LiDAR projection (colour by distance) | Camera YOLOv8 (green boxes) | Feature fusion (orange) |
| **Row 2** | LiDAR clusters (blue) | Decision fusion (red) | Method comparison |

### `statistics.csv`

Per-frame statistics: LiDAR point count, visible points, detections per method, processing time.

### `metrics.csv`

Complete metrics table (precision / recall / F1 / TP / FP / FN) for every combination of method × class × distance range.

### `report.txt`

Text report: active methods, LiDAR statistics, average detections, FPS, accuracy against ground truth broken down by class and distance.

---

## Metrics and evaluation

Evaluation is performed by greedy matching of detections against ground-truth annotations, starting from the highest confidence. A detection counts as a true positive when its IoU is at least 0.5 with a ground-truth object that has not yet been matched. Metrics are aggregated:

- **By class:** Car, Pedestrian, Cyclist, ALL
- **By distance:** 0–20 m, 20–40 m, 40–80 m, > 80 m, ALL
- **By method:** camera, lidar, decision_fusion, feature_fusion

Ground-truth annotations with truncation above 0.8 or occlusion above 2 are automatically excluded as unreliable.
