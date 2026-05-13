# 🚗 Illegal Parking Zone Violation Detection

> **Real-time parking violation detector** using YOLOv8 + DeepSORT + pytesseract.  
> Draw your own illegal zone on any video — vehicles that park there are automatically logged.

---

## 📐 System Architecture

```
Video Frame
    │
    ▼
┌─────────────┐     bounding boxes     ┌──────────────┐
│  YOLOv8     │ ─────────────────────► │  DeepSORT    │  persistent Track IDs
│  (detect)   │                        │  (track)     │
└─────────────┘                        └──────┬───────┘
                                              │ tracks
                                              ▼
                                    ┌─────────────────────┐
                                    │  Zone Check          │  point-in-polygon
                                    │  (centroid inside?)  │
                                    └──────────┬──────────┘
                                               │ inside + timer
                                               ▼
                                    ┌─────────────────────┐
                                    │  pytesseract OCR     │  licence plate text
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  ViolationLogger     │  CSV + snapshot
                                    └─────────────────────┘
```

---

## 🗂 Project Structure

```
Ilegal parking zone violation detection/
├── main.py            # ← Entry point — run this
├── config.py          # ← All settings (paths, thresholds, colours)
├── zone_drawer.py     # Interactive polygon zone drawing tool
├── tracker.py         # DeepSORT wrapper
├── ocr.py             # Pytesseract licence plate OCR
├── logger.py          # CSV + snapshot violation logger
├── environment.yml    # Conda environment definition
└── violations/        # Auto-created — output goes here
    ├── output.mp4
    ├── violations.csv
    └── snapshots/
```

---

## ⚙️ Setup (Conda — recommended)

### 1. Create the conda environment
```bash
conda env create -f environment.yml
```

### 2. Activate it
```bash
conda activate parking-violation
```

### 3. Verify tesseract is available
```bash
tesseract --version
```
> If not found, install manually:  
> **Windows** → [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)  
> After installing, set `TESSERACT_CMD` in `config.py` to the full `.exe` path.

---

## ▶️ Running the System

### Basic usage (video set in config.py)
```bash
python main.py
```

### Override video at runtime
```bash
python main.py path/to/your_video.mp4
```

### Step-by-step walkthrough

| Step | What happens |
|------|--------------|
| **1** | First frame opens — click to draw your illegal parking zone |
| **2** | Press **ENTER** to confirm the zone |
| **3** | Detection starts — vehicles are tracked with coloured boxes |
| **4** | Orange box → vehicle is inside the zone (timer counting) |
| **5** | Red box + `!! VIOLATION` → vehicle exceeded the time limit |
| **6** | Violation is saved to CSV + snapshot image |
| **7** | Press **Q** at any time to stop |

---

## 🖱️ Zone Drawing Controls

| Key / Action | Effect |
|---|---|
| `Left Click` | Add polygon point |
| `Right Click` | Undo last point |
| `ENTER` or `C` | Confirm zone (need ≥ 3 points) |
| `R` | Reset — clear all points |
| `Q` | Quit zone drawer |

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `"sample.mp4"` | Input video file |
| `OUTPUT_PATH` | `"violations/output.mp4"` | Annotated output |
| `YOLO_MODEL` | `"yolov8n.pt"` | `n`=fast, `s`/`m`=more accurate |
| `PARK_DURATION_THRESHOLD` | `3.0` | Seconds in zone before violation |
| `CONFIDENCE_THRESHOLD` | `0.4` | Detection confidence cutoff |
| `TESSERACT_CMD` | `None` | Path to tesseract.exe (Windows) |

---

## 📄 Output Files

### `violations/violations.csv`
```
Timestamp,Track_ID,Plate_Number,Duration_In_Zone_s,Snapshot_File
2025-01-15 10:32:45,3,ABC1234,5.20,violation_ID3_20250115_103245.jpg
2025-01-15 10:33:10,7,UNKNOWN,4.80,violation_ID7_20250115_103310.jpg
```

### `violations/snapshots/`
One `.jpg` image per violation — the full annotated frame at the moment the violation was recorded.

---

## 🧩 How Each Module Works

### `zone_drawer.py` — Draw Zone
Interactive OpenCV window on the **first video frame**. Click to add polygon points. The zone is passed to `main.py` as a numpy array.

### `tracker.py` — DeepSORT
Wraps `deep_sort_realtime`. Converts YOLO detections to the `[left, top, w, h]` format DeepSORT expects, returns confirmed tracks with persistent IDs.

### `ocr.py` — Plate OCR
Crops the **bottom 40%** of each vehicle bounding box (plate region), upscales 2×, applies bilateral filter + Otsu threshold, then runs pytesseract in single-word alphanumeric mode. Results are cached per track ID.

### `logger.py` — Violation Logger
Each vehicle is logged **at most once** (keyed by track ID). Saves a snapshot of the full frame and appends a row to the CSV.

### `main.py` — Orchestrator
Ties everything together in a frame-by-frame loop:
`detect → track → zone check → OCR → log → draw → display`

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 vehicle detection |
| `deep-sort-realtime` | Multi-object tracking |
| `pytesseract` | Licence plate OCR |
| `opencv-python` | Video I/O, drawing, zone tool |
| `numpy` | Array math |

---

## 🎓 Learning Notes

- **Why DeepSORT?** It combines Kalman filtering (motion prediction) with a deep appearance feature extractor — giving stable IDs even when vehicles are briefly occluded.
- **Why centroid-in-polygon?** Simple and fast. `cv2.pointPolygonTest` returns ≥0 if inside. More robust methods (IoU with zone) can be added in `main.py`.
- **Why bottom-40% crop for OCR?** Plates are almost always in the lower half of a vehicle bounding box. Narrowing the crop reduces noise for Tesseract.
- **Why log once per track ID?** DeepSORT maintains persistent IDs across frames, so we can reliably identify "same vehicle" and avoid duplicate log entries.
