# 🎥 Real-Time Webcam AI Object Detection

A production-style Python application that opens your webcam and detects objects in real time using [YOLO](https://docs.ultralytics.com/) and [OpenCV](https://opencv.org/).  Detected objects are highlighted with bounding boxes, class names, confidence scores, a live FPS counter, and an object count — all rendered in the video window as you watch.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Running the Application](#4-running-the-application)
5. [Command-Line Options](#5-command-line-options)
6. [How the Application Works](#6-how-the-application-works)
7. [Computer Vision Concepts Explained](#7-computer-vision-concepts-explained)
8. [Project Structure](#8-project-structure)
9. [Module Overview](#9-module-overview)
10. [Running the Tests](#10-running-the-tests)
11. [About the YOLO Model File](#11-about-the-yolo-model-file)
12. [Troubleshooting](#12-troubleshooting)
13. [Performance Tips](#13-performance-tips)

---

## 1. What This Project Does

| Feature | Details |
|---------|---------|
| **Live detection** | Runs YOLO inference on every webcam frame |
| **Bounding boxes** | Each detected object gets a coloured rectangle |
| **Labels** | Class name + confidence score on every box |
| **HUD** | Real-time FPS and object count overlay |
| **Safe exit** | Press **Q** or close the window to quit cleanly |
| **Configurable** | Camera index, confidence threshold, model, image size |
| **Tested** | Pure-Python unit tests that need no webcam or GPU |

---

## 2. Prerequisites

### Required software

| Tool | Minimum version | How to install |
|------|----------------|----------------|
| **Python** | 3.11 | [python.org](https://www.python.org/downloads/) |
| **uv** | 0.4+ | `pip install uv` or [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| **Webcam** | Any USB or built-in camera | Plug in / enable |

### Operating system notes

- **Windows**: Works out of the box.  OpenCV uses the DirectShow backend.
- **macOS**: Grant terminal camera permission in *System Settings → Privacy → Camera*.
- **Linux**: Make sure your user is in the `video` group (`sudo usermod -aG video $USER`).

> **No GPU required.**  The default model (`yolo11n.pt`) runs comfortably on CPU.
> A CUDA-capable NVIDIA GPU will automatically be used if available for higher FPS.

---

## 3. Installation

Clone or download this repository, then run **one command** inside the project folder:

```bash
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates a `.venv` virtual environment, and installs every dependency — including OpenCV, Ultralytics YOLO, and NumPy — in a single step.

> **Do not use `pip install`.**  This project is managed entirely with **uv**.

---

## 4. Running the Application

```bash
uv run python -m webcam_ai.main
```

What happens on first run:

1. YOLO automatically downloads `yolo11n.pt` (~6 MB) to the Ultralytics cache.
2. Your webcam opens.
3. A window titled *"Real-Time Webcam AI Object Detection  |  Press Q to quit"* appears.
4. Detected objects are highlighted in real time.

**Press Q** (or close the window) to exit.

---

## 5. Command-Line Options

```
usage: webcam_ai [-h] [--camera INDEX] [--confidence FLOAT] [--model NAME] [--imgsz PIXELS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera INDEX` | `0` | Which webcam to open (0 = default, 1 = second camera, …) |
| `--confidence FLOAT` | `0.5` | Minimum confidence to show a detection (0.0–1.0) |
| `--model NAME` | `yolo11n.pt` | YOLO weights file (downloaded automatically if not found) |
| `--imgsz PIXELS` | `640` | Inference image size in pixels (smaller = faster) |

### Examples

```bash
# Default settings
uv run python -m webcam_ai.main

# Use an external USB camera
uv run python -m webcam_ai.main --camera 1

# Stricter confidence (only show high-confidence detections)
uv run python -m webcam_ai.main --confidence 0.7

# Faster but slightly less accurate model
uv run python -m webcam_ai.main --model yolo11n.pt --imgsz 320

# Combine flags
uv run python -m webcam_ai.main --camera 0 --confidence 0.5 --model yolo11n.pt
```

---

## 6. How the Application Works

```
┌─────────────┐
│   Webcam    │  OpenCV captures one frame (~30 times per second)
└──────┬──────┘
       │  BGR NumPy array
       ▼
┌─────────────┐
│   OpenCV    │  VideoCapture.read() grabs the raw pixel data
└──────┬──────┘
       │  frame (height × width × 3 bytes)
       ▼
┌─────────────┐
│    YOLO     │  Runs a neural network on the frame
└──────┬──────┘
       │  list of Detection objects
       ▼
┌──────────────────┐
│ Object Detection │  Filters results below the confidence threshold
└──────┬───────────┘
       │  filtered detections
       ▼
┌──────────────────┐
│  Bounding Boxes  │  Draws coloured rectangles + labels on the frame
└──────┬───────────┘
       │  annotated frame
       ▼
┌──────────────────┐
│  OpenCV Display  │  imshow() renders the frame in a window
└──────────────────┘
```

The loop runs once per frame.  After drawing, the application checks whether `Q` was pressed (or the window was closed) and either repeats or exits cleanly.

---

## 7. Computer Vision Concepts Explained

### Frame
A **frame** is a single still image captured from the webcam.  Video is just a rapid sequence of frames — usually 24–60 per second.  Each frame is stored as a 3D array of numbers: height × width × 3 colour channels (Blue, Green, Red in OpenCV).

### FPS (Frames Per Second)
**FPS** measures how many frames the application processes per second.  Higher FPS = smoother video.  30 FPS is generally considered smooth for most people.  This application displays the FPS live in the top-left corner.

### Bounding Box
A **bounding box** is a rectangle drawn around a detected object.  It is defined by four numbers: the x and y co-ordinates of the top-left corner, and the x and y co-ordinates of the bottom-right corner.  The box tells you *where* the object is in the frame.

### Confidence Score
The **confidence score** is a number between 0.0 and 1.0 that tells you how certain the model is about a detection.  `0.94` means the model is 94% sure it found what it claims to have found.  This application only shows detections above a configurable threshold (default 50%).

### Object Detection
**Object detection** is a computer vision task that answers two questions at once:
- *What* objects are present in an image? (classification)
- *Where* exactly are they? (localisation via bounding boxes)

### Inference
**Inference** is the process of running a trained neural network on new data to get predictions.  In this app, inference happens every frame: the YOLO network analyses the frame and predicts what objects are present and where.

### Real-Time Processing
**Real-time processing** means the system responds fast enough for a human to perceive it as instant — typically ≥20–30 FPS.  This application processes each frame as soon as it arrives and displays the result immediately.

---

## 8. Project Structure

```
real-time-webcam-ai/
│
├── pyproject.toml          ← project metadata, dependencies (managed by uv)
├── uv.lock                 ← exact dependency versions (auto-generated)
├── README.md               ← this file
├── .gitignore              ← files excluded from version control
│
├── src/
│   └── webcam_ai/
│       ├── __init__.py     ← package exports
│       ├── config.py       ← configuration dataclass
│       ├── detection.py    ← Detection dataclass
│       ├── fps.py          ← FPS counter
│       ├── camera.py       ← webcam abstraction
│       ├── detector.py     ← YOLO wrapper
│       ├── visualization.py← drawing functions
│       └── main.py         ← entry-point + CLI
│
├── tests/
│   ├── __init__.py
│   ├── test_fps.py         ← FPS counter tests
│   ├── test_config.py      ← Config tests
│   └── test_detection.py   ← Detection dataclass tests
│
├── models/                 ← (optional) place .pt files here
└── output/                 ← (optional) saved screenshots/recordings
```

---

## 9. Module Overview

### `config.py` — Configuration
Holds all tunable settings in a single `Config` dataclass.  No magic strings or numbers scattered through the code.  Pass a `Config` instance to every function that needs settings.

### `detection.py` — Detection Data Structure
A frozen dataclass representing one detected object.  Contains `class_id`, `class_name`, `confidence`, and bounding-box co-ordinates (`x1 y1 x2 y2`).  Has convenience properties for `width`, `height`, `area`, and `centre`.

### `fps.py` — FPS Counter
Tracks frame durations in a rolling deque.  Averaging over the last *N* frames produces a stable, readable FPS number without wild jumps.

### `camera.py` — Webcam Abstraction
Wraps OpenCV `VideoCapture` with clean open/release lifecycle, a context manager (`with Camera() as cam:`), a `frames()` generator, and friendly error messages when the camera is unavailable.

### `detector.py` — YOLO Detector
Loads the YOLO model and runs inference.  Converts Ultralytics result objects into plain `Detection` instances so the rest of the app has no Ultralytics dependency.

### `visualization.py` — Drawing Functions
All OpenCV drawing lives here: bounding boxes, label backgrounds, class names, confidence scores, and the HUD panel (FPS + object count).  Uses semi-transparent overlays for the HUD.

### `main.py` — Entry-Point
Parses CLI arguments, builds a `Config`, loads the detector and camera, runs the frame loop, and handles every error case with a helpful message.

---

## 10. Running the Tests

```bash
uv run pytest
```

Expected output:

```
collected N items

tests/test_fps.py::TestFPSCounterConstruction::test_default_window PASSED
tests/test_fps.py::TestFPSCalculation::test_30fps_two_ticks PASSED
...
tests/test_config.py::TestConfigDefaults::test_camera_index_default PASSED
...
tests/test_detection.py::TestDetectionFields::test_class_id PASSED
...

====== N passed in 0.XXs ======
```

> All tests are pure Python.  No webcam, no GPU, no model download required.

---

## 11. About the YOLO Model File

The default model is `yolo11n.pt` ("nano" variant — smallest and fastest).  It is **not** included in this repository because model files are large binaries that do not belong in Git.

**What happens on first run:**

1. Ultralytics checks if the file exists in its local cache (`~/.cache/ultralytics/` on Linux/macOS, `%LOCALAPPDATA%\Ultralytics\` on Windows).
2. If not found, it downloads the file automatically (~6 MB — takes a few seconds).
3. On subsequent runs the cached file is used instantly.

**Alternative models** (larger = more accurate but slower):

| Model | Size | Speed (CPU) | Best for |
|-------|------|-------------|----------|
| `yolo11n.pt` | ~6 MB | Fast | Real-time on CPU |
| `yolo11s.pt` | ~22 MB | Medium | Balanced |
| `yolo11m.pt` | ~49 MB | Slower | Higher accuracy |
| `yolo11l.pt` | ~87 MB | Slow | Best accuracy |

Switch models with `--model yolo11s.pt` etc.

---

## 12. Troubleshooting

### Camera not opening

```
Camera error: Could not open camera at index 0.
```

**Causes & fixes:**
- Another application (Zoom, Teams, browser) is already using the camera → close it.
- Wrong camera index → try `--camera 1` or `--camera 2`.
- On Linux: run `ls /dev/video*` to confirm the device exists.
- On macOS: grant camera permission to Terminal in *System Settings → Privacy → Camera*.

### Black screen / green screen

The camera opened but frames are corrupted:
- Unplug and re-plug the USB camera.
- Try a lower resolution or different backend: some cameras need `cv2.CAP_DSHOW` (Windows) or `cv2.CAP_V4L2` (Linux). You can modify `camera.py` to pass the backend flag.

### Model download fails

```
Failed to load YOLO model 'yolo11n.pt'. Check internet connection…
```

- Ensure you have an internet connection on first run.
- If behind a proxy set `HTTP_PROXY` / `HTTPS_PROXY` environment variables.
- Manually download from [github.com/ultralytics/assets](https://github.com/ultralytics/assets/releases) and place in the project's `models/` folder, then run `--model models/yolo11n.pt`.

### Low FPS (< 10)

- Use a smaller model: `--model yolo11n.pt`
- Reduce inference size: `--imgsz 320`
- Close background applications consuming CPU.
- If you have an NVIDIA GPU, install CUDA and `uv add torch torchvision --index https://download.pytorch.org/whl/cu121` — YOLO will use it automatically.

### Permission denied (Linux/macOS)

```bash
# Linux: add yourself to the video group (requires logout/login)
sudo usermod -aG video $USER

# macOS: approve Terminal in System Settings → Privacy → Camera
```

### CPU vs GPU performance

| Hardware | Expected FPS (yolo11n 640px) |
|----------|------------------------------|
| Modern CPU (Intel/AMD) | 10–30 FPS |
| Apple M-series (MPS) | 30–60 FPS |
| NVIDIA GPU (CUDA) | 60+ FPS |

YOLO auto-detects the best available device.  No code changes needed.

### `uv sync` fails

Make sure you are using Python 3.11 or newer:
```bash
python --version   # should be 3.11+
uv python install 3.11
uv sync
```

---

## Licence

MIT — use freely.
