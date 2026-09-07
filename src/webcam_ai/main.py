"""
This is the prince gondaliya project
This is a simple pt
Main entry-point for the Real-Time Webcam AI Object Detection application.
This project only detect few objects
Pipeline
--------
::

    Start application
           ↓
    Parse CLI arguments
           ↓
    Build Config from defaults + args
           ↓
    Load YOLO model  (DetectorError → friendly message + exit)
           ↓
    Open webcam      (CameraError  → friendly message + exit)
           ↓
    ┌──────────────────────────────────────┐
    │  Capture frame                       │
    │       ↓                              │
    │  Run YOLO inference                  │
    │       ↓                              │
    │  Process detections                  │
    │       ↓                              │
    │  Draw bounding boxes                 │
    │       ↓                              │
    │  Draw FPS + object-count HUD         │
    │       ↓                              │
    │  Show frame                          │
    │       ↓                              │
    │  Q pressed? ─── Yes ──► break        │
    │       │                              │
    │       No                             │
    └───────┘
           ↓
    Release webcam + destroy OpenCV windows
           ↓
    Exit

Run the application
-------------------
::

    uv run python -m webcam_ai.main
    uv run python -m webcam_ai.main --camera 1
    uv run python -m webcam_ai.main --confidence 0.4
    uv run python -m webcam_ai.main --model yolo11n.pt
    uv run python -m webcam_ai.main --camera 0 --confidence 0.5 --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from webcam_ai.camera import Camera, CameraError
from webcam_ai.config import Config
from webcam_ai.detector import DetectorError, YOLODetector
from webcam_ai.fps import FPSCounter
from webcam_ai.visualization import render_frame


# ---------------------------------------------------------------------------
# Logging setup – INFO level so progress messages reach the terminal
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="webcam_ai",
        description=(
            "Real-Time Webcam AI Object Detection using YOLO.\n\n"
            "Press Q inside the video window to quit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="INDEX",
        help="Webcam index to open (default: %(default)s → uses config default 0).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum confidence threshold 0.0–1.0 (default: uses config default 0.5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="YOLO model weights filename, e.g. yolo11n.pt (default: uses config default).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        metavar="PIXELS",
        help="Inference image size in pixels (default: uses config default 640).",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_config(args: argparse.Namespace) -> Config:
    """Merge CLI arguments into a Config, falling back to defaults."""
    kwargs: dict = {}
    if args.camera is not None:
        kwargs["camera_index"] = args.camera
    if args.confidence is not None:
        if not (0.0 <= args.confidence <= 1.0):
            logger.error(
                "Invalid --confidence %s: must be between 0.0 and 1.0.", args.confidence
            )
            sys.exit(1)
        kwargs["confidence_threshold"] = args.confidence
    if args.model is not None:
        kwargs["model_name"] = args.model
    if args.imgsz is not None:
        kwargs["image_size"] = args.imgsz
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# Core application loop
# ---------------------------------------------------------------------------

def run(config: Config) -> None:
    """Execute the main detection loop.

    Opens the YOLO model and the webcam, then continuously captures frames,
    runs inference, annotates the frame, and displays it until the user
    presses **Q** or the camera fails.

    Args:
        config: Fully constructed :class:`~webcam_ai.config.Config` instance.

    Raises:
        DetectorError: Re-raised if the YOLO model cannot be loaded.
        CameraError:   Re-raised if the camera cannot be opened.
    """
    # --- 1. Load YOLO model -----------------------------------------------
    logger.info("Loading YOLO model '%s' …", config.model_name)
    detector = YOLODetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
        image_size=config.image_size,
    )

    # --- 2. Open webcam + start main loop ----------------------------------
    fps_counter = FPSCounter(window=config.fps_smoothing)

    with Camera(camera_index=config.camera_index) as camera:
        logger.info("Starting detection loop. Press Q in the video window to quit.")

        for frame in camera.frames():
            # --- 3. Run inference -----------------------------------------
            detections = detector.detect(frame)

            # --- 4. Tick FPS before rendering (measures full pipeline) -----
            fps_counter.tick()

            # --- 5. Annotate frame ----------------------------------------
            render_frame(
                frame=frame,
                detections=detections,
                fps=fps_counter.get(),
                colour_fn=config.colour_for,
                font_scale=config.font_scale,
                box_thickness=config.box_thickness,
                show_confidence=config.show_confidence,
            )

            # --- 6. Display -----------------------------------------------
            cv2.imshow(config.window_title, frame)

            # --- 7. Quit on Q (waitKey returns -1 if no key pressed) -------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Q pressed – exiting.")
                break

            # Also quit if the user closes the window via the X button.
            if cv2.getWindowProperty(config.window_title, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed – exiting.")
                break

    # Camera context manager already called release(); destroy OpenCV windows.
    cv2.destroyAllWindows()
    logger.info("All resources released. Goodbye!")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build config, and run the application."""
    args = _parse_args(argv)
    config = _build_config(args)

    logger.info(
        "Configuration: camera=%d  model=%s  confidence=%.2f  imgsz=%d",
        config.camera_index,
        config.model_name,
        config.confidence_threshold,
        config.image_size,
    )

    try:
        run(config)
    except DetectorError as exc:
        logger.error("Model error:\n%s", exc)
        sys.exit(1)
    except CameraError as exc:
        logger.error("Camera error:\n%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    main()
