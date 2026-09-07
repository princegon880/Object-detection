"""
YOLO object detector module.

Wraps Ultralytics YOLO so the rest of the application works with plain
Detection dataclass instances instead of Ultralytics-specific result objects.
This keeps the AI inference layer isolated: swap the model or the framework
here without touching camera.py, visualization.py, or main.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from webcam_ai.detection import Detection

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or used."""


class YOLODetector:
    """Loads a YOLO model and runs per-frame inference.

    The model file is downloaded automatically by Ultralytics if it is not
    found locally (requires an internet connection on the first run).

    Args:
        model_name: Filename or full path to a ``.pt`` weights file.
        confidence_threshold: Detections below this score are discarded.
        image_size: Resolution (pixels, square) for YOLO inference.

    Raises:
        DetectorError: If the model file cannot be loaded.

    Example::

        detector = YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.5)
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        image_size: int = 640,
    ) -> None:
        self._model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self._model = self._load_model(model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_name: str):  # type: ignore[return]
        """Load the YOLO weights, raising DetectorError on failure.

        Args:
            model_name: Filename or path to the weights file.

        Returns:
            A loaded ``ultralytics.YOLO`` instance.

        Raises:
            DetectorError: On any loading problem.
        """
        try:
            from ultralytics import YOLO  # local import keeps startup fast
        except ImportError as exc:
            raise DetectorError(
                "Could not import 'ultralytics'. "
                "Make sure it is installed: uv add ultralytics"
            ) from exc

        model_path = Path(model_name)

        # If not an absolute or relative file path, treat as a named preset
        # that Ultralytics knows how to download automatically.
        try:
            logger.info("Loading YOLO model '%s' …", model_name)
            model = YOLO(str(model_path))
            logger.info("Model loaded successfully (%s).", model_name)
            return model
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(
                f"Failed to load YOLO model '{model_name}'.\n"
                "  • Check that the filename is correct (e.g. yolo11n.pt).\n"
                "  • Ensure you have an internet connection for the first download.\n"
                f"  Details: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO inference on a single BGR frame.

        Args:
            frame: A BGR ``numpy`` array as returned by OpenCV.

        Returns:
            List of :class:`~webcam_ai.detection.Detection` objects, one per
            detected object that meets the confidence threshold.  The list is
            empty when nothing is detected.
        """
        try:
            results = self._model(
                frame,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                verbose=False,  # silence per-frame console spam
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference error: %s", exc)
            return []

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = result.names.get(class_id, str(class_id))
                confidence = float(box.conf[0])
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

        return detections

    @property
    def model_name(self) -> str:
        """The model name/path that was loaded."""
        return self._model_name
