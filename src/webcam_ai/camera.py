"""
Camera abstraction module.

Wraps OpenCV's VideoCapture in a class with:
  * Clear success/failure feedback at open time.
  * A context-manager interface (``with Camera() as cam``) that guarantees
    the camera is released even when an exception occurs.
  * Per-frame error handling that does not crash the application.
"""

from __future__ import annotations

import logging
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or has an unrecoverable error."""


class Camera:
    """Manages a single webcam via OpenCV VideoCapture.

    Args:
        camera_index: Integer index of the camera to open (0 = default).

    Raises:
        CameraError: If the camera cannot be opened after construction.

    Example::

        with Camera(camera_index=0) as cam:
            for frame in cam.frames():
                process(frame)
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._index = camera_index
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the webcam.

        Raises:
            CameraError: If the device cannot be opened or is not a valid
                capture source (e.g. wrong index, device busy, no permission).
        """
        logger.info("Opening camera index %d …", self._index)
        cap = cv2.VideoCapture(self._index)

        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera at index {self._index}.\n"
                "  • Make sure a webcam is connected and not used by another application.\n"
                "  • On Linux/macOS grant camera permission to the terminal / Python process.\n"
                "  • Try a different --camera index (0, 1, 2 …)."
            )

        self._cap = cap
        logger.info(
            "Camera %d opened  (%d×%d @ %.0f fps reported by driver)",
            self._index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )

    def release(self) -> None:
        """Release the camera resource (safe to call multiple times)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self._index)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Capture a single frame from the webcam.

        Returns:
            ``(True, frame)`` on success, or ``(False, None)`` if the frame
            could not be grabbed (e.g. camera disconnected mid-run).
        """
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            logger.warning("Failed to grab frame from camera %d.", self._index)
            return False, None

        return True, frame

    def frames(self, max_failures: int = 5) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until the camera fails or is released.

        Args:
            max_failures: Number of consecutive failed reads before the
                generator stops (protects against an infinite loop on
                a silently broken camera).

        Yields:
            BGR ``numpy`` frames as returned by OpenCV.
        """
        failures = 0
        while True:
            ok, frame = self.read_frame()
            if not ok or frame is None:
                failures += 1
                logger.warning("Consecutive read failure %d/%d.", failures, max_failures)
                if failures >= max_failures:
                    logger.error("Too many consecutive frame read failures – stopping.")
                    break
                continue
            failures = 0
            yield frame  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """``True`` if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_width(self) -> int:
        """Reported frame width in pixels (may differ from actual capture)."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Reported frame height in pixels."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
