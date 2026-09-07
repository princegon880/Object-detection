"""
Detection data structures.

Keeping detection data in a plain dataclass decouples the YOLO detector from
the visualisation layer.  Any component can work with a list of Detection
objects without importing Ultralytics or OpenCV.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """Represents a single detected object in a video frame.

    All co-ordinates are in **pixel space** relative to the frame that was
    passed to the detector.

    Attributes:
        class_id: Integer index into the model's class list.
        class_name: Human-readable label (e.g. ``"person"``, ``"car"``).
        confidence: Model confidence in the range [0.0, 1.0].
        x1: Left edge of the bounding box (pixels).
        y1: Top edge of the bounding box (pixels).
        x2: Right edge of the bounding box (pixels).
        y2: Bottom edge of the bounding box (pixels).
    """

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def centre(self) -> tuple[int, int]:
        """(cx, cy) centre of the bounding box."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def __str__(self) -> str:
        return (
            f"{self.class_name} {self.confidence:.2f} "
            f"[({self.x1},{self.y1})->({self.x2},{self.y2})]"
        )
