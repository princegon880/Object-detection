"""
Configuration module for the Real-Time Webcam AI application.

Centralizes all tunable settings so nothing is hard-coded throughout the rest
of the codebase.  Values can be overridden at runtime via CLI arguments (see
main.py) or by creating a Config instance with custom keyword arguments.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Application-wide configuration with sensible defaults.

    Attributes:
        camera_index: Which webcam to open (0 = default system camera).
        confidence_threshold: Minimum YOLO confidence [0.0 – 1.0] for a
            detection to be displayed.
        image_size: Pixel size fed to YOLO during inference (square).
        model_name: Filename of the YOLO weights to load.  YOLO will download
            the file automatically the first time if it is not found locally.
        window_title: Title of the OpenCV display window.
        fps_smoothing: Number of recent frame-times used to compute a smooth
            rolling FPS average.  Higher = smoother but slower to react.
        box_thickness: Pixel thickness of bounding-box rectangles.
        font_scale: OpenCV font scale used for all on-screen labels.
        show_confidence: Whether to include the confidence score next to the
            class name on screen.
    """

    camera_index: int = 0
    confidence_threshold: float = 0.5
    image_size: int = 640
    model_name: str = "yolo11n.pt"
    window_title: str = "Real-Time Webcam AI Object Detection  |  Press Q to quit"
    fps_smoothing: int = 30
    box_thickness: int = 2
    font_scale: float = 0.6
    show_confidence: bool = True

    # Colour palette used when drawing boxes (BGR tuples).
    # One colour per class-id bucket (wraps around for >20 classes).
    _colour_palette: list = field(
        default_factory=lambda: [
            (56, 56, 255),    # 0  – red-ish
            (151, 157, 255),  # 1  – lavender
            (31, 112, 255),   # 2  – orange
            (29, 178, 255),   # 3  – gold
            (49, 210, 207),   # 4  – teal
            (10, 249, 72),    # 5  – green
            (23, 204, 146),   # 6  – sea-green
            (134, 219, 61),   # 7  – lime
            (52, 147, 26),    # 8  – dark green
            (187, 212, 0),    # 9  – yellow-green
            (168, 153, 44),   # 10 – olive
            (220, 144, 41),   # 11 – amber
            (147, 103, 189),  # 12 – purple
            (72, 144, 204),   # 13 – steel blue
            (229, 181, 103),  # 14 – sandy
            (194, 186, 113),  # 15 – tan
            (111, 186, 0),    # 16 – chartreuse
            (0, 175, 177),    # 17 – cyan
            (212, 83, 0),     # 18 – deep orange
            (110, 0, 211),    # 19 – violet
        ]
    )

    def colour_for(self, class_id: int) -> tuple[int, int, int]:
        """Return a consistent BGR colour for the given class ID."""
        return self._colour_palette[class_id % len(self._colour_palette)]
