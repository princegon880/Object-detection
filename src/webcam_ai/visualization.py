"""
Visualisation helpers for the Real-Time Webcam AI application.

All drawing functions operate on a BGR ``numpy`` frame (as produced by
OpenCV) and modify it **in-place** for efficiency.  They return the same
frame so calls can be chained.

Nothing in this module imports Ultralytics, the camera, or the FPS counter –
it only depends on OpenCV, NumPy, and the Detection dataclass.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from webcam_ai.detection import Detection


# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------

# BGR colours used for the HUD overlay (FPS, object count, etc.)
_HUD_BG_COLOUR = (0, 0, 0)          # black semi-transparent panel
_HUD_TEXT_COLOUR = (255, 255, 255)   # white text
_HUD_FPS_COLOUR = (0, 255, 128)      # bright green for FPS
_HUD_COUNT_COLOUR = (0, 200, 255)    # amber-ish for object count

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE_TYPE = cv2.LINE_AA              # anti-aliased for smoother text


# ---------------------------------------------------------------------------
# Bounding-box drawing
# ---------------------------------------------------------------------------

def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    colour: tuple[int, int, int],
    font_scale: float = 0.6,
    box_thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw a single bounding box with a label on *frame*.

    The label is drawn on a filled rectangle above the bounding box so it
    remains readable even on busy backgrounds.

    Args:
        frame: BGR image array (modified in-place).
        detection: The object to draw.
        colour: BGR tuple for the box and label background.
        font_scale: OpenCV font scale for the label text.
        box_thickness: Pixel width of the bounding-box border.
        show_confidence: Append the confidence percentage to the label.

    Returns:
        The same *frame* with drawing applied.
    """
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2

    # --- bounding rectangle ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thickness, _LINE_TYPE)

    # --- label text ---
    if show_confidence:
        label = f"{detection.class_name}  {detection.confidence:.0%}"
    else:
        label = detection.class_name

    (tw, th), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)

    # Position: just above the top-left corner of the box, clamped to frame.
    label_y = max(y1 - baseline - 4, th + baseline)
    label_bg_y1 = label_y - th - baseline
    label_bg_y2 = label_y + baseline

    # Filled background rectangle for readability.
    cv2.rectangle(
        frame,
        (x1, label_bg_y1),
        (x1 + tw + 4, label_bg_y2),
        colour,
        cv2.FILLED,
    )

    # White text on coloured background.
    cv2.putText(
        frame,
        label,
        (x1 + 2, label_y),
        _FONT,
        font_scale,
        (255, 255, 255),
        1,
        _LINE_TYPE,
    )

    return frame


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    colour_fn: Callable[[int], tuple[int, int, int]],
    font_scale: float = 0.6,
    box_thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw all detections on *frame*.

    Args:
        frame: BGR image array (modified in-place).
        detections: List of Detection objects to draw.
        colour_fn: Callable ``(class_id) -> BGR tuple`` for consistent colours.
        font_scale: See :func:`draw_detection`.
        box_thickness: See :func:`draw_detection`.
        show_confidence: See :func:`draw_detection`.

    Returns:
        The same *frame* after all boxes are drawn.
    """
    for det in detections:
        colour = colour_fn(det.class_id)
        draw_detection(frame, det, colour, font_scale, box_thickness, show_confidence)
    return frame


# ---------------------------------------------------------------------------
# HUD overlay (FPS + object count)
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    fps: float,
    object_count: int,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a heads-up display panel with FPS and object count.

    The panel is painted in the top-left corner on a semi-transparent dark
    background so it remains legible regardless of the scene.

    Args:
        frame: BGR image array (modified in-place).
        fps: Current FPS value to display.
        object_count: Number of detections in the current frame.
        font_scale: OpenCV font scale for HUD text.

    Returns:
        The same *frame* with the HUD overlay applied.
    """
    fps_text = f"FPS: {fps:.1f}"
    count_text = f"Objects: {object_count}"

    line_height = int(28 * font_scale / 0.6)   # scale with font
    padding = 8
    panel_width = 160
    panel_height = 2 * line_height + 3 * padding

    # Semi-transparent background via addWeighted overlay.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), _HUD_BG_COLOUR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS line.
    cv2.putText(
        frame,
        fps_text,
        (padding, padding + line_height),
        _FONT,
        font_scale,
        _HUD_FPS_COLOUR,
        1,
        _LINE_TYPE,
    )

    # Object-count line.
    cv2.putText(
        frame,
        count_text,
        (padding, 2 * padding + 2 * line_height),
        _FONT,
        font_scale,
        _HUD_COUNT_COLOUR,
        1,
        _LINE_TYPE,
    )

    return frame


# ---------------------------------------------------------------------------
# Convenience: draw everything in one call
# ---------------------------------------------------------------------------

def render_frame(
    frame: np.ndarray,
    detections: list[Detection],
    fps: float,
    colour_fn: Callable[[int], tuple[int, int, int]],
    font_scale: float = 0.6,
    box_thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """Apply all visualisation layers to *frame* in the correct order.

    Order: bounding boxes first, then HUD overlay on top.

    Args:
        frame: BGR image array (modified in-place).
        detections: Detected objects for the current frame.
        fps: Current FPS reading.
        colour_fn: Class-id → BGR colour callable.
        font_scale: Uniform font scale for all text.
        box_thickness: Bounding-box border thickness.
        show_confidence: Whether to include confidence in box labels.

    Returns:
        The annotated *frame*.
    """
    draw_detections(frame, detections, colour_fn, font_scale, box_thickness, show_confidence)
    draw_hud(frame, fps, len(detections), font_scale)
    return frame
