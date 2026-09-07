"""
Unit tests for Config defaults and overrides.

No webcam, GPU, or model download required.
"""

from __future__ import annotations

import pytest

from webcam_ai.config import Config


class TestConfigDefaults:
    """Verify that defaults match documented values."""

    def test_camera_index_default(self) -> None:
        assert Config().camera_index == 0

    def test_confidence_threshold_default(self) -> None:
        assert Config().confidence_threshold == 0.5

    def test_image_size_default(self) -> None:
        assert Config().image_size == 640

    def test_model_name_default(self) -> None:
        assert Config().model_name == "yolo11n.pt"

    def test_window_title_non_empty(self) -> None:
        assert len(Config().window_title) > 0

    def test_fps_smoothing_positive(self) -> None:
        assert Config().fps_smoothing > 0

    def test_box_thickness_positive(self) -> None:
        assert Config().box_thickness > 0

    def test_font_scale_positive(self) -> None:
        assert Config().font_scale > 0.0

    def test_show_confidence_default_true(self) -> None:
        assert Config().show_confidence is True


class TestConfigOverrides:
    """Ensure fields can be overridden at construction time."""

    def test_override_camera_index(self) -> None:
        cfg = Config(camera_index=2)
        assert cfg.camera_index == 2

    def test_override_confidence_threshold(self) -> None:
        cfg = Config(confidence_threshold=0.7)
        assert cfg.confidence_threshold == pytest.approx(0.7)

    def test_override_model_name(self) -> None:
        cfg = Config(model_name="yolo11s.pt")
        assert cfg.model_name == "yolo11s.pt"

    def test_override_image_size(self) -> None:
        cfg = Config(image_size=320)
        assert cfg.image_size == 320


class TestConfigColourPalette:
    """colour_for should always return a valid BGR triple."""

    def test_colour_for_class_0(self) -> None:
        colour = Config().colour_for(0)
        assert len(colour) == 3
        assert all(0 <= c <= 255 for c in colour)

    def test_colour_for_wraps_large_class_id(self) -> None:
        cfg = Config()
        # Should not raise even for a class_id > palette size.
        colour = cfg.colour_for(9999)
        assert len(colour) == 3

    def test_colour_for_is_consistent(self) -> None:
        """Same class_id always produces the same colour."""
        cfg = Config()
        assert cfg.colour_for(5) == cfg.colour_for(5)
