"""
Unit tests for the Detection dataclass.

No webcam, GPU, or model download required.
"""

from __future__ import annotations

import pytest

from webcam_ai.detection import Detection


@pytest.fixture()
def sample_detection() -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.94,
        x1=100,
        y1=50,
        x2=300,
        y2=400,
    )


class TestDetectionFields:
    def test_class_id(self, sample_detection: Detection) -> None:
        assert sample_detection.class_id == 0

    def test_class_name(self, sample_detection: Detection) -> None:
        assert sample_detection.class_name == "person"

    def test_confidence(self, sample_detection: Detection) -> None:
        assert sample_detection.confidence == pytest.approx(0.94)

    def test_bounding_box_coordinates(self, sample_detection: Detection) -> None:
        assert sample_detection.x1 == 100
        assert sample_detection.y1 == 50
        assert sample_detection.x2 == 300
        assert sample_detection.y2 == 400


class TestDetectionProperties:
    def test_width(self, sample_detection: Detection) -> None:
        assert sample_detection.width == 200  # 300 - 100

    def test_height(self, sample_detection: Detection) -> None:
        assert sample_detection.height == 350  # 400 - 50

    def test_area(self, sample_detection: Detection) -> None:
        assert sample_detection.area == 200 * 350

    def test_centre(self, sample_detection: Detection) -> None:
        cx, cy = sample_detection.centre
        assert cx == 200  # 100 + 200//2
        assert cy == 225  # 50  + 350//2


class TestDetectionImmutability:
    def test_frozen_prevents_attribute_assignment(self, sample_detection: Detection) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_detection.class_id = 99  # type: ignore[misc]


class TestDetectionEquality:
    def test_equal_detections(self) -> None:
        d1 = Detection(0, "car", 0.8, 10, 20, 100, 200)
        d2 = Detection(0, "car", 0.8, 10, 20, 100, 200)
        assert d1 == d2

    def test_different_class_id_not_equal(self) -> None:
        d1 = Detection(0, "car", 0.8, 10, 20, 100, 200)
        d2 = Detection(1, "car", 0.8, 10, 20, 100, 200)
        assert d1 != d2


class TestDetectionStr:
    def test_str_contains_class_name(self, sample_detection: Detection) -> None:
        assert "person" in str(sample_detection)

    def test_str_contains_confidence(self, sample_detection: Detection) -> None:
        assert "0.94" in str(sample_detection)
