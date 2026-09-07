"""
Unit tests for the FPSCounter class.

These tests do NOT require a webcam, GPU, or any external model download.
All timing is controlled with mock.patch so the tests are deterministic and
fast.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from webcam_ai.fps import FPSCounter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFPSCounterConstruction:
    def test_default_window(self) -> None:
        fps = FPSCounter()
        assert fps.frame_count == 0

    def test_custom_window(self) -> None:
        fps = FPSCounter(window=10)
        assert fps.frame_count == 0

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            FPSCounter(window=0)

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError):
            FPSCounter(window=-5)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestFPSCounterInitialState:
    def test_get_before_any_tick_returns_zero(self) -> None:
        fps = FPSCounter()
        assert fps.get() == 0.0

    def test_get_after_single_tick_returns_zero(self) -> None:
        """One tick only seeds the timer; a delta requires at least two."""
        fps = FPSCounter()
        fps.tick()
        assert fps.get() == 0.0

    def test_frame_count_zero_before_ticks(self) -> None:
        fps = FPSCounter()
        assert fps.frame_count == 0


# ---------------------------------------------------------------------------
# FPS calculation accuracy
# ---------------------------------------------------------------------------

class TestFPSCalculation:
    """Patch time.perf_counter to deliver precise, deterministic timings."""

    def test_30fps_two_ticks(self) -> None:
        """Two ticks 1/30 s apart → ~30 FPS."""
        t = [0.0, 1 / 30]
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = t
            fps = FPSCounter(window=30)
            fps.tick()  # t=0.0  (seeds timer)
            fps.tick()  # t=1/30 (records delta)
            result = fps.get()

        assert abs(result - 30.0) < 0.01, f"Expected ~30 FPS, got {result}"

    def test_60fps(self) -> None:
        """Ticks spaced 1/60 s apart should yield ~60 FPS."""
        times = [i / 60 for i in range(5)]
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = times
            fps = FPSCounter(window=10)
            for _ in times:
                fps.tick()
            result = fps.get()

        assert abs(result - 60.0) < 0.1, f"Expected ~60 FPS, got {result}"

    def test_1fps(self) -> None:
        """One tick per second → 1 FPS."""
        t = [0.0, 1.0]
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = t
            fps = FPSCounter(window=5)
            fps.tick()
            fps.tick()
            result = fps.get()

        assert abs(result - 1.0) < 0.001

    def test_rolling_window_discards_old_samples(self) -> None:
        """After the window fills the oldest sample is dropped."""
        # window=2: only the last 2 intervals matter.
        # Ticks at t=0, 1, 2, 3 → deltas 1,1,1 → window keeps last 2 → FPS=1.
        # Then tick at t=3.1 → delta 0.1 → window=[1, 0.1] → avg=0.55 → FPS≈1.82
        times = [0.0, 1.0, 2.0, 3.0, 3.1]
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = times
            fps = FPSCounter(window=2)
            for _ in times:
                fps.tick()
            result = fps.get()

        # avg of [1.0, 0.1] = 0.55 → fps ≈ 1.818
        expected = 1.0 / ((1.0 + 0.1) / 2)
        assert abs(result - expected) < 0.01


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

class TestFPSCounterReset:
    def test_reset_clears_data(self) -> None:
        times = [0.0, 0.1, 0.2]
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = times
            fps = FPSCounter(window=10)
            fps.tick()
            fps.tick()
            fps.tick()

        fps.reset()
        assert fps.get() == 0.0
        assert fps.frame_count == 0

    def test_tick_after_reset_works(self) -> None:
        times_before = [0.0, 0.1]
        times_after = [1.0, 1.05]  # 20 FPS
        with patch("webcam_ai.fps.time") as mock_time:
            mock_time.perf_counter.side_effect = times_before + times_after
            fps = FPSCounter(window=5)
            fps.tick()
            fps.tick()
            fps.reset()
            fps.tick()
            fps.tick()
            result = fps.get()

        assert abs(result - 20.0) < 0.1
