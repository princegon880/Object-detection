"""
FPS (Frames Per Second) counter using a rolling-window average.

Computing FPS as 1 / single-frame-time produces values that jump around
wildly with even small jitter.  Instead we maintain a deque of recent
frame durations and average them, giving a smooth, stable reading while
still reacting reasonably quickly to genuine speed changes.
"""

import time
from collections import deque


class FPSCounter:
    """Tracks processing speed with a rolling-window average.

    Usage::

        fps = FPSCounter(window=30)
        while True:
            fps.tick()
            frame = capture()
            process(frame)
            print(fps.get())

    Args:
        window: Number of recent frame-times to include in the average.
            Larger values are smoother; smaller values react faster.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        self._window = window
        self._times: deque[float] = deque(maxlen=window)
        self._last_tick: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Record the current moment as the end of one frame.

        Call this once per processed frame **before** reading :meth:`get`.
        The very first call seeds the timer; FPS is only meaningful from the
        second call onwards.
        """
        now = time.perf_counter()
        if self._last_tick is not None:
            elapsed = now - self._last_tick
            if elapsed > 0:
                self._times.append(elapsed)
        self._last_tick = now

    def get(self) -> float:
        """Return the current rolling-average FPS.

        Returns:
            Smoothed FPS value, or ``0.0`` if fewer than two ticks have been
            recorded (to avoid division-by-zero on startup).
        """
        if not self._times:
            return 0.0
        avg_elapsed = sum(self._times) / len(self._times)
        if avg_elapsed <= 0:
            return 0.0
        return 1.0 / avg_elapsed

    def reset(self) -> None:
        """Clear all recorded frame times and restart measurement."""
        self._times.clear()
        self._last_tick = None

    @property
    def frame_count(self) -> int:
        """Number of frame intervals stored in the current window."""
        return len(self._times)
