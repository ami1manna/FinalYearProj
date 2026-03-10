"""
speed/optical_flow.py
======================
Vehicle speed estimation using sparse Lucas-Kanade optical flow.

How it works
------------
1. Detect Shi-Tomasi corners in the lower 50–90% of each frame
   (ground plane region — avoids sky, hood, and moving objects).
2. Track those corners into the next frame using Lucas-Kanade pyramidal flow.
3. For each successfully tracked point, convert BOTH (prev, curr) pixel
   positions to ground-plane metres via IPMEstimator.pixel_to_ground().
4. Take the median forward (X-axis) displacement across all tracked points.
5. speed = |median_displacement| / frame_dt
6. Smooth over a rolling buffer to suppress jitter.

Usage
-----
    from speed import OpticalFlowSpeed
    from distance import IPMEstimator
    from config import CameraConfig

    ipm = IPMEstimator.from_config(CameraConfig())
    estimator = OpticalFlowSpeed(ipm, fps=30)

    # Inside video loop:
    speed_mps = estimator.update(bgr_frame)          # may return None on first frame
    speed_kmh = estimator.speed_kmh                  # cached km/h property
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import Optional


class OpticalFlowSpeed:
    """
    Sparse optical flow vehicle speed estimator.

    Parameters
    ----------
    ipm : IPMEstimator
        A calibrated IPM instance used to convert pixels to metres.
    fps : float
        Camera frame rate (Hz). Used to compute per-frame time delta.
    buffer_size : int
        Number of speed samples to keep for median smoothing.
    """

    # Shi-Tomasi corner detection settings
    _FEATURE_PARAMS = dict(
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=7,
    )

    # Lucas-Kanade pyramidal optical flow settings
    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Ground-plane ROI: rows between these fractions of image height
    _ROI_TOP = 0.50    # start at mid-frame (exclude sky / far road)
    _ROI_BOT = 0.90    # stop near bottom  (exclude vehicle hood)

    # Minimum tracked features before re-detecting
    _MIN_FEATURES = 20

    def __init__(self, ipm, fps: float = 30.0, buffer_size: int = 10):
        self._ipm = ipm
        self._dt = 1.0 / fps
        self._buffer: deque = deque(maxlen=buffer_size)

        # Tracking state — reset on each new video
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._last_speed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> Optional[float]:
        """
        Process one frame and return an updated speed estimate.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR video frame.

        Returns
        -------
        speed_mps : float or None
            Estimated speed in metres/second.
            Returns None on the very first frame (no previous frame yet).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── First frame: initialise and return nothing ─────────────────
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            return None

        # ── Not enough features to track: re-detect ────────────────────
        if self._prev_pts is None or len(self._prev_pts) < 5:
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            return self._last_speed if self._buffer else None

        # ── Track features ─────────────────────────────────────────────
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._LK_PARAMS
        )

        good_prev = self._prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 5:
            self._reset(gray)
            return self._last_speed if self._buffer else None

        # ── Convert pixel displacements to ground-plane metres ──────────
        displacements = self._pixel_displacements_to_metres(good_prev, good_curr)

        if len(displacements) < 3:
            self._reset(gray)
            return self._last_speed if self._buffer else None

        # ── Median displacement → speed ─────────────────────────────────
        median_disp = float(np.median(displacements))
        speed_mps = abs(median_disp) / self._dt
        self._buffer.append(speed_mps)
        self._last_speed = float(np.median(self._buffer))

        # ── Update state ────────────────────────────────────────────────
        self._prev_gray = gray
        self._prev_pts = good_curr.reshape(-1, 1, 2)

        # Refresh features when count is low
        if len(self._prev_pts) < self._MIN_FEATURES:
            new_pts = self._detect_features(gray)
            if new_pts is not None:
                self._prev_pts = new_pts

        return self._last_speed

    def reset(self):
        """Clear all tracking state. Call when switching video sources."""
        self._prev_gray = None
        self._prev_pts = None
        self._buffer.clear()
        self._last_speed = 0.0

    @property
    def speed_kmh(self) -> float:
        """Last estimated speed converted to km/h."""
        return self._last_speed * 3.6

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect Shi-Tomasi corners in the ground-plane ROI.

        The ROI covers the lower portion of the frame where road texture
        is visible and optical flow reflects vehicle forward motion.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image.

        Returns
        -------
        points : np.ndarray or None
            Array of shape (N, 1, 2) with detected corner locations in
            full-frame coordinates. None if no corners found.
        """
        h, w = gray.shape
        y_top = int(h * self._ROI_TOP)
        y_bot = int(h * self._ROI_BOT)

        roi = gray[y_top:y_bot, :]
        pts = cv2.goodFeaturesToTrack(roi, mask=None, **self._FEATURE_PARAMS)

        if pts is not None:
            pts[:, 0, 1] += y_top   # shift y back to full-frame coords

        return pts

    def _pixel_displacements_to_metres(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
    ) -> list:
        """
        Convert tracked pixel pairs to forward ground-plane displacements.

        Parameters
        ----------
        prev_pts, curr_pts : np.ndarray, shape (N, 2)
            Matched pixel coordinates in previous and current frames.

        Returns
        -------
        displacements : list of float
            Forward (X-axis) displacement in metres for each tracked point.
        """
        displacements = []

        for p_prev, p_curr in zip(prev_pts, curr_pts):
            X_prev, _, _ = self._ipm.pixel_to_ground(p_prev[0], p_prev[1])
            X_curr, _, _ = self._ipm.pixel_to_ground(p_curr[0], p_curr[1])

            if X_prev is None or X_curr is None:
                continue

            displacements.append(X_curr - X_prev)

        return displacements

    def _reset(self, gray: np.ndarray):
        """Soft reset: keep gray frame but re-detect features."""
        self._prev_gray = gray
        self._prev_pts = self._detect_features(gray)
