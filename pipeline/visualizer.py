"""
pipeline/visualizer.py
=======================
All OpenCV drawing and overlay code in one place.

Responsibilities
----------------
- Draw bounding boxes with severity-coded colours.
- Render HUD text (speed, frame count).
- Draw the suspension damping level bar.
- Print per-pothole console log lines.

This module contains ZERO business logic. It only reads FrameResult
dicts produced by FrameProcessor and annotates a BGR image.

Usage
-----
    from pipeline import Visualizer

    viz = Visualizer()
    annotated = viz.draw(frame_result)
    cv2.imshow("Output", annotated)
    viz.log(frame_result)   # optional console output
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any

from control import RoadSeverity


class Visualizer:
    """
    Stateless frame annotator.

    All methods are pure: they take a FrameResult dict and a frame
    and return an annotated copy. No internal state is maintained.
    """

    # ── Style constants ────────────────────────────────────────────────
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _HUD_COLOR = (255, 255, 255)   # white
    _BAR_BG    = (50, 50, 50)      # dark grey background
    _BAR_MAX_W = 300               # pixels
    _BAR_H     = 20
    _C_MIN     = 800.0
    _C_MAX     = 4000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(self, result: Dict[str, Any]) -> np.ndarray:
        """
        Produce a fully annotated copy of the frame.

        Runs every annotation layer in order:
        1. Bounding boxes with severity labels
        2. HUD text (speed, frame counter)
        3. Suspension damping bar

        Parameters
        ----------
        result : dict
            FrameResult from FrameProcessor.process().

        Returns
        -------
        annotated : np.ndarray
            Annotated BGR frame (copy of result["frame"]).
        """
        canvas = result["frame"].copy()

        self._draw_bboxes(canvas, result["potholes"])
        self._draw_hud(canvas, result)
        self._draw_damping_bar(canvas, result["potholes"])

        return canvas

    def log(self, result: Dict[str, Any], frame_id: int = 0):
        """
        Print one console line per detected pothole.

        Parameters
        ----------
        result : dict   FrameResult.
        frame_id : int  Frame counter (for context).
        """
        for p in result["potholes"]:
            print(
                f"  Pothole | {p['severity'].name:8s} ({p['score']:3d}/100)"
                f" | dist={p['distance_m']:5.1f}m"
                f" | t={p['t_impact']:.2f}s"
                f" | c={p['damping_coeff']:.0f} Ns/m"
                f" | conf={p['confidence']:.2f}"
            )

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_bboxes(self, canvas: np.ndarray, potholes: list):
        """
        Draw a bounding box and label for each detected pothole.

        Colour is determined by RoadSeverity.color_bgr.
        Label format: "P{score} {SEVERITY} {distance}m"
        """
        for p in potholes:
            x1, y1, x2, y2 = map(int, p["bbox"])
            color = p["severity"].color_bgr

            # Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"P{p['score']} {p['severity'].name} {p['distance_m']:.1f}m"
            (lw, lh), _ = cv2.getTextSize(label, self._FONT, 0.5, 1)
            cv2.rectangle(
                canvas,
                (x1, y1 - lh - 10),
                (x1 + lw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                canvas, label,
                (x1 + 2, y1 - 6),
                self._FONT, 0.5, (0, 0, 0), 1,
            )

    def _draw_hud(self, canvas: np.ndarray, result: Dict[str, Any]):
        """
        Render top-left HUD: speed and pothole count.
        """
        lines = [
            f"Speed : {result['speed_kmh']:5.1f} km/h",
            f"Potholes: {len(result['potholes'])}",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                canvas, line,
                (10, y), self._FONT, 0.65, self._HUD_COLOR, 2,
            )
            y += 28

    def _draw_damping_bar(self, canvas: np.ndarray, potholes: list):
        """
        Draw a colour-coded horizontal bar showing suspension damping level.

        - Green  : c < 1500  (soft)
        - Yellow : 1500 ≤ c < 2500  (medium)
        - Red    : c ≥ 2500  (hard)

        Bar is anchored to the bottom-left of the frame.
        """
        h, w = canvas.shape[:2]

        # Use most severe pothole's c, else minimum
        c = max(
            (p["damping_coeff"] for p in potholes),
            default=self._C_MIN,
        )

        bar_x = 30
        bar_y = h - 55
        fill_w = int(np.clip(
            (c - self._C_MIN) / (self._C_MAX - self._C_MIN),
            0.0, 1.0
        ) * self._BAR_MAX_W)

        # Colour zones
        if c < 1500:
            bar_color = (0, 220, 0)     # green
        elif c < 2500:
            bar_color = (0, 220, 220)   # yellow
        else:
            bar_color = (0, 60, 220)    # red

        # Background track
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + self._BAR_MAX_W, bar_y + self._BAR_H),
            self._BAR_BG, -1,
        )

        # Filled portion
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x, bar_y),
                (bar_x + fill_w, bar_y + self._BAR_H),
                bar_color, -1,
            )

        # Border
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + self._BAR_MAX_W, bar_y + self._BAR_H),
            self._HUD_COLOR, 1,
        )

        # Labels
        cv2.putText(
            canvas, "Suspension",
            (bar_x, bar_y - 8), self._FONT, 0.5, self._HUD_COLOR, 1,
        )
        cv2.putText(
            canvas, f"c = {c:.0f} Ns/m",
            (bar_x, bar_y + self._BAR_H + 16),
            self._FONT, 0.55, self._HUD_COLOR, 1,
        )
