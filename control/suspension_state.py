"""
control/suspension_state.py
============================
Pure data models for the suspension control subsystem.

Keeping data structures in a separate file means:
- SkyhookController can import them without circular dependencies.
- Other modules (visualizer, frame_processor) can use the types
  without importing the controller.

Contents
--------
SuspensionState  : physical state of one suspension corner
RoadSeverity     : enum of four disturbance levels
classify_severity: maps a 0-100 score to a RoadSeverity value
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Physical state ─────────────────────────────────────────────────────────────

@dataclass
class SuspensionState:
    """
    Instantaneous dynamic state of one suspension corner.

    All values are in SI units.

    Attributes
    ----------
    z_s     : float — Sprung mass (vehicle body) vertical displacement (m)
    z_s_dot : float — Sprung mass vertical velocity (m/s)
    z_u     : float — Unsprung mass (wheel + axle) vertical displacement (m)
    z_u_dot : float — Unsprung mass vertical velocity (m/s)
    """

    z_s: float = 0.0
    z_s_dot: float = 0.0
    z_u: float = 0.0
    z_u_dot: float = 0.0

    @property
    def relative_velocity(self) -> float:
        """
        Relative velocity across the damper (sprung minus unsprung).

        Positive = suspension compressing.
        """
        return self.z_s_dot - self.z_u_dot

    def reset(self):
        """Zero all state variables (e.g. at simulation start)."""
        self.z_s = self.z_s_dot = self.z_u = self.z_u_dot = 0.0


# ── Severity classification ────────────────────────────────────────────────────

class RoadSeverity(Enum):
    """Categorical severity level of a road disturbance."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def color_bgr(self) -> tuple:
        """
        OpenCV BGR colour for each severity level.
        Used by the visualizer module.
        """
        _colors = {
            RoadSeverity.LOW:      (0,   255, 0  ),   # green
            RoadSeverity.MEDIUM:   (0,   255, 255),   # yellow
            RoadSeverity.HIGH:     (0,   165, 255),   # orange
            RoadSeverity.CRITICAL: (0,   0,   255),   # red
        }
        return _colors[self]


def classify_severity(score: int) -> RoadSeverity:
    """
    Map a 0–100 severity score to a RoadSeverity enum value.

    Boundaries
    ----------
    0  – 24  → LOW
    25 – 49  → MEDIUM
    50 – 74  → HIGH
    75 – 100 → CRITICAL

    Parameters
    ----------
    score : int
        Severity score (0–100).

    Returns
    -------
    RoadSeverity
    """
    if score < 25:
        return RoadSeverity.LOW
    elif score < 50:
        return RoadSeverity.MEDIUM
    elif score < 75:
        return RoadSeverity.HIGH
    else:
        return RoadSeverity.CRITICAL
