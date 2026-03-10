"""
config/camera_config.py
========================
All camera calibration and mounting parameters in one place.

To calibrate your camera:
  - focal_length : use lane markings (3 m apart) or camera specs (pixels)
  - camera_height: measure from ground to camera lens center (metres)
  - pitch_angle  : measure downtilt from horizontal (degrees, positive = looking down)

Typical dashcam values are provided as defaults.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CameraConfig:
    """
    Camera intrinsic and extrinsic parameters.

    All distance / speed calculations depend on accurate values here.
    Wrong focal_length is the #1 cause of bad distance estimates.
    """

    # ── Image dimensions ──────────────────────────────────────────────
    width: int = 640          # pixels
    height: int = 384         # pixels

    # ── Intrinsics ────────────────────────────────────────────────────
    focal_length: float = 640.0   # pixels  (update with your calibrated value)

    # ── Extrinsics ────────────────────────────────────────────────────
    camera_height: float = 1.2    # metres above ground
    pitch_angle: float = 15.0     # degrees, positive = looking downward

    # ── Video source ──────────────────────────────────────────────────
    fps: float = 30.0

    def to_matrix(self) -> np.ndarray:
        """
        Build the 3x3 camera intrinsic matrix K.

        Returns
        -------
        K : np.ndarray, shape (3, 3), dtype float32
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        cx = self.width / 2.0
        cy = self.height / 2.0
        K = np.array(
            [
                [self.focal_length, 0.0,              cx],
                [0.0,              self.focal_length,  cy],
                [0.0,              0.0,                1.0],
            ],
            dtype=np.float32,
        )
        return K

    def no_distortion(self) -> np.ndarray:
        """Return a zero distortion coefficient vector (assumes no lens distortion)."""
        return np.zeros((4, 1), dtype=np.float32)
