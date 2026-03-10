from __future__ import annotations
"""
distance/ipm_estimator.py
==========================
Inverse Perspective Mapping (IPM) distance estimator.

Responsibilities
----------------
- Convert image-plane pixel coordinates to real-world ground-plane
  coordinates (metres) using a flat-road assumption.
- Estimate the distance from the camera to any detected bounding box
  using the bottom-centre point (ground contact).
- Provide a classmethod to build an instance directly from CameraConfig.

Theory
------
For a calibrated camera mounted at height H with pitch angle θ:

    distance = (H * fx) / |v - cy_effective|

where  cy_effective = cy - fy * tan(θ)
and    (v) is the y-coordinate of the bottom-centre of the bounding box.

Usage
-----
    from distance import IPMEstimator
    from config import CameraConfig

    cam = CameraConfig()
    ipm = IPMEstimator.from_config(cam)

    distance_m = ipm.bbox_distance(x1, y1, x2, y2)
    X, Y, dist = ipm.pixel_to_ground(u, v)
"""

import numpy as np
import cv2

from typing import Optional, Tuple


class IPMEstimator:
    """
    Monocular distance estimator using Inverse Perspective Mapping.

    Assumes a flat road surface and a calibrated camera.

    Parameters
    ----------
    camera_matrix : np.ndarray, shape (3, 3)
        Camera intrinsic matrix K.
    dist_coeffs : np.ndarray
        Lens distortion coefficients. Pass zeros if uncalibrated.
    camera_height : float
        Camera height above the ground plane (metres).
    pitch_angle : float
        Camera downward tilt from horizontal (degrees, positive = down).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        camera_height: float,
        pitch_angle: float,
    ):
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_height = camera_height
        self.pitch_rad = np.deg2rad(pitch_angle)

        # Pre-compute effective horizon shift and homography
        self._cy_eff = self._compute_effective_cy()
        self._H_inv = self._compute_homography()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cam) -> "IPMEstimator":
        """
        Build an IPMEstimator directly from a CameraConfig instance.

        Parameters
        ----------
        cam : CameraConfig
            Populated camera configuration dataclass.

        Returns
        -------
        IPMEstimator
        """
        return cls(
            camera_matrix=cam.to_matrix(),
            dist_coeffs=cam.no_distortion(),
            camera_height=cam.camera_height,
            pitch_angle=cam.pitch_angle,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bbox_distance(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Optional[float]:
        """
        Estimate distance to an object using its bounding box.

        Uses the bottom-centre point of the box, which corresponds to
        the object's ground contact point.

        Parameters
        ----------
        x1, y1, x2, y2 : float
            Bounding box corner coordinates (pixels).

        Returns
        -------
        distance : float or None
            Distance in metres. None if the point is at/above the horizon.
        """
        u = (x1 + x2) / 2.0   # horizontal centre
        v = y2                  # bottom edge = ground contact

        fy = self.K[1, 1]

        dv = abs(v - self._cy_eff)
        if dv < 1e-6:
            return None   # point is on the horizon — undefined distance

        distance = (self.camera_height * self.K[0, 0]) / dv
        return float(distance)

    def pixel_to_ground(
        self, u: float, v: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Convert a single image pixel to ground-plane coordinates.

        Parameters
        ----------
        u, v : float
            Image coordinates (pixels).

        Returns
        -------
        X : float or None   — forward distance (metres, positive ahead)
        Y : float or None   — lateral offset (metres, positive right)
        dist : float or None — Euclidean distance from camera base (metres)
        """
        # Optional lens undistortion
        pts = np.array([[[u, v]]], dtype=np.float32)
        pts_ud = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        u_ud, v_ud = pts_ud[0, 0]

        p_img = np.array([u_ud, v_ud, 1.0])
        p_gnd = self._H_inv @ p_img

        w = p_gnd[2]
        if abs(w) < 1e-9:
            return None, None, None

        X = float(p_gnd[0] / w)
        Y = float(p_gnd[1] / w)
        dist = float(np.sqrt(X ** 2 + Y ** 2))

        return X, Y, dist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_effective_cy(self) -> float:
        """
        Shift the principal point y-coordinate to account for camera pitch.

        The horizon line in the image moves with camera tilt, so all
        distance calculations use this adjusted cy rather than the raw one.
        """
        fy = self.K[1, 1]
        cy = self.K[1, 2]
        return cy - fy * np.tan(self.pitch_rad)

    def _compute_homography(self) -> np.ndarray:
        """
        Compute the 3x3 inverse homography matrix (image → ground plane).

        Encodes camera height and focal length for fast per-pixel transforms.

        Returns
        -------
        H_inv : np.ndarray, shape (3, 3)
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        h = self.camera_height

        H_inv = np.array(
            [
                [ h / fx,    0.0,    -cx * h / fx          ],
                [ 0.0,       h / fy, -self._cy_eff * h / fy],
                [ 0.0,       0.0,    1.0                    ],
            ]
        )
        return H_inv
