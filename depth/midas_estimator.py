"""
depth/midas_estimator.py
=========================
Intel MiDaS monocular depth estimation wrapper.

Responsibilities
----------------
- Download / cache MiDaS weights from PyTorch Hub on first use.
- Convert a BGR frame into a percentile-normalised inverse depth map.
- Provide a helper to compute the *relative* depth of a bounding box
  versus the surrounding road surface.

Notes
-----
- MiDaS produces *inverse* depth: higher value = closer to camera.
- Normalisation uses the 5th/95th percentile so extreme pixels do not
  skew the [0, 1] range.
- Real-time use on CPU is ~1–2 FPS; enable GPU or increase
  SystemConfig.depth_interval to compensate.

Usage
-----
    from depth import MiDaSEstimator

    estimator = MiDaSEstimator()
    depth_map = estimator.estimate(bgr_frame)        # shape (H, W), float32 in [0,1]
    rel_depth  = estimator.relative_depth(depth_map, (x1, y1, x2, y2))
    score      = estimator.depth_to_score(rel_depth)
"""

import numpy as np
import cv2
import torch


class MiDaSEstimator:
    """
    Monocular depth estimator using Intel MiDaS.

    Parameters
    ----------
    model_type : str
        MiDaS variant. Options:
        - 'MiDaS_small'   : fastest, lowest accuracy  (recommended for CPU)
        - 'DPT_Large'     : slowest, highest accuracy  (requires GPU)
    """

    def __init__(self, model_type: str = "MiDaS_small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        print(f"[MiDaSEstimator] Loaded {model_type} on {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a normalised inverse depth map from a BGR frame.

        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR format (H x W x 3).

        Returns
        -------
        depth : np.ndarray
            Float32 array, shape (H, W), values in [0, 1].
            1.0 = closest, 0.0 = furthest.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy()

        # Percentile normalisation — robust to outlier depths
        p5, p95 = np.percentile(depth, [5, 95])
        depth = np.clip((depth - p5) / (p95 - p5 + 1e-6), 0.0, 1.0)

        return depth.astype(np.float32)

    def relative_depth(
        self,
        depth_map: np.ndarray,
        box: tuple,
        road_margin: int = 30,
    ) -> float:
        """
        Compute how much deeper the pothole is vs the road above it.

        Compares the mean depth inside the bounding box against a
        reference strip of road immediately above the box.

        Parameters
        ----------
        depth_map : np.ndarray
            Normalised depth map from estimate().
        box : tuple
            Bounding box (x1, y1, x2, y2) in pixels.
        road_margin : int
            Height in pixels of the road reference strip above the box.

        Returns
        -------
        rel_depth : float
            |mean_pothole_depth - mean_road_depth|.
            Returns 0.0 if the box is too small or near the image edge.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_map.shape

        # Clamp to image bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        pothole_mean = float(np.mean(depth_map[y1:y2, x1:x2]))

        # Road reference — strip directly above the box
        ry1, ry2 = max(0, y1 - road_margin), y1
        if ry2 <= ry1:
            return 0.0

        road_mean = float(np.mean(depth_map[ry1:ry2, x1:x2]))

        return abs(pothole_mean - road_mean)

    def depth_to_score(
        self,
        rel_depth: float,
        min_d: float = 0.02,
        max_d: float = 0.18,
    ) -> int:
        """
        Map a relative depth value to an integer severity score [0, 100].

        Uses min-max normalisation clamped to [min_d, max_d].

        Parameters
        ----------
        rel_depth : float
            Raw relative depth from relative_depth().
        min_d : float
            Lower bound — anything below maps to 0.
        max_d : float
            Upper bound — anything above maps to 100.

        Returns
        -------
        score : int
            Severity score in the range [0, 100].
        """
        clamped = max(min(rel_depth, max_d), min_d)
        return int(100.0 * (clamped - min_d) / (max_d - min_d))

    @staticmethod
    def placeholder_map(height: int, width: int, value: float = 0.1) -> np.ndarray:
        """
        Return a constant-value depth map for use when MiDaS is disabled.

        Parameters
        ----------
        height, width : int
            Dimensions of the output map.
        value : float
            Fill value (default 0.1).

        Returns
        -------
        depth : np.ndarray, shape (height, width), float32
        """
        return np.full((height, width), value, dtype=np.float32)
