"""
pipeline/frame_processor.py
============================
Per-frame processing pipeline — zero drawing code lives here.

Responsibilities
----------------
- Accept a raw BGR frame.
- Run every subsystem in order: detect → depth → distance → speed → control.
- Return a structured FrameResult dict so that the Visualizer and the
  main loop can consume data without knowing internal module details.

FrameResult schema
------------------
{
    "frame"      : np.ndarray        # resized BGR frame (unmodified)
    "speed_mps"  : float             # estimated vehicle speed
    "speed_kmh"  : float             # same, in km/h
    "potholes"   : list[PotholeData] # one entry per detected pothole
}

PotholeData schema
------------------
{
    "bbox"          : (x1, y1, x2, y2)  # pixel coordinates
    "confidence"    : float              # YOLO confidence [0, 1]
    "score"         : int                # severity [0, 100]
    "severity"      : RoadSeverity       # enum
    "distance_m"    : float              # metres to pothole
    "t_impact"      : float              # seconds until impact
    "damping_coeff" : float              # scheduled c (N·s/m)
    "force"         : float              # Skyhook force (N)
}

Usage
-----
    processor = FrameProcessor(camera_cfg, system_cfg, controller_cfg)
    result = processor.process(bgr_frame)
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import Optional

from config import CameraConfig, SystemConfig, ControllerConfig
from detection import PotholeDetector
from depth import MiDaSEstimator
from distance import IPMEstimator
from speed import OpticalFlowSpeed
from control import SkyhookController, SuspensionState, classify_severity


class FrameProcessor:
    """
    Stateful per-frame processing pipeline.

    Holds all subsystem instances and inter-frame state (buffers, frame counter).

    Parameters
    ----------
    cam_cfg  : CameraConfig
    sys_cfg  : SystemConfig
    ctrl_cfg : ControllerConfig
    model_path : str
        Path to YOLO weights file.
    """

    def __init__(
        self,
        cam_cfg: CameraConfig,
        sys_cfg: SystemConfig,
        ctrl_cfg: ControllerConfig,
        model_path: str,
    ):
        self._sys = sys_cfg
        self._frame_id = 0

        # ── Subsystems ──────────────────────────────────────────────────
        self._detector = PotholeDetector(model_path, conf=sys_cfg.detection_conf)

        self._depth_est = MiDaSEstimator() if sys_cfg.use_real_depth else None

        self._ipm = IPMEstimator.from_config(cam_cfg)

        self._speed_est = OpticalFlowSpeed(
            ipm=self._ipm,
            fps=cam_cfg.fps,
            buffer_size=sys_cfg.speed_buffer_len,
        )

        self._controller = SkyhookController(ctrl_cfg)

        # Shared suspension state — updated per frame (demo / simulation)
        self._susp_state = SuspensionState()

        # Rolling buffers
        self._score_buf: deque = deque(maxlen=sys_cfg.score_buffer_len)

        # Cached depth map reused between MiDaS intervals
        self._depth_map: Optional[np.ndarray] = None

        print("[FrameProcessor] All subsystems initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_frame: np.ndarray) -> dict:
        """
        Run the complete pipeline on one video frame.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR frame from cv2.VideoCapture (any resolution).

        Returns
        -------
        result : dict
            FrameResult as described in the module docstring.
        """
        self._frame_id += 1

        # 1 ── Resize to standard processing resolution
        frame = cv2.resize(
            raw_frame,
            (self._sys.resize_width, self._sys.resize_height),
        )

        # 2 ── Pothole detection
        _, det_results = self._detector.detect_frame(frame)

        # 3 ── Speed estimation
        speed_mps = self._speed_est.update(frame) or 0.0

        # 4 ── Depth map (throttled or placeholder)
        depth_map = self._get_depth_map(frame)

        # 5 ── Per-pothole processing
        potholes = self._process_detections(
            det_results, depth_map, speed_mps
        )

        return {
            "frame":      frame,
            "speed_mps":  speed_mps,
            "speed_kmh":  speed_mps * 3.6,
            "potholes":   potholes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a depth map for the current frame.

        - If use_real_depth is False: returns a constant placeholder map.
        - If use_real_depth is True: runs MiDaS every depth_interval frames,
          caches and reuses the result in between.

        Parameters
        ----------
        frame : np.ndarray   Resized BGR frame.

        Returns
        -------
        depth_map : np.ndarray, shape (H, W), float32
        """
        if not self._sys.use_real_depth:
            return MiDaSEstimator.placeholder_map(
                self._sys.resize_height, self._sys.resize_width
            )

        if (
            self._depth_map is None
            or self._frame_id % self._sys.depth_interval == 0
        ):
            self._depth_map = self._depth_est.estimate(frame)

        return self._depth_map

    def _process_detections(
        self,
        results,
        depth_map: np.ndarray,
        speed_mps: float,
    ) -> list:
        """
        Build PotholeData dicts for every detection above confidence threshold.

        Limited to SystemConfig.max_potholes per frame for performance.

        Parameters
        ----------
        results : ultralytics Results
        depth_map : np.ndarray
        speed_mps : float

        Returns
        -------
        potholes : list[dict]
        """
        potholes = []

        if results.boxes is None:
            return potholes

        for i, box in enumerate(results.boxes):
            if i >= self._sys.max_potholes:
                break

            conf = float(box.conf[0])
            if conf < self._sys.detection_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = (float(x1), float(y1), float(x2), float(y2))

            # ── Severity score ─────────────────────────────────────────
            if self._sys.use_real_depth:
                rel_d = self._depth_est.relative_depth(depth_map, bbox)
                raw_score = self._depth_est.depth_to_score(
                    rel_d,
                    self._sys.depth_score_min,
                    self._sys.depth_score_max,
                )
            else:
                raw_score = int(conf * 100)   # confidence proxy

            self._score_buf.append(raw_score)
            smoothed_score = int(np.mean(self._score_buf))

            # ── Distance ───────────────────────────────────────────────
            distance_m = self._ipm.bbox_distance(*bbox) or (10.0 + i * 5.0)

            # ── Control ────────────────────────────────────────────────
            F, c = self._controller.compute(
                state=self._susp_state,
                severity=smoothed_score,
                speed_mps=speed_mps,
                distance_m=distance_m,
            )

            # ── Time to impact ─────────────────────────────────────────
            t_impact = distance_m / max(speed_mps, 0.1)

            potholes.append(
                {
                    "bbox":          bbox,
                    "confidence":    conf,
                    "score":         smoothed_score,
                    "severity":      classify_severity(smoothed_score),
                    "distance_m":    distance_m,
                    "t_impact":      t_impact,
                    "damping_coeff": c,
                    "force":         F,
                }
            )

        return potholes
