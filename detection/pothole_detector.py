"""
detection/pothole_detector.py
==============================
YOLOv8 pothole detection wrapper.

Responsibilities
----------------
- Load custom-trained YOLOv8 weights once at startup.
- Expose two clean methods: one for images, one for live video frames.
- Return raw Ultralytics Results objects so callers can access boxes,
  confidences, and class IDs without knowing YOLO internals.

Usage
-----
    from detection import PotholeDetector

    detector = PotholeDetector("models/pothole.pt", conf=0.4)
    frame_copy, results = detector.detect_frame(bgr_frame)
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])
"""

import cv2
import numpy as np
from ultralytics import YOLO


class PotholeDetector:
    """
    Thin wrapper around Ultralytics YOLOv8 for pothole detection.

    Parameters
    ----------
    model_path : str
        Path to the .pt weights file (e.g. 'models/pothole.pt').
    conf : float, optional
        Minimum confidence threshold for a detection to be kept.
        Default is 0.25; recommended 0.4 for this model.
    device : str, optional
        Inference device — 'cpu' or '0' for first CUDA GPU.
    """

    def __init__(self, model_path: str, conf: float = 0.25, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        print(f"[PotholeDetector] Loaded model: {model_path}")
        print(f"[PotholeDetector] Classes: {self.model.names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_image(self, image_path: str):
        """
        Run detection on a single image file.

        Parameters
        ----------
        image_path : str
            Path to the image on disk.

        Returns
        -------
        image : np.ndarray
            Copy of the loaded BGR image (unmodified).
        results : ultralytics.engine.results.Results
            YOLO results object. Access detections via results.boxes.

        Raises
        ------
        ValueError
            If the image cannot be loaded (bad path or corrupt file).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"[PotholeDetector] Cannot load image: {image_path}")

        results = self._run_inference(image)
        return image.copy(), results

    def detect_frame(self, frame: np.ndarray):
        """
        Run detection on a single BGR video frame (real-time path).

        Parameters
        ----------
        frame : np.ndarray
            BGR image array from cv2.VideoCapture or similar.

        Returns
        -------
        frame_copy : np.ndarray
            Unmodified copy of the input frame.
        results : ultralytics.engine.results.Results
            YOLO results object. Access detections via results.boxes.
        """
        results = self._run_inference(frame)
        return frame.copy(), results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(self, source):
        """
        Shared inference call used by both public methods.

        Parameters
        ----------
        source : np.ndarray or str
            Image array or file path accepted by YOLO.predict().

        Returns
        -------
        results : ultralytics.engine.results.Results
            First element of the results list returned by YOLO.
        """
        results = self.model.predict(
            source=source,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        return results[0]
