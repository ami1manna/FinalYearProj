import sys
import os
import cv2
import numpy as np

# -------------------------------------------------
# Fix Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pothole_detector import PotholeDetector
from midas.midas_utils import MiDaSDepthEstimator


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def compute_box_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float(np.mean(depth_map[y1:y2, x1:x2]))


def classify_severity(depth):
    if depth < 0.25:
        return "LOW"
    elif depth < 0.45:
        return "MEDIUM"
    elif depth < 0.65:
        return "HIGH"
    else:
        return "CRITICAL"


def severity_color(severity):
    return {
        "LOW": (0, 255, 0),
        "MEDIUM": (0, 255, 255),
        "HIGH": (0, 165, 255),
        "CRITICAL": (0, 0, 255)
    }[severity]


# -------------------------------------------------
# REAL-TIME VIDEO PIPELINE
# -------------------------------------------------
if __name__ == "__main__":

    VIDEO_PATH = "data/videos/road.mp4"   # <-- your video
    MODEL_PATH = "models/pothole.pt"

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    detector = PotholeDetector(MODEL_PATH, conf=0.25)
    depth_estimator = MiDaSDepthEstimator("MiDaS_small")

    frame_count = 0
    depth_map = None
    DEPTH_INTERVAL = 5  # run MiDaS every 5 frames (CPU optimization)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize for speed (IMPORTANT)
        frame = cv2.resize(frame, (640, 384))

        # YOLO detection
        output_frame, results = detector.detect_frame(frame)

        # MiDaS depth (not every frame)
        if depth_map is None or frame_count % DEPTH_INTERVAL == 0:
            depth_map = depth_estimator.estimate_depth(frame)

        # Draw detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            mean_depth = compute_box_depth(depth_map, (x1, y1, x2, y2))
            severity = classify_severity(mean_depth)
            color = severity_color(severity)

            cv2.rectangle(
                output_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            cv2.putText(
                output_frame,
                f"{severity} | {mean_depth:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        cv2.imshow("Real-Time Pothole Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
