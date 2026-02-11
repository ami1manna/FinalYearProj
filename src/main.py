import sys
import os
import cv2
import numpy as np
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pothole_detector import PotholeDetector
from midas.midas_utils import MiDaSDepthEstimator



def compute_relative_depth(depth_map, box, margin=30):
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    pothole_depth = np.mean(depth_map[y1:y2, x1:x2])

    # Road region ONLY above pothole
    ry1 = max(0, y1 - margin)
    ry2 = y1
    if ry2 <= ry1:
        return 0.0

    road_depth = np.mean(depth_map[ry1:ry2, x1:x2])

    return abs(road_depth - pothole_depth)


def severity_score(d, min_d=0.02, max_d=0.18):
    """
    Converts MiDaS relative depth into stable 0–100 score
    """
    d = max(min(d, max_d), min_d)
    return int(100 * (d - min_d) / (max_d - min_d))


def classify_severity(score):
    if score < 25:
        return "LOW"
    elif score < 50:
        return "MEDIUM"
    elif score < 75:
        return "HIGH"
    else:
        return "CRITICAL"


def severity_color(sev):
    return {
        "LOW": (0, 255, 0),
        "MEDIUM": (0, 255, 255),
        "HIGH": (0, 165, 255),
        "CRITICAL": (0, 0, 255)
    }[sev]


# -------------------------------------------------
# VIDEO PIPELINE
# -------------------------------------------------
if __name__ == "__main__":

    VIDEO_PATH = "data/videos/road.mp4"
    MODEL_PATH = "models/pothole.pt"

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    detector = PotholeDetector(MODEL_PATH, conf=0.4)
    depth_estimator = MiDaSDepthEstimator()

    frame_id = 0
    depth_map = None
    DEPTH_INTERVAL = 5

    # Temporal buffer for stability
    score_buffer = deque(maxlen=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (640, 384))

        output, results = detector.detect_frame(frame)

        if depth_map is None or frame_id % DEPTH_INTERVAL == 0:
            depth_map = depth_estimator.estimate_depth(frame)

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            rel_depth = compute_relative_depth(depth_map, (x1, y1, x2, y2))
            score = severity_score(rel_depth)

            score_buffer.append(score)
            avg_score = int(sum(score_buffer) / len(score_buffer))

            severity = classify_severity(avg_score)
            color = severity_color(severity)

            print(f"ΔDepth={rel_depth:.4f} | Score={avg_score}")

            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                output,
                f"{severity} ({avg_score})",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        cv2.imshow("Pothole Severity (Stable)", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()