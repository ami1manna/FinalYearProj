"""
main.py
========
Entry point for the Adaptive Suspension System.

This file does exactly three things:
1. Configure the system (camera + system + controller params).
2. Initialise the pipeline.
3. Run the video loop.

All logic lives in the modules under config/, pipeline/, etc.
To change behaviour, edit the config dataclasses — not this file.

Run
---
    python main.py
    python main.py --video data/videos/road.mp4
    python main.py --video data/videos/road.mp4 --model models/pothole.pt
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from config import CameraConfig, SystemConfig, ControllerConfig
from pipeline import FrameProcessor, Visualizer


# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_VIDEO = "data/videos/road.mp4"
DEFAULT_MODEL = "models/pothole.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Suspension System")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path")
    return parser.parse_args()


def build_configs():
    """
    Central configuration.

    Edit values here (or load from a YAML/JSON file in future) to tune
    the system without touching any module files.
    """
    cam = CameraConfig(
        width=640,
        height=384,
        focal_length=640.0,    # ← calibrate for your camera
        camera_height=1.2,     # ← measure for your mount
        pitch_angle=15.0,      # ← measure for your mount
        fps=30.0,
    )

    sys = SystemConfig(
        detection_conf=0.4,
        max_potholes=3,
        depth_interval=15,
        use_real_depth=False,  # set True to enable MiDaS (needs GPU for realtime)
        speed_buffer_len=10,
        score_buffer_len=7,
    )

    ctrl = ControllerConfig(
        c_min=800.0,
        c_max=4000.0,
        v_max=25.0,
        ramp_window=0.300,
    )

    return cam, sys, ctrl


def run(video_path: str, model_path: str):
    """
    Main video processing loop.

    Parameters
    ----------
    video_path : str   Path to input video.
    model_path : str   Path to YOLO .pt weights.
    """
    cam_cfg, sys_cfg, ctrl_cfg = build_configs()

    # ── Initialise pipeline ────────────────────────────────────────────
    processor = FrameProcessor(cam_cfg, sys_cfg, ctrl_cfg, model_path)
    visualizer = Visualizer()

    # ── Open video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    print(f"\nProcessing: {video_path}")
    print("Press 'q' to quit.\n")

    fps_buf = deque(maxlen=30)
    frame_id = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Process ────────────────────────────────────────────────────
        result = processor.process(raw_frame)

        # ── Draw ───────────────────────────────────────────────────────
        annotated = visualizer.draw(result)

        # ── Log (console) ──────────────────────────────────────────────
        if result["potholes"]:
            visualizer.log(result, frame_id)

        # ── FPS overlay ────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        fps_buf.append(elapsed)
        fps = 1.0 / np.mean(fps_buf)
        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (annotated.shape[1] - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
        )

        # ── Display ────────────────────────────────────────────────────
        cv2.imshow("Adaptive Suspension System", annotated)

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  Frame {frame_id:5d} | FPS {fps:.1f} | Speed {result['speed_kmh']:.1f} km/h")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_id} frames.")


if __name__ == "__main__":
    args = parse_args()
    run(args.video, args.model)
