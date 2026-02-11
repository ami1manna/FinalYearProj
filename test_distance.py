#!/usr/bin/env python3
"""
Test Distance Estimation Alone
==============================
Tests the IPM distance estimation with pothole detection.
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pothole_detector import PotholeDetector
from ipm_distance import IPMDistanceEstimator, create_default_camera_matrix

def test_distance_estimation():
    """Test distance estimation on video frames."""
    
    print("=== Distance Estimation Test ===\n")
    
    # Configuration
    VIDEO_PATH = "data/videos/road.mp4"
    MODEL_PATH = "models/pothole.pt"
    
    # Camera configuration (use your calibrated values)
    camera_config = {
        'width': 640,
        'height': 384,
        'focal_length': 640,  # pixels
        'camera_height': 1.2,  # meters
        'pitch_angle': 15,     # degrees
        'fps': 30
    }
    
    print("1. Initializing models...")
    detector = PotholeDetector(MODEL_PATH, conf=0.25)
    camera_matrix = create_default_camera_matrix(
        camera_config['width'], 
        camera_config['height'], 
        camera_config['focal_length']
    )
    dist_coeffs = np.zeros((4, 1))  # No distortion assumed
    distance_estimator = IPMDistanceEstimator(
        camera_matrix, 
        dist_coeffs, 
        camera_config['camera_height'], 
        camera_config['pitch_angle']
    )
    print("   ✓ Models initialized")
    
    print("2. Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    
    # Process first 10 frames for testing
    frame_count = 0
    max_frames = 10
    
    print(f"3. Processing {max_frames} frames...\n")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to match camera config
        frame = cv2.resize(frame, (camera_config['width'], camera_config['height']))
        
        # Detect potholes
        _, results = detector.detect_frame(frame)
        
        if len(results.boxes) > 0:
            print(f"Frame {frame_count}: Found {len(results.boxes)} pothole(s)")
            
            for i, box in enumerate(results.boxes):
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Estimate distance using bottom-center of bounding box
                distance_m = distance_estimator.bbox_bottom_distance(x1, y1, x2, y2)
                
                # Calculate severity based on depth (placeholder)
                severity = min(100, int(confidence * 100))
                
                print(f"  Pothole {i+1}:")
                print(f"    BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                print(f"    Confidence: {confidence:.2f}")
                print(f"    Center: ({cx}, {cy})")
                print(f"    Distance: {distance_m:.1f}m")
                print(f"    Severity: {severity}/100")
                
                # Validation check
                if distance_m < 0 or distance_m > 100:
                    print(f"    ⚠️  WARNING: Distance seems unrealistic ({distance_m:.1f}m)")
                elif distance_m < 5:
                    print(f"    ℹ️  Very close pothole ({distance_m:.1f}m)")
                elif distance_m > 50:
                    print(f"    ℹ️  Distant pothole ({distance_m:.1f}m)")
                else:
                    print(f"    ✅ Reasonable distance ({distance_m:.1f}m)")
                
                print()
        else:
            print(f"Frame {frame_count}: No potholes detected")
    
    cap.release()
    
    print("=== Test Complete ===")
    print("\nValidation Guidelines:")
    print("- Distances should be in 5-50m range for typical detection")
    print("- If distances are >100m or negative, check focal_length calibration")
    print("- If all distances are very similar, check pitch_angle calibration")
    print("- Use lane markings (3m spacing) for ground truth validation")

if __name__ == "__main__":
    test_distance_estimation()
