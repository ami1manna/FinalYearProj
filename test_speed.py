#!/usr/bin/env python3
"""
Test Speed Estimation Alone
===========================
Tests the optical flow speed estimation.
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from speed_estimator import OpticalFlowSpeedEstimator
from ipm_distance import IPMDistanceEstimator, create_default_camera_matrix

def test_speed_estimation():
    """Test speed estimation on video."""
    
    print("=== Speed Estimation Test ===\n")
    
    # Configuration
    VIDEO_PATH = "data/videos/road.mp4"
    
    # Camera configuration
    camera_config = {
        'width': 640,
        'height': 384,
        'focal_length': 640,  # pixels
        'camera_height': 1.2,  # meters
        'pitch_angle': 15,     # degrees
        'fps': 30
    }
    
    print("1. Initializing speed estimator...")
    
    # Create IPM estimator first
    camera_matrix = create_default_camera_matrix(
        camera_config['width'], 
        camera_config['height'], 
        camera_config['focal_length']
    )
    dist_coeffs = np.zeros((4, 1))
    ipm_estimator = IPMDistanceEstimator(
        camera_matrix, 
        dist_coeffs, 
        camera_config['camera_height'], 
        camera_config['pitch_angle']
    )
    
    estimator = OpticalFlowSpeedEstimator(ipm_estimator, fps=camera_config['fps'])
    print("   [OK] Speed estimator initialized")
    
    print("2. Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    
    print("3. Processing video for speed estimation...\n")
    
    frame_count = 0
    speed_readings = []
    
    # Process first 100 frames to get stable readings
    max_frames = 100
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to match camera config
        frame = cv2.resize(frame, (camera_config['width'], camera_config['height']))
        
        # Estimate speed
        speed_mps = estimator.estimate_speed(frame)
        confidence = 1.0 if speed_mps is not None else 0.0
        
        if speed_mps is not None:
            speed_kmh = speed_mps * 3.6  # Convert to km/h
        
        if confidence > 0.3:  # Only use confident readings
            speed_readings.append(speed_kmh)
            
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: {speed_kmh:.1f} km/h (confidence: {confidence:.2f})")
        
        # Display frame with speed info (optional)
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames, {len(speed_readings)} valid speed readings")
    
    cap.release()
    
    print(f"\n=== Speed Analysis ===")
    
    if speed_readings:
        avg_speed = np.mean(speed_readings)
        std_speed = np.std(speed_readings)
        min_speed = np.min(speed_readings)
        max_speed = np.max(speed_readings)
        
        print(f"Total valid readings: {len(speed_readings)}")
        print(f"Average speed: {avg_speed:.1f} km/h")
        print(f"Speed std dev: {std_speed:.1f} km/h")
        print(f"Speed range: {min_speed:.1f} - {max_speed:.1f} km/h")
        
        # Validation
        print(f"\n=== Validation ===")
        
        if avg_speed < 5 or avg_speed > 120:
            print(f"⚠️  WARNING: Average speed seems unrealistic ({avg_speed:.1f} km/h)")
            print("   - Check if vehicle is actually moving in the video")
            print("   - Verify focal_length calibration")
        else:
            print(f"✅ Average speed seems reasonable ({avg_speed:.1f} km/h)")
        
        if std_speed > 20:
            print(f"⚠️  WARNING: High speed variability (std: {std_speed:.1f} km/h)")
            print("   - Optical flow may be unstable")
            print("   - Road may have insufficient features")
        else:
            print(f"✅ Speed estimation seems stable (std: {std_speed:.1f} km/h)")
        
        # Typical speed ranges
        if 20 <= avg_speed <= 80:
            print(f"✅ Speed within typical urban driving range")
        elif avg_speed > 80:
            print(f"ℹ️  High speed detected - possibly highway driving")
        else:
            print(f"ℹ️  Low speed detected - possibly traffic or residential area")
    
    else:
        print("❌ No valid speed readings obtained")
        print("   - Video may have insufficient optical flow features")
        print("   - Camera may be stationary")
        print("   - Try lowering confidence threshold")
    
    print(f"\n=== Recommendations ===")
    print("- Compare with known vehicle speed if available")
    print("- For testing, you can use a fixed speed (e.g., 50 km/h)")
    print("- Ensure video has sufficient texture for optical flow")

if __name__ == "__main__":
    test_speed_estimation()
