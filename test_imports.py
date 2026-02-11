#!/usr/bin/env python3
"""Test all imports for the adaptive suspension system"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    print("1. Testing basic imports...")
    import cv2
    import numpy as np
    print("   ✓ OpenCV and NumPy imported")
    
    print("2. Testing YOLO...")
    from ultralytics import YOLO
    print("   ✓ YOLO imported")
    
    print("3. Testing pothole detector...")
    from pothole_detector import PotholeDetector
    print("   ✓ PotholeDetector imported")
    
    print("4. Testing MiDaS...")
    from midas.midas_utils import MiDaSDepthEstimator
    print("   ✓ MiDaS imported")
    
    print("5. Testing IPM distance...")
    from ipm_distance import IPMDistanceEstimator, create_default_camera_matrix
    print("   ✓ IPM distance estimator imported")
    
    print("6. Testing speed estimator...")
    from speed_estimator import OpticalFlowSpeedEstimator
    print("   ✓ Speed estimator imported")
    
    print("7. Testing skyhook controller...")
    from skyhook_controller import (
        GainScheduledSkyhookController, 
        SuspensionState,
        classify_severity
    )
    print("   ✓ Skyhook controller imported")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
