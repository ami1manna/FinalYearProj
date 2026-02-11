#!/usr/bin/env python3
"""
Camera Calibration Helper
=========================
Helps calibrate camera parameters for accurate distance estimation.
"""

import sys
import os
import cv2
import numpy as np
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ipm_distance import calibrate_from_reference, create_default_camera_matrix

def measure_camera_parameters():
    """Interactive camera parameter measurement guide."""
    
    print("=== Camera Calibration Guide ===\n")
    
    print("This script will help you measure and calibrate your camera parameters.")
    print("You will need:\n")
    print("1. Tape measure (for camera height)")
    print("2. Smartphone with inclinometer app (for pitch angle)")
    print("3. Video with lane markings or known objects (for focal length)")
    print()
    
    input("Press Enter to continue...")
    
    print("\n=== Step 1: Camera Height Measurement ===")
    print("Instructions:")
    print("- Measure from the GROUND to the CENTER of your camera lens")
    print("- Use a tape measure for accuracy")
    print("- Typical dashcam: 1.0-1.5 meters")
    print()
    
    while True:
        try:
            height = float(input("Enter camera height in meters (e.g., 1.2): "))
            if 0.5 <= height <= 3.0:
                break
            else:
                print("Please enter a reasonable height (0.5-3.0 meters)")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\n✓ Camera height: {height} meters")
    
    print("\n=== Step 2: Pitch Angle Measurement ===")
    print("Instructions:")
    print("- Download an inclinometer app on your smartphone")
    print("- Place your phone flat against the camera mount")
    print("- Measure the angle from HORIZONTAL (downward is positive)")
    print("- Typical dashcam: 10-20 degrees")
    print()
    
    while True:
        try:
            angle = float(input("Enter pitch angle in degrees (e.g., 15): "))
            if -45 <= angle <= 45:
                break
            else:
                print("Please enter a reasonable angle (-45 to 45 degrees)")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\n✓ Pitch angle: {angle} degrees")
    
    print("\n=== Step 3: Focal Length Calibration ===")
    print("Choose your calibration method:")
    print("1. Use lane markings (recommended)")
    print("2. Use default (image width)")
    print("3. Enter custom value")
    print()
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    focal_length = None
    
    if choice == '1':
        print("\n--- Lane Marking Calibration ---")
        print("Instructions:")
        print("- Find a frame with clear lane markings")
        print("- Standard lane dash: 3 meters long")
        print("- Count pixels between lane dash ends")
        print()
        
        try:
            pixel_distance = float(input("Enter pixel distance between lane markings: "))
            real_distance = 3.0  # Standard lane marking length
            
            # Calculate focal length using similar triangles
            # f = (pixel_distance * camera_height) / (real_distance * sin(pitch_angle))
            focal_length = (pixel_distance * height) / (real_distance * math.sin(math.radians(angle)))
            
            print(f"\n✓ Calculated focal length: {focal_length:.0f} pixels")
            
        except ValueError:
            print("Invalid input, using default")
            focal_length = 640
    
    elif choice == '2':
        focal_length = 640  # Default to image width
        print(f"\n✓ Using default focal length: {focal_length} pixels")
    
    else:  # choice == '3'
        while True:
            try:
                focal_length = float(input("Enter focal length in pixels: "))
                if 100 <= focal_length <= 2000:
                    break
                else:
                    print("Please enter a reasonable focal length (100-2000 pixels)")
            except ValueError:
                print("Please enter a valid number")
    
    print("\n=== Calibration Summary ===")
    print(f"Camera height: {height} meters")
    print(f"Pitch angle: {angle} degrees")
    print(f"Focal length: {focal_length:.0f} pixels")
    print()
    
    # Save to config file
    config_content = f"""# Camera Configuration for Adaptive Suspension System
# Auto-generated calibration - {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Video/Image dimensions
width: 640
height: 384

# Camera intrinsic parameters
focal_length: {focal_length:.0f}  # pixels - CALIBRATED VALUE

# Camera extrinsic parameters
camera_height: {height}  # meters - MEASURED VALUE
pitch_angle: {angle}     # degrees - MEASURED VALUE

# Video settings
fps: 30

# Calibration notes:
# - Camera height measured from ground to lens center
# - Pitch angle measured from horizontal using inclinometer
# - Focal length calibrated using lane markings
# - Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('camera_config.txt', 'w') as f:
        f.write(config_content)
    
    print("✓ Configuration saved to camera_config.txt")
    
    print("\n=== Next Steps ===")
    print("1. Run 'python test_distance.py' to validate distance estimation")
    print("2. Run 'python test_speed.py' to validate speed estimation")
    print("3. Run 'python adaptive_suspension_main.py' for full system")
    print()
    
    # Test calibration
    print("=== Quick Calibration Test ===")
    camera_config = {
        'width': 640,
        'height': 384,
        'focal_length': focal_length,
        'camera_height': height,
        'pitch_angle': angle,
        'fps': 30
    }
    
    # Test distance estimation for typical scenarios
    camera_matrix = create_default_camera_matrix(
        camera_config['width'], 
        camera_config['height'], 
        camera_config['focal_length']
    )
    
    test_points = [
        (320, 200),  # Center bottom
        (320, 100),  # Center top
        (160, 200),  # Left bottom
        (480, 200),  # Right bottom
    ]
    
    print("\nTest distance estimates:")
    for x, y in test_points:
        # Simple distance calculation for testing
        # This is a simplified version of the IPM calculation
        y_rel = camera_config['height'] - y
        if y_rel > 0:
            distance = (camera_config['focal_length'] * camera_config['camera_height']) / y_rel
            print(f"  Point ({x}, {y}): {distance:.1f}m")
        else:
            print(f"  Point ({x}, {y}): Invalid (above horizon)")
    
    print("\n✓ Calibration complete!")

if __name__ == "__main__":
    measure_camera_parameters()
