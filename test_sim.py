#!/usr/bin/env python3
"""Test script for quarter car simulation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from quarter_car_sim import compare_controllers, generate_pothole
    print("✓ Successfully imported simulation modules")
    
    # Define a simple pothole
    depth = 0.05  # 5cm deep
    width = 0.6   # 60cm wide
    severity = 0.7
    
    print("✓ Generating pothole profile...")
    road = generate_pothole(depth, width, severity)
    
    # Define severity preview
    def severity_sched(t, t_pothole=10.0/15.0, preview_window=0.3):
        if t < t_pothole - preview_window:
            return 0.0
        elif t < t_pothole:
            return (t - (t_pothole - preview_window)) / preview_window
        else:
            return 1.0
    
    print("✓ Running controller comparison...")
    compare_controllers(road, severity_sched)
    print("✓ Simulation completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
