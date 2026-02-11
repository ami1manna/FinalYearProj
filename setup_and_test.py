#!/usr/bin/env python3
"""
Complete Setup and Testing Script
=================================
Guides through the complete setup and testing process.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    
    print("=== Checking Dependencies ===\n")
    
    # Map package names to import names
    package_imports = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'timm': 'timm',
        'pillow': 'PIL'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_files():
    """Check if all required files are present."""
    
    print("\n=== Checking Required Files ===\n")
    
    required_files = [
        'data/videos/road.mp4',
        'models/pothole.pt',
        'src/pothole_detector.py',
        'src/depth_estimator.py',
        'midas/midas_utils.py',
        'ipm_distance.py',
        'speed_estimator.py',
        'skyhook_controller.py',
        'adaptive_suspension_main.py',
        'quarter_car_sim.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)} files")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def run_test(test_name, test_file):
    """Run a test script and capture output."""
    
    print(f"\n=== Running {test_name} ===\n")
    
    try:
        # Try to run the test
        result = subprocess.run([
            'pothole-env\\Scripts\\python.exe', test_file
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Test failed")
            if result.stderr.strip():
                print("Error:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (60s)")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    
    return True

def main():
    """Main setup and testing workflow."""
    
    print("🚗 Adaptive Suspension System - Setup & Testing")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Step 2: Check files
    if not check_files():
        print("\n❌ Please ensure all required files are present")
        return
    
    # Step 3: Interactive calibration
    print("\n" + "=" * 50)
    print("📐 Camera Calibration")
    print("=" * 50)
    
    calibrate = input("\nHave you calibrated your camera parameters? (y/n): ").strip().lower()
    
    if calibrate != 'y':
        print("\nRunning camera calibration...")
        run_test("Camera Calibration", "calibrate_camera.py")
    else:
        print("✓ Using existing camera_config.txt")
    
    # Step 4: Component testing
    print("\n" + "=" * 50)
    print("🧪 Component Testing")
    print("=" * 50)
    
    tests = [
        ("Distance Estimation", "test_distance.py"),
        ("Speed Estimation", "test_speed.py"),
        ("Quarter Car Simulation", "quarter_car_sim.py")
    ]
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            success = run_test(test_name, test_file)
            if not success:
                print(f"\n⚠️  {test_name} test failed - check configuration")
        else:
            print(f"\n⚠️  Test file {test_file} not found")
    
    # Step 5: Full system test
    print("\n" + "=" * 50)
    print("🎯 Full System Test")
    print("=" * 50)
    
    run_full = input("\nRun full adaptive suspension system? (y/n): ").strip().lower()
    
    if run_full == 'y':
        print("\nStarting full system...")
        print("Press 'q' to quit the video display")
        run_test("Full System", "adaptive_suspension_main.py")
    
    # Step 6: Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Review test results above")
    print("2. If distance/speed estimates are inaccurate, re-run calibration")
    print("3. Run the full system with: python adaptive_suspension_main.py")
    print("4. Check the generated suspension_comparison.png from simulation")
    
    print("\n📖 For detailed guidance, see:")
    print("- camera_config.txt (your calibration values)")
    print("- test_distance.py output (distance validation)")
    print("- test_speed.py output (speed validation)")
    print("- suspension_comparison.png (controller performance)")

if __name__ == "__main__":
    main()
