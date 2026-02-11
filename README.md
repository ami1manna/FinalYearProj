# Adaptive Suspension System with Computer Vision

An intelligent vehicle suspension system that detects potholes using computer vision and controls suspension damping in real-time for improved ride comfort and safety.

## 🚗 Overview

This system combines:
- **YOLO-based pothole detection** for identifying road defects
- **MiDaS depth estimation** for assessing pothole severity
- **Inverse Perspective Mapping (IPM)** for accurate distance calculation
- **Optical flow speed estimation** for vehicle velocity
- **Preview-based Skyhook control** for adaptive suspension damping

## 📁 Project Structure

```
FinalYearProj/
├── data/
│   ├── videos/
│   │   ├── road.mp4          # Input road video
│   │   └── output.mp4        # Processed output
│   └── images/                # Test images
├── models/
│   └── pothole.pt            # Trained YOLO model
├── src/
│   ├── pothole_detector.py   # YOLO detection module
│   ├── depth_estimator.py    # MiDaS depth estimation
│   └── main.py               # Original main file
├── midas/
│   └── midas_utils.py        # MiDaS utilities
├── pothole-env/              # Virtual environment
├── camera_config.txt         # Camera calibration parameters
├── adaptive_suspension_main.py # Main integrated system
├── ipm_distance.py           # Distance estimation
├── speed_estimator.py        # Speed estimation
├── skyhook_controller.py     # Suspension control
├── quarter_car_sim.py        # Simulation/validation
├── calibrate_camera.py       # Camera calibration tool
├── test_distance.py          # Distance testing
├── test_speed.py             # Speed testing
└── setup_and_test.py         # Complete setup guide
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment support

### Setup Steps

1. **Clone/Download the project**
2. **Create virtual environment:**
   ```bash
   python -m venv pothole-env
   pothole-env\Scripts\activate  # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies
```
torch
torchvision
ultralytics
opencv-python
numpy
matplotlib
timm
pillow
```

## 📐 Camera Calibration (CRITICAL)

Before running the system, you MUST calibrate your camera:

### Method 1: Automated Calibration
```bash
python calibrate_camera.py
```
Follow the interactive guide to measure:
- **Camera height**: Distance from ground to lens center (1.0-1.5m typical)
- **Pitch angle**: Camera angle from horizontal (10-20° typical)
- **Focal length**: Using lane markings or camera specs

### Method 2: Manual Configuration
Edit `camera_config.txt`:
```txt
width: 640
height: 384
focal_length: 640      # pixels - YOUR CALIBRATED VALUE
camera_height: 1.2      # meters - YOUR MEASUREMENT
pitch_angle: 15         # degrees - YOUR MEASUREMENT
fps: 30
```

## 🧪 Testing Components

### Complete Setup & Testing
```bash
python setup_and_test.py
```
This guides you through:
- Dependency checking
- File verification
- Camera calibration
- Component testing
- Full system validation

### Individual Component Tests

**Distance Estimation:**
```bash
python test_distance.py
```
Validates distance measurements are reasonable (5-50m range).

**Speed Estimation:**
```bash
python test_speed.py
```
Validates optical flow speed estimation.

**Suspension Simulation:**
```bash
python quarter_car_sim.py
```
Generates `suspension_comparison.png` showing controller performance.

## 🚀 Running the System

### Full Adaptive Suspension System
```bash
python adaptive_suspension_main.py
```

### Original Pothole Detection (Backup)
```bash
python src/main.py
```

## 📊 System Output

The main system displays:
- **Real-time video** with pothole bounding boxes
- **Distance estimates** for each detected pothole
- **Severity classification** (LOW/MEDIUM/HIGH/CRITICAL)
- **Damping coefficients** applied by controller
- **Vehicle speed** estimation
- **Console output** with control actions

Example console output:
```
Pothole: HIGH (75/100) | d=12.3m | t=0.89s | c=2500 N·s/m
FPS: 28.5
```

## 🔧 Configuration

### Camera Parameters
- **focal_length**: Most critical for distance accuracy
- **camera_height**: Affects distance scaling
- **pitch_angle**: Affects distance perspective

### Controller Parameters (skyhook_controller.py)
- **c_min/c_max**: Damping coefficient range (800-4000 N·s/m)
- **actuator_latency**: Hardware response delay
- **ramp_window**: Preview timing (300ms default)

## 📈 Performance Metrics

### Expected Ranges
- **Distance**: 5-50 meters (typical detection range)
- **Speed**: 20-80 km/h (urban driving)
- **Damping**: 800-4000 N·s/m (adjustable)
- **FPS**: 25-30 (real-time processing)

### Validation
- Use lane markings (3m spacing) for distance ground truth
- Compare speed estimates with known vehicle speed
- Check suspension_comparison.png for controller effectiveness

## 🐛 Troubleshooting

### Distance Issues
- **Problem**: All distances similar or unrealistic
- **Solution**: Re-calibrate focal_length using lane markings

### Speed Issues  
- **Problem**: No speed readings or erratic values
- **Solution**: Ensure video has sufficient texture for optical flow

### Detection Issues
- **Problem**: No potholes detected
- **Solution**: Check model path and confidence threshold

### Performance Issues
- **Problem**: Low FPS
- **Solution**: Reduce video resolution or use GPU acceleration

## 📚 Advanced Features

### Pothole Tracking (Optional)
Install tracking library:
```bash
pip install filterpy
```
Enables Kalman filtering for smoother distance estimates.

### Data Logging
The system can log:
- Timestamps
- Detection results
- Control actions
- Performance metrics

### Real-time Visualization
Add dashboard showing:
- Current speed
- Upcoming potholes
- Damping coefficient history
- Suspension state

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review test outputs
3. Verify camera calibration
4. Ensure all dependencies are installed

---

**Note**: This system requires accurate camera calibration for reliable distance estimation. Take time to measure your camera parameters carefully before deployment.