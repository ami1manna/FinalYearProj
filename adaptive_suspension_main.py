"""
Integrated Adaptive Suspension System
======================================
Combines pothole detection, distance estimation, speed estimation,
and preview-based suspension control.
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
import time

# Import your existing modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pothole_detector import PotholeDetector
from midas.midas_utils import MiDaSDepthEstimator

# Import new modules
from ipm_distance import IPMDistanceEstimator, create_default_camera_matrix
from speed_estimator import OpticalFlowSpeedEstimator
from skyhook_controller import (
    GainScheduledSkyhookController, 
    SuspensionState,
    classify_severity
)


class AdaptiveSuspensionSystem:
    """
    Complete adaptive suspension system with vision-based preview control.
    """
    
    def __init__(self, model_path, camera_config):
        """
        Initialize the adaptive suspension system.
        
        Parameters:
        -----------
        model_path : str
            Path to YOLO pothole detection model
        camera_config : dict
            Camera calibration and mounting parameters:
            - 'width': image width
            - 'height': image height
            - 'focal_length': focal length (pixels)
            - 'camera_height': height above ground (m)
            - 'pitch_angle': pitch angle (degrees)
            - 'fps': frame rate
        """
        print("Initializing Adaptive Suspension System...")
        
        # Pothole detection
        self.detector = PotholeDetector(model_path, conf=0.4)
        print("✓ Pothole detector loaded")
        
        # Depth estimation
        self.depth_estimator = MiDaSDepthEstimator()
        print("✓ Depth estimator loaded")
        
        # Distance estimation (IPM)
        K = create_default_camera_matrix(
            camera_config['width'],
            camera_config['height'],
            camera_config.get('focal_length', camera_config['width'])
        )
        
        self.distance_estimator = IPMDistanceEstimator(
            camera_matrix=K,
            dist_coeffs=None,  # Add distortion coefficients if available
            camera_height=camera_config['camera_height'],
            pitch_angle=camera_config['pitch_angle']
        )
        print("✓ Distance estimator initialized")
        
        # Speed estimation
        self.speed_estimator = OpticalFlowSpeedEstimator(
            self.distance_estimator,
            fps=camera_config.get('fps', 30)
        )
        print("✓ Speed estimator initialized")
        
        # Suspension controller
        self.controller = GainScheduledSkyhookController()
        print("✓ Suspension controller initialized")
        
        # State management
        self.depth_map = None
        self.score_buffer = deque(maxlen=7)
        self.speed_buffer = deque(maxlen=10)
        self.current_speed = 0.0  # m/s
        
        # Depth computation interval
        self.DEPTH_INTERVAL = 5
        self.frame_id = 0
        
        # Simulation state (for demo)
        self.suspension_state = SuspensionState(
            z_s=0.0, z_s_dot=0.0,
            z_u=0.0, z_u_dot=0.0
        )
        
        print("System ready!\n")
    
    def compute_relative_depth(self, depth_map, box, margin=30):
        """
        Compute relative depth of pothole.
        
        Parameters:
        -----------
        depth_map : np.ndarray
            MiDaS depth map
        box : tuple
            Bounding box (x1, y1, x2, y2)
        margin : int
            Margin for road reference region
            
        Returns:
        --------
        rel_depth : float
            Relative depth value
        """
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
    
    def severity_score(self, d, min_d=0.02, max_d=0.18):
        """
        Convert relative depth to 0-100 severity score.
        
        Parameters:
        -----------
        d : float
            Relative depth value
        min_d, max_d : float
            Depth range for normalization
            
        Returns:
        --------
        score : int
            Severity score (0-100)
        """
        d = max(min(d, max_d), min_d)
        return int(100 * (d - min_d) / (max_d - min_d))
    
    def process_frame(self, frame):
        """
        Process single frame through complete pipeline.
        
        Parameters:
        -----------
        frame : np.ndarray
            Input BGR frame
            
        Returns:
        --------
        output : np.ndarray
            Annotated frame
        control_data : dict
            Control information for logging/display
        """
        self.frame_id += 1
        
        # Resize frame
        frame = cv2.resize(frame, (640, 384))
        
        # 1. DETECT POTHOLES
        output, results = self.detector.detect_frame(frame)
        
        # 2. ESTIMATE SPEED
        speed_mps = self.speed_estimator.estimate_speed(frame)
        if speed_mps is not None:
            self.speed_buffer.append(speed_mps)
            self.current_speed = np.median(list(self.speed_buffer))
        
        # 3. COMPUTE DEPTH MAP (periodic)
        if self.depth_map is None or self.frame_id % self.DEPTH_INTERVAL == 0:
            self.depth_map = self.depth_estimator.estimate_depth(frame)
        
        # 4. PROCESS EACH DETECTED POTHOLE
        control_data = {
            'speed_mps': self.current_speed,
            'speed_kmh': self.current_speed * 3.6,
            'potholes': []
        }
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Compute severity
            rel_depth = self.compute_relative_depth(
                self.depth_map, (x1, y1, x2, y2)
            )
            score = self.severity_score(rel_depth)
            
            self.score_buffer.append(score)
            avg_score = int(sum(self.score_buffer) / len(self.score_buffer))
            
            # Estimate distance
            distance_m = self.distance_estimator.bbox_bottom_distance(
                x1, y1, x2, y2
            )
            
            # Compute control action
            if distance_m and self.current_speed > 0.1:
                F, c = self.controller.compute_control(
                    self.suspension_state,
                    severity_score=avg_score,
                    speed_mps=self.current_speed,
                    distance_m=distance_m
                )
                
                # Time to impact
                t_impact = distance_m / max(self.current_speed, 0.1)
                
                pothole_data = {
                    'severity': avg_score,
                    'severity_class': classify_severity(avg_score).name,
                    'distance_m': distance_m,
                    't_impact': t_impact,
                    'damping_coeff': c,
                    'control_force': F,
                    'bbox': (x1, y1, x2, y2)
                }
                
                control_data['potholes'].append(pothole_data)
                
                # Draw annotations
                severity = classify_severity(avg_score)
                color = self._severity_color(severity.name)
                
                cv2.rectangle(output, (int(x1), int(y1)), 
                            (int(x2), int(y2)), color, 2)
                
                # Label with distance and control info
                label = f"{severity.name} ({avg_score})"
                sublabel = f"d={distance_m:.1f}m c={c:.0f}"
                
                cv2.putText(output, label,
                          (int(x1), int(y1) - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(output, sublabel,
                          (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display overall info
        info_text = [
            f"Speed: {control_data['speed_kmh']:.1f} km/h",
            f"Frame: {self.frame_id}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(output, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return output, control_data
    
    def _severity_color(self, severity_name):
        """Get color for severity level."""
        colors = {
            "LOW": (0, 255, 0),
            "MEDIUM": (0, 255, 255),
            "HIGH": (0, 165, 255),
            "CRITICAL": (0, 0, 255)
        }
        return colors.get(severity_name, (255, 255, 255))


def main():
    """Run the adaptive suspension system on video."""
    
    VIDEO_PATH = "data/videos/road.mp4"
    MODEL_PATH = "models/pothole.pt"
    
    # Camera configuration (MUST BE CALIBRATED for your setup!)
    camera_config = {
        'width': 640,
        'height': 384,
        'focal_length': 500,  # pixels (estimate, needs calibration)
        'camera_height': 1.2,  # meters above ground
        'pitch_angle': 15,  # degrees down from horizontal
        'fps': 30
    }
    
    # Initialize system
    system = AdaptiveSuspensionSystem(MODEL_PATH, camera_config)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    
    print("Processing video... Press 'q' to quit\n")
    
    # Performance tracking
    frame_times = deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Process frame
        output, control_data = system.process_frame(frame)
        
        # Log control actions
        if control_data['potholes']:
            for p in control_data['potholes']:
                print(f"Pothole: {p['severity_class']} ({p['severity']}/100) "
                      f"| d={p['distance_m']:.1f}m | t={p['t_impact']:.2f}s "
                      f"| c={p['damping_coeff']:.0f} N·s/m")
        
        # Display
        cv2.imshow("Adaptive Suspension with Preview Control", output)
        
        # FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        fps = 1.0 / np.mean(frame_times) if frame_times else 0
        
        if self.frame_id % 30 == 0:
            print(f"FPS: {fps:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
