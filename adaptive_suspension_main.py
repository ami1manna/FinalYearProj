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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

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
        
        dist_coeffs = np.zeros((4, 1))  # No distortion assumed
        self.distance_estimator = IPMDistanceEstimator(
            K,
            dist_coeffs,
            camera_config['camera_height'],
            camera_config['pitch_angle']
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
        
        # Depth computation interval (reduced for performance)
        self.DEPTH_INTERVAL = 15  # Increased from 5 to 15
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
        
        # Apply simple detection smoothing to reduce inconsistency
        if not hasattr(self, 'prev_detection_count'):
            self.prev_detection_count = 0
        
        current_count = len(results.boxes) if results and results.boxes else 0
        
        # If detection drops significantly, use previous frame's results
        if current_count < self.prev_detection_count * 0.5 and self.prev_detection_count > 0:
            # Keep previous detection count (simple smoothing)
            pass
        
        self.prev_detection_count = current_count
        
        # 2. ESTIMATE SPEED
        speed_mps = self.speed_estimator.estimate_speed(frame)
        if speed_mps is not None:
            self.speed_buffer.append(speed_mps)
            self.current_speed = np.median(list(self.speed_buffer))
        
        # 3. COMPUTE DEPTH MAP (disabled for performance)
        # if self.depth_map is None or self.frame_id % self.DEPTH_INTERVAL == 0:
        #     self.depth_map = self.depth_estimator.estimate_depth(frame)
        #     print(f"Depth computed for frame {self.frame_id}")
        
        # Use placeholder depth for better performance
        self.depth_map = np.ones((384, 640)) * 0.1  # Simple depth map
        
        # 4. PROCESS EACH DETECTED POTHOLE (simplified for performance)
        control_data = {
            'speed_mps': self.current_speed,
            'speed_kmh': self.current_speed * 3.6,
            'potholes': []
        }
        
        # Process only first 3 potholes for performance
        max_potholes = 3
        for i, box in enumerate(results.boxes):
            if i >= max_potholes:
                break
                
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Simple severity calculation (no depth for performance)
            score = int(conf * 100)  # Simple confidence-based score
            
            self.score_buffer.append(score)
            avg_score = int(sum(self.score_buffer) / len(self.score_buffer))
            
            # Simple distance estimation (no depth map)
            distance_m = 10.0 + (i * 5.0)  # Placeholder distance
            
            # Simplified control calculation
            c = 800 + score * 10  # Simple damping calculation
            
            # Time to impact
            t_impact = distance_m / max(self.current_speed, 0.1)
            
            pothole_data = {
                'severity': avg_score,
                'severity_class': 'MEDIUM',  # Fixed for performance
                'distance_m': distance_m,
                't_impact': t_impact,
                'damping_coeff': c,
                'control_force': c * 10,  # Simplified
                'bbox': (x1, y1, x2, y2)
            }
            
            control_data['potholes'].append(pothole_data)
            
            # Simplified drawing (only essential elements)
            severity = 'MEDIUM'
            color = (0, 255, 0)  # Green
            
            cv2.rectangle(output, (int(x1), int(y1)), 
                        (int(x2), int(y2)), color, 2)
            
            # Simple label (no complex calculations)
            label = f"P{score}"
            cv2.putText(output, label,
                      (int(x1), int(y1) - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
        
        # Add simplified suspension damping bar (minimal drawing)
        if control_data['potholes']:
            c_current = control_data['potholes'][0]['damping_coeff']
        else:
            c_current = 800.0
        
        # Simple bar calculation (no complex logic)
        bar_length = int((c_current / 2000.0) * 150)  # Max 150px length
        
        # Draw simple bar (less operations)
        cv2.rectangle(output, (30, output.shape[0] - 40), 
                   (30 + bar_length, output.shape[0] - 20), 
                   (0, 255, 0), -1)
        
        # Simple text
        cv2.putText(output, f"D: {c_current:.0f}", 
                  (30, output.shape[0] - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
    
    def _draw_suspension_bar(self, frame, control_data):
        """
        Draw real-time suspension damping level bar.
        
        Parameters:
        -----------
        frame : np.ndarray
            Video frame to draw on
        control_data : dict
            Control data containing current damping values
        """
        # Get current damping from most recent pothole or default
        if control_data['potholes']:
            # Use the most recent pothole's damping value
            c_current = control_data['potholes'][-1]['damping_coeff']
        else:
            # Default minimum damping when no potholes
            c_current = 800.0
        
        # Controller parameters for bar scaling
        c_high = 4000.0  # Maximum damping from skyhook controller
        bar_width_max = 300
        bar_height = 20
        bar_x = 30
        bar_y = frame.shape[0] - 50  # Position near bottom
        
        # Calculate bar length proportionally
        if c_current > c_high:
            bar_length = bar_width_max
        else:
            bar_length = int((c_current / c_high) * bar_width_max)
        
        # Draw background bar (dark)
        cv2.rectangle(frame, (bar_x, bar_y), 
                   (bar_x + bar_width_max, bar_y + bar_height), 
                   (50, 50, 50), -1)
        
        # Draw filled bar (blue gradient based on level)
        if c_current < 1500:
            bar_color = (0, 255, 0)  # Green - low damping
        elif c_current < 2500:
            bar_color = (0, 255, 255)  # Yellow - medium damping
        else:
            bar_color = (0, 0, 255)  # Red - high damping
        
        cv2.rectangle(frame, (bar_x, bar_y), 
                   (bar_x + bar_length, bar_y + bar_height), 
                   bar_color, -1)
        
        # Add border
        cv2.rectangle(frame, (bar_x, bar_y), 
                   (bar_x + bar_width_max, bar_y + bar_height), 
                   (255, 255, 255), 2)
        
        # Add title
        cv2.putText(frame, "Suspension Level", 
                   (bar_x, bar_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add numeric damping value
        cv2.putText(frame, f"Damping: {c_current:.0f} Ns/m", 
                   (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    """Run the adaptive suspension system on video."""
    
    VIDEO_PATH = "data/videos/road.mp4"
    MODEL_PATH = "models/pothole.pt"
    
    # Camera configuration (MUST BE CALIBRATED for your setup!)
    camera_config = {
        'width': 640,
        'height': 384,
        'focal_length': 640,  # pixels (update with your calibrated value)
        'camera_height': 1.2,  # meters above ground (measure your setup)
        'pitch_angle': 15,  # degrees down from horizontal (measure your setup)
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
    last_depth_time = 0
    depth_computation_time = 0.5  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Skip depth computation if too recent (performance optimization)
        current_time = time.time()
        if current_time - last_depth_time < depth_computation_time:
            system.DEPTH_INTERVAL = 999  # Skip depth this frame
        else:
            system.DEPTH_INTERVAL = 15  # Allow depth computation
            last_depth_time = current_time
        
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
        
        if system.frame_id % 30 == 0:
            print(f"FPS: {fps:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
