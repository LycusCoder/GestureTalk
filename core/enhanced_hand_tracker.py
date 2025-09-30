"""
Enhanced Hand Tracker Module - Improved hand rigging visualization & performance
Advanced MediaPipe hand tracking with enhanced skeleton rendering and real-time performance
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
import time


class EnhancedHandTracker:
    """
    Enhanced hand tracker dengan improved rigging visualization dan performance
    
    Features:
    - Enhanced 21-point hand skeleton dengan detailed joints
    - Real-time performance optimization
    - Advanced visualization dengan bone connections
    - Multiple visualization modes (skeleton, landmarks, full)
    - Improved landmark stability dan filtering
    - Better webcam compatibility
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 visualization_mode: str = 'full'):
        """
        Initialize EnhancedHandTracker dengan advanced settings
        
        Args:
            static_image_mode: False untuk video stream (lebih efisien)
            max_num_hands: Maksimal tangan yang bisa dideteksi 
            min_detection_confidence: Confidence threshold untuk deteksi
            min_tracking_confidence: Confidence threshold untuk tracking
            visualization_mode: 'skeleton', 'landmarks', 'full', 'minimal'
        """
        # MediaPipe Hands initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Visualization settings
        self.visualization_mode = visualization_mode
        self._setup_hand_anatomy()
        self._setup_enhanced_styles()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.avg_processing_time = 0.0
        
        # Landmark stability
        self.landmark_history = []
        self.stability_window = 5
        
        print(f"🚀 EnhancedHandTracker initialized with {visualization_mode} mode")
    
    def _setup_hand_anatomy(self):
        """Setup detailed hand anatomy untuk enhanced visualization"""
        # MediaPipe hand landmark indices
        self.LANDMARK_NAMES = {
            0: 'WRIST',
            # Thumb
            1: 'THUMB_CMC', 2: 'THUMB_MCP', 3: 'THUMB_IP', 4: 'THUMB_TIP',
            # Index finger
            5: 'INDEX_FINGER_MCP', 6: 'INDEX_FINGER_PIP', 7: 'INDEX_FINGER_DIP', 8: 'INDEX_FINGER_TIP',
            # Middle finger
            9: 'MIDDLE_FINGER_MCP', 10: 'MIDDLE_FINGER_PIP', 11: 'MIDDLE_FINGER_DIP', 12: 'MIDDLE_FINGER_TIP',
            # Ring finger
            13: 'RING_FINGER_MCP', 14: 'RING_FINGER_PIP', 15: 'RING_FINGER_DIP', 16: 'RING_FINGER_TIP',
            # Pinky
            17: 'PINKY_MCP', 18: 'PINKY_PIP', 19: 'PINKY_DIP', 20: 'PINKY_TIP'
        }
        
        # Enhanced bone connections untuk detailed skeleton
        self.ENHANCED_CONNECTIONS = [
            # Wrist to palm base
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # Thumb bones
            (1, 2), (2, 3), (3, 4),
            # Index finger bones  
            (5, 6), (6, 7), (7, 8),
            # Middle finger bones
            (9, 10), (10, 11), (11, 12),
            # Ring finger bones
            (13, 14), (14, 15), (15, 16),
            # Pinky bones
            (17, 18), (18, 19), (19, 20),
            # Palm connections
            (1, 5), (5, 9), (9, 13), (13, 17)
        ]
        
        # Finger tip indices untuk special highlighting
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        
        # Joint types untuk different visualization
        self.JOINT_TYPES = {
            'wrist': [0],
            'mcp': [1, 5, 9, 13, 17],  # Metacarpophalangeal joints
            'pip': [6, 10, 14, 18],    # Proximal interphalangeal joints
            'dip': [7, 11, 15, 19],    # Distal interphalangeal joints
            'tip': [4, 8, 12, 16, 20]  # Finger tips
        }
    
    def _setup_enhanced_styles(self):
        """Setup enhanced drawing styles untuk better visualization"""
        self.styles = {
            'skeleton': {
                'connection_color': (0, 255, 0),      # Green bones
                'connection_thickness': 3,
                'landmark_color': (255, 0, 0),        # Blue landmarks
                'landmark_radius': 4,
                'tip_color': (0, 0, 255),             # Red fingertips
                'tip_radius': 6
            },
            'landmarks': {
                'landmark_color': (255, 255, 0),      # Yellow landmarks
                'landmark_radius': 5,
                'tip_color': (0, 255, 255),           # Cyan fingertips
                'tip_radius': 8
            },
            'full': {
                'connection_color': (0, 200, 100),    # Green-blue bones
                'connection_thickness': 2,
                'landmark_color': (255, 100, 100),    # Light red landmarks
                'landmark_radius': 3,
                'tip_color': (0, 100, 255),           # Orange-red fingertips
                'tip_radius': 5,
                'joint_colors': {
                    'wrist': (255, 255, 255),         # White wrist
                    'mcp': (255, 200, 0),             # Orange MCP joints
                    'pip': (255, 150, 100),           # Light orange PIP joints
                    'dip': (255, 100, 150),           # Pink DIP joints
                    'tip': (0, 0, 255)                # Red tips
                }
            },
            'minimal': {
                'connection_color': (100, 100, 100),  # Gray bones
                'connection_thickness': 1,
                'landmark_color': (200, 200, 200),    # Light gray landmarks
                'landmark_radius': 2
            }
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Enhanced frame processing dengan performance monitoring
        
        Args:
            frame: Input frame dari camera (BGR format)
            
        Returns:
            Tuple[processed_frame, landmarks_list]:
            - processed_frame: Frame dengan enhanced hand visualization
            - landmarks_list: List of normalized landmarks
        """
        start_time = time.time()
        
        # Convert BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process dengan MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Copy frame untuk drawing
        output_frame = frame.copy()
        landmarks_list = None
        
        # Process hand detection results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Enhanced visualization
                self._draw_enhanced_hand(output_frame, hand_landmarks)
                
                # Extract dan stabilize landmarks
                landmarks_list = self._extract_and_stabilize_landmarks(hand_landmarks)
                break  # Process first hand only
        
        # Performance monitoring
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        return output_frame, landmarks_list
    
    def _draw_enhanced_hand(self, frame: np.ndarray, hand_landmarks):
        """
        Draw enhanced hand visualization dengan detailed anatomy
        
        Args:
            frame: Output frame untuk drawing
            hand_landmarks: MediaPipe hand landmarks
        """
        h, w, c = frame.shape
        style = self.styles.get(self.visualization_mode, self.styles['full'])
        
        # Convert landmarks ke pixel coordinates
        landmark_points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append((x, y))
        
        if self.visualization_mode in ['skeleton', 'full']:
            self._draw_skeleton(frame, landmark_points, style)
        
        if self.visualization_mode in ['landmarks', 'full']:
            self._draw_landmarks(frame, landmark_points, style)
        
        if self.visualization_mode == 'full':
            self._draw_enhanced_features(frame, landmark_points, style)
    
    def _draw_skeleton(self, frame: np.ndarray, points: List[Tuple], style: Dict):
        """Draw skeleton connections dengan enhanced bone structure"""
        connection_color = style['connection_color']
        thickness = style.get('connection_thickness', 2)
        
        # Draw bone connections
        for connection in self.ENHANCED_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                # Vary thickness based on bone type
                if start_idx == 0 or end_idx == 0:  # Wrist connections
                    cv2.line(frame, start_point, end_point, connection_color, thickness + 1)
                else:
                    cv2.line(frame, start_point, end_point, connection_color, thickness)
    
    def _draw_landmarks(self, frame: np.ndarray, points: List[Tuple], style: Dict):
        """Draw landmark points dengan different styles untuk joint types"""
        if self.visualization_mode == 'full':
            # Enhanced joint-specific coloring
            joint_colors = style.get('joint_colors', {})
            
            for joint_type, indices in self.JOINT_TYPES.items():
                color = joint_colors.get(joint_type, style['landmark_color'])
                radius = style['landmark_radius']
                
                if joint_type == 'tip':
                    radius = style.get('tip_radius', radius + 2)
                
                for idx in indices:
                    if idx < len(points):
                        cv2.circle(frame, points[idx], radius, color, -1)
                        cv2.circle(frame, points[idx], radius + 1, (255, 255, 255), 1)
        else:
            # Standard landmark drawing
            landmark_color = style['landmark_color']
            landmark_radius = style['landmark_radius']
            
            for i, point in enumerate(points):
                if i in self.FINGER_TIPS:
                    # Highlight finger tips
                    tip_color = style.get('tip_color', landmark_color)
                    tip_radius = style.get('tip_radius', landmark_radius + 2)
                    cv2.circle(frame, point, tip_radius, tip_color, -1)
                else:
                    cv2.circle(frame, point, landmark_radius, landmark_color, -1)
    
    def _draw_enhanced_features(self, frame: np.ndarray, points: List[Tuple], style: Dict):
        """Draw additional enhanced features seperti gesture direction arrows"""
        if len(points) < 21:
            return
        
        # Draw palm center
        wrist = points[0]
        middle_mcp = points[9]
        palm_center = ((wrist[0] + middle_mcp[0]) // 2, (wrist[1] + middle_mcp[1]) // 2)
        cv2.circle(frame, palm_center, 8, (255, 255, 255), 2)
        
        # Draw gesture direction arrow (from wrist to middle finger)
        middle_tip = points[12]
        cv2.arrowedLine(frame, palm_center, middle_tip, (255, 255, 0), 2, tipLength=0.3)
        
        # Draw landmark indices (untuk debugging)
        if self.visualization_mode == 'full':
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            for i, point in enumerate(points):
                cv2.putText(frame, str(i), (point[0] + 10, point[1]), 
                           font, font_scale, (255, 255, 255), 1)
    
    def _extract_and_stabilize_landmarks(self, hand_landmarks) -> List[float]:
        """
        Extract landmarks dengan stability filtering untuk smoother tracking
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            List[float]: Stabilized normalized coordinates
        """
        # Extract current landmarks
        current_landmarks = []
        landmark_array = []
        
        for landmark in hand_landmarks.landmark:
            landmark_array.append([landmark.x, landmark.y])
        landmark_array = np.array(landmark_array)
        
        # Normalize relatif terhadap wrist
        wrist = landmark_array[0]
        for i, (x, y) in enumerate(landmark_array):
            norm_x = x - wrist[0]
            norm_y = y - wrist[1]
            current_landmarks.extend([norm_x, norm_y])
        
        # Apply stability filtering
        if len(self.landmark_history) >= self.stability_window:
            # Remove oldest entry
            self.landmark_history.pop(0)
        
        # Add current landmarks
        self.landmark_history.append(current_landmarks)
        
        # Calculate stabilized landmarks (moving average)
        if len(self.landmark_history) >= 3:  # Need at least 3 frames
            stabilized = np.mean(self.landmark_history, axis=0).tolist()
            return stabilized
        else:
            return current_landmarks
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics untuk monitoring"""
        self.frame_count += 1
        self.processing_times.append(processing_time)
        
        # Keep only recent times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        # Update average
        self.avg_processing_time = np.mean(self.processing_times)
    
    def draw_performance_info(self, frame: np.ndarray, landmarks: Optional[List], 
                            gesture: str = "Unknown") -> np.ndarray:
        """
        Draw comprehensive performance dan hand info overlay
        
        Args:
            frame: Input frame
            landmarks: Current landmarks
            gesture: Detected gesture name
            
        Returns:
            Frame dengan performance info
        """
        height, width = frame.shape[:2]
        
        # Enhanced semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        
        # Enhanced gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
        
        # Hand detection status dengan detail
        hand_status = "Hand Detected" if landmarks else "No Hand"
        status_color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, f"Status: {hand_status}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Performance metrics
        fps_text = f"Processing: {1/self.avg_processing_time:.1f} FPS" if self.avg_processing_time > 0 else "Processing: -- FPS"
        cv2.putText(frame, fps_text, (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Visualization mode
        cv2.putText(frame, f"Mode: {self.visualization_mode.title()}", (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
        
        # Landmark count (untuk debugging)
        if landmarks:
            landmark_count = len(landmarks) // 2  # x,y pairs
            cv2.putText(frame, f"Landmarks: {landmark_count}", (250, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Enhanced controls info
        cv2.putText(frame, "Controls: ESC=Exit, SPACE=Screenshot, M=Mode", (20, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def switch_visualization_mode(self):
        """Cycle through visualization modes"""
        modes = ['full', 'skeleton', 'landmarks', 'minimal']
        current_idx = modes.index(self.visualization_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.visualization_mode = modes[next_idx]
        print(f"🎨 Switched to {self.visualization_mode} mode")
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        return {
            'avg_processing_time': self.avg_processing_time,
            'processing_fps': 1/self.avg_processing_time if self.avg_processing_time > 0 else 0,
            'frame_count': self.frame_count,
            'visualization_mode': self.visualization_mode,
            'landmark_history_length': len(self.landmark_history),
            'stability_window': self.stability_window
        }
    
    def is_hand_detected(self) -> bool:
        """Check apakah ada tangan yang terdeteksi dalam recent frames"""
        return len(self.landmark_history) > 0
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        self.landmark_history.clear()
        self.processing_times.clear()


# Test function untuk enhanced hand tracker
def test_enhanced_hand_tracker():
    """
    Test function untuk verify EnhancedHandTracker
    """
    print("🚀 Testing EnhancedHandTracker...")
    
    # Initialize enhanced tracker
    tracker = EnhancedHandTracker(visualization_mode='full')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Camera not available")
        return
    
    print("✅ Enhanced hand tracker ready")
    print("📱 Show your hand to camera...")
    print("⌨️  Controls:")
    print("   ESC: Exit")
    print("   M: Switch visualization mode")
    print("   SPACE: Take screenshot")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame dengan enhanced tracker
            processed_frame, landmarks = tracker.process_frame(frame)
            
            # Add performance info
            display_frame = tracker.draw_performance_info(
                processed_frame, landmarks, "Enhanced Test Mode"
            )
            
            # Show frame
            cv2.imshow('Enhanced Hand Tracker Test', display_frame)
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                stats = tracker.get_performance_stats()
                print(f"📊 FPS: {fps:.1f} | Processing: {stats['processing_fps']:.1f} | "
                      f"Mode: {stats['visualization_mode']}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('m') or key == ord('M'):  # Switch mode
                tracker.switch_visualization_mode()
            elif key == 32:  # SPACE - screenshot
                cv2.imwrite(f'enhanced_hand_screenshot_{int(time.time())}.jpg', display_frame)
                print("📸 Screenshot saved!")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.cleanup()
        
        # Show final stats
        final_stats = tracker.get_performance_stats()
        print("\n📊 ENHANCED HAND TRACKER FINAL STATS:")
        print(f"   Total frames: {final_stats['frame_count']}")
        print(f"   Processing FPS: {final_stats['processing_fps']:.1f}")
        print(f"   Average processing time: {final_stats['avg_processing_time']*1000:.1f}ms")
        print("✅ Enhanced hand tracker test completed")


if __name__ == "__main__":
    test_enhanced_hand_tracker()