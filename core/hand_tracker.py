"""
Hand Tracker Module - Core component for hand detection using MediaPipe
Handles real-time hand landmark detection and coordinate normalization
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict


class HandTracker:
    """
    MediaPipe-based hand tracker for real-time hand detection and landmark extraction
    
    Features:
    - Real-time hand detection dari webcam
    - 21-point hand landmarks extraction  
    - Coordinate normalization (relative to wrist)
    - Multi-hand support
    - Drawing utilities untuk visualization
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize HandTracker dengan MediaPipe Hands solution
        
        Args:
            static_image_mode: False untuk video stream (lebih efisien)
            max_num_hands: Maksimal tangan yang bisa dideteksi (1 untuk assistive app)
            min_detection_confidence: Confidence threshold untuk deteksi (0.7 optimal)
            min_tracking_confidence: Confidence threshold untuk tracking (0.5 optimal)
        """
        # Initialize MediaPipe Hands
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
        
        # Hand landmarks indices untuk referensi
        self.LANDMARK_NAMES = {
            0: 'WRIST',
            1: 'THUMB_CMC', 2: 'THUMB_MCP', 3: 'THUMB_IP', 4: 'THUMB_TIP',
            5: 'INDEX_FINGER_MCP', 6: 'INDEX_FINGER_PIP', 7: 'INDEX_FINGER_DIP', 8: 'INDEX_FINGER_TIP',
            9: 'MIDDLE_FINGER_MCP', 10: 'MIDDLE_FINGER_PIP', 11: 'MIDDLE_FINGER_DIP', 12: 'MIDDLE_FINGER_TIP',
            13: 'RING_FINGER_MCP', 14: 'RING_FINGER_PIP', 15: 'RING_FINGER_DIP', 16: 'RING_FINGER_TIP',
            17: 'PINKY_MCP', 18: 'PINKY_PIP', 19: 'PINKY_DIP', 20: 'PINKY_TIP'
        }
        
        # Status tracking
        self.last_landmarks = None
        self.detection_confidence = 0.0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process single frame untuk hand detection
        
        Args:
            frame: Input frame dari camera (BGR format)
            
        Returns:
            Tuple[processed_frame, landmarks_list]:
            - processed_frame: Frame dengan drawings (jika ada hand detected)
            - landmarks_list: List of normalized landmarks atau None
        """
        # Convert BGR ke RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process dengan MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Copy frame untuk drawing
        output_frame = frame.copy()
        landmarks_list = None
        
        # Jika ada hand detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton dengan style yang bagus
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract dan normalize landmarks
                landmarks_list = self._extract_landmarks(hand_landmarks)
                break  # Ambil hand pertama saja
        
        return output_frame, landmarks_list
    
    def _extract_landmarks(self, hand_landmarks) -> List[float]:
        """
        Extract dan normalize landmarks untuk ML model
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            List[float]: 42 normalized coordinates [x1,y1,x2,y2,...,x21,y21]
        """
        landmarks = []
        
        # Convert MediaPipe landmarks ke numpy array
        landmark_array = []
        for landmark in hand_landmarks.landmark:
            landmark_array.append([landmark.x, landmark.y])
        landmark_array = np.array(landmark_array)
        
        # Normalize relatif terhadap wrist (landmark 0)
        wrist = landmark_array[0]  # Koordinat pergelangan tangan
        
        for i, (x, y) in enumerate(landmark_array):
            # Koordinat relatif terhadap wrist
            norm_x = x - wrist[0]
            norm_y = y - wrist[1]
            
            landmarks.extend([norm_x, norm_y])
        
        # Store for debugging
        self.last_landmarks = landmarks.copy()
        
        return landmarks
    
    def draw_info(self, frame: np.ndarray, landmarks: Optional[List], gesture: str = "Unknown") -> np.ndarray:
        """
        Draw informasi tambahan di frame (gesture name, confidence, dll)
        
        Args:
            frame: Input frame
            landmarks: Current landmarks
            gesture: Detected gesture name
            
        Returns:
            Frame dengan info text
        """
        height, width = frame.shape[:2]
        
        # Background semi-transparent untuk text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Status info
        status = "Tangan Terdeteksi" if landmarks else "Tidak Ada Tangan"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Instructions
        cv2.putText(frame, "ESC: Keluar | SPACE: Screenshot", (20, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_landmark_info(self, landmarks: List[float], index: int) -> Tuple[float, float, str]:
        """
        Get informasi specific landmark by index
        
        Args:
            landmarks: List of landmarks
            index: Landmark index (0-20)
            
        Returns:
            Tuple[x, y, name]: Koordinat dan nama landmark
        """
        if not landmarks or index < 0 or index > 20:
            return 0.0, 0.0, "INVALID"
        
        x = landmarks[index * 2]
        y = landmarks[index * 2 + 1]
        name = self.LANDMARK_NAMES.get(index, f"UNKNOWN_{index}")
        
        return x, y, name
    
    def is_hand_detected(self) -> bool:
        """Check apakah ada tangan yang terdeteksi di frame terakhir"""
        return self.last_landmarks is not None
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


# Test function untuk development
def test_hand_tracker():
    """
    Test function untuk verify HandTracker berfungsi dengan baik
    """
    import time
    
    print("🚀 Testing HandTracker...")
    
    # Initialize tracker
    tracker = HandTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka camera")
        return
    
    print("✅ Camera berhasil diakses")
    print("📱 Tunjukkan tangan kamu ke camera...")
    print("⌨️  Tekan ESC untuk keluar")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, landmarks = tracker.process_frame(frame)
            
            # Add info
            gesture_name = "Testing Mode"
            display_frame = tracker.draw_info(processed_frame, landmarks, gesture_name)
            
            # Show frame
            cv2.imshow('HandTracker Test', display_frame)
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 FPS: {fps:.1f} | Hand Detected: {tracker.is_hand_detected()}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key - take screenshot
                cv2.imwrite(f'/app/hand_screenshot_{int(time.time())}.jpg', display_frame)
                print("📸 Screenshot saved!")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.cleanup()
        print("✅ HandTracker test selesai")


if __name__ == "__main__":
    test_hand_tracker()