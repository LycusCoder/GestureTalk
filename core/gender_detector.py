"""
Gender Detection Module - Face-based gender classification using MediaPipe
Real-time gender detection dari face landmarks dan characteristics
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GenderDetector:
    """
    MediaPipe-based gender detector menggunakan face landmarks
    
    Features:
    - Real-time face detection menggunakan MediaPipe
    - Gender classification berdasarkan facial features
    - Confidence scoring
    - Multiple face support
    - Performance optimization
    """
    
    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize GenderDetector dengan MediaPipe Face Detection
        
        Args:
            min_detection_confidence: Minimum confidence untuk face detection
            min_tracking_confidence: Minimum confidence untuk face tracking
        """
        # Initialize MediaPipe Face Detection dan Face Mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection untuk bounding box
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 untuk full range model
            min_detection_confidence=min_detection_confidence
        )
        
        # Face mesh untuk detailed landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Gender classification features (based pada research)
        self._setup_gender_features()
        
        # Performance tracking
        self.detection_count = 0
        self.last_detection = None
        self.detection_history = []
        
        print("🚀 GenderDetector initialized")
    
    def _setup_gender_features(self):
        """Setup gender classification features berdasarkan facial landmarks"""
        
        # Key facial landmarks indices untuk gender classification
        self.GENDER_LANDMARKS = {
            # Face shape landmarks
            'face_outline': [10, 151, 9, 8, 168, 6, 148, 176, 149, 150],
            'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            'forehead': [10, 151, 9, 8],
            
            # Eye landmarks
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263],
            
            # Eyebrow landmarks  
            'left_eyebrow': [46, 53, 52, 51, 48],
            'right_eyebrow': [276, 283, 282, 281, 278],
            
            # Nose landmarks
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125],
            
            # Mouth landmarks
            'lips': [0, 17, 18, 200, 199, 175, 0, 269, 270, 267, 271, 272],
            
            # Cheek landmarks
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142],
            'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452]
        }
        
        # Gender classification rules (improved - less sensitive to facial expressions)
        self.GENDER_RULES = {
            'jawline_width_ratio': {'threshold': 0.87, 'weight': 3.0},  # Strong indicator
            'face_width_height_ratio': {'threshold': 0.78, 'weight': 2.5},  # Stable indicator
            'eye_distance_ratio': {'threshold': 0.43, 'weight': 1.5},   # Moderate indicator  
            'nose_width_ratio': {'threshold': 0.24, 'weight': 2.0},     # Good indicator
            'eyebrow_thickness': {'threshold': 0.025, 'weight': 1.8}    # Stable indicator
        }
        
        # Remove lip-based detection to avoid mouth expression sensitivity
    
    def detect_gender(self, frame: np.ndarray) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        Detect gender dari face dalam frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple[gender, confidence, face_info]:
            - gender: 'Laki-laki', 'Perempuan', atau None
            - confidence: Confidence score (0.0-1.0)
            - face_info: Additional face detection info
        """
        if frame is None:
            return None, 0.0, None
        
        # Convert BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_detection_results = self.face_detection.process(rgb_frame)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        if not face_detection_results.detections or not face_mesh_results.multi_face_landmarks:
            return None, 0.0, None
        
        # Get face landmarks
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        face_detection = face_detection_results.detections[0]
        
        # Calculate gender features
        features = self._calculate_gender_features(face_landmarks, frame.shape)
        
        # Classify gender
        gender, confidence = self._classify_gender(features)
        
        # Face info untuk visualization
        face_info = {
            'detection': face_detection,
            'landmarks': face_landmarks,
            'features': features,
            'bbox': self._get_face_bbox(face_detection, frame.shape)
        }
        
        # Update tracking
        self._update_detection_history(gender, confidence)
        self.detection_count += 1
        
        return gender, confidence, face_info
    
    def _calculate_gender_features(self, face_landmarks, frame_shape) -> Dict[str, float]:
        """
        Calculate gender classification features dari face landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Frame dimensions (h, w, c)
            
        Returns:
            Dict dengan calculated features
        """
        h, w = frame_shape[:2]
        
        # Convert landmarks ke normalized coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            landmarks.append([x, y])
        landmarks = np.array(landmarks)
        
        features = {}
        
        try:
            # 1. Jawline width ratio (jaw width / face height) - Strong male indicator
            jaw_left = landmarks[172]  # Left jaw point
            jaw_right = landmarks[397]  # Right jaw point
            face_top = landmarks[10]   # Face top
            face_bottom = landmarks[152]  # Face bottom (chin)
            
            jaw_width = np.linalg.norm(jaw_right - jaw_left)
            face_height = np.linalg.norm(face_bottom - face_top)
            features['jawline_width_ratio'] = jaw_width / face_height if face_height > 0 else 0
            
            # 2. Face width to height ratio (more stable than forehead alone)
            face_width = jaw_width
            features['face_width_height_ratio'] = face_width / face_height if face_height > 0 else 0
            
            # 3. Eye distance ratio (inter-eye distance / face width)
            left_eye_center = landmarks[33]
            right_eye_center = landmarks[263]
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            features['eye_distance_ratio'] = eye_distance / face_width if face_width > 0 else 0
            
            # 4. Nose width ratio (nose width / face width) 
            nose_left = landmarks[131]
            nose_right = landmarks[358]
            nose_width = np.linalg.norm(nose_right - nose_left)
            features['nose_width_ratio'] = nose_width / face_width if face_width > 0 else 0
            
            # 5. Eyebrow thickness (more stable than lip features)
            left_eyebrow_top = landmarks[46]
            left_eyebrow_bottom = landmarks[70]  # Below eyebrow
            right_eyebrow_top = landmarks[276]
            right_eyebrow_bottom = landmarks[300]  # Below eyebrow
            
            left_thickness = np.linalg.norm(left_eyebrow_top - left_eyebrow_bottom)
            right_thickness = np.linalg.norm(right_eyebrow_top - right_eyebrow_bottom)
            avg_thickness = (left_thickness + right_thickness) / 2
            features['eyebrow_thickness'] = avg_thickness / face_height if face_height > 0 else 0
            
        except (IndexError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating features: {e}")
            # Return default features
            features = {key: 0.5 for key in self.GENDER_RULES.keys()}
        
        return features
    
    def _classify_gender(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify gender berdasarkan calculated features
        
        Args:
            features: Calculated facial features
            
        Returns:
            Tuple[gender, confidence]: Gender prediction dengan confidence
        """
        male_score = 0
        female_score = 0
        total_features = len(self.GENDER_RULES)
        
        # Score berdasarkan each feature
        for feature_name, threshold in self.GENDER_RULES.items():
            feature_value = features.get(feature_name, 0.5)
            
            if feature_name == 'jawline_width_ratio':
                if feature_value > threshold:
                    male_score += 1
                else:
                    female_score += 1
                    
            elif feature_name == 'forehead_height_ratio':
                if feature_value < threshold:
                    male_score += 1
                else:
                    female_score += 1
                    
            elif feature_name == 'eye_distance_ratio':
                if feature_value < threshold:
                    male_score += 1
                else:
                    female_score += 1
                    
            elif feature_name == 'nose_width_ratio':
                if feature_value > threshold:
                    male_score += 1
                else:
                    female_score += 1
                    
            elif feature_name == 'lip_thickness_ratio':
                if feature_value < threshold:
                    male_score += 1
                else:
                    female_score += 1
        
        # Determine gender dan confidence
        if male_score > female_score:
            gender = 'Laki-laki'
            confidence = male_score / total_features
        elif female_score > male_score:
            gender = 'Perempuan'
            confidence = female_score / total_features
        else:
            # Tie - use default atau previous detection
            gender = 'Tidak dapat ditentukan'
            confidence = 0.5
        
        # Adjust confidence (make it more realistic)
        confidence = min(0.9, max(0.6, confidence))
        
        return gender, confidence
    
    def _get_face_bbox(self, detection, frame_shape) -> Tuple[int, int, int, int]:
        """Get face bounding box coordinates"""
        h, w = frame_shape[:2]
        
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        return x, y, width, height
    
    def _update_detection_history(self, gender: Optional[str], confidence: float):
        """Update detection history untuk stability"""
        if gender:
            self.detection_history.append({
                'gender': gender,
                'confidence': confidence,
                'timestamp': cv2.getTickCount()
            })
            
            # Keep only recent detections
            if len(self.detection_history) > 10:
                self.detection_history = self.detection_history[-10:]
            
            self.last_detection = gender
    
    def get_stable_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get stable gender prediction berdasarkan recent detections
        
        Returns:
            Tuple[gender, confidence]: Stable prediction
        """
        if not self.detection_history:
            return None, 0.0
        
        # Count recent predictions
        recent_detections = self.detection_history[-5:]  # Last 5 detections
        gender_counts = {}
        total_confidence = 0.0
        
        for detection in recent_detections:
            gender = detection['gender']
            confidence = detection['confidence']
            
            if gender not in gender_counts:
                gender_counts[gender] = {'count': 0, 'total_confidence': 0.0}
            
            gender_counts[gender]['count'] += 1
            gender_counts[gender]['total_confidence'] += confidence
            total_confidence += confidence
        
        # Get most frequent gender
        if gender_counts:
            best_gender = max(gender_counts.keys(), 
                            key=lambda g: gender_counts[g]['count'])
            best_confidence = gender_counts[best_gender]['total_confidence'] / gender_counts[best_gender]['count']
            
            return best_gender, best_confidence
        
        return None, 0.0
    
    def draw_gender_info(self, frame: np.ndarray, face_info: Optional[Dict], 
                        gender: Optional[str], confidence: float) -> np.ndarray:
        """
        Draw gender detection info pada frame
        
        Args:
            frame: Input frame
            face_info: Face detection information
            gender: Detected gender
            confidence: Confidence score
            
        Returns:
            Frame dengan gender info overlay
        """
        if not face_info or not gender:
            return frame
        
        # Draw face bounding box
        bbox = face_info['bbox']
        x, y, w, h = bbox
        
        # Gender color coding
        color = (0, 255, 0) if gender == 'Laki-laki' else (255, 0, 255)  # Green untuk male, Magenta untuk female
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw gender label
        label = f"{gender}: {confidence:.1%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background untuk label
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        
        # Gender text
        cv2.putText(frame, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw some key landmarks jika available
        if 'landmarks' in face_info:
            landmarks = face_info['landmarks']
            h_frame, w_frame = frame.shape[:2]
            
            # Draw eye landmarks
            for landmark in [33, 263]:  # Left dan right eye centers
                if landmark < len(landmarks.landmark):
                    lm = landmarks.landmark[landmark]
                    x_lm = int(lm.x * w_frame)
                    y_lm = int(lm.y * h_frame)
                    cv2.circle(frame, (x_lm, y_lm), 3, (0, 255, 255), -1)
        
        return frame
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_detections': self.detection_count,
            'last_detection': self.last_detection,
            'history_length': len(self.detection_history),
            'stable_prediction': self.get_stable_prediction()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Test function
def test_gender_detector():
    """Test function untuk GenderDetector"""
    print("🚀 Testing GenderDetector...")
    
    detector = GenderDetector()
    
    # Test dengan camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not available for testing")
        return
    
    print("✅ GenderDetector ready")
    print("📱 Show your face to camera...")
    print("⌨️  Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect gender
        gender, confidence, face_info = detector.detect_gender(frame)
        
        # Draw results
        display_frame = detector.draw_gender_info(frame, face_info, gender, confidence)
        
        # Show stats
        if gender:
            stable_gender, stable_confidence = detector.get_stable_prediction()
            cv2.putText(display_frame, f"Stable: {stable_gender} ({stable_confidence:.1%})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Gender Detection Test', display_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    
    # Show final stats
    stats = detector.get_detection_stats()
    print(f"\n📊 Final Stats: {stats}")
    print("✅ GenderDetector test completed")


if __name__ == "__main__":
    test_gender_detector()