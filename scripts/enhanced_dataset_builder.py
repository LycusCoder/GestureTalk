"""
Enhanced Dataset Builder - Create better gesture dataset dengan ASL-inspired gestures
Advanced data generation, augmentation, dan quality improvement untuk gesture recognition
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import cv2
from typing import List, Dict, Tuple, Optional
import argparse

# Add parent directory ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_hand_tracker import EnhancedHandTracker
from core.enhanced_camera_handler import EnhancedCameraHandler


class EnhancedDatasetBuilder:
    """
    Enhanced dataset builder untuk creating high-quality gesture recognition datasets
    
    Features:
    - ASL-inspired gesture definitions
    - Advanced data augmentation
    - Quality validation dan filtering
    - Real-time gesture recording dengan enhanced visualization
    - Multi-session data collection support
    - Automatic data balancing
    - Export dalam multiple formats
    """
    
    def __init__(self, output_dir: str = '/app/data'):
        """
        Initialize EnhancedDatasetBuilder
        
        Args:
            output_dir: Directory untuk save datasets
        """
        self.output_dir = output_dir
        self.current_session = None
        
        # Enhanced gesture definitions dengan ASL-inspired signs
        self.enhanced_gestures = {
            # Emergency & Basic Communication
            'tolong': {
                'description': '🆘 Help gesture - One hand raised with palm facing forward, or both hands waving',
                'category': 'emergency',
                'priority': 'high',
                'variations': ['open_palm_raised', 'both_hands_waving', 'pointing_up']
            },
            'halo': {
                'description': '👋 Hello gesture - Hand waving side to side',
                'category': 'greeting',
                'priority': 'high',
                'variations': ['open_hand_wave', 'small_wave', 'big_wave']
            },
            'terima_kasih': {
                'description': '🙏 Thank you - Hand on chest or both hands together',
                'category': 'courtesy',
                'priority': 'high',
                'variations': ['hand_on_chest', 'both_hands_together', 'slight_bow']
            },
            
            # Yes/No responses
            'ya': {
                'description': '👍 Yes gesture - Thumbs up or nodding hand motion',
                'category': 'response',
                'priority': 'high',
                'variations': ['thumbs_up', 'ok_sign', 'fist_up']
            },
            'tidak': {
                'description': '❌ No gesture - Hand shake, stop sign, or finger wag',
                'category': 'response',
                'priority': 'high',
                'variations': ['stop_sign', 'finger_wag', 'hand_shake']
            },
            'mungkin': {
                'description': '🤷 Maybe gesture - Hand tilted or shrugging motion',
                'category': 'response',
                'priority': 'medium',
                'variations': ['hand_tilt', 'shoulder_shrug', 'uncertain_wave']
            },
            
            # Numbers (ASL-inspired)
            'satu': {
                'description': '1️⃣ Number one - Index finger extended',
                'category': 'numbers',
                'priority': 'medium',
                'variations': ['index_up', 'pointer_gesture']
            },
            'dua': {
                'description': '2️⃣ Number two - Peace sign or two fingers up',
                'category': 'numbers', 
                'priority': 'medium',
                'variations': ['peace_sign', 'two_fingers_up', 'v_sign']
            },
            'tiga': {
                'description': '3️⃣ Number three - Three fingers extended',
                'category': 'numbers',
                'priority': 'medium',
                'variations': ['three_fingers_up', 'ok_three']
            },
            'lima': {
                'description': '5️⃣ Number five - All fingers extended (high five)',
                'category': 'numbers',
                'priority': 'medium',
                'variations': ['high_five', 'open_hand', 'five_fingers']
            },
            
            # Daily needs
            'makan': {
                'description': '🍽️ Eat gesture - Hand to mouth motion',
                'category': 'daily_needs',
                'priority': 'medium',
                'variations': ['hand_to_mouth', 'eating_motion']
            },
            'minum': {
                'description': '🥤 Drink gesture - Tilted hand to mouth like holding cup',
                'category': 'daily_needs',
                'priority': 'medium',
                'variations': ['cup_to_mouth', 'drinking_motion']
            },
            'tidur': {
                'description': '😴 Sleep gesture - Hand on side of head (pillow)',
                'category': 'daily_needs',
                'priority': 'medium',
                'variations': ['head_on_hand', 'sleeping_pose']
            },
            
            # Emotions
            'senang': {
                'description': '😊 Happy gesture - Upward gesture or clapping',
                'category': 'emotions',
                'priority': 'low',
                'variations': ['thumbs_up', 'clapping_motion', 'upward_gesture']
            },
            'sedih': {
                'description': '😢 Sad gesture - Downward motion or wiping eyes',
                'category': 'emotions',
                'priority': 'low',
                'variations': ['downward_motion', 'wiping_eyes']
            }
        }
        
        # Data collection settings
        self.samples_per_gesture = 150  # Increased dari 100
        self.recording_duration = 6.0   # Increased untuk better quality
        self.stability_threshold = 0.8   # Landmark stability requirement
        
        # Quality metrics
        self.quality_metrics = {
            'min_landmarks_detected': 0.9,  # 90% of frames must have landmarks
            'landmark_stability': 0.8,       # Stability score requirement
            'min_gesture_duration': 3.0,     # Minimum gesture hold time
            'max_noise_ratio': 0.2          # Maximum noise dalam landmarks
        }
        
        # Initialize components
        self.hand_tracker = None
        self.camera_handler = None
        
        # Data storage
        self.session_data = []
        self.session_stats = {
            'total_samples': 0,
            'gestures_recorded': {},
            'quality_scores': {},
            'session_start': time.time()
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 EnhancedDatasetBuilder initialized")
        print(f"📁 Output directory: {output_dir}")
        print(f"🤚 Enhanced gestures available: {len(self.enhanced_gestures)}")
    
    def initialize_hardware(self) -> bool:
        """Initialize enhanced hardware components"""
        print("🔧 Initializing enhanced hardware components...")
        
        # Initialize enhanced hand tracker
        self.hand_tracker = EnhancedHandTracker(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            visualization_mode='full'
        )
        
        # Initialize enhanced camera handler
        self.camera_handler = EnhancedCameraHandler(
            target_fps=30,
            resolution=(640, 480),
            auto_optimize=True
        )
        
        # Test camera initialization
        camera_success = self.camera_handler.initialize_camera()
        if not camera_success:
            print("❌ Enhanced camera initialization failed")
            return False
        
        # Start camera capture
        capture_success = self.camera_handler.start_capture()
        if not capture_success:
            print("❌ Enhanced camera capture failed")
            return False
        
        print("✅ Enhanced hardware initialized successfully")
        
        # Show camera info
        camera_info = self.camera_handler.get_enhanced_camera_info()
        print(f"📊 Camera backend: {camera_info['backend']}")
        print(f"📊 Resolution: {camera_info['resolution']['actual']['width']}x{camera_info['resolution']['actual']['height']}")
        
        return True
    
    def show_enhanced_menu(self):
        """Show enhanced main menu dengan gesture categories"""
        print("\n" + "="*60)
        print("🤚 ENHANCED GESTURE DATASET BUILDER")
        print("="*60)
        
        # Group gestures by category
        categories = {}
        for gesture, info in self.enhanced_gestures.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(gesture)
        
        # Show available gestures by category
        print("📋 Available Gesture Categories:")
        for i, (category, gestures) in enumerate(categories.items(), 1):
            gesture_list = ", ".join(gestures)
            print(f"  {i}. {category.title()}: {gesture_list}")
        
        print("\n🎯 Options:")
        print("1. Record specific gesture")
        print("2. Record high-priority gestures (emergency + basic)")
        print("3. Record by category")
        print("4. Record all gestures")
        print("5. View existing data")
        print("6. Data quality analysis")
        print("7. Test enhanced hand tracking")
        print("8. Exit")
        print("="*60)
    
    def record_specific_gesture(self, gesture_name: str, target_samples: int = None) -> int:
        """
        Record data untuk specific gesture dengan enhanced quality control
        
        Args:
            gesture_name: Name of gesture to record
            target_samples: Target number of samples
            
        Returns:
            int: Number of high-quality samples recorded
        """
        if target_samples is None:
            target_samples = self.samples_per_gesture
        
        if gesture_name not in self.enhanced_gestures:
            print(f"❌ Gesture '{gesture_name}' not found in enhanced gesture list")
            return 0
        
        gesture_info = self.enhanced_gestures[gesture_name]
        
        print(f"\n📹 Recording Enhanced Gesture: '{gesture_name}'")
        print("="*50)
        print(f"💡 {gesture_info['description']}")
        print(f"🏷️  Category: {gesture_info['category'].title()}")
        print(f"⭐ Priority: {gesture_info['priority'].title()}")
        print(f"🎯 Target samples: {target_samples}")
        print(f"⏱️  Recording duration: {self.recording_duration} seconds")
        
        if 'variations' in gesture_info:
            print(f"🔄 Variations to try: {', '.join(gesture_info['variations'])}")
        
        print("\n📋 Enhanced Recording Instructions:")
        print("- Position hand clearly dalam camera frame")
        print("- Hold gesture steady selama recording")
        print("- Try different angles dan distances")
        print("- Maintain good lighting")
        print("- Record akan otomatis start setelah countdown")
        
        input("\n⌨️  Press Enter to start enhanced recording...")
        
        # Enhanced countdown dengan gesture preview
        self._show_enhanced_countdown(gesture_name)
        
        # Enhanced recording dengan quality monitoring
        recorded_samples = self._record_enhanced_gesture_data(gesture_name, target_samples)
        
        print(f"\n✅ Enhanced recording complete! Recorded {recorded_samples} high-quality samples")
        
        # Update session stats
        if gesture_name not in self.session_stats['gestures_recorded']:
            self.session_stats['gestures_recorded'][gesture_name] = 0
        self.session_stats['gestures_recorded'][gesture_name] += recorded_samples
        self.session_stats['total_samples'] += recorded_samples
        
        return recorded_samples
    
    def _show_enhanced_countdown(self, gesture_name: str):
        """Enhanced countdown dengan gesture preview dan quality indicators"""
        print("\n⏰ Enhanced countdown starting...")
        
        countdown_duration = 5  # Extended countdown
        countdown_start = time.time()
        
        while True:
            # Get current frame
            frame = self.camera_handler.get_frame()
            if frame is None:
                continue
            
            # Calculate remaining time
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed
            
            if remaining <= 0:
                break
            
            # Process frame dengan enhanced hand tracker
            processed_frame, landmarks = self.hand_tracker.process_frame(frame)
            
            # Enhanced countdown overlay
            overlay = processed_frame.copy()
            height, width = overlay.shape[:2]
            
            # Large countdown number dengan glow effect
            countdown_num = int(remaining) + 1
            text_size = cv2.getTextSize(str(countdown_num), cv2.FONT_HERSHEY_SIMPLEX, 6, 15)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            # Glow effect
            cv2.putText(overlay, str(countdown_num), (text_x+5, text_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20)  # Shadow
            cv2.putText(overlay, str(countdown_num), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 15)  # Main number
            
            # Gesture information
            cv2.putText(overlay, f"Prepare: {gesture_name}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            
            # Hand detection quality indicator
            if landmarks:
                quality_score = self._calculate_landmark_quality(landmarks)
                quality_text = f"Hand Quality: {quality_score:.1%}"
                color = (0, 255, 0) if quality_score > 0.8 else (0, 255, 255) if quality_score > 0.5 else (0, 0, 255)
                cv2.putText(overlay, quality_text, (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(overlay, "✅ HAND DETECTED - READY!", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(overlay, "❌ NO HAND - SHOW YOUR HAND", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Recording tip
            cv2.putText(overlay, "TIP: Hold gesture steady throughout recording", 
                       (50, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced Gesture Recording', overlay)
            
            # Handle exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return False
        
        # Final "RECORDING!" message dengan enhanced effect
        frame = self.camera_handler.get_frame()
        if frame is not None:
            overlay = frame.copy()
            height, width = overlay.shape[:2]
            
            # Pulsing effect background
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 20)
            overlay = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
            
            # Large recording text
            cv2.putText(overlay, "RECORDING NOW!", 
                       (width//2 - 200, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 8)
            cv2.putText(overlay, f"Gesture: {gesture_name}", 
                       (width//2 - 150, height//2 + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            cv2.imshow('Enhanced Gesture Recording', overlay)
            cv2.waitKey(1000)  # Show for 1 second
        
        return True
    
    def _record_enhanced_gesture_data(self, gesture_name: str, target_samples: int) -> int:
        """
        Record gesture data dengan enhanced quality monitoring dan real-time feedback
        
        Args:
            gesture_name: Name of gesture
            target_samples: Target number of samples
            
        Returns:
            int: Number of high-quality samples recorded
        """
        recorded_samples = 0
        high_quality_samples = 0
        recording_start = time.time()
        last_sample_time = 0
        
        # Quality tracking
        quality_scores = []
        landmark_stability_history = []
        
        # Adaptive sample interval
        sample_interval = self.recording_duration / (target_samples * 1.5)  # Over-sample untuk quality filtering
        
        while True:
            current_time = time.time()
            elapsed = current_time - recording_start
            
            # Check if recording finished
            if elapsed >= self.recording_duration:
                break
            
            # Get enhanced frame
            frame = self.camera_handler.get_frame()
            if frame is None:
                continue
            
            # Process dengan enhanced hand tracker
            processed_frame, landmarks = self.hand_tracker.process_frame(frame)
            
            # Record sample jika conditions met
            if (landmarks and 
                current_time - last_sample_time >= sample_interval):
                
                # Calculate quality metrics
                quality_score = self._calculate_landmark_quality(landmarks)
                stability_score = self._calculate_stability_score(landmarks, landmark_stability_history)
                
                # Only record high-quality samples
                if quality_score >= self.quality_metrics['min_landmarks_detected']:
                    data_row = [gesture_name] + landmarks
                    self.session_data.append({
                        'data': data_row,
                        'quality_score': quality_score,
                        'stability_score': stability_score,
                        'timestamp': current_time
                    })
                    
                    recorded_samples += 1
                    
                    if quality_score >= 0.9 and stability_score >= 0.8:
                        high_quality_samples += 1
                    
                    last_sample_time = current_time
                    quality_scores.append(quality_score)
            
            # Enhanced real-time feedback overlay
            display_frame = self._add_enhanced_recording_overlay(
                processed_frame, gesture_name, elapsed, 
                recorded_samples, target_samples, landmarks,
                quality_scores
            )
            
            cv2.imshow('Enhanced Gesture Recording', display_frame)
            
            # Handle exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        # Store quality metrics
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            self.session_stats['quality_scores'][gesture_name] = {
                'average_quality': avg_quality,
                'total_samples': recorded_samples,
                'high_quality_samples': high_quality_samples,
                'quality_percentage': (high_quality_samples / recorded_samples) * 100 if recorded_samples > 0 else 0
            }
        
        return high_quality_samples  # Return only high-quality samples
    
    def _calculate_landmark_quality(self, landmarks: List[float]) -> float:
        """Calculate quality score untuk landmark data"""
        if not landmarks or len(landmarks) != 42:
            return 0.0
        
        # Convert ke numpy array untuk easier processing
        landmark_array = np.array(landmarks).reshape(21, 2)
        
        # Quality factors
        quality_factors = []
        
        # 1. Landmark spread (hand should span reasonable area)
        x_coords = landmark_array[:, 0]
        y_coords = landmark_array[:, 1]
        x_spread = np.max(x_coords) - np.min(x_coords)
        y_spread = np.max(y_coords) - np.min(y_coords)
        
        # Reasonable hand size (normalized coordinates)
        spread_score = min(1.0, (x_spread + y_spread) / 0.5)  # Expected total spread ~0.5
        quality_factors.append(spread_score)
        
        # 2. Landmark consistency (check for outliers)
        distances_from_wrist = np.linalg.norm(landmark_array - landmark_array[0], axis=1)
        reasonable_distances = np.sum(distances_from_wrist < 0.8) / len(distances_from_wrist)  # Within reasonable range
        quality_factors.append(reasonable_distances)
        
        # 3. Finger structure integrity
        finger_tips = [4, 8, 12, 16, 20]  # Fingertip indices
        fingertip_distances = []
        for tip_idx in finger_tips:
            distance = np.linalg.norm(landmark_array[tip_idx] - landmark_array[0])  # Distance from wrist
            fingertip_distances.append(distance)
        
        # Reasonable fingertip spread
        fingertip_score = 1.0 if 0.2 < np.mean(fingertip_distances) < 0.7 else 0.5
        quality_factors.append(fingertip_score)
        
        # Overall quality score
        overall_quality = np.mean(quality_factors)
        return overall_quality
    
    def _calculate_stability_score(self, current_landmarks: List[float], 
                                 history: List[List[float]]) -> float:
        """Calculate stability score based on landmark history"""
        if len(history) < 3:  # Need history untuk stability
            history.append(current_landmarks)
            return 0.5  # Neutral score
        
        # Add current landmarks
        history.append(current_landmarks)
        
        # Keep only recent history
        if len(history) > 10:
            history.pop(0)
        
        # Calculate stability as inverse of average movement
        movements = []
        for i in range(1, len(history)):
            prev_landmarks = np.array(history[i-1])
            curr_landmarks = np.array(history[i])
            movement = np.mean(np.abs(curr_landmarks - prev_landmarks))
            movements.append(movement)
        
        if movements:
            avg_movement = np.mean(movements)
            # Stability score: lower movement = higher stability
            stability_score = max(0.0, 1.0 - (avg_movement / 0.1))  # Normalize by expected max movement
            return min(1.0, stability_score)
        
        return 0.5
    
    def _add_enhanced_recording_overlay(self, frame: np.ndarray, gesture_name: str,
                                      elapsed: float, recorded_samples: int, 
                                      target_samples: int, landmarks: Optional[List],
                                      quality_scores: List[float]) -> np.ndarray:
        """Add comprehensive recording overlay dengan real-time quality metrics"""
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Enhanced semi-transparent background
        cv2.rectangle(overlay, (10, 10), (width-10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Recording progress bar
        progress = elapsed / self.recording_duration
        bar_width = int((width - 40) * progress)
        cv2.rectangle(frame, (20, 25), (width-20, 55), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 25), (20 + bar_width, 55), (0, 255, 0), -1)
        
        # Recording info
        cv2.putText(frame, f"Recording: {gesture_name}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s / {self.recording_duration}s", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Samples: {recorded_samples} / {target_samples}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Quality indicators
        if landmarks:
            current_quality = self._calculate_landmark_quality(landmarks)
            quality_color = (0, 255, 0) if current_quality > 0.8 else (0, 255, 255) if current_quality > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"Quality: {current_quality:.1%}", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
            
            # Average quality
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                cv2.putText(frame, f"Avg Quality: {avg_quality:.1%}", (300, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recording status indicator
        if landmarks:
            cv2.putText(frame, "🔴 RECORDING", (width-200, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "⏸️ PAUSED - NO HAND", (width-250, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Hold gesture steady | ESC to stop early", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def record_high_priority_gestures(self):
        """Record high-priority gestures (emergency + basic communication)"""
        high_priority = [name for name, info in self.enhanced_gestures.items() 
                        if info['priority'] == 'high']
        
        print(f"\n🚨 Recording High-Priority Gestures ({len(high_priority)} gestures)")
        print("="*60)
        
        for gesture in high_priority:
            print(f"📍 Next: {gesture}")
            response = input(f"⌨️  Record '{gesture}'? (y/n/skip): ").lower()
            
            if response == 'n':
                break
            elif response == 'skip':
                continue
            
            samples = self.record_specific_gesture(gesture)
            print(f"✅ Recorded {samples} samples for '{gesture}'")
        
        print("\n✅ High-priority gesture recording completed!")
    
    def save_enhanced_dataset(self, filename: str = None) -> bool:
        """
        Save enhanced dataset dengan quality metrics
        
        Args:
            filename: Custom filename untuk dataset
            
        Returns:
            bool: True jika berhasil save
        """
        if not self.session_data:
            print("⚠️  No data to save")
            return False
        
        try:
            # Filter dan prepare data
            high_quality_data = []
            quality_stats = {'total': 0, 'high_quality': 0, 'medium_quality': 0, 'low_quality': 0}
            
            for sample in self.session_data:
                quality_stats['total'] += 1
                
                # Quality filtering
                if sample['quality_score'] >= 0.9:
                    quality_stats['high_quality'] += 1
                    high_quality_data.append(sample['data'])
                elif sample['quality_score'] >= 0.7:
                    quality_stats['medium_quality'] += 1
                    high_quality_data.append(sample['data'])  # Include medium quality
                else:
                    quality_stats['low_quality'] += 1
                    # Skip low quality samples
            
            if not high_quality_data:
                print("❌ No high-quality samples to save")
                return False
            
            # Create enhanced dataset
            columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            df = pd.DataFrame(high_quality_data, columns=columns)
            
            # Generate filename
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'enhanced_gestures_{timestamp}.csv'
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Load existing data jika ada
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                df = pd.concat([existing_df, df], ignore_index=True)
                print(f"📝 Added {len(high_quality_data)} high-quality samples to existing data")
            else:
                print(f"📝 Created new dataset with {len(high_quality_data)} high-quality samples")
            
            # Save dataset
            df.to_csv(filepath, index=False)
            
            # Save quality report
            self._save_quality_report(filepath.replace('.csv', '_quality_report.json'), quality_stats)
            
            print(f"✅ Enhanced dataset saved to {filepath}")
            print(f"📊 Quality distribution: High={quality_stats['high_quality']}, "
                  f"Medium={quality_stats['medium_quality']}, Low={quality_stats['low_quality']}")
            
            # Show dataset statistics
            self._show_dataset_statistics(df)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving enhanced dataset: {e}")
            return False
    
    def _save_quality_report(self, filepath: str, quality_stats: Dict):
        """Save quality report dengan detailed metrics"""
        import json
        
        report = {
            'session_info': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_recording_time': time.time() - self.session_stats['session_start'],
                'gestures_recorded': list(self.session_stats['gestures_recorded'].keys())
            },
            'quality_stats': quality_stats,
            'gesture_quality': self.session_stats['quality_scores'],
            'recording_settings': {
                'samples_per_gesture': self.samples_per_gesture,
                'recording_duration': self.recording_duration,
                'stability_threshold': self.stability_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Quality report saved to {filepath}")
    
    def _show_dataset_statistics(self, df: pd.DataFrame):
        """Show comprehensive dataset statistics"""
        print("\n📊 ENHANCED DATASET STATISTICS:")
        print("="*50)
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns) - 1}")
        
        # Gesture distribution
        print("\nGesture distribution:")
        gesture_counts = df['label'].value_counts()
        for gesture, count in gesture_counts.items():
            percentage = (count / len(df)) * 100
            category = self.enhanced_gestures.get(gesture, {}).get('category', 'unknown')
            print(f"  {gesture} ({category}): {count} samples ({percentage:.1f}%)")
        
        print(f"\nUnique gestures: {len(gesture_counts)}")
        print(f"Most common: {gesture_counts.index[0]} ({gesture_counts.iloc[0]} samples)")
        print(f"Least common: {gesture_counts.index[-1]} ({gesture_counts.iloc[-1]} samples)")
        
        # Data balance score
        min_samples = gesture_counts.min()
        max_samples = gesture_counts.max()
        balance_score = min_samples / max_samples if max_samples > 0 else 0
        print(f"Data balance score: {balance_score:.3f} (1.0 = perfectly balanced)")
    
    def analyze_data_quality(self, filepath: str = None):
        """Analyze existing dataset quality"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'gestures.csv')
        
        if not os.path.exists(filepath):
            print(f"❌ Dataset not found: {filepath}")
            return
        
        try:
            df = pd.read_csv(filepath)
            print(f"\n🔍 ANALYZING DATASET QUALITY: {filepath}")
            print("="*60)
            
            # Basic statistics
            self._show_dataset_statistics(df)
            
            # Quality analysis
            print("\n🔍 QUALITY ANALYSIS:")
            print("-"*30)
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            print(f"Missing values: {missing_values}")
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            print(f"Duplicate rows: {duplicates} ({(duplicates/len(df)*100):.1f}%)")
            
            # Landmark range analysis
            feature_cols = [col for col in df.columns if col != 'label']
            
            print(f"\nLandmark coordinate analysis:")
            for col in feature_cols[:5]:  # Show first 5 features
                col_min, col_max = df[col].min(), df[col].max()
                col_std = df[col].std()
                print(f"  {col}: range=[{col_min:.3f}, {col_max:.3f}], std={col_std:.3f}")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            print("-"*30)
            
            if duplicates > len(df) * 0.1:
                print("⚠️  High duplicate rate - consider data cleaning")
            if missing_values > 0:
                print("⚠️  Missing values found - requires cleaning")
            
            balance_score = df['label'].value_counts().min() / df['label'].value_counts().max()
            if balance_score < 0.5:
                print("⚠️  Imbalanced dataset - collect more data untuk underrepresented gestures")
            else:
                print("✅ Dataset is reasonably balanced")
            
            if len(df) < 500:
                print("⚠️  Small dataset - consider collecting more samples")
            else:
                print("✅ Dataset size is adequate for training")
                
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
    
    def test_enhanced_tracking(self):
        """Test enhanced hand tracking dengan real-time visualization"""
        print("\n🧪 Enhanced Hand Tracking Test")
        print("⌨️  Controls: ESC=Exit, M=Switch mode, SPACE=Screenshot")
        
        if not self.hand_tracker or not self.camera_handler:
            if not self.initialize_hardware():
                print("❌ Hardware initialization failed")
                return
        
        try:
            while True:
                frame = self.camera_handler.get_frame()
                if frame is None:
                    continue
                
                # Process dengan enhanced hand tracker
                processed_frame, landmarks = self.hand_tracker.process_frame(frame)
                
                # Add enhanced performance info
                display_frame = self.hand_tracker.draw_performance_info(
                    processed_frame, landmarks, "Enhanced Tracking Test"
                )
                
                cv2.imshow('Enhanced Hand Tracking Test', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('m') or key == ord('M'):
                    self.hand_tracker.switch_visualization_mode()
                elif key == 32:  # SPACE
                    cv2.imwrite(f'enhanced_tracking_test_{int(time.time())}.jpg', display_frame)
                    print("📸 Screenshot saved!")
        
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted")
        
        finally:
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🗑️  Cleaning up enhanced resources...")
        
        cv2.destroyAllWindows()
        
        if self.camera_handler:
            self.camera_handler.release()
        
        if self.hand_tracker:
            self.hand_tracker.cleanup()
        
        print("✅ Enhanced cleanup complete")
    
    def run_interactive_session(self):
        """Run enhanced interactive session"""
        if not self.initialize_hardware():
            print("❌ Enhanced hardware initialization failed")
            return
        
        try:
            while True:
                self.show_enhanced_menu()
                choice = input("\n⌨️  Select option (1-8): ").strip()
                
                if choice == '1':
                    gesture_name = input("🏷️  Enter gesture name: ").strip()
                    if gesture_name in self.enhanced_gestures:
                        self.record_specific_gesture(gesture_name)
                        
                        # Ask untuk save
                        save_choice = input("\n💾 Save data now? (y/n): ").lower()
                        if save_choice == 'y':
                            self.save_enhanced_dataset()
                    else:
                        print(f"❌ Gesture '{gesture_name}' not found in enhanced gesture list")
                        print(f"Available: {list(self.enhanced_gestures.keys())}")
                
                elif choice == '2':
                    self.record_high_priority_gestures()
                    if self.session_data:
                        self.save_enhanced_dataset()
                
                elif choice == '3':
                    # Record by category
                    categories = set(info['category'] for info in self.enhanced_gestures.values())
                    print(f"📋 Available categories: {', '.join(categories)}")
                    category = input("🏷️  Enter category: ").strip().lower()
                    
                    category_gestures = [name for name, info in self.enhanced_gestures.items() 
                                       if info['category'] == category]
                    
                    if category_gestures:
                        print(f"🎯 Recording {len(category_gestures)} gestures dalam category '{category}'")
                        for gesture in category_gestures:
                            response = input(f"Record '{gesture}'? (y/n/skip): ").lower()
                            if response == 'y':
                                self.record_specific_gesture(gesture)
                            elif response == 'n':
                                break
                        
                        if self.session_data:
                            self.save_enhanced_dataset()
                    else:
                        print(f"❌ No gestures found dalam category '{category}'")
                
                elif choice == '4':
                    print("🎯 Recording ALL enhanced gestures...")
                    for gesture_name in self.enhanced_gestures.keys():
                        response = input(f"Record '{gesture_name}'? (y/n/stop): ").lower()
                        if response == 'y':
                            self.record_specific_gesture(gesture_name)
                        elif response == 'stop':
                            break
                    
                    if self.session_data:
                        self.save_enhanced_dataset()
                
                elif choice == '5':
                    self.analyze_data_quality()
                
                elif choice == '6':
                    filepath = input("📁 Enter dataset path (or press Enter for default): ").strip()
                    self.analyze_data_quality(filepath if filepath else None)
                
                elif choice == '7':
                    self.test_enhanced_tracking()
                
                elif choice == '8':
                    break
                
                else:
                    print("❌ Invalid option")
        
        except KeyboardInterrupt:
            print("\n⏹️  Session interrupted by user")
        
        finally:
            # Show session summary
            if self.session_data:
                print("\n📊 ENHANCED SESSION SUMMARY:")
                print("="*40)
                elapsed = time.time() - self.session_stats['session_start']
                print(f"Session duration: {elapsed/60:.1f} minutes")
                print(f"Total samples collected: {len(self.session_data)}")
                
                if self.session_stats['gestures_recorded']:
                    print("Gestures recorded:")
                    for gesture, count in self.session_stats['gestures_recorded'].items():
                        print(f"   {gesture}: {count} samples")
                
                # Auto-save remaining data
                print("\n💾 Auto-saving session data...")
                self.save_enhanced_dataset()
            
            # Cleanup
            self.cleanup()


def main():
    """Main function dengan enhanced CLI"""
    parser = argparse.ArgumentParser(description='Enhanced Gesture Dataset Builder')
    parser.add_argument('--output', '-o', 
                       default='/app/data',
                       help='Output directory untuk enhanced datasets')
    parser.add_argument('--samples', '-s', 
                       type=int, default=150,
                       help='Target samples per gesture (enhanced default: 150)')
    parser.add_argument('--duration', '-d', 
                       type=float, default=6.0,
                       help='Recording duration per gesture (enhanced default: 6.0s)')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Gesture Dataset Builder")
    print("="*50)
    print(f"📁 Output directory: {args.output}")
    print(f"🎯 Target samples per gesture: {args.samples}")
    print(f"⏱️  Recording duration: {args.duration}s")
    print("🎨 Features: Enhanced tracking, Quality control, ASL-inspired gestures")
    
    # Create enhanced builder
    builder = EnhancedDatasetBuilder(output_dir=args.output)
    builder.samples_per_gesture = args.samples
    builder.recording_duration = args.duration
    
    # Run enhanced interactive session
    builder.run_interactive_session()


if __name__ == "__main__":
    main()