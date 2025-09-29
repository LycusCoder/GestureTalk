#!/usr/bin/env python3
"""
GestureTalk Terminal Demo - Console version untuk testing tanpa GUI
Real-time gesture recognition dengan output di terminal
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add modules to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Core imports
from core.hand_tracker import HandTracker
from core.camera_handler import CameraHandler
from core.tts_handler import TTSHandler
from core.gesture_predictor import GesturePredictor


class GestureTalkTerminal:
    """
    Terminal version dari GestureTalk untuk testing dan demo
    """
    
    def __init__(self):
        """Initialize terminal application"""
        self.is_running = False
        self.gesture_count = 0
        self.last_gesture = None
        self.gesture_history = []
        
        # Core components
        self.hand_tracker = None
        self.camera_handler = None
        self.tts_handler = None
        self.gesture_predictor = None
        
        print("🚀 GestureTalk Terminal Demo")
        print("=" * 50)
    
    def initialize_systems(self):
        """Initialize all core systems"""
        print("🔧 Initializing systems...")
        
        try:
            # Initialize hand tracker
            print("  • Hand Tracker...", end="")
            self.hand_tracker = HandTracker(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print(" ✅")
            
            # Initialize camera
            print("  • Camera Handler...", end="")
            self.camera_handler = CameraHandler(
                target_fps=30,
                resolution=(640, 480)
            )
            success = self.camera_handler.initialize_camera()
            if success:
                print(" ✅")
            else:
                print(" ❌ (Camera not available)")
                return False
            
            # Initialize TTS
            print("  • Text-to-Speech...", end="")
            self.tts_handler = TTSHandler(language='id', rate=150)
            self.tts_handler.start_speech_service()
            print(" ✅" if self.tts_handler.status.name != 'ERROR' else " ⚠️")
            
            # Initialize gesture predictor
            print("  • Gesture Predictor...", end="")
            self.gesture_predictor = GesturePredictor(confidence_threshold=0.6)
            if self.gesture_predictor.is_loaded:
                print(" ✅")
                stats = self.gesture_predictor.get_prediction_stats()
                print(f"    Model: {stats['model_type']}")
                print(f"    Classes: {stats['gesture_classes']}")
            else:
                print(" ❌ (No trained model)")
                return False
            
            return True
            
        except Exception as e:
            print(f"\n❌ System initialization error: {e}")
            return False
    
    def show_instructions(self):
        """Show usage instructions"""
        print("\n📋 INSTRUCTIONS:")
        print("-" * 30)
        print("• Position your hand dalam frame camera")
        print("• Perform one of these gestures:")
        
        if self.gesture_predictor:
            stats = self.gesture_predictor.get_prediction_stats()
            for i, gesture in enumerate(stats['gesture_classes'], 1):
                print(f"  {i}. {gesture}")
        
        print("• Hold gesture untuk 2-3 detik")
        print("• Press Ctrl+C untuk exit")
        print("\n🎯 Starting detection...")
        print("=" * 50)
    
    def run_detection_loop(self):
        """Main detection loop"""
        self.is_running = True
        
        # Start camera capture
        if not self.camera_handler.start_capture():
            print("❌ Failed to start camera capture")
            return
        
        print("📹 Camera started - detecting gestures...")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.is_running:
                # Get frame
                frame = self.camera_handler.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process dengan hand tracker
                processed_frame, landmarks = self.hand_tracker.process_frame(frame)
                
                # Gesture prediction
                gesture_name = None
                confidence = 0.0
                
                if landmarks:
                    gesture_name, confidence = self.gesture_predictor.predict_gesture(landmarks)
                
                # Handle gesture detection
                if gesture_name and gesture_name not in ["Tidak ada", "Tidak dikenali"]:
                    self._handle_gesture_detection(gesture_name, confidence)
                
                # Show periodic status
                frame_count += 1
                if frame_count % 60 == 0:  # Every 2 seconds at 30fps
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    
                    print(f"📊 Status: {frame_count} frames | FPS: {fps:.1f} | "
                          f"Last: {gesture_name or 'None'} ({confidence:.2f})")
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        except Exception as e:
            print(f"\n❌ Detection error: {e}")
        
        finally:
            self.cleanup()
    
    def _handle_gesture_detection(self, gesture_name, confidence):
        """Handle detected gesture"""
        current_time = time.time()
        
        # Avoid duplicate detections
        if not hasattr(self, '_last_detection_time'):
            self._last_detection_time = 0
            self._last_detected_gesture = ""
        
        # Only process jika gesture berbeda atau sudah lewat 3 detik
        if (gesture_name != self._last_detected_gesture or 
            current_time - self._last_detection_time > 3.0):
            
            if confidence >= 0.5:  # Minimum confidence
                # Update counters
                self.gesture_count += 1
                self.last_gesture = gesture_name
                
                # Add to history
                self.gesture_history.append({
                    'gesture': gesture_name,
                    'confidence': confidence,
                    'time': current_time
                })
                
                # Keep only recent 10 gestures
                if len(self.gesture_history) > 10:
                    self.gesture_history = self.gesture_history[-10:]
                
                # Print detection
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n🤚 [{timestamp}] GESTURE DETECTED:")
                print(f"   Gesture: {gesture_name}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Total detected: {self.gesture_count}")
                
                # TTS output
                if self.tts_handler and confidence >= 0.6:
                    print(f"🔊 Speaking: {gesture_name}")
                    self.tts_handler.speak_gesture(gesture_name)
                
                # Update tracking
                self._last_detection_time = current_time
                self._last_detected_gesture = gesture_name
    
    def show_summary(self):
        """Show detection summary"""
        print("\n" + "=" * 50)
        print("📊 DETECTION SUMMARY")
        print("=" * 50)
        print(f"Total gestures detected: {self.gesture_count}")
        print(f"Last gesture: {self.last_gesture or 'None'}")
        
        if self.gesture_history:
            print("\nRecent detections:")
            for i, detection in enumerate(self.gesture_history[-5:], 1):
                timestamp = time.strftime("%H:%M:%S", time.localtime(detection['time']))
                print(f"  {i}. [{timestamp}] {detection['gesture']} "
                      f"({detection['confidence']:.1%})")
        
        print("=" * 50)
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🗑️  Cleaning up resources...")
        
        self.is_running = False
        
        if self.camera_handler:
            self.camera_handler.release()
        
        if self.hand_tracker:
            self.hand_tracker.cleanup()
        
        if self.tts_handler:
            self.tts_handler.cleanup()
        
        print("✅ Cleanup completed")
    
    def run(self):
        """Run the terminal application"""
        try:
            # Initialize systems
            if not self.initialize_systems():
                print("\n❌ Failed to initialize systems")
                return 1
            
            # Show instructions
            self.show_instructions()
            
            # Wait untuk user ready
            input("Press Enter to start detection...")
            
            # Run detection loop
            self.run_detection_loop()
            
            # Show summary
            self.show_summary()
            
            return 0
            
        except Exception as e:
            print(f"\n❌ Application error: {e}")
            return 1


def main():
    """Main function"""
    try:
        # Check basic requirements
        print("🧪 Checking requirements...")
        
        # Check model exists
        model_path = Path(__file__).parent / 'models' / 'gesture_model.pkl'
        if not model_path.exists():
            print("❌ No trained model found")
            print("   Run: python scripts/train_model.py")
            return 1
        
        print("✅ Model found")
        
        # Create dan run terminal app
        app = GestureTalkTerminal()
        return app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Application interrupted")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())