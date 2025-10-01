"""
Test Components - Test individual components tanpa GUI
Testing hand tracker, gender detector, dan gesture predictor
"""

import cv2
import numpy as np
import time
from core.enhanced_hand_tracker import EnhancedHandTracker
from core.gender_detector import GenderDetector
from core.gesture_predictor import GesturePredictor


def test_hand_tracker():
    """Test enhanced hand tracker"""
    print("🤚 Testing Enhanced Hand Tracker...")
    
    try:
        tracker = EnhancedHandTracker(visualization_mode='full')
        print("✅ EnhancedHandTracker initialized")
        
        # Test dengan dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame, landmarks = tracker.process_frame(dummy_frame)
        
        print(f"✅ Frame processing test completed")
        print(f"   Landmarks detected: {landmarks is not None}")
        
        # Get performance stats
        stats = tracker.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        tracker.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ HandTracker error: {e}")
        return False


def test_gender_detector():
    """Test gender detector"""
    print("👤 Testing Gender Detector...")
    
    try:
        detector = GenderDetector()
        print("✅ GenderDetector initialized")
        
        # Test dengan dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gender, confidence, face_info = detector.detect_gender(dummy_frame)
        
        print(f"✅ Gender detection test completed")
        print(f"   Gender: {gender}, Confidence: {confidence}")
        print(f"   Face detected: {face_info is not None}")
        
        # Get detection stats
        stats = detector.get_detection_stats()
        print(f"✅ Detection stats: {stats}")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ GenderDetector error: {e}")
        return False


def test_gesture_predictor():
    """Test gesture predictor"""
    print("🤖 Testing Gesture Predictor...")
    
    try:
        predictor = GesturePredictor()
        
        if not predictor.is_loaded:
            print("❌ Model not loaded")
            return False
        
        print("✅ GesturePredictor loaded successfully")
        
        # Test predictions dengan dummy landmarks
        dummy_landmarks = np.random.normal(0, 0.2, 42).tolist()
        gesture, confidence = predictor.predict_gesture(dummy_landmarks)
        
        print(f"✅ Prediction test completed")
        print(f"   Gesture: {gesture}, Confidence: {confidence:.3f}")
        
        # Get prediction stats
        stats = predictor.get_prediction_stats()
        print(f"✅ Prediction stats:")
        for key, value in stats.items():
            print(f"      {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ GesturePredictor error: {e}")
        return False


def test_camera_access():
    """Test camera access"""
    print("📹 Testing Camera Access...")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  Camera not available (normal untuk container environment)")
            cap.release()
            return True  # This is expected dalam container
        
        print("✅ Camera accessible")
        
        # Try to read frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame captured: {frame.shape}")
        else:
            print("⚠️  Could not capture frame")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera access error: {e}")
        return False


def test_integrated_pipeline():
    """Test integrated pipeline dengan all components"""
    print("🔄 Testing Integrated Pipeline...")
    
    try:
        # Initialize all components
        hand_tracker = EnhancedHandTracker(visualization_mode='full')
        gender_detector = GenderDetector()
        gesture_predictor = GesturePredictor()
        
        print("✅ All components initialized")
        
        # Create test frame dengan some pattern
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process dengan hand tracker
        processed_frame, landmarks = hand_tracker.process_frame(test_frame)
        print(f"✅ Hand tracking: landmarks={landmarks is not None}")
        
        # Process dengan gender detector
        gender, g_confidence, face_info = gender_detector.detect_gender(test_frame)
        print(f"✅ Gender detection: {gender} ({g_confidence:.2f})")
        
        # Process gesture jika ada landmarks
        if landmarks:
            gesture, h_confidence = gesture_predictor.predict_gesture(landmarks)
            print(f"✅ Gesture prediction: {gesture} ({h_confidence:.2f})")
        else:
            # Test dengan dummy landmarks
            dummy_landmarks = np.random.normal(0, 0.2, 42).tolist()
            gesture, h_confidence = gesture_predictor.predict_gesture(dummy_landmarks)
            print(f"✅ Gesture prediction (dummy): {gesture} ({h_confidence:.2f})")
        
        # Cleanup
        hand_tracker.cleanup()
        gender_detector.cleanup()
        
        print("✅ Integrated pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 GESTURETALK COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Hand Tracker", test_hand_tracker),
        ("Gender Detector", test_gender_detector),
        ("Gesture Predictor", test_gesture_predictor),
        ("Camera Access", test_camera_access),
        ("Integrated Pipeline", test_integrated_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name.upper()}")
        print("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! Core components ready!")
        print("✅ You can run 'python app.py' to start the full application")
    elif success_rate >= 60:
        print("👍 GOOD! Most components working")
        print("⚠️  Some issues detected, but core functionality available")
    else:
        print("🔧 NEEDS WORK! Major issues detected")
        print("❌ Please check the errors above")
    
    print(f"\n💡 Next Steps:")
    print("   1. Run 'python app.py' untuk start full GUI application")
    print("   2. Test dengan webcam jika available")
    print("   3. Observe hand rigging dan gender detection dalam action")


if __name__ == "__main__":
    main()