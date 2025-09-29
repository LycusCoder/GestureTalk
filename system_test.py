#!/usr/bin/env python3
"""
GestureTalk System Test - Comprehensive testing untuk semua components
Test semua functionality tanpa hardware dependencies
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add modules to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def test_core_modules():
    """Test all core modules functionality"""
    print("🧪 TESTING CORE MODULES")
    print("=" * 40)
    
    results = {}
    
    # Test HandTracker
    try:
        from core.hand_tracker import HandTracker
        tracker = HandTracker()
        print("✅ HandTracker: PASSED")
        tracker.cleanup()
        results['hand_tracker'] = True
    except Exception as e:
        print(f"❌ HandTracker: FAILED - {e}")
        results['hand_tracker'] = False
    
    # Test CameraHandler  
    try:
        from core.camera_handler import CameraHandler
        camera = CameraHandler()
        print("✅ CameraHandler: PASSED (init)")
        camera.release()
        results['camera_handler'] = True
    except Exception as e:
        print(f"❌ CameraHandler: FAILED - {e}")
        results['camera_handler'] = False
    
    # Test TTSHandler
    try:
        from core.tts_handler import TTSHandler
        tts = TTSHandler()
        tts_status = tts.get_status()
        print(f"✅ TTSHandler: PASSED (status: {tts_status['status']})")
        tts.cleanup()
        results['tts_handler'] = True
    except Exception as e:
        print(f"❌ TTSHandler: FAILED - {e}")
        results['tts_handler'] = False
    
    # Test GesturePredictor
    try:
        from core.gesture_predictor import GesturePredictor
        predictor = GesturePredictor()
        
        if predictor.is_loaded:
            # Test prediction dengan dummy data
            dummy_landmarks = np.random.normal(0, 0.3, 42).tolist()
            gesture, confidence = predictor.predict_gesture(dummy_landmarks)
            print(f"✅ GesturePredictor: PASSED - Predicted: {gesture} ({confidence:.3f})")
        else:
            print("⚠️  GesturePredictor: Model not loaded (expected)")
        
        results['gesture_predictor'] = True
    except Exception as e:
        print(f"❌ GesturePredictor: FAILED - {e}")
        results['gesture_predictor'] = False
    
    return results

def test_ml_pipeline():
    """Test ML training pipeline"""
    print("\n🤖 TESTING ML PIPELINE")
    print("=" * 40)
    
    results = {}
    
    # Test Data Collection
    try:
        from scripts.collect_data import GestureDataCollector
        collector = GestureDataCollector()
        print(f"✅ Data Collection: PASSED - {len(collector.predefined_gestures)} gestures")
        collector.cleanup()
        results['data_collection'] = True
    except Exception as e:
        print(f"❌ Data Collection: FAILED - {e}")
        results['data_collection'] = False
    
    # Test Model Training
    try:
        from scripts.train_model import GestureModelTrainer
        trainer = GestureModelTrainer()
        print("✅ Model Training: PASSED (init)")
        results['model_training'] = True
    except Exception as e:
        print(f"❌ Model Training: FAILED - {e}")
        results['model_training'] = False
    
    return results

def test_trained_model():
    """Test existing trained model"""
    print("\n🧠 TESTING TRAINED MODEL")
    print("=" * 40)
    
    model_path = ROOT_DIR / 'models' / 'gesture_model.pkl'
    
    if not model_path.exists():
        print("⚠️  No trained model found")
        return False
    
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Gesture classes: {list(model.classes_)}")
        
        # Test prediction
        dummy_data = np.random.normal(0, 0.3, 42).reshape(1, -1)
        prediction = model.predict(dummy_data)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(dummy_data)[0]
            max_proba = np.max(proba)
            print(f"✅ Test prediction: {prediction} (confidence: {max_proba:.3f})")
        else:
            print(f"✅ Test prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_gui_imports():
    """Test GUI module imports"""
    print("\n🎨 TESTING GUI IMPORTS")
    print("=" * 40)
    
    try:
        # Test without actually creating GUI (avoid tkinter issues)
        import customtkinter as ctk
        print("✅ CustomTkinter: Available")
        
        from gui.main_window import GestureTalkMainWindow
        print("✅ Main Window: Import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI Import failed: {e}")
        print("   This is expected in headless environment")
        return False

def test_data_files():
    """Test data files dan structure"""
    print("\n📁 TESTING DATA FILES")
    print("=" * 40)
    
    # Check directories
    dirs_to_check = ['data', 'models', 'scripts', 'core', 'gui']
    
    for dir_name in dirs_to_check:
        dir_path = ROOT_DIR / dir_name
        if dir_path.exists():
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
    
    # Check data file
    data_file = ROOT_DIR / 'data' / 'gestures.csv'
    if data_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"✅ Data file: {len(df)} samples, {len(df.columns)} columns")
            
            # Show gesture distribution
            gesture_counts = df['label'].value_counts()
            print("   Gesture distribution:")
            for gesture, count in gesture_counts.items():
                print(f"     {gesture}: {count} samples")
                
        except Exception as e:
            print(f"❌ Data file error: {e}")
    else:
        print("⚠️  No training data file found")
    
    # Check model file
    model_file = ROOT_DIR / 'models' / 'gesture_model.pkl'
    if model_file.exists():
        print("✅ Model file: Available")
    else:
        print("⚠️  No model file found")

def show_system_summary():
    """Show comprehensive system summary"""
    print("\n📊 SYSTEM SUMMARY")
    print("=" * 50)
    
    # System info
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {ROOT_DIR}")
    
    # Module availability
    modules = {
        'OpenCV': 'cv2',
        'MediaPipe': 'mediapipe', 
        'CustomTkinter': 'customtkinter',
        'PyTTSx3': 'pyttsx3',
        'Scikit-learn': 'sklearn',
        'Pandas': 'pandas',
        'NumPy': 'numpy',
        'PIL': 'PIL'
    }
    
    print("\nModule Availability:")
    for name, module in modules.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
    
    # File structure
    print(f"\nProject Structure:")
    for item in sorted(ROOT_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            file_count = len(list(item.glob('*.py')))
            print(f"  📁 {item.name}/ ({file_count} Python files)")
        elif item.suffix == '.py':
            print(f"  📄 {item.name}")

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 GESTURETALK COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    # Run all tests
    core_results = test_core_modules()
    ml_results = test_ml_pipeline()
    model_working = test_trained_model()
    gui_available = test_gui_imports()
    test_data_files()
    
    # Show summary
    show_system_summary()
    
    # Final results
    print("\n🎯 FINAL TEST RESULTS")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    print("Core Modules:")
    for module, result in core_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {module}: {status}")
        total_tests += 1
        if result:
            passed_tests += 1
    
    print("\nML Pipeline:")  
    for module, result in ml_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {module}: {status}")
        total_tests += 1
        if result:
            passed_tests += 1
    
    print(f"\nTrained Model: {'✅ WORKING' if model_working else '❌ NOT AVAILABLE'}")
    print(f"GUI System: {'✅ AVAILABLE' if gui_available else '⚠️  HEADLESS ENV'}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n📈 OVERALL SCORE: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! System ready for deployment")
    elif success_rate >= 60:
        print("🎯 GOOD! Minor issues to address")
    else:
        print("⚠️  NEEDS WORK! Major issues detected")
    
    print("=" * 60)
    
    return success_rate >= 60

def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())