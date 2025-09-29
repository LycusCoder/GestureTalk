#!/usr/bin/env python3
"""
Quick Improvements Script - Immediate enhancements untuk GestureTalk
Apply quick fixes dan improvements untuk better performance
"""

import sys
import os
from pathlib import Path
import json
import pickle
import numpy as np

# Add modules to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))


def improve_model_settings():
    """Improve model settings untuk better performance"""
    print("🔧 IMPROVING MODEL SETTINGS")
    print("=" * 40)
    
    try:
        from core.gesture_predictor import GesturePredictor
        
        # Test current settings
        predictor = GesturePredictor()
        if predictor.is_loaded:
            print("✅ Current model loaded successfully")
            
            # Get current stats
            stats = predictor.get_prediction_stats()
            print(f"   Model: {stats['model_type']}")
            print(f"   Classes: {stats['gesture_classes']}")
            print(f"   Confidence threshold: {stats['confidence_threshold']}")
            
            # Suggest better settings
            print("\n💡 Recommended settings:")
            print("   - Lower confidence threshold: 0.4 (dari 0.6)")
            print("   - Increase stability window: 5 (dari 3)")
            print("   - Add gesture filtering: Remove low-confidence predictions")
            
            return True
        else:
            print("❌ No model loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_better_dummy_data():
    """Create better dummy data dengan more realistic patterns"""
    print("\n🎲 CREATING BETTER DUMMY DATA")
    print("=" * 40)
    
    try:
        import pandas as pd
        
        # Enhanced gesture patterns
        gesture_patterns = {
            'tolong': {
                'pattern': 'raised_hand',  # Tangan terangkat tinggi
                'key_points': [8, 12, 16, 20],  # Fingertips
                'y_offset': -0.3  # Lebih tinggi
            },
            'halo': {
                'pattern': 'waving',  # Gerakan melambai
                'key_points': [8, 12, 16, 20],
                'x_variation': 0.2  # Variasi horizontal
            },
            'terima_kasih': {
                'pattern': 'prayer',  # Gesture berdoa
                'key_points': [4, 8, 12, 16, 20],
                'center_bias': True  # Terpusat di tengah
            },
            'ya': {
                'pattern': 'thumbs_up',
                'key_points': [4],  # Ibu jari
                'y_offset': -0.2
            },
            'tidak': {
                'pattern': 'stop_hand',
                'key_points': [8, 12, 16, 20],
                'palm_forward': True
            }
        }
        
        print("🎯 Generating enhanced gesture patterns...")
        
        data = []
        samples_per_gesture = 150  # Increase dari 30
        
        np.random.seed(42)  # Consistent results
        
        for gesture, config in gesture_patterns.items():
            print(f"   Generating {gesture}...")
            
            for sample in range(samples_per_gesture):
                # Base coordinates (21 landmarks × 2 coords = 42 features)
                landmarks = np.random.normal(0, 0.15, 42)  # Smaller variance
                
                # Apply gesture-specific patterns
                if config['pattern'] == 'raised_hand':
                    # Make fingertips higher
                    for point_idx in config['key_points']:
                        y_idx = point_idx * 2 + 1  # Y coordinate
                        landmarks[y_idx] += config['y_offset']
                        
                elif config['pattern'] == 'waving':
                    # Add horizontal variation
                    for point_idx in config['key_points']:
                        x_idx = point_idx * 2  # X coordinate
                        landmarks[x_idx] += np.random.normal(0, config['x_variation'])
                        
                elif config['pattern'] == 'prayer':
                    # Center all points
                    if config.get('center_bias'):
                        landmarks = landmarks * 0.7  # Smaller spread
                        
                elif config['pattern'] == 'thumbs_up':
                    # Emphasize thumb
                    for point_idx in config['key_points']:
                        y_idx = point_idx * 2 + 1
                        landmarks[y_idx] += config['y_offset']
                        
                elif config['pattern'] == 'stop_hand':
                    # Palm forward (fingertips aligned)
                    if config.get('palm_forward'):
                        for point_idx in config['key_points']:
                            x_idx = point_idx * 2
                            landmarks[x_idx] = np.random.normal(0, 0.1)  # Aligned X
                
                # Add to dataset
                data.append([gesture] + landmarks.tolist())
        
        # Create DataFrame
        columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        df = pd.DataFrame(data, columns=columns)
        
        # Save enhanced data
        data_path = ROOT_DIR / 'data' / 'gestures_enhanced.csv'
        df.to_csv(data_path, index=False)
        
        print(f"✅ Enhanced dataset created: {len(df)} samples")
        print(f"   File: {data_path}")
        
        # Show distribution
        print("\n📊 Gesture distribution:")
        for gesture, count in df['label'].value_counts().items():
            print(f"   {gesture}: {count} samples")
        
        return str(data_path)
        
    except Exception as e:
        print(f"❌ Error creating enhanced data: {e}")
        return None


def train_better_model(data_path):
    """Train better model dengan enhanced data"""
    print("\n🤖 TRAINING BETTER MODEL")
    print("=" * 40)
    
    try:
        from scripts.train_model import GestureModelTrainer
        
        # Create trainer dengan enhanced data
        trainer = GestureModelTrainer(data_file=data_path)
        
        # Run training pipeline
        success = trainer.run_full_training_pipeline()
        
        if success:
            print("✅ Enhanced model training completed!")
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False


def optimize_camera_settings():
    """Create optimized camera configuration"""
    print("\n📹 OPTIMIZING CAMERA SETTINGS")  
    print("=" * 40)
    
    # Camera optimization config
    camera_config = {
        'resolution_priority': [
            (1920, 1080),  # FullHD
            (1280, 720),   # HD  
            (800, 600),    # SVGA
            (640, 480),    # VGA
        ],
        'fps_target': 30,
        'buffer_size': 1,  # Minimize latency
        'auto_exposure': True,
        'auto_focus': True,
        'brightness': 0,
        'contrast': 0,
        'saturation': 0,
    }
    
    # Save config
    config_path = ROOT_DIR / 'camera_config.json'
    with open(config_path, 'w') as f:
        json.dump(camera_config, f, indent=2)
    
    print("✅ Camera configuration created")
    print(f"   File: {config_path}")
    print("   Features:")
    print("   - Multiple resolution fallbacks")
    print("   - Optimized for 30 FPS")
    print("   - Minimal latency buffer")
    print("   - Auto-adjustment enabled")
    
    return str(config_path)


def create_performance_monitor():
    """Create performance monitoring script"""
    print("\n📊 CREATING PERFORMANCE MONITOR")
    print("=" * 40)
    
    monitor_script = '''#!/usr/bin/env python3
"""
Performance Monitor - Real-time performance tracking untuk GestureTalk
"""

import time
import psutil
import json
from collections import deque
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.fps_history = deque(maxlen=60)  # Last 60 FPS readings
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.start_time = time.time()
        
    def update_fps(self, current_fps):
        self.fps_history.append(current_fps)
        
    def update_system_metrics(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        
    def get_performance_report(self):
        if not self.fps_history:
            return {"status": "No data"}
            
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_minutes": (time.time() - self.start_time) / 60,
            "fps": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": sum(self.fps_history) / len(self.fps_history),
                "min": min(self.fps_history),
                "max": max(self.fps_history),
            },
            "cpu": {
                "current": self.cpu_history[-1] if self.cpu_history else 0,
                "average": sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            },
            "memory": {
                "current": self.memory_history[-1] if self.memory_history else 0,
                "average": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            }
        }
    
    def print_status(self):
        report = self.get_performance_report()
        if report.get("status") == "No data":
            print("📊 No performance data yet...")
            return
            
        print(f"📊 PERFORMANCE STATUS")
        print(f"⏱️  Uptime: {report['uptime_minutes']:.1f} minutes")
        print(f"🎬 FPS: {report['fps']['current']:.1f} (avg: {report['fps']['average']:.1f})")
        print(f"💻 CPU: {report['cpu']['current']:.1f}% (avg: {report['cpu']['average']:.1f}%)")
        print(f"🧠 Memory: {report['memory']['current']:.1f}%")

# Usage example:
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Simulate monitoring loop
    for i in range(10):
        monitor.update_fps(28 + i % 5)  # Simulate varying FPS
        monitor.update_system_metrics()
        monitor.print_status()
        time.sleep(1)
'''

    monitor_path = ROOT_DIR / 'performance_monitor.py'
    with open(monitor_path, 'w') as f:
        f.write(monitor_script)
        
    print("✅ Performance monitor created")
    print(f"   File: {monitor_path}")
    print("   Features:")
    print("   - Real-time FPS tracking")
    print("   - CPU/Memory monitoring") 
    print("   - Performance history")
    print("   - Status reporting")
    
    return str(monitor_path)


def create_gesture_config():
    """Create customizable gesture configuration"""
    print("\n🤚 CREATING GESTURE CONFIGURATION")
    print("=" * 40)
    
    gesture_config = {
        "gestures": {
            "tolong": {
                "phrase": "Saya butuh bantuan",
                "priority": "high",
                "confidence_threshold": 0.6,
                "description": "Gesture untuk meminta bantuan darurat"
            },
            "halo": {
                "phrase": "Halo, apa kabar?",
                "priority": "normal", 
                "confidence_threshold": 0.5,
                "description": "Gesture sapaan ramah"
            },
            "terima_kasih": {
                "phrase": "Terima kasih banyak",
                "priority": "normal",
                "confidence_threshold": 0.5,
                "description": "Gesture berterima kasih"
            },
            "ya": {
                "phrase": "Ya, benar",
                "priority": "normal",
                "confidence_threshold": 0.6,
                "description": "Gesture persetujuan"
            },
            "tidak": {
                "phrase": "Tidak, salah",
                "priority": "normal",
                "confidence_threshold": 0.6, 
                "description": "Gesture penolakan"
            }
        },
        "settings": {
            "default_confidence_threshold": 0.5,
            "stability_window": 5,
            "speech_delay": 2.0,
            "max_gesture_history": 10
        }
    }
    
    config_path = ROOT_DIR / 'gesture_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(gesture_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Gesture configuration created")
    print(f"   File: {config_path}")
    print("   Features:")
    print("   - Customizable phrases")
    print("   - Priority levels")
    print("   - Individual confidence thresholds")
    print("   - Flexible settings")
    
    return str(config_path)


def run_quick_improvements():
    """Run all quick improvements"""
    print("🚀 GESTURETALK QUICK IMPROVEMENTS")
    print("=" * 60)
    
    results = {}
    
    # 1. Model settings analysis
    results['model_analysis'] = improve_model_settings()
    
    # 2. Create better dummy data
    enhanced_data_path = create_better_dummy_data()
    results['enhanced_data'] = enhanced_data_path is not None
    
    # 3. Train better model
    if enhanced_data_path:
        results['better_model'] = train_better_model(enhanced_data_path)
    else:
        results['better_model'] = False
    
    # 4. Camera optimization
    camera_config_path = optimize_camera_settings()
    results['camera_config'] = camera_config_path is not None
    
    # 5. Performance monitoring
    monitor_path = create_performance_monitor()
    results['performance_monitor'] = monitor_path is not None
    
    # 6. Gesture configuration
    gesture_config_path = create_gesture_config()
    results['gesture_config'] = gesture_config_path is not None
    
    # Summary
    print("\n🎯 IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    total_improvements = len(results)
    successful_improvements = sum(results.values())
    
    for improvement, success in results.items():
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"   {improvement}: {status}")
    
    success_rate = (successful_improvements / total_improvements) * 100
    print(f"\n📈 SUCCESS RATE: {successful_improvements}/{total_improvements} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! Most improvements applied successfully")
    elif success_rate >= 60:
        print("🎯 GOOD! Major improvements completed")
    else:
        print("⚠️  PARTIAL SUCCESS: Some improvements failed")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Test enhanced model dengan GUI aplikasi")
    print("   2. Collect real gesture data dari users")
    print("   3. Fine-tune performance settings")
    print("   4. Implement advanced features")
    
    return success_rate >= 60


def main():
    """Main improvement function"""
    try:
        success = run_quick_improvements()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Improvement script error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())