#!/usr/bin/env python3
"""
GestureTalk - Main Application Entry Point
Assistive communication system dengan gesture recognition dan TTS
"""

import sys
import os
import logging
from pathlib import Path

# Add modules ke Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT_DIR / 'gesturetalk.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check apakah semua dependencies tersedia"""
    required_modules = [
        'cv2', 'mediapipe', 'customtkinter', 'pyttsx3', 
        'sklearn', 'pandas', 'numpy', 'PIL'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required modules: {missing_modules}")
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("\n📦 Please install missing dependencies:")
        print("pip install opencv-python mediapipe customtkinter pyttsx3 scikit-learn pandas numpy pillow")
        return False
    
    logger.info("✅ All dependencies available")
    return True


def check_system_requirements():
    """Check system requirements"""
    try:
        import cv2
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("⚠️  Camera tidak tersedia, tapi aplikasi tetap bisa jalan")
            print("⚠️  Warning: Camera tidak terdeteksi")
            print("   - Pastikan webcam terhubung")
            print("   - Check permission camera")
        else:
            logger.info("✅ Camera available")
            cap.release()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System requirement check failed: {e}")
        return False


def setup_directories():
    """Setup required directories"""
    directories = [
        ROOT_DIR / 'data',
        ROOT_DIR / 'models',
        ROOT_DIR / 'logs'
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        logger.info(f"✅ Directory ready: {directory}")


def show_welcome_message():
    """Show welcome message"""
    print("\n" + "="*60)
    print("🤚 GESTURETALK - ASSISTIVE COMMUNICATION SYSTEM")
    print("="*60)
    print("📋 Features:")
    print("  • Real-time hand gesture recognition")
    print("  • Text-to-Speech dalam Bahasa Indonesia")
    print("  • Modern GUI interface")
    print("  • Assistive communication support")
    print("\n🎯 Target Gestures:")
    print("  • tolong     - Gesture meminta bantuan")
    print("  • halo       - Gesture menyapa")
    print("  • terima_kasih - Gesture berterima kasih")
    print("\n💡 Tips:")
    print("  • Pastikan pencahayaan cukup")
    print("  • Posisikan tangan dalam frame camera")
    print("  • Tahan gesture 2-3 detik untuk deteksi stabil")
    print("="*60)


def main():
    """Main application function"""
    try:
        # Show welcome message
        show_welcome_message()
        
        logger.info("🚀 Starting GestureTalk application...")
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Check system requirements
        if not check_system_requirements():
            logger.warning("⚠️  Some system requirements not met, continuing anyway...")
        
        # Setup directories
        setup_directories()
        
        # Import dan start GUI (after dependency check)
        from gui.main_window import GestureTalkMainWindow
        
        logger.info("🎨 Initializing GUI...")
        
        # Create dan run application
        app = GestureTalkMainWindow()
        
        logger.info("✅ GestureTalk GUI ready")
        print("\n🚀 Launching GestureTalk GUI...")
        print("💡 Check the GUI window untuk mulai menggunakan aplikasi")
        
        # Run application
        app.run()
        
        logger.info("👋 GestureTalk application closed")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Application interrupted by user")
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        error_msg = f"❌ Application error: {e}"
        print(error_msg)
        logger.error(error_msg)
        
        # Try to show error in messagebox if GUI available
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("GestureTalk Error", error_msg)
        except:
            pass
        
        return 1


if __name__ == "__main__":
    exit(main())