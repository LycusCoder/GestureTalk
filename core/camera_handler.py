"""
Camera Handler Module - Manages webcam access and video frame processing
Handles camera initialization, frame capture, and video stream management
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraHandler:
    """
    Robust camera handler untuk webcam management dan video processing
    
    Features:
    - Multi-resolution support dengan auto-fallback
    - Threaded frame capture untuk smooth performance
    - Error handling dan auto-reconnection
    - FPS monitoring dan optimization
    - Frame preprocessing untuk ML models
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 target_fps: int = 30,
                 resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize CameraHandler
        
        Args:
            camera_index: Camera device index (0 untuk default webcam)
            target_fps: Target FPS untuk video capture
            resolution: Target resolution (width, height)
        """
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.resolution = resolution
        
        # Camera state
        self.cap = None
        self.is_opened = False
        self.is_running = False
        
        # Threading untuk smooth capture
        self.capture_thread = None
        self.thread_lock = threading.Lock()
        self.latest_frame = None
        self.frame_ready = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        
        # Error tracking
        self.connection_attempts = 0
        self.max_reconnect_attempts = 3
        
    def initialize_camera(self) -> bool:
        """
        Initialize webcam dengan error handling dan resolution fallback
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        logger.info(f"🔧 Initializing camera {self.camera_index}...")
        
        try:
            # Release previous camera jika ada
            if self.cap is not None:
                self.cap.release()
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Tidak bisa membuka camera {self.camera_index}")
                return False
            
            # Set resolution dengan fallback
            success = self._set_optimal_resolution()
            if not success:
                logger.warning("⚠️  Menggunakan default resolution")
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Verify camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"✅ Camera initialized:")
            logger.info(f"   Resolution: {actual_width}x{actual_height}")
            logger.info(f"   FPS: {actual_fps}")
            
            self.is_opened = True
            self.connection_attempts = 0
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera initialization error: {e}")
            return False
    
    def _set_optimal_resolution(self) -> bool:
        """
        Set optimal resolution dengan fallback ke resolutions yang didukung
        
        Returns:
            bool: True jika berhasil set resolution
        """
        # Daftar resolutions untuk fallback (dari tinggi ke rendah)
        fallback_resolutions = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (1024, 768),   # Standard
            (800, 600),    # SVGA
            (640, 480),    # VGA (most compatible)
            (320, 240)     # QVGA (fallback terakhir)
        ]
        
        # Coba resolution yang diminta dulu
        target_resolutions = [self.resolution] + fallback_resolutions
        
        for width, height in target_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify apakah berhasil
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                logger.info(f"✅ Resolution set to {width}x{height}")
                self.resolution = (width, height)
                return True
            
        logger.warning("⚠️  Tidak bisa set resolution yang diinginkan")
        return False
    
    def start_capture(self) -> bool:
        """
        Start threaded frame capture untuk smooth performance
        
        Returns:
            bool: True jika berhasil start
        """
        if not self.is_opened:
            logger.error("❌ Camera belum diinisialisasi")
            return False
        
        if self.is_running:
            logger.warning("⚠️  Capture sudah berjalan")
            return True
        
        self.is_running = True
        self.fps_start_time = time.time()
        self.fps_counter = 0
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("🎥 Camera capture started")
        return True
    
    def _capture_loop(self):
        """
        Main capture loop yang berjalan di thread terpisah
        """
        frame_delay = 1.0 / self.target_fps  # Delay antar frame
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("❌ Gagal membaca frame dari camera")
                    if self.connection_attempts < self.max_reconnect_attempts:
                        self._attempt_reconnection()
                    else:
                        break
                    continue
                
                # Flip frame horizontal untuk mirror effect (lebih natural)
                frame = cv2.flip(frame, 1)
                
                # Update latest frame dengan thread safety
                with self.thread_lock:
                    self.latest_frame = frame.copy()
                    self.frame_ready = True
                
                # FPS monitoring
                self._update_fps_counter()
                
                # Control frame rate
                time.sleep(max(0, frame_delay - 0.001))  # Small buffer
                
            except Exception as e:
                logger.error(f"❌ Error dalam capture loop: {e}")
                break
        
        logger.info("🛑 Capture loop stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame dari camera dengan thread safety
        
        Returns:
            Optional[np.ndarray]: Latest frame atau None jika tidak ada
        """
        if not self.frame_ready:
            return None
        
        with self.thread_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def _update_fps_counter(self):
        """Update FPS counter dan calculate actual FPS"""
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            elapsed_time = time.time() - self.fps_start_time
            self.actual_fps = self.fps_counter / elapsed_time
            
            # Reset counter untuk measurement berikutnya
            if self.fps_counter >= 300:  # Reset setiap 300 frames
                self.fps_counter = 0
                self.fps_start_time = time.time()
    
    def _attempt_reconnection(self):
        """
        Attempt untuk reconnect camera ketika ada error
        """
        logger.info(f"🔄 Attempting camera reconnection ({self.connection_attempts + 1}/{self.max_reconnect_attempts})")
        
        self.connection_attempts += 1
        time.sleep(1)  # Wait before retry
        
        # Release dan reinitialize
        if self.cap:
            self.cap.release()
        
        success = self.initialize_camera()
        if success:
            logger.info("✅ Camera reconnection berhasil")
        else:
            logger.error("❌ Camera reconnection gagal")
    
    def get_camera_info(self) -> dict:
        """
        Get informasi detail tentang camera settings
        
        Returns:
            dict: Camera information
        """
        if not self.cap or not self.is_opened:
            return {"status": "Camera not initialized"}
        
        return {
            "status": "active" if self.is_running else "inactive",
            "resolution": {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            "fps": {
                "target": self.target_fps,
                "actual": round(self.actual_fps, 1)
            },
            "camera_index": self.camera_index,
            "frame_ready": self.frame_ready,
            "connection_attempts": self.connection_attempts
        }
    
    def stop_capture(self):
        """Stop camera capture dan cleanup threads"""
        logger.info("🛑 Stopping camera capture...")
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        logger.info("✅ Camera capture stopped")
    
    def release(self):
        """
        Release camera resources completely
        """
        logger.info("🗑️  Releasing camera resources...")
        
        # Stop capture first
        self.stop_capture()
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset state
        self.is_opened = False
        self.frame_ready = False
        self.latest_frame = None
        
        logger.info("✅ Camera resources released")
    
    def is_camera_ready(self) -> bool:
        """Check apakah camera ready untuk digunakan"""
        return self.is_opened and self.is_running and self.frame_ready


# Test function untuk development
def test_camera_handler():
    """
    Test function untuk verify CameraHandler berfungsi dengan baik
    """
    print("🚀 Testing CameraHandler...")
    
    # Initialize camera handler
    camera = CameraHandler(target_fps=30)
    
    # Test initialization
    success = camera.initialize_camera()
    if not success:
        print("❌ Camera initialization gagal")
        return
    
    print("✅ Camera berhasil diinisialisasi")
    
    # Print camera info
    info = camera.get_camera_info()
    print(f"📊 Camera Info: {info}")
    
    # Start capture
    success = camera.start_capture()
    if not success:
        print("❌ Gagal start camera capture")
        camera.release()
        return
    
    print("🎥 Camera capture started")
    print("⌨️  Tekan ESC untuk keluar")
    
    # Test frame capture
    frame_count = 0
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # Add info overlay
                info = camera.get_camera_info()
                fps_text = f"FPS: {info['fps']['actual']}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "CameraHandler Test", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.imshow('CameraHandler Test', frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"📊 {frame_count} frames captured | FPS: {info['fps']['actual']}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera.release()
        print("✅ CameraHandler test selesai")


if __name__ == "__main__":
    test_camera_handler()