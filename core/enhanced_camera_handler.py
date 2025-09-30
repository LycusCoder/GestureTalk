"""
Enhanced Camera Handler - Universal webcam compatibility & performance optimization
Advanced webcam management dengan support untuk berbagai driver dan enhanced performance
"""

import cv2
import numpy as np
import threading
import time
import platform
import os
from typing import Optional, Callable, Tuple, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedCameraHandler:
    """
    Enhanced camera handler dengan universal webcam compatibility dan advanced features
    
    Features:
    - Universal driver compatibility (V4L2, DirectShow, GStreamer)
    - Multiple camera backend support
    - Auto-detection dan fallback system
    - Enhanced resolution dan FPS handling
    - Advanced threading untuk smooth performance
    - Real-time camera settings optimization
    - Multi-camera support
    - Error recovery dan reconnection
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 target_fps: int = 30,
                 resolution: Tuple[int, int] = (640, 480),
                 auto_optimize: bool = True):
        """
        Initialize EnhancedCameraHandler dengan universal compatibility
        
        Args:
            camera_index: Camera device index
            target_fps: Target FPS untuk video capture  
            resolution: Target resolution (width, height)
            auto_optimize: Enable automatic optimization
        """
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.resolution = resolution
        self.auto_optimize = auto_optimize
        
        # Camera state
        self.cap = None
        self.is_opened = False
        self.is_running = False
        self.current_backend = None
        
        # Threading untuk smooth capture
        self.capture_thread = None
        self.thread_lock = threading.Lock()
        self.latest_frame = None
        self.frame_ready = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        self.frame_drops = 0
        
        # Error tracking dan recovery
        self.connection_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_error = None
        
        # Camera capabilities
        self.supported_resolutions = []
        self.supported_fps_ranges = []
        self.camera_info = {}
        
        # Platform-specific settings
        self._setup_platform_specifics()
        self._detect_available_backends()
        
        print(f"🚀 EnhancedCameraHandler initialized for camera {camera_index}")
    
    def _setup_platform_specifics(self):
        """Setup platform-specific configurations"""
        self.platform = platform.system().lower()
        
        if self.platform == 'linux':
            # Linux: V4L2 support
            self.preferred_backends = [
                cv2.CAP_V4L2,
                cv2.CAP_GSTREAMER,
                cv2.CAP_FFMPEG,
                cv2.CAP_ANY
            ]
            self.backend_names = {
                cv2.CAP_V4L2: "V4L2",
                cv2.CAP_GSTREAMER: "GStreamer", 
                cv2.CAP_FFMPEG: "FFmpeg",
                cv2.CAP_ANY: "Any"
            }
        elif self.platform == 'windows':
            # Windows: DirectShow, Media Foundation
            self.preferred_backends = [
                cv2.CAP_DSHOW,
                cv2.CAP_MSMF,
                cv2.CAP_VFW,
                cv2.CAP_ANY
            ]
            self.backend_names = {
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Media Foundation",
                cv2.CAP_VFW: "Video for Windows",
                cv2.CAP_ANY: "Any"
            }
        elif self.platform == 'darwin':
            # macOS: AVFoundation
            self.preferred_backends = [
                cv2.CAP_AVFOUNDATION,
                cv2.CAP_QTKIT,
                cv2.CAP_ANY
            ]
            self.backend_names = {
                cv2.CAP_AVFOUNDATION: "AVFoundation",
                cv2.CAP_QTKIT: "QTKit",
                cv2.CAP_ANY: "Any"
            }
        else:
            # Generic fallback
            self.preferred_backends = [cv2.CAP_ANY]
            self.backend_names = {cv2.CAP_ANY: "Any"}
    
    def _detect_available_backends(self):
        """Detect available camera backends pada sistem"""
        logger.info("🔍 Detecting available camera backends...")
        
        self.available_backends = []
        
        for backend in self.preferred_backends:
            backend_name = self.backend_names.get(backend, f"Backend_{backend}")
            
            try:
                # Quick test untuk check backend availability
                test_cap = cv2.VideoCapture(self.camera_index, backend)
                if test_cap.isOpened():
                    self.available_backends.append(backend)
                    logger.info(f"✅ {backend_name} available")
                    test_cap.release()
                else:
                    logger.info(f"❌ {backend_name} not available")
            except Exception as e:
                logger.info(f"❌ {backend_name} failed: {e}")
        
        if not self.available_backends:
            logger.warning("⚠️  No specific backends available, using default")
            self.available_backends = [cv2.CAP_ANY]
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera dengan universal compatibility dan auto-fallback
        
        Returns:
            bool: True jika berhasil initialize camera
        """
        logger.info(f"🔧 Initializing camera {self.camera_index} with universal compatibility...")
        
        # Release previous camera jika ada
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Try each available backend
        for backend in self.available_backends:
            backend_name = self.backend_names.get(backend, f"Backend_{backend}")
            logger.info(f"🔄 Trying {backend_name}...")
            
            try:
                if backend == cv2.CAP_ANY:
                    self.cap = cv2.VideoCapture(self.camera_index)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                
                if self.cap.isOpened():
                    self.current_backend = backend
                    logger.info(f"✅ Camera initialized with {backend_name}")
                    
                    # Setup camera properties
                    success = self._setup_camera_properties()
                    if success:
                        self._detect_camera_capabilities()
                        self.is_opened = True
                        self.connection_attempts = 0
                        return True
                    else:
                        logger.warning(f"⚠️  {backend_name} opened but setup failed")
                        self.cap.release()
                        self.cap = None
                
            except Exception as e:
                logger.warning(f"❌ {backend_name} initialization error: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        logger.error("❌ Failed to initialize camera dengan semua backends")
        return False
    
    def _setup_camera_properties(self) -> bool:
        """Setup camera properties dengan error handling"""
        try:
            # Set resolution dengan fallback
            success = self._set_optimal_resolution()
            if not success:
                logger.warning("⚠️  Resolution setup failed, using default")
            
            # Set FPS dengan fallback
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Platform-specific optimizations
            self._apply_platform_optimizations()
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📊 Camera settings:")
            logger.info(f"   Resolution: {actual_width}x{actual_height}")
            logger.info(f"   FPS: {actual_fps}")
            logger.info(f"   Backend: {self.backend_names.get(self.current_backend, 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera property setup error: {e}")
            return False
    
    def _set_optimal_resolution(self) -> bool:
        """Set optimal resolution dengan comprehensive fallback system"""
        # Extended fallback resolutions untuk maximum compatibility
        fallback_resolutions = [
            # Try requested resolution first
            self.resolution,
            # Common HD resolutions
            (1920, 1080), (1280, 720), (1024, 768),
            # Standard resolutions
            (800, 600), (640, 480), (320, 240),
            # Wide resolutions
            (1366, 768), (1600, 900),
            # 4:3 ratios
            (1024, 768), (800, 600), (640, 480),
            # Minimal fallback
            (160, 120)
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_resolutions = []
        for res in fallback_resolutions:
            if res not in seen:
                unique_resolutions.append(res)
                seen.add(res)
        
        for width, height in unique_resolutions:
            try:
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify setting
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Accept if close enough (some cameras adjust resolution)
                if abs(actual_width - width) <= 50 and abs(actual_height - height) <= 50:
                    logger.info(f"✅ Resolution set to {actual_width}x{actual_height} (requested {width}x{height})")
                    self.resolution = (actual_width, actual_height)
                    return True
                    
            except Exception as e:
                logger.debug(f"Resolution {width}x{height} failed: {e}")
                continue
        
        # If all failed, use whatever camera provides
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (actual_width, actual_height)
        logger.warning(f"⚠️  Using camera default resolution: {actual_width}x{actual_height}")
        return False
    
    def _apply_platform_optimizations(self):
        """Apply platform-specific optimizations"""
        try:
            if self.platform == 'linux':
                # V4L2 specific optimizations
                if self.current_backend == cv2.CAP_V4L2:
                    # Set buffer size untuk reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Auto exposure
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            elif self.platform == 'windows':
                # DirectShow optimizations
                if self.current_backend == cv2.CAP_DSHOW:
                    # Reduce buffer size
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Set exposure mode
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            elif self.platform == 'darwin':
                # macOS AVFoundation optimizations
                if self.current_backend == cv2.CAP_AVFOUNDATION:
                    # Buffer optimization
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Universal optimizations
            if self.auto_optimize:
                # Auto white balance
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                # Auto focus (if supported)
                try:
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                except:
                    pass  # Not all cameras support autofocus
                
        except Exception as e:
            logger.debug(f"Platform optimization error: {e}")
    
    def _detect_camera_capabilities(self):
        """Detect camera capabilities untuk optimization"""
        try:
            self.camera_info = {
                'backend': self.backend_names.get(self.current_backend, 'Unknown'),
                'resolution': self.resolution,
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE)
            }
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                self.camera_info['test_capture'] = 'Success'
                self.camera_info['frame_shape'] = test_frame.shape
            else:
                self.camera_info['test_capture'] = 'Failed'
                
        except Exception as e:
            logger.debug(f"Camera capability detection error: {e}")
    
    def start_capture(self) -> bool:
        """
        Start enhanced threaded frame capture
        
        Returns:
            bool: True jika berhasil start capture
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
        self.frame_drops = 0
        
        # Start enhanced capture thread
        self.capture_thread = threading.Thread(target=self._enhanced_capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("🎥 Enhanced camera capture started")
        return True
    
    def _enhanced_capture_loop(self):
        """Enhanced capture loop dengan advanced error handling dan optimization"""
        frame_delay = 1.0 / self.target_fps
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.is_running:
            try:
                frame_start = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"❌ Frame capture failed (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("❌ Too many consecutive failures, attempting reconnection")
                        if not self._attempt_camera_recovery():
                            break
                    
                    time.sleep(0.1)  # Brief delay before retry
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Flip frame horizontal untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                # Apply enhancements jika diperlukan
                if self.auto_optimize:
                    frame = self._apply_frame_enhancements(frame)
                
                # Update latest frame dengan thread safety
                with self.thread_lock:
                    self.latest_frame = frame.copy()
                    self.frame_ready = True
                
                # FPS monitoring
                self._update_fps_counter()
                
                # Frame rate control dengan adaptive delay
                processing_time = time.time() - frame_start
                sleep_time = max(0, frame_delay - processing_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Frame took too long, increment drop counter
                    self.frame_drops += 1
                
            except Exception as e:
                logger.error(f"❌ Error dalam enhanced capture loop: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    break
                
                time.sleep(0.1)
        
        logger.info("🛑 Enhanced capture loop stopped")
    
    def _apply_frame_enhancements(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic frame enhancements"""
        try:
            # Simple brightness/contrast enhancement
            # Convert ke LAB color space untuk better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) ke L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels kembali
            enhanced = cv2.merge([l, a, b])
            enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced_frame
            
        except Exception as e:
            logger.debug(f"Frame enhancement error: {e}")
            return frame  # Return original jika enhancement gagal
    
    def _attempt_camera_recovery(self) -> bool:
        """Attempt camera recovery ketika ada persistent errors"""
        logger.info("🔄 Attempting camera recovery...")
        
        self.connection_attempts += 1
        
        if self.connection_attempts > self.max_reconnect_attempts:
            logger.error("❌ Max reconnection attempts exceeded")
            return False
        
        try:
            # Release current connection
            if self.cap:
                self.cap.release()
                self.cap = None
            
            time.sleep(1)  # Brief delay
            
            # Try to reinitialize
            success = self.initialize_camera()
            
            if success:
                logger.info("✅ Camera recovery successful")
                return True
            else:
                logger.error("❌ Camera recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Camera recovery error: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame dengan enhanced thread safety
        
        Returns:
            Optional[np.ndarray]: Latest frame atau None
        """
        if not self.frame_ready:
            return None
        
        with self.thread_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def _update_fps_counter(self):
        """Enhanced FPS counter dengan drop tracking"""
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            elapsed_time = time.time() - self.fps_start_time
            self.actual_fps = self.fps_counter / elapsed_time
            
            # Reset counter untuk next measurement
            if self.fps_counter >= 300:  # Reset every 300 frames
                self.fps_counter = 0
                self.fps_start_time = time.time()
                self.frame_drops = 0  # Reset drop counter
    
    def get_enhanced_camera_info(self) -> Dict:
        """Get comprehensive camera information"""
        if not self.cap or not self.is_opened:
            return {"status": "Camera not initialized"}
        
        info = {
            "status": "active" if self.is_running else "inactive",
            "platform": self.platform,
            "backend": self.backend_names.get(self.current_backend, "Unknown"),
            "camera_index": self.camera_index,
            "resolution": {
                "requested": self.resolution,
                "actual": {
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
            },
            "fps": {
                "target": self.target_fps,
                "actual": round(self.actual_fps, 1),
                "camera_reported": self.cap.get(cv2.CAP_PROP_FPS)
            },
            "performance": {
                "frame_ready": self.frame_ready,
                "frame_drops": self.frame_drops,
                "connection_attempts": self.connection_attempts
            },
            "capabilities": self.camera_info,
            "available_backends": [self.backend_names.get(b, str(b)) for b in self.available_backends]
        }
        
        return info
    
    def stop_capture(self):
        """Stop camera capture dan cleanup threads"""
        logger.info("🛑 Stopping enhanced camera capture...")
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
            
            if self.capture_thread.is_alive():
                logger.warning("⚠️  Capture thread did not stop gracefully")
        
        logger.info("✅ Enhanced camera capture stopped")
    
    def release(self):
        """Release all camera resources"""
        logger.info("🗑️  Releasing enhanced camera resources...")
        
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
        self.current_backend = None
        
        logger.info("✅ Enhanced camera resources released")
    
    def is_camera_ready(self) -> bool:
        """Check camera readiness"""
        return self.is_opened and self.is_running and self.frame_ready


# Test function untuk enhanced camera handler
def test_enhanced_camera_handler():
    """Test function untuk verify enhanced camera compatibility"""
    print("🚀 Testing EnhancedCameraHandler...")
    
    # Initialize enhanced camera handler
    camera = EnhancedCameraHandler(
        target_fps=30,
        resolution=(640, 480),
        auto_optimize=True
    )
    
    # Test initialization
    success = camera.initialize_camera()
    if not success:
        print("❌ Enhanced camera initialization failed")
        return
    
    print("✅ Enhanced camera initialized successfully")
    
    # Print comprehensive camera info
    info = camera.get_enhanced_camera_info()
    print(f"\n📊 Enhanced Camera Info:")
    print(f"   Platform: {info['platform']}")
    print(f"   Backend: {info['backend']}")
    print(f"   Resolution: {info['resolution']['actual']['width']}x{info['resolution']['actual']['height']}")
    print(f"   Available backends: {info['available_backends']}")
    
    # Start capture
    success = camera.start_capture()
    if not success:
        print("❌ Failed to start enhanced camera capture")
        camera.release()
        return
    
    print("🎥 Enhanced camera capture started")
    print("⌨️  Press ESC to exit, SPACE for screenshot, I for info")
    
    # Test frame capture dengan enhanced features
    frame_count = 0
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # Add enhanced info overlay
                info = camera.get_enhanced_camera_info()
                
                # Enhanced info display
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                cv2.putText(frame, f"Enhanced Camera Test", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Backend: {info['backend']}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"FPS: {info['fps']['actual']}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Drops: {info['performance']['frame_drops']}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Enhanced Camera Test', frame)
                frame_count += 1
                
                if frame_count % 60 == 0:  # Every 2 seconds
                    print(f"📊 {frame_count} frames | FPS: {info['fps']['actual']:.1f} | "
                          f"Drops: {info['performance']['frame_drops']}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE - screenshot
                if frame is not None:
                    cv2.imwrite(f'enhanced_camera_test_{int(time.time())}.jpg', frame)
                    print("📸 Screenshot saved!")
            elif key == ord('i') or key == ord('I'):  # Info
                info = camera.get_enhanced_camera_info()
                print("\n📊 DETAILED CAMERA INFO:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera.release()
        
        # Final stats
        final_info = camera.get_enhanced_camera_info()
        print("\n📊 ENHANCED CAMERA TEST SUMMARY:")
        print(f"   Total frames captured: {frame_count}")
        print(f"   Final FPS: {final_info.get('fps', {}).get('actual', 0):.1f}")
        print(f"   Frame drops: {final_info.get('performance', {}).get('frame_drops', 0)}")
        print("✅ Enhanced camera test completed")


if __name__ == "__main__":
    test_enhanced_camera_handler()