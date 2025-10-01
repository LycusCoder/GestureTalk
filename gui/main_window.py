"""
Main Window GUI - Desktop interface untuk GestureTalk assistive application
Modern CustomTkinter interface dengan real-time webcam dan gesture recognition
"""

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
from PIL import Image, ImageTk
from typing import Optional, Dict
import sys
import os

# Add parent directory ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracker import HandTracker
from core.enhanced_hand_tracker import EnhancedHandTracker
from core.camera_handler import CameraHandler
from core.tts_handler import TTSHandler
from core.gesture_predictor import GesturePredictor
from core.gender_detector import GenderDetector


class GestureTalkMainWindow:
    """
    Main GUI application untuk GestureTalk assistive communication system
    
    Features:
    - Real-time webcam video feed
    - Hand detection dan gesture recognition
    - Text-to-Speech dengan Bahasa Indonesia
    - Modern CustomTkinter interface
    - Settings panel untuk customization
    - Performance monitoring
    - Model management
    """
    
    def __init__(self):
        """Initialize main application window"""
        # Set CustomTkinter theme
        ctk.set_appearance_mode("dark")  # "light" or "dark"
        ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("GestureTalk - Assistive Communication System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Application state
        self.is_running = False
        self.is_camera_active = False
        self.current_gesture = "Tidak ada"
        self.current_confidence = 0.0
        self.gesture_history = []
        self.current_gender = "Tidak diketahui"
        self.current_gender_confidence = 0.0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Core components
        self.hand_tracker = None
        self.camera_handler = None
        self.tts_handler = None
        self.gesture_predictor = None
        self.gender_detector = None
        
        # GUI components
        self.video_label = None
        self.gesture_display = None
        self.confidence_display = None
        self.status_display = None
        self.settings_frame = None
        
        # Threading
        self.video_thread = None
        self.video_thread_running = False
        
        # Initialize GUI
        self._setup_gui()
        
        # Initialize core systems
        self._initialize_systems()
        
        print("🚀 GestureTalk GUI initialized")
    
    def _setup_gui(self):
        """Setup main GUI layout dan components"""
        # Configure grid weight
        self.root.grid_columnconfigure(0, weight=2)  # Video column
        self.root.grid_columnconfigure(1, weight=1)  # Control column
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self._create_video_frame()
        self._create_control_frame()
        
        # Setup window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_video_frame(self):
        """Create video display frame"""
        # Main video frame
        self.video_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(1, weight=1)
        
        # Video title
        title_label = ctk.CTkLabel(
            self.video_frame, 
            text="🎥 Real-time Webcam Feed",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(20, 10))
        
        # Video display area
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Kamera akan muncul di sini...\n\nTekan 'Start Camera' untuk memulai",
            font=ctk.CTkFont(size=16),
            width=640,
            height=480
        )
        self.video_label.grid(row=1, column=0, padx=20, pady=20)
        
        # Video info frame
        self.video_info_frame = ctk.CTkFrame(self.video_frame)
        self.video_info_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # FPS display
        self.fps_label = ctk.CTkLabel(
            self.video_info_frame,
            text="FPS: --",
            font=ctk.CTkFont(size=14)
        )
        self.fps_label.pack(side="left", padx=10, pady=10)
        
        # Resolution display
        self.resolution_label = ctk.CTkLabel(
            self.video_info_frame,
            text="Resolution: --",
            font=ctk.CTkFont(size=14)
        )
        self.resolution_label.pack(side="left", padx=10, pady=10)
    
    def _create_control_frame(self):
        """Create control panel frame"""
        # Main control frame
        self.control_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # Control title
        control_title = ctk.CTkLabel(
            self.control_frame,
            text="🎛️ Control Panel",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        control_title.grid(row=0, column=0, pady=(20, 10))
        
        # Camera controls
        self._create_camera_controls()
        
        # Gesture detection display
        self._create_gesture_display()
        
        # TTS controls
        self._create_tts_controls()
        
        # Settings
        self._create_settings_panel()
        
        # Status display
        self._create_status_panel()
    
    def _create_camera_controls(self):
        """Create camera control buttons"""
        camera_frame = ctk.CTkFrame(self.control_frame)
        camera_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        # Camera control label
        camera_label = ctk.CTkLabel(
            camera_frame,
            text="📹 Camera Control",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        camera_label.pack(pady=(15, 5))
        
        # Start/Stop camera button
        self.camera_button = ctk.CTkButton(
            camera_frame,
            text="▶️ Start Camera",
            command=self._toggle_camera,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.camera_button.pack(pady=10)
        
        # Camera status
        self.camera_status = ctk.CTkLabel(
            camera_frame,
            text="Status: Camera Off",
            font=ctk.CTkFont(size=12)
        )
        self.camera_status.pack(pady=(0, 15))
    
    def _create_gesture_display(self):
        """Create gesture detection display"""
        gesture_frame = ctk.CTkFrame(self.control_frame)
        gesture_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Gesture display label
        gesture_label = ctk.CTkLabel(
            gesture_frame,
            text="🤚 Gesture Detection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        gesture_label.pack(pady=(15, 5))
        
        # Current gesture display
        self.gesture_display = ctk.CTkLabel(
            gesture_frame,
            text="Gesture: Tidak ada",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#1f6aa5", "#0066cc")
        )
        self.gesture_display.pack(pady=5)
        
        # Confidence display
        self.confidence_display = ctk.CTkLabel(
            gesture_frame,
            text="Confidence: 0.0%",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_display.pack(pady=5)
        
        # Confidence progress bar
        self.confidence_progress = ctk.CTkProgressBar(
            gesture_frame,
            width=200,
            height=20
        )
        self.confidence_progress.pack(pady=10)
        self.confidence_progress.set(0)
        
        # Gesture history
        self.history_label = ctk.CTkLabel(
            gesture_frame,
            text="Recent: -",
            font=ctk.CTkFont(size=12)
        )
        self.history_label.pack(pady=(0, 15))
    
    def _create_tts_controls(self):
        """Create TTS control panel"""
        tts_frame = ctk.CTkFrame(self.control_frame)
        tts_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # TTS label
        tts_label = ctk.CTkLabel(
            tts_frame,
            text="🔊 Text-to-Speech",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        tts_label.pack(pady=(15, 5))
        
        # TTS status
        self.tts_status = ctk.CTkLabel(
            tts_frame,
            text="Status: Inactive",
            font=ctk.CTkFont(size=12)
        )
        self.tts_status.pack(pady=5)
        
        # Manual TTS test
        self.tts_test_button = ctk.CTkButton(
            tts_frame,
            text="🎤 Test Voice",
            command=self._test_tts,
            width=150,
            height=35
        )
        self.tts_test_button.pack(pady=10)
        
        # TTS settings
        tts_volume_label = ctk.CTkLabel(tts_frame, text="Volume:")
        tts_volume_label.pack(pady=(10, 0))
        
        self.tts_volume_slider = ctk.CTkSlider(
            tts_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=9,
            command=self._update_tts_volume
        )
        self.tts_volume_slider.pack(pady=5)
        self.tts_volume_slider.set(0.9)
        
        tts_speed_label = ctk.CTkLabel(tts_frame, text="Speed:")
        tts_speed_label.pack(pady=(10, 0))
        
        self.tts_speed_slider = ctk.CTkSlider(
            tts_frame,
            from_=100,
            to=250,
            number_of_steps=15,
            command=self._update_tts_speed
        )
        self.tts_speed_slider.pack(pady=(5, 15))
        self.tts_speed_slider.set(150)
    
    def _create_settings_panel(self):
        """Create settings panel"""
        settings_frame = ctk.CTkFrame(self.control_frame)
        settings_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        # Settings label
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        settings_label.pack(pady=(15, 5))
        
        # Confidence threshold
        confidence_label = ctk.CTkLabel(settings_frame, text="Confidence Threshold:")
        confidence_label.pack(pady=(5, 0))
        
        self.confidence_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.3,
            to=0.9,
            number_of_steps=6,
            command=self._update_confidence_threshold
        )
        self.confidence_slider.pack(pady=5)
        self.confidence_slider.set(0.6)
        
        # Model info
        self.model_info = ctk.CTkLabel(
            settings_frame,
            text="Model: Loading...",
            font=ctk.CTkFont(size=12)
        )
        self.model_info.pack(pady=(10, 15))
    
    def _create_status_panel(self):
        """Create status display panel"""
        status_frame = ctk.CTkFrame(self.control_frame)
        status_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        # Status label
        status_label = ctk.CTkLabel(
            status_frame,
            text="📊 System Status",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        status_label.pack(pady=(15, 5))
        
        # System status
        self.system_status = ctk.CTkLabel(
            status_frame,
            text="Initializing systems...",
            font=ctk.CTkFont(size=12)
        )
        self.system_status.pack(pady=(5, 15))
    
    def _initialize_systems(self):
        """Initialize core systems (camera, TTS, etc.)"""
        try:
            # Initialize in separate thread untuk avoid UI blocking
            init_thread = threading.Thread(target=self._init_systems_worker, daemon=True)
            init_thread.start()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize systems: {e}")
    
    def _init_systems_worker(self):
        """Worker thread untuk system initialization"""
        try:
            # Initialize hand tracker
            self.hand_tracker = HandTracker(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Initialize camera handler
            self.camera_handler = CameraHandler(
                target_fps=30,
                resolution=(640, 480)
            )
            
            # Initialize TTS handler
            self.tts_handler = TTSHandler(
                language='id',
                rate=150,
                volume=0.9
            )
            
            # Start TTS service
            self.tts_handler.start_speech_service()
            
            # Initialize gesture predictor
            self.gesture_predictor = GesturePredictor(
                confidence_threshold=0.6,
                stability_window=3
            )
            
            # Update GUI status
            self.root.after(0, self._update_system_status)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("System Error", f"System initialization failed: {error_msg}"))
    
    def _update_system_status(self):
        """Update system status display"""
        # Check system states
        camera_ok = self.camera_handler is not None
        tts_ok = self.tts_handler is not None and self.tts_handler.status.name != 'ERROR'
        model_ok = self.gesture_predictor is not None and self.gesture_predictor.is_loaded
        
        # Update status text
        status_text = "System Status:\n"
        status_text += f"📹 Camera: {'✅' if camera_ok else '❌'}\n"
        status_text += f"🔊 TTS: {'✅' if tts_ok else '❌'}\n"
        status_text += f"🤖 Model: {'✅' if model_ok else '❌'}"
        
        self.system_status.configure(text=status_text)
        
        # Update TTS status
        if tts_ok:
            self.tts_status.configure(text="Status: Ready")
        else:
            self.tts_status.configure(text="Status: Error")
        
        # Update model info
        if model_ok and self.gesture_predictor:
            stats = self.gesture_predictor.get_prediction_stats()
            model_text = f"Model: {stats['model_type']}\n"
            model_text += f"Classes: {len(stats['gesture_classes'])}"
            self.model_info.configure(text=model_text)
        else:
            self.model_info.configure(text="Model: Not loaded")
    
    def _toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_active:
            self._start_camera()
        else:
            self._stop_camera()
    
    def _start_camera(self):
        """Start camera dan video processing"""
        try:
            if not self.camera_handler:
                messagebox.showerror("Error", "Camera handler not initialized")
                return
            
            # Initialize camera
            if not self.camera_handler.initialize_camera():
                messagebox.showerror("Camera Error", "Failed to initialize camera")
                return
            
            # Start camera capture
            if not self.camera_handler.start_capture():
                messagebox.showerror("Camera Error", "Failed to start camera capture")
                return
            
            # Start video processing thread
            self.video_thread_running = True
            self.video_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
            self.video_thread.start()
            
            # Update UI
            self.is_camera_active = True
            self.camera_button.configure(text="⏹️ Stop Camera")
            self.camera_status.configure(text="Status: Camera Active")
            
            print("✅ Camera started successfully")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def _stop_camera(self):
        """Stop camera dan video processing"""
        try:
            # Stop video thread
            self.video_thread_running = False
            
            # Wait for thread to finish
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)
            
            # Stop camera
            if self.camera_handler:
                self.camera_handler.stop_capture()
            
            # Update UI
            self.is_camera_active = False
            self.camera_button.configure(text="▶️ Start Camera")
            self.camera_status.configure(text="Status: Camera Off")
            
            # Clear video display
            self.video_label.configure(
                image=None,
                text="Kamera dihentikan.\n\nTekan 'Start Camera' untuk memulai lagi"
            )
            
            print("✅ Camera stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping camera: {e}")
    
    def _video_processing_loop(self):
        """Main video processing loop"""
        while self.video_thread_running:
            try:
                # Get frame dari camera
                frame = self.camera_handler.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame dengan hand tracker
                processed_frame, landmarks = self.hand_tracker.process_frame(frame)
                
                # Gesture prediction jika ada landmarks
                gesture_name = "Tidak ada"
                confidence = 0.0
                
                if landmarks and self.gesture_predictor and self.gesture_predictor.is_loaded:
                    gesture_name, confidence = self.gesture_predictor.predict_gesture(landmarks)
                    
                    if gesture_name is None:
                        gesture_name = "Tidak dikenali"
                        confidence = 0.0
                
                # Add info overlay ke frame
                display_frame = self._add_info_overlay(processed_frame, gesture_name, confidence, landmarks is not None)
                
                # Update GUI dengan frame baru
                self.root.after(0, self._update_video_display, display_frame, gesture_name, confidence)
                
                # TTS untuk gesture yang terdeteksi
                if gesture_name and gesture_name not in ["Tidak ada", "Tidak dikenali"] and confidence > 0.7:
                    self._handle_gesture_detection(gesture_name, confidence)
                
                # FPS calculation
                self._update_fps_counter()
                
                # Small delay untuk control frame rate
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Video processing error: {e}")
                time.sleep(0.1)
    
    def _add_info_overlay(self, frame, gesture_name, confidence, hand_detected):
        """Add informational overlay ke video frame"""
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Background untuk text
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Gesture info
        cv2.putText(overlay, f"Gesture: {gesture_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Confidence
        cv2.putText(overlay, f"Confidence: {confidence:.1%}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Hand detection status
        status_text = "Tangan Terdeteksi" if hand_detected else "Tidak Ada Tangan"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(overlay, status_text, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return overlay
    
    def _update_video_display(self, frame, gesture_name, confidence):
        """Update video display di GUI"""
        try:
            # Convert frame ke format yang bisa ditampilkan di tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update video label
            self.video_label.configure(image=frame_tk, text="")
            self.video_label.image = frame_tk  # Keep reference
            
            # Update gesture display
            self.gesture_display.configure(text=f"Gesture: {gesture_name}")
            self.confidence_display.configure(text=f"Confidence: {confidence:.1%}")
            self.confidence_progress.set(confidence)
            
            # Update gesture history
            self._update_gesture_history(gesture_name, confidence)
            
        except Exception as e:
            print(f"❌ Error updating video display: {e}")
    
    def _update_gesture_history(self, gesture_name, confidence):
        """Update gesture history display"""
        if gesture_name not in ["Tidak ada", "Tidak dikenali"] and confidence > 0.5:
            # Add ke history
            self.gesture_history.append(gesture_name)
            
            # Keep only recent 3 gestures
            if len(self.gesture_history) > 3:
                self.gesture_history = self.gesture_history[-3:]
            
            # Update display
            history_text = "Recent: " + " → ".join(self.gesture_history)
            self.history_label.configure(text=history_text)
    
    def _handle_gesture_detection(self, gesture_name, confidence):
        """Handle detected gesture (TTS, etc.)"""
        if self.tts_handler and gesture_name:
            # Avoid spam dengan checking jika gesture sama dengan previous
            current_time = time.time()
            
            if not hasattr(self, '_last_tts_time'):
                self._last_tts_time = 0
                self._last_tts_gesture = ""
            
            # Only speak jika gesture berbeda atau sudah lewat 3 detik
            if (gesture_name != self._last_tts_gesture or 
                current_time - self._last_tts_time > 3.0):
                
                self.tts_handler.speak_gesture(gesture_name)
                self._last_tts_time = current_time
                self._last_tts_gesture = gesture_name
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            
            # Update FPS display
            self.root.after(0, lambda: self.fps_label.configure(text=f"FPS: {self.current_fps:.1f}"))
            
            # Reset counter
            if self.fps_counter >= 300:
                self.fps_counter = 0
                self.fps_start_time = time.time()
    
    def _test_tts(self):
        """Test TTS functionality"""
        if self.tts_handler:
            test_message = "Halo, sistem Text to Speech berfungsi dengan baik"
            self.tts_handler.speak(test_message, priority='high')
        else:
            messagebox.showerror("TTS Error", "TTS system not initialized")
    
    def _update_tts_volume(self, value):
        """Update TTS volume"""
        if self.tts_handler:
            self.tts_handler.update_settings(volume=value)
    
    def _update_tts_speed(self, value):
        """Update TTS speed"""
        if self.tts_handler:
            self.tts_handler.update_settings(rate=int(value))
    
    def _update_confidence_threshold(self, value):
        """Update confidence threshold"""
        if self.gesture_predictor:
            self.gesture_predictor.update_settings(confidence_threshold=value)
    
    def _on_closing(self):
        """Handle application closing"""
        try:
            # Stop camera
            if self.is_camera_active:
                self._stop_camera()
            
            # Cleanup systems
            if self.camera_handler:
                self.camera_handler.release()
            
            if self.hand_tracker:
                self.hand_tracker.cleanup()
            
            if self.tts_handler:
                self.tts_handler.cleanup()
            
            print("✅ Application cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
        
        finally:
            self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        print("🚀 Starting GestureTalk GUI...")
        self.root.mainloop()


# Test function
def test_main_window():
    """Test function untuk GUI"""
    try:
        app = GestureTalkMainWindow()
        app.run()
    except Exception as e:
        print(f"❌ GUI Error: {e}")


if __name__ == "__main__":
    test_main_window()