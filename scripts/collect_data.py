"""
Data Collection Script - Collect gesture training data untuk ML model
Interactive script untuk merekam hand landmark data dari webcam
"""

import cv2
import pandas as pd
import numpy as np
import time
import os
import sys
from typing import List, Dict
import argparse

# Add parent directory ke path untuk import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_tracker import HandTracker
from core.camera_handler import CameraHandler


class GestureDataCollector:
    """
    Interactive data collector untuk gesture recognition training
    
    Features:
    - Multiple gesture recording dalam satu session
    - Real-time preview dengan countdown
    - Data validation dan cleaning
    - CSV export dengan normalized landmarks
    - Progress tracking dan statistics
    """
    
    def __init__(self, output_file: str = '/app/GestureTalk/data/gestures.csv'):
        """
        Initialize GestureDataCollector
        
        Args:
            output_file: Path untuk save CSV data
        """
        self.output_file = output_file
        self.data_directory = os.path.dirname(output_file)
        
        # Initialize core components
        self.hand_tracker = HandTracker(min_detection_confidence=0.8)
        self.camera = CameraHandler(target_fps=30)
        
        # Data storage
        self.collected_data = []
        self.session_stats = {
            'total_samples': 0,
            'gestures_recorded': {},
            'session_start': time.time()
        }
        
        # Recording settings
        self.samples_per_gesture = 100  # Target samples per gesture
        self.recording_duration = 5.0   # Seconds untuk setiap gesture recording
        self.countdown_duration = 3     # Countdown sebelum recording
        
        # Predefined gestures untuk assistive communication
        self.predefined_gestures = [
            'tolong',           # Tangan menunjuk ke depan atau gesture SOS
            'halo',             # Tangan melambai
            'terima_kasih',     # Tangan di dada atau gesture thankful
            'maaf',             # Tangan menghadap ke atas, gesture apologetic
            'ya',               # Thumbs up atau nod gesture
            'tidak',            # Tangan menggeleng atau stop gesture
        ]
        
        # Create data directory jika belum ada
        os.makedirs(self.data_directory, exist_ok=True)
        
        print("🚀 GestureDataCollector initialized")
        print(f"📁 Data akan disimpan di: {self.output_file}")
    
    def initialize_hardware(self) -> bool:
        """
        Initialize camera dan hand tracker
        
        Returns:
            bool: True jika berhasil initialize semua components
        """
        print("🔧 Initializing hardware components...")
        
        # Initialize camera
        camera_success = self.camera.initialize_camera()
        if not camera_success:
            print("❌ Gagal initialize camera")
            return False
        
        # Start camera capture
        capture_success = self.camera.start_capture()
        if not capture_success:
            print("❌ Gagal start camera capture")
            return False
        
        # Test hand tracker
        print("✅ Camera ready")
        print("✅ Hand tracker ready")
        
        # Wait untuk camera warm-up
        print("🔥 Warming up camera...")
        time.sleep(2)
        
        return True
    
    def show_main_menu(self):
        """Show main menu untuk data collection options"""
        print("\n" + "="*50)
        print("🤚 GESTURE DATA COLLECTION SYSTEM")
        print("="*50)
        print("Pilih opsi:")
        print("1. Record gesture baru")
        print("2. Record predefined gestures (assistive)")
        print("3. Lihat data yang sudah dikumpulkan")
        print("4. Test hand detection")
        print("5. Exit")
        print("="*50)
    
    def record_single_gesture(self, gesture_name: str, target_samples: int = None) -> int:
        """
        Record data untuk satu gesture specific
        
        Args:
            gesture_name: Nama gesture yang akan direcord
            target_samples: Target jumlah samples (default dari class setting)
            
        Returns:
            int: Jumlah samples yang berhasil direcord
        """
        if target_samples is None:
            target_samples = self.samples_per_gesture
        
        print(f"\n📹 Recording gesture: '{gesture_name}'")
        print(f"🎯 Target samples: {target_samples}")
        print(f"⏱️  Recording duration: {self.recording_duration} detik")
        print("\n📋 Instructions:")
        print("- Posisikan tangan kamu dalam frame")
        print("- Bersiap untuk melakukan gesture")
        print("- Recording akan mulai setelah countdown")
        print("- Tahan gesture selama recording berlangsung")
        
        input("\n⌨️  Tekan Enter untuk mulai countdown...")
        
        # Countdown phase
        self._show_countdown()
        
        # Recording phase
        recorded_samples = self._record_gesture_data(gesture_name, target_samples)
        
        print(f"\n✅ Recording selesai! Berhasil record {recorded_samples} samples")
        
        # Update session stats
        if gesture_name not in self.session_stats['gestures_recorded']:
            self.session_stats['gestures_recorded'][gesture_name] = 0
        self.session_stats['gestures_recorded'][gesture_name] += recorded_samples
        self.session_stats['total_samples'] += recorded_samples
        
        return recorded_samples
    
    def _show_countdown(self):
        """Show countdown dengan visual feedback di OpenCV window"""
        print("\n⏰ Countdown dimulai...")
        
        countdown_start = time.time()
        
        while True:
            # Get current frame
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Calculate remaining time
            elapsed = time.time() - countdown_start
            remaining = self.countdown_duration - elapsed
            
            if remaining <= 0:
                break
            
            # Process frame untuk hand detection
            processed_frame, landmarks = self.hand_tracker.process_frame(frame)
            
            # Add countdown overlay
            overlay = processed_frame.copy()
            height, width = overlay.shape[:2]
            
            # Large countdown number
            countdown_num = int(remaining) + 1
            cv2.putText(overlay, str(countdown_num), 
                       (width//2 - 50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            
            # Status info
            hand_status = "✅ TANGAN TERDETEKSI" if landmarks else "❌ TIDAK ADA TANGAN"
            cv2.putText(overlay, hand_status, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if landmarks else (0, 0, 255), 2)
            
            cv2.putText(overlay, "BERSIAP UNTUK RECORDING...", (50, height-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow('Gesture Data Collection', overlay)
            
            # Handle exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return False
        
        # Final "GO!" message
        frame = self.camera.get_frame()
        if frame is not None:
            overlay = frame.copy()
            height, width = overlay.shape[:2]
            cv2.putText(overlay, "RECORDING!", 
                       (width//2 - 150, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
            cv2.imshow('Gesture Data Collection', overlay)
            cv2.waitKey(500)
        
        return True
    
    def _record_gesture_data(self, gesture_name: str, target_samples: int) -> int:
        """
        Record actual gesture data dengan real-time feedback
        
        Args:
            gesture_name: Nama gesture
            target_samples: Target samples
            
        Returns:
            int: Actual samples recorded
        """
        recorded_samples = 0
        recording_start = time.time()
        last_sample_time = 0
        sample_interval = self.recording_duration / target_samples  # Interval antar samples
        
        while True:
            current_time = time.time()
            elapsed = current_time - recording_start
            
            # Check apakah recording sudah selesai
            if elapsed >= self.recording_duration:
                break
            
            # Get frame
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Process frame
            processed_frame, landmarks = self.hand_tracker.process_frame(frame)
            
            # Record sample jika ada hand detected dan interval sudah lewat
            if (landmarks and 
                current_time - last_sample_time >= sample_interval):
                
                # Create data row
                data_row = [gesture_name] + landmarks
                self.collected_data.append(data_row)
                
                recorded_samples += 1
                last_sample_time = current_time
            
            # Add recording overlay
            overlay = processed_frame.copy()
            height, width = overlay.shape[:2]
            
            # Progress bar
            progress = elapsed / self.recording_duration
            bar_width = int(400 * progress)
            cv2.rectangle(overlay, (50, 100), (450, 130), (100, 100, 100), -1)
            cv2.rectangle(overlay, (50, 100), (50 + bar_width, 130), (0, 255, 0), -1)
            
            # Stats
            cv2.putText(overlay, f"Gesture: {gesture_name}", (50, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay, f"Samples: {recorded_samples}/{target_samples}", (50, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(overlay, f"Time: {elapsed:.1f}s/{self.recording_duration}s", (50, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Hand detection status
            status_color = (0, 255, 0) if landmarks else (0, 0, 255)
            status_text = "RECORDING" if landmarks else "NO HAND"
            cv2.putText(overlay, status_text, (50, height-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('Gesture Data Collection', overlay)
            
            # Handle exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        return recorded_samples
    
    def record_predefined_gestures(self):
        """Record semua predefined gestures untuk assistive communication"""
        print(f"\n🤚 Recording {len(self.predefined_gestures)} predefined gestures...")
        
        for i, gesture in enumerate(self.predefined_gestures):
            print(f"\n📍 Progress: {i+1}/{len(self.predefined_gestures)}")
            print(f"🎯 Current gesture: '{gesture}'")
            
            # Explanation untuk setiap gesture
            self._show_gesture_explanation(gesture)
            
            # Ask user jika ready
            response = input(f"\n⌨️  Ready untuk record '{gesture}'? (y/n/skip): ").lower()
            
            if response == 'n':
                break
            elif response == 'skip':
                continue
            
            # Record gesture
            samples = self.record_single_gesture(gesture)
            
            if samples < self.samples_per_gesture * 0.5:  # Kurang dari 50% target
                print("⚠️  Samples kurang, mau coba lagi? (y/n)")
                retry = input().lower()
                if retry == 'y':
                    self.record_single_gesture(gesture)
        
        print("\n✅ Predefined gestures recording selesai!")
    
    def _show_gesture_explanation(self, gesture: str):
        """Show explanation untuk setiap gesture"""
        explanations = {
            'tolong': '🆘 Gesture TOLONG: Tangan menunjuk ke depan atau tangan di atas kepala seperti meminta bantuan',
            'halo': '👋 Gesture HALO: Tangan melambai dari kiri ke kanan seperti menyapa',
            'terima_kasih': '🙏 Gesture TERIMA KASIH: Kedua tangan di depan dada seperti berdoa atau satu tangan di dada',
            'maaf': '🤲 Gesture MAAF: Tangan terbuka menghadap ke atas seperti meminta maaf',
            'ya': '👍 Gesture YA: Thumbs up atau tangan mengangguk',
            'tidak': '❌ Gesture TIDAK: Tangan menggeleng atau gesture stop'
        }
        
        explanation = explanations.get(gesture, f"Gesture {gesture}")
        print(f"💡 {explanation}")
    
    def save_data_to_csv(self) -> bool:
        """
        Save collected data ke CSV file
        
        Returns:
            bool: True jika berhasil save
        """
        if not self.collected_data:
            print("⚠️  Tidak ada data untuk disimpan")
            return False
        
        try:
            # Create DataFrame
            columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            df = pd.DataFrame(self.collected_data, columns=columns)
            
            # Load existing data jika file sudah ada
            if os.path.exists(self.output_file):
                existing_df = pd.read_csv(self.output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
                print(f"📝 Menambahkan {len(self.collected_data)} samples ke existing data")
            else:
                print(f"📝 Membuat file baru dengan {len(self.collected_data)} samples")
            
            # Save ke CSV
            df.to_csv(self.output_file, index=False)
            
            print(f"✅ Data berhasil disimpan ke {self.output_file}")
            print(f"📊 Total samples di file: {len(df)}")
            
            # Show data statistics
            self._show_data_statistics(df)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False
    
    def _show_data_statistics(self, df: pd.DataFrame):
        """Show statistics dari collected data"""
        print("\n📊 DATA STATISTICS:")
        print("-" * 30)
        
        # Gesture counts
        gesture_counts = df['label'].value_counts()
        for gesture, count in gesture_counts.items():
            print(f"  {gesture}: {count} samples")
        
        print(f"\n  Total samples: {len(df)}")
        print(f"  Total gestures: {len(gesture_counts)}")
    
    def view_existing_data(self):
        """View existing data di CSV file"""
        if not os.path.exists(self.output_file):
            print("📁 Belum ada data yang tersimpan")
            return
        
        try:
            df = pd.read_csv(self.output_file)
            print(f"\n📊 Existing Data: {self.output_file}")
            self._show_data_statistics(df)
            
        except Exception as e:
            print(f"❌ Error reading data: {e}")
    
    def test_hand_detection(self):
        """Test mode untuk verify hand detection bekerja dengan baik"""
        print("\n🧪 Hand Detection Test Mode")
        print("⌨️  Tekan ESC untuk keluar")
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            processed_frame, landmarks = self.hand_tracker.process_frame(frame)
            
            # Add test info
            display_frame = self.hand_tracker.draw_info(
                processed_frame, landmarks, "Test Mode"
            )
            
            cv2.imshow('Hand Detection Test', display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
    
    def show_session_summary(self):
        """Show summary dari current session"""
        elapsed_time = time.time() - self.session_stats['session_start']
        
        print("\n" + "="*40)
        print("📊 SESSION SUMMARY")
        print("="*40)
        print(f"⏱️  Session duration: {elapsed_time/60:.1f} menit")
        print(f"📈 Total samples collected: {self.session_stats['total_samples']}")
        
        if self.session_stats['gestures_recorded']:
            print("🤚 Gestures recorded:")
            for gesture, count in self.session_stats['gestures_recorded'].items():
                print(f"   {gesture}: {count} samples")
        else:
            print("🤚 No gestures recorded this session")
        
        print("="*40)
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🗑️  Cleaning up resources...")
        
        cv2.destroyAllWindows()
        self.camera.release()
        self.hand_tracker.cleanup()
        
        print("✅ Cleanup complete")
    
    def run_interactive_session(self):
        """Main interactive session untuk data collection"""
        if not self.initialize_hardware():
            print("❌ Hardware initialization gagal")
            return
        
        try:
            while True:
                self.show_main_menu()
                choice = input("\n⌨️  Pilih opsi (1-5): ").strip()
                
                if choice == '1':
                    gesture_name = input("🏷️  Masukkan nama gesture: ").strip()
                    if gesture_name:
                        self.record_single_gesture(gesture_name)
                        
                        # Ask untuk save
                        save_choice = input("\n💾 Save data sekarang? (y/n): ").lower()
                        if save_choice == 'y':
                            self.save_data_to_csv()
                
                elif choice == '2':
                    self.record_predefined_gestures()
                    
                    # Auto save after predefined gestures
                    if self.collected_data:
                        self.save_data_to_csv()
                
                elif choice == '3':
                    self.view_existing_data()
                
                elif choice == '4':
                    self.test_hand_detection()
                
                elif choice == '5':
                    break
                
                else:
                    print("❌ Pilihan tidak valid")
        
        except KeyboardInterrupt:
            print("\n⏹️  Session interrupted by user")
        
        finally:
            # Show summary
            self.show_session_summary()
            
            # Save remaining data
            if self.collected_data:
                print("\n💾 Saving remaining data...")
                self.save_data_to_csv()
            
            # Cleanup
            self.cleanup()


def main():
    """Main function dengan argument parsing"""
    parser = argparse.ArgumentParser(description='Gesture Data Collection System')
    parser.add_argument('--output', '-o', 
                       default='/app/GestureTalk/data/gestures.csv',
                       help='Output CSV file path')
    parser.add_argument('--samples', '-s', 
                       type=int, default=100,
                       help='Target samples per gesture')
    parser.add_argument('--duration', '-d', 
                       type=float, default=5.0,
                       help='Recording duration per gesture (seconds)')
    
    args = parser.parse_args()
    
    # Create collector
    collector = GestureDataCollector(output_file=args.output)
    collector.samples_per_gesture = args.samples
    collector.recording_duration = args.duration
    
    print("🚀 Starting Gesture Data Collection System...")
    print(f"📁 Output file: {args.output}")
    print(f"🎯 Target samples per gesture: {args.samples}")
    print(f"⏱️  Recording duration: {args.duration}s")
    
    # Run interactive session
    collector.run_interactive_session()


if __name__ == "__main__":
    main()