"""
TTS Handler Module - Text-to-Speech untuk komunikasi assistive
Handles speech synthesis dengan fokus pada Bahasa Indonesia
"""

import pyttsx3
import threading
import queue
import time
import logging
from typing import Optional, Dict, List
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSStatus(Enum):
    """Status enumeration untuk TTS operations"""
    IDLE = "idle"
    SPEAKING = "speaking"
    ERROR = "error"
    DISABLED = "disabled"


class TTSHandler:
    """
    Text-to-Speech handler untuk komunikasi assistive dengan Bahasa Indonesia
    
    Features:
    - Bahasa Indonesia support dengan voice optimization
    - Non-blocking speech dengan threading
    - Speech queue management untuk multiple requests
    - Voice customization (speed, volume)
    - Emergency phrases untuk assistive communication
    - Error handling dan fallback mechanisms
    """
    
    def __init__(self, 
                 language: str = 'id',
                 rate: int = 150,
                 volume: float = 0.9):
        """
        Initialize TTSHandler
        
        Args:
            language: Language code ('id' untuk Bahasa Indonesia)
            rate: Speech rate (150 optimal untuk Indonesian)
            volume: Volume level (0.0 - 1.0)
        """
        self.language = language
        self.rate = rate
        self.volume = volume
        
        # TTS engine
        self.engine = None
        self.status = TTSStatus.IDLE
        
        # Threading untuk non-blocking speech
        self.speech_thread = None
        self.speech_queue = queue.Queue()
        self.is_running = False
        self.current_speech = None
        
        # Voice settings
        self.available_voices = []
        self.selected_voice = None
        
        # Predefined phrases untuk assistive communication
        self.emergency_phrases = {
            "tolong": "Saya butuh bantuan",
            "halo": "Halo, apa kabar?",
            "terima_kasih": "Terima kasih banyak",
            "maaf": "Maaf, permisi",
            "ya": "Ya, benar",
            "tidak": "Tidak, salah",
            "stop": "Berhenti",
            "lanjut": "Lanjutkan",
            "repeat": "Ulangi lagi",
            "help": "Saya perlu bantuan"
        }
        
        # Initialize TTS engine
        self._initialize_tts()
        
    def _initialize_tts(self) -> bool:
        """
        Initialize pyttsx3 engine dengan optimization untuk Bahasa Indonesia
        
        Returns:
            bool: True jika berhasil initialize
        """
        try:
            logger.info("🔧 Initializing TTS engine...")
            
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            if not self.engine:
                logger.error("❌ Gagal initialize TTS engine")
                self.status = TTSStatus.ERROR
                return False
            
            # Set basic properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Get available voices
            self._load_available_voices()
            
            # Set Indonesian voice jika tersedia
            self._set_indonesian_voice()
            
            # Test TTS engine
            test_success = self._test_tts_engine()
            if not test_success:
                logger.warning("⚠️  TTS engine test gagal, tapi engine tetap aktif")
            
            self.status = TTSStatus.IDLE
            logger.info("✅ TTS engine berhasil diinisialisasi")
            return True
            
        except Exception as e:
            logger.error(f"❌ TTS initialization error: {e}")
            self.status = TTSStatus.ERROR
            return False
    
    def _load_available_voices(self):
        """Load dan analyze available voices di sistem"""
        try:
            voices = self.engine.getProperty('voices')
            self.available_voices = []
            
            if voices:
                for i, voice in enumerate(voices):
                    voice_info = {
                        'index': i,
                        'id': voice.id,
                        'name': voice.name,
                        'language': getattr(voice, 'languages', []),
                        'gender': getattr(voice, 'gender', 'unknown')
                    }
                    self.available_voices.append(voice_info)
                    
                logger.info(f"📊 Found {len(self.available_voices)} voices")
                
                # Log voice details
                for voice in self.available_voices:
                    logger.info(f"   Voice: {voice['name']} | Lang: {voice['language']}")
                    
        except Exception as e:
            logger.error(f"❌ Error loading voices: {e}")
    
    def _set_indonesian_voice(self):
        """
        Set voice yang paling cocok untuk Bahasa Indonesia
        """
        # Cari voice Indonesia
        indonesian_voice = None
        for voice in self.available_voices:
            if 'id' in str(voice['language']).lower() or 'indonesia' in voice['name'].lower():
                indonesian_voice = voice
                break
        
        # Jika tidak ada, cari voice yang neutral/universal
        if not indonesian_voice:
            for voice in self.available_voices:
                if 'english' in voice['name'].lower() and 'female' in voice['name'].lower():
                    indonesian_voice = voice
                    break
        
        # Set voice
        if indonesian_voice:
            try:
                self.engine.setProperty('voice', indonesian_voice['id'])
                self.selected_voice = indonesian_voice
                logger.info(f"✅ Selected voice: {indonesian_voice['name']}")
            except Exception as e:
                logger.error(f"❌ Error setting voice: {e}")
        else:
            logger.warning("⚠️  Tidak ada Indonesian voice, menggunakan default")
    
    def _test_tts_engine(self) -> bool:
        """
        Test TTS engine dengan phrase sederhana
        
        Returns:
            bool: True jika test berhasil
        """
        try:
            # Disable output untuk test
            self.engine.say("Test")
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"❌ TTS test error: {e}")
            return False
    
    def start_speech_service(self):
        """
        Start threaded speech service untuk non-blocking TTS
        """
        if self.is_running:
            logger.warning("⚠️  Speech service sudah berjalan")
            return
        
        if self.status == TTSStatus.ERROR:
            logger.error("❌ TTS engine error, tidak bisa start service")
            return
        
        self.is_running = True
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
        logger.info("🎤 Speech service started")
    
    def _speech_worker(self):
        """
        Worker thread untuk handle speech requests dari queue
        """
        while self.is_running:
            try:
                # Get speech request dari queue (blocking dengan timeout)
                speech_request = self.speech_queue.get(timeout=1)
                
                if speech_request is None:  # Shutdown signal
                    break
                
                text = speech_request.get('text', '')
                priority = speech_request.get('priority', 'normal')
                
                if text.strip():
                    self._speak_text_sync(text)
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"❌ Speech worker error: {e}")
        
        logger.info("🛑 Speech worker stopped")
    
    def _speak_text_sync(self, text: str):
        """
        Synchronous speech execution (runs di worker thread)
        
        Args:
            text: Text untuk diucapkan
        """
        try:
            self.status = TTSStatus.SPEAKING
            self.current_speech = text
            
            logger.info(f"🎤 Speaking: '{text}'")
            
            self.engine.say(text)
            self.engine.runAndWait()
            
            self.status = TTSStatus.IDLE
            self.current_speech = None
            
        except Exception as e:
            logger.error(f"❌ Speech error: {e}")
            self.status = TTSStatus.ERROR
    
    def speak(self, text: str, priority: str = 'normal') -> bool:
        """
        Add speech request ke queue (non-blocking)
        
        Args:
            text: Text untuk diucapkan
            priority: Priority level ('high', 'normal', 'low')
            
        Returns:
            bool: True jika berhasil add ke queue
        """
        if not self.is_running:
            logger.error("❌ Speech service belum distart")
            return False
        
        if self.status == TTSStatus.ERROR:
            logger.error("❌ TTS engine error")
            return False
        
        # Validate text
        if not text or not text.strip():
            logger.warning("⚠️  Text kosong, skip speech")
            return False
        
        # Clean text
        clean_text = self._clean_text(text)
        
        try:
            # Add ke queue
            speech_request = {
                'text': clean_text,
                'priority': priority,
                'timestamp': time.time()
            }
            
            # Clear queue jika high priority
            if priority == 'high':
                self._clear_queue()
            
            self.speech_queue.put(speech_request)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding speech to queue: {e}")
            return False
    
    def speak_gesture(self, gesture_name: str) -> bool:
        """
        Speak predefined phrase untuk recognized gesture
        
        Args:
            gesture_name: Nama gesture yang dideteksi
            
        Returns:
            bool: True jika berhasil
        """
        # Cek emergency phrases
        phrase = self.emergency_phrases.get(gesture_name.lower())
        
        if phrase:
            return self.speak(phrase, priority='high')
        else:
            # Fallback ke gesture name
            fallback_text = f"Gesture {gesture_name} terdeteksi"
            return self.speak(fallback_text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean dan optimize text untuk Indonesian TTS
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Replace common abbreviations untuk better pronunciation
        replacements = {
            'ok': 'oke',
            'thx': 'terima kasih',
            'ty': 'terima kasih',
            'plz': 'tolong',
            'pls': 'tolong'
        }
        
        for abbrev, full in replacements.items():
            text = text.replace(abbrev, full)
        
        return text
    
    def _clear_queue(self):
        """Clear speech queue untuk high priority requests"""
        try:
            while not self.speech_queue.empty():
                self.speech_queue.get_nowait()
        except queue.Empty:
            pass
    
    def stop_current_speech(self):
        """Stop current speech jika sedang berbicara"""
        try:
            if self.status == TTSStatus.SPEAKING and self.engine:
                self.engine.stop()
                logger.info("🛑 Speech stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping speech: {e}")
    
    def get_status(self) -> Dict:
        """
        Get status lengkap dari TTS system
        
        Returns:
            Dict: Status information
        """
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'current_speech': self.current_speech,
            'queue_size': self.speech_queue.qsize(),
            'selected_voice': self.selected_voice['name'] if self.selected_voice else 'default',
            'settings': {
                'rate': self.rate,
                'volume': self.volume,
                'language': self.language
            },
            'available_voices': len(self.available_voices),
            'emergency_phrases': len(self.emergency_phrases)
        }
    
    def update_settings(self, rate: Optional[int] = None, volume: Optional[float] = None):
        """
        Update TTS settings secara real-time
        
        Args:
            rate: New speech rate
            volume: New volume level
        """
        try:
            if rate is not None and 50 <= rate <= 300:
                self.rate = rate
                self.engine.setProperty('rate', rate)
                logger.info(f"✅ Speech rate updated to {rate}")
            
            if volume is not None and 0.0 <= volume <= 1.0:
                self.volume = volume
                self.engine.setProperty('volume', volume)
                logger.info(f"✅ Volume updated to {volume}")
                
        except Exception as e:
            logger.error(f"❌ Error updating TTS settings: {e}")
    
    def add_custom_phrase(self, gesture: str, phrase: str):
        """
        Add custom phrase untuk specific gesture
        
        Args:
            gesture: Gesture name
            phrase: Phrase yang akan diucapkan
        """
        self.emergency_phrases[gesture.lower()] = phrase
        logger.info(f"✅ Added custom phrase: {gesture} -> {phrase}")
    
    def stop_speech_service(self):
        """Stop speech service dan cleanup"""
        logger.info("🛑 Stopping speech service...")
        
        self.is_running = False
        
        # Clear queue
        self._clear_queue()
        
        # Add shutdown signal
        self.speech_queue.put(None)
        
        # Wait for thread to finish
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=3)
        
        logger.info("✅ Speech service stopped")
    
    def cleanup(self):
        """
        Complete cleanup TTS resources
        """
        logger.info("🗑️  Cleaning up TTS resources...")
        
        # Stop service
        self.stop_speech_service()
        
        # Cleanup engine
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
            self.engine = None
        
        self.status = TTSStatus.DISABLED
        logger.info("✅ TTS resources cleaned up")


# Test function untuk development
def test_tts_handler():
    """
    Test function untuk verify TTSHandler berfungsi dengan baik
    """
    print("🚀 Testing TTSHandler...")
    
    # Initialize TTS handler
    tts = TTSHandler()
    
    if tts.status == TTSStatus.ERROR:
        print("❌ TTS initialization gagal")
        return
    
    print("✅ TTS berhasil diinisialisasi")
    
    # Print TTS info
    status = tts.get_status()
    print(f"📊 TTS Status: {status}")
    
    # Start speech service
    tts.start_speech_service()
    
    # Test basic speech
    print("🎤 Testing basic speech...")
    tts.speak("Halo, ini adalah test sistem Text to Speech")
    
    time.sleep(3)
    
    # Test gesture phrases
    print("🤚 Testing gesture phrases...")
    test_gestures = ['tolong', 'halo', 'terima_kasih']
    
    for gesture in test_gestures:
        print(f"   Testing gesture: {gesture}")
        tts.speak_gesture(gesture)
        time.sleep(2)
    
    print("⌨️  Press Enter to continue...")
    input()
    
    # Cleanup
    tts.cleanup()
    print("✅ TTSHandler test selesai")


if __name__ == "__main__":
    test_tts_handler()