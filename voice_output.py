#!/usr/bin/env python3
"""
Voice Output Module
Text-to-speech using pyttsx3 for the Vision AI Assistant
"""

import pyttsx3
import threading
import queue
import time
from typing import Optional


class VoiceOutput:
    """Text-to-speech voice output using pyttsx3"""
    
    def __init__(self, 
                 rate=150, 
                 volume=0.8, 
                 voice_id=None,
                 queue_size=10):
        """
        Initialize voice output
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_id: Specific voice ID to use
            queue_size: Maximum queue size for speech requests
        """
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.queue_size = queue_size
        
        # TTS engine
        self.engine = None
        
        # Threading
        self.speech_queue = queue.Queue(maxsize=queue_size)
        self.speech_thread = None
        self.is_running = False
        self.is_speaking = False
        
        # Speech control
        self.speech_enabled = True
        self.last_speech_time = 0
        self.min_speech_interval = 0.5  # Minimum time between speeches
        
        # Initialize TTS engine
        self._initialize_tts()
        self._start_speech_thread()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        print("Initializing voice output...")
        
        try:
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Set speech properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice if specified
            if self.voice_id:
                voices = self.engine.getProperty('voices')
                if voices and self.voice_id < len(voices):
                    self.engine.setProperty('voice', voices[self.voice_id].id)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                print(f"Available voices: {len(voices)}")
                for i, voice in enumerate(voices):
                    print(f"  {i}: {voice.name} ({voice.languages})")
            
            print("✅ Voice output initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing voice output: {e}")
            raise
    
    def _start_speech_thread(self):
        """Start background speech thread"""
        self.is_running = True
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        print("✅ Speech thread started!")
    
    def _speech_worker(self):
        """Background worker for speech processing"""
        while self.is_running:
            try:
                # Get speech request from queue
                speech_request = self.speech_queue.get(timeout=1.0)
                
                if speech_request is None:  # Shutdown signal
                    break
                
                text, priority = speech_request
                
                # Check if speech is enabled and enough time has passed
                if (self.speech_enabled and 
                    time.time() - self.last_speech_time >= self.min_speech_interval):
                    
                    self._speak_text(text)
                    self.last_speech_time = time.time()
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
    
    def _speak_text(self, text: str):
        """Actually speak the text"""
        try:
            self.is_speaking = True
            print(f"🗣️ Speaking: {text}")
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            self.is_speaking = False
            
        except Exception as e:
            print(f"Error speaking text: {e}")
            self.is_speaking = False
    
    def speak(self, text: str, priority: int = 0):
        """
        Queue text for speech
        
        Args:
            text: Text to speak
            priority: Priority level (higher = more important)
        """
        if not text or not text.strip():
            return
        
        if not self.speech_enabled:
            print(f"Speech disabled, would say: {text}")
            return
        
        try:
            # Add to queue with priority
            self.speech_queue.put((text.strip(), priority), block=False)
        except queue.Full:
            print("Speech queue is full, dropping message")
    
    def speak_immediate(self, text: str):
        """
        Speak text immediately (blocking)
        
        Args:
            text: Text to speak
        """
        if not text or not text.strip():
            return
        
        if not self.speech_enabled:
            print(f"Speech disabled, would say: {text}")
            return
        
        # Stop current speech if any
        self.stop_speaking()
        
        # Speak immediately
        self._speak_text(text.strip())
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.is_speaking:
            try:
                self.engine.stop()
                self.is_speaking = False
            except Exception as e:
                print(f"Error stopping speech: {e}")
    
    def clear_queue(self):
        """Clear speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """Set speech volume"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)
    
    def set_voice(self, voice_id: int):
        """Set voice by ID"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            if voices and 0 <= voice_id < len(voices):
                self.engine.setProperty('voice', voices[voice_id].id)
                self.voice_id = voice_id
    
    def enable_speech(self):
        """Enable speech output"""
        self.speech_enabled = True
        print("✅ Speech output enabled")
    
    def disable_speech(self):
        """Disable speech output"""
        self.speech_enabled = False
        self.stop_speaking()
        self.clear_queue()
        print("🔇 Speech output disabled")
    
    def get_status(self):
        """Get current status"""
        return {
            'is_running': self.is_running,
            'is_speaking': self.is_speaking,
            'speech_enabled': self.speech_enabled,
            'queue_size': self.speech_queue.qsize(),
            'rate': self.rate,
            'volume': self.volume,
            'voice_id': self.voice_id
        }
    
    def get_available_voices(self):
        """Get list of available voices"""
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        if not voices:
            return []
        
        return [
            {
                'id': i,
                'name': voice.name,
                'languages': voice.languages,
                'gender': getattr(voice, 'gender', 'unknown')
            }
            for i, voice in enumerate(voices)
        ]
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up voice output...")
        
        self.is_running = False
        
        # Stop current speech
        self.stop_speaking()
        
        # Clear queue and send shutdown signal
        self.clear_queue()
        try:
            self.speech_queue.put(None, block=False)
        except queue.Full:
            pass
        
        # Wait for speech thread to finish
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2.0)
        
        # Clean up engine
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        print("✅ Voice output cleanup completed!")


class SpeechAnnouncer:
    """Specialized speech announcer for Vision AI Assistant"""
    
    def __init__(self, voice_output: VoiceOutput):
        """
        Initialize speech announcer
        
        Args:
            voice_output: VoiceOutput instance
        """
        self.voice = voice_output
        self.announcement_queue = queue.Queue()
        self.is_announcing = False
    
    def announce_detection(self, detections, depths, directions):
        """
        Announce object detections
        
        Args:
            detections: List of detected objects
            depths: List of depth information
            directions: List of direction information
        """
        if not detections:
            self.voice.speak("No objects detected.")
            return
        
        # Count objects by type
        object_counts = {}
        for det in detections:
            label = det['label']
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Create announcement
        announcements = []
        for label, count in object_counts.items():
            if count == 1:
                announcements.append(f"1 {label}")
            else:
                announcements.append(f"{count} {label}s")
        
        announcement = f"Detected: {', '.join(announcements)}."
        self.voice.speak(announcement)
    
    def announce_detailed_detection(self, detection, depth, direction):
        """
        Announce detailed detection information
        
        Args:
            detection: Single detection object
            depth: Depth information
            direction: Direction information
        """
        label = detection['label']
        distance_desc = depth.get('distance_description', f"{depth['distance_steps']} steps ahead")
        direction_desc = direction.get('combined', 'center')
        
        announcement = f"{label}, {distance_desc} {direction_desc}."
        self.voice.speak(announcement)
    
    def announce_navigation(self, route_info):
        """
        Announce navigation information
        
        Args:
            route_info: Route information dictionary
        """
        if not route_info:
            self.voice.speak("No route information available.")
            return
        
        distance = route_info.get('distance_formatted', 'unknown distance')
        duration = route_info.get('duration_formatted', 'unknown time')
        steps = route_info.get('steps', 0)
        
        announcement = f"Route found. Distance: {distance}, about {steps} steps. Estimated time: {duration}."
        self.voice.speak(announcement)
    
    def announce_navigation_step(self, step_info):
        """
        Announce navigation step
        
        Args:
            step_info: Navigation step information
        """
        instruction = step_info.get('instruction', 'Continue straight')
        distance = step_info.get('distance', '')
        
        if distance:
            announcement = f"{instruction} for {distance}."
        else:
            announcement = instruction
        
        self.voice.speak(announcement)
    
    def announce_error(self, error_message):
        """Announce error message"""
        self.voice.speak(f"Error: {error_message}")
    
    def announce_status(self, status_message):
        """Announce status message"""
        self.voice.speak(status_message)


def test_voice_output():
    """Test voice output module"""
    print("Testing voice output module...")
    
    try:
        # Initialize voice output
        voice = VoiceOutput(rate=150, volume=0.8)
        
        # Test basic speech
        print("Testing basic speech...")
        voice.speak("Hello, this is a test of the voice output system.")
        
        time.sleep(2)
        
        # Test multiple speeches
        print("Testing multiple speeches...")
        voice.speak("First message")
        voice.speak("Second message")
        voice.speak("Third message")
        
        time.sleep(5)
        
        # Test immediate speech
        print("Testing immediate speech...")
        voice.speak_immediate("This is an immediate speech test.")
        
        time.sleep(2)
        
        # Test status
        status = voice.get_status()
        print(f"Voice status: {status}")
        
        # Test available voices
        voices = voice.get_available_voices()
        print(f"Available voices: {len(voices)}")
        for voice_info in voices[:3]:  # Show first 3
            print(f"  {voice_info['name']} ({voice_info['languages']})")
        
        time.sleep(2)
        
        # Test speech announcer
        print("Testing speech announcer...")
        announcer = SpeechAnnouncer(voice)
        
        # Test detection announcement
        test_detections = [
            {'label': 'person', 'confidence': 0.9},
            {'label': 'chair', 'confidence': 0.8}
        ]
        test_depths = [
            {'distance_steps': 3, 'distance_description': '3 steps ahead'},
            {'distance_steps': 2, 'distance_description': '2 steps ahead'}
        ]
        test_directions = [
            {'combined': 'center'},
            {'combined': 'left'}
        ]
        
        announcer.announce_detection(test_detections, test_depths, test_directions)
        
        time.sleep(3)
        
        # Test detailed announcement
        announcer.announce_detailed_detection(
            test_detections[0], test_depths[0], test_directions[0]
        )
        
        time.sleep(3)
        
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        if 'voice' in locals():
            voice.cleanup()
    
    print("✅ Voice output test completed!")


if __name__ == "__main__":
    test_voice_output()