#!/usr/bin/env python3
"""
Voice Output Module for Vision AI Assistant
Provides offline text-to-speech functionality using pyttsx3
"""

import pyttsx3
import threading
import time
from typing import Optional


class VoiceOutput:
    """
    Voice output handler for Vision AI Assistant
    Provides offline text-to-speech with configurable settings
    """
    
    def __init__(self, 
                 rate: int = 150, 
                 volume: float = 0.8, 
                 voice_index: int = 0,
                 enable_console_output: bool = True):
        """
        Initialize voice output engine
        
        Args:
            rate: Speech rate (words per minute, default 150)
            volume: Volume level (0.0 to 1.0, default 0.8)
            voice_index: Voice index (0=male, 1=female, etc.)
            enable_console_output: Whether to print to console
        """
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.enable_console_output = enable_console_output
        self.engine = None
        self.is_speaking = False
        self.speech_queue = []
        self.thread_lock = threading.Lock()
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 engine with configured settings"""
        try:
            self.engine = pyttsx3.init()
            
            # Set voice properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice (if available)
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
            
            print("✅ Voice output engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing voice engine: {e}")
            self.engine = None
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak text aloud and optionally print to console
        
        Args:
            text: Text to speak
            blocking: Whether to wait for speech to complete
        """
        if not self.engine:
            print(f"[Assistant]: {text} (Voice engine not available)")
            return
        
        if not text or not text.strip():
            return
        
        # Print to console if enabled
        if self.enable_console_output:
            print(f"[Assistant]: {text}")
        
        try:
            with self.thread_lock:
                self.is_speaking = True
            
            # Speak the text
            self.engine.say(text)
            
            if blocking:
                self.engine.runAndWait()
            else:
                # Non-blocking mode - start in separate thread
                def speak_thread():
                    self.engine.runAndWait()
                    with self.thread_lock:
                        self.is_speaking = False
                
                thread = threading.Thread(target=speak_thread)
                thread.daemon = True
                thread.start()
                
        except Exception as e:
            print(f"❌ Error speaking text: {e}")
        finally:
            if blocking:
                with self.thread_lock:
                    self.is_speaking = False
    
    def speak_async(self, text: str):
        """Speak text asynchronously (non-blocking)"""
        self.speak(text, blocking=False)
    
    def speak_detection(self, label: str, steps: int, direction: str):
        """
        Speak a detection result in a standardized format
        
        Args:
            label: Object label (e.g., "person", "chair")
            steps: Distance in steps
            direction: Direction ("ahead", "on your left", "on your right")
        """
        if direction == "center":
            direction_desc = "ahead"
        elif direction == "left":
            direction_desc = "on your left"
        else:  # right
            direction_desc = "on your right"
        
        message = f"{label} detected, about {steps} steps {direction_desc}"
        self.speak(message)
    
    def speak_multiple_detections(self, detections: list):
        """
        Speak multiple detections in sequence
        
        Args:
            detections: List of detection dictionaries with 'label', 'steps', 'direction'
        """
        if not detections:
            return
        
        # Create combined message
        messages = []
        for det in detections:
            label = det.get('label', 'object')
            steps = det.get('steps', 1)
            direction = det.get('direction', 'ahead')
            
            if direction == "center":
                direction_desc = "ahead"
            elif direction == "left":
                direction_desc = "on your left"
            else:  # right
                direction_desc = "on your right"
            
            messages.append(f"{label} detected, about {steps} steps {direction_desc}")
        
        # Speak combined message
        combined_text = ". ".join(messages) + "."
        self.speak(combined_text)
    
    def speak_ai_response(self, response: str):
        """
        Speak a response from AI agent
        
        Args:
            response: AI response text
        """
        self.speak(response)
    
    def speak_error(self, error_message: str):
        """
        Speak an error message
        
        Args:
            error_message: Error description
        """
        self.speak(f"Error: {error_message}")
    
    def speak_status(self, status_message: str):
        """
        Speak a status message
        
        Args:
            status_message: Status description
        """
        self.speak(status_message)
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """Set volume level"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.volume)
    
    def set_voice(self, voice_index: int):
        """Set voice index"""
        if not self.engine:
            return
        
        voices = self.engine.getProperty('voices')
        if voices and len(voices) > voice_index:
            self.engine.setProperty('voice', voices[voice_index].id)
            self.voice_index = voice_index
    
    def get_available_voices(self):
        """Get list of available voices"""
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        return [(i, voice.name, voice.id) for i, voice in enumerate(voices)] if voices else []
    
    def is_available(self):
        """Check if voice engine is available"""
        return self.engine is not None
    
    def stop(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def cleanup(self):
        """Clean up voice engine"""
        if self.engine:
            try:
                self.engine.stop()
                del self.engine
                self.engine = None
            except:
                pass


# -------------------
# Convenience Functions
# -------------------

# Global voice output instance
_voice_output = None

def initialize_voice_output(rate: int = 150, 
                          volume: float = 0.8, 
                          voice_index: int = 0,
                          enable_console_output: bool = True):
    """Initialize global voice output instance"""
    global _voice_output
    _voice_output = VoiceOutput(rate, volume, voice_index, enable_console_output)
    return _voice_output

def speak(text: str, blocking: bool = True):
    """
    Convenience function to speak text using global voice output
    
    Args:
        text: Text to speak
        blocking: Whether to wait for speech to complete
    """
    global _voice_output
    if _voice_output is None:
        _voice_output = initialize_voice_output()
    
    _voice_output.speak(text, blocking)

def speak_detection(label: str, steps: int, direction: str):
    """
    Convenience function to speak detection result
    
    Args:
        label: Object label
        steps: Distance in steps
        direction: Direction
    """
    global _voice_output
    if _voice_output is None:
        _voice_output = initialize_voice_output()
    
    _voice_output.speak_detection(label, steps, direction)

def speak_ai_response(response: str):
    """Convenience function to speak AI response"""
    global _voice_output
    if _voice_output is None:
        _voice_output = initialize_voice_output()
    
    _voice_output.speak_ai_response(response)

def get_voice_output():
    """Get global voice output instance"""
    global _voice_output
    if _voice_output is None:
        _voice_output = initialize_voice_output()
    return _voice_output


# -------------------
# Example Usage
# -------------------

if __name__ == "__main__":
    print("Voice Output Module - Test")
    print("=" * 40)
    
    # Initialize voice output
    voice = VoiceOutput(rate=150, volume=0.8)
    
    if voice.is_available():
        print("✅ Voice engine ready!")
        
        # Test basic speech
        voice.speak("Vision AI assistant is ready.")
        
        # Test detection speech
        voice.speak_detection("person", 3, "ahead")
        voice.speak_detection("chair", 5, "left")
        voice.speak_detection("laptop", 2, "right")
        
        # Test multiple detections
        detections = [
            {"label": "person", "steps": 1, "direction": "ahead"},
            {"label": "chair", "steps": 4, "direction": "left"},
            {"label": "laptop", "steps": 6, "direction": "right"}
        ]
        voice.speak_multiple_detections(detections)
        
        # Test AI response
        voice.speak_ai_response("I can see several objects around you. There's a person ahead, a chair on your left, and a laptop on your right.")
        
        print("✅ Voice output test completed!")
        
    else:
        print("❌ Voice engine not available!")
    
    # Cleanup
    voice.cleanup()
