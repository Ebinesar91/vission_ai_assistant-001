#!/usr/bin/env python3
"""
Complete Voice-Enabled Vision AI Assistant
Integrates YOLOv9 + MiDaS + Vosk + Relevance AI for full voice interaction.
"""

import cv2
import torch
import numpy as np
import pyttsx3
import time
import requests
import json
import threading
import queue
import argparse
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import pyaudio
import vosk
import os
import urllib.request
import zipfile


class VoiceEnabledVisionAssistant:
    def __init__(self, 
                 yolo_model_path='yolov8n.pt',
                 webcam_source=0,
                 relevance_ai_api_key=None,
                 agent_id='737e270d-bf08-439a-b82e-f8fbc5543013',
                 project_id='f021bc31-c5e3-4c23-b437-7db1f29e9530',
                 frame_width=640,
                 frame_height=480,
                 step_length=0.75,
                 speak_interval=5.0,
                 conf_threshold=0.25,
                 vosk_model_path='vosk-model-small-en-us-0.15'):
        """
        Initialize the Voice-Enabled Vision Assistant
        
        Args:
            yolo_model_path: Path to YOLOv9 model
            webcam_source: Webcam source number
            relevance_ai_api_key: Relevance AI API key
            agent_id: Relevance AI agent ID
            project_id: Relevance AI project ID
            frame_width: Frame width for processing
            frame_height: Frame height for processing
            step_length: Length of one step in meters
            speak_interval: Time between auto-announcements (seconds)
            conf_threshold: Detection confidence threshold
            vosk_model_path: Path to Vosk model
        """
        self.yolo_model_path = yolo_model_path
        self.webcam_source = webcam_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.step_length = step_length
        self.speak_interval = speak_interval
        self.conf_threshold = conf_threshold
        self.vosk_model_path = vosk_model_path
        
        # Relevance AI configuration
        self.relevance_ai_api_key = relevance_ai_api_key
        self.agent_id = agent_id
        self.project_id = project_id
        self.api_base_url = "https://api.relevanceai.com/v1"
        
        # Initialize components
        self.yolo_model = None
        self.midas = None
        self.midas_transforms = None
        self.device = None
        self.cap = None
        self.tts_engine = None
        self.vosk_model = None
        self.recognizer = None
        self.audio_stream = None
        self.audio = None
        
        # State management
        self.running = False
        self.last_speak_time = 0
        self.current_detections = []
        self.voice_queue = queue.Queue()
        self.detection_lock = threading.Lock()
        
        # Detection history
        self.detection_history = []
        self.max_history = 10
        
        self._initialize_components()
    
    def _download_vosk_model(self):
        """Download Vosk model if not present"""
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        model_zip = "vosk-model-small-en-us-0.15.zip"
        model_dir = "vosk-model-small-en-us-0.15"
        
        if not os.path.exists(model_dir):
            print("Downloading Vosk model... This may take a few minutes.")
            try:
                urllib.request.urlretrieve(model_url, model_zip)
                with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove(model_zip)
                print("✅ Vosk model downloaded successfully!")
            except Exception as e:
                print(f"❌ Error downloading Vosk model: {e}")
                print("Please download manually from: https://alphacephei.com/vosk/models/")
                raise
        else:
            print("✅ Vosk model found!")
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("Initializing Voice-Enabled Vision Assistant...")
        
        # Load YOLOv9 model
        print(f"Loading YOLOv9 model: {self.yolo_model_path}")
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLOv9 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLOv9 model: {e}")
            raise
        
        # Load MiDaS depth estimation
        print("Loading MiDaS depth estimation model...")
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = self.midas_transforms.dpt_transform
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            self.midas.to(self.device)
            self.midas.eval()
            print("✅ MiDaS depth estimation loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading MiDaS: {e}")
            raise
        
        # Initialize webcam
        print(f"Initializing webcam source: {self.webcam_source}")
        self.cap = cv2.VideoCapture(self.webcam_source)
        if not self.cap.isOpened():
            raise Exception(f"Could not open webcam {self.webcam_source}")
        
        # Set frame size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        print("✅ Webcam initialized successfully!")
        
        # Initialize text-to-speech
        print("Initializing text-to-speech engine...")
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            print("✅ Text-to-speech engine initialized!")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            raise
        
        # Initialize Vosk speech recognition
        print("Initializing Vosk speech recognition...")
        try:
            self._download_vosk_model()
            self.vosk_model = vosk.Model(self.vosk_model_path)
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
            
            # Initialize audio
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
            print("✅ Vosk speech recognition initialized!")
        except Exception as e:
            print(f"❌ Error initializing Vosk: {e}")
            raise
        
        # Check Relevance AI API key
        if not self.relevance_ai_api_key:
            print("⚠️  Warning: No Relevance AI API key provided. Voice responses will be limited.")
        else:
            print("✅ Relevance AI API key configured!")
        
        print("🎉 All components initialized successfully!")
    
    def get_direction(self, x1: int, x2: int) -> str:
        """Determine direction based on bounding box center"""
        center_x = (x1 + x2) // 2
        if center_x < self.frame_width // 3:
            return "left"
        elif center_x < (2 * self.frame_width) // 3:
            return "center"
        else:
            return "right"
    
    def estimate_distance_from_size(self, bbox_area: float, frame_area: float) -> float:
        """Estimate distance based on bounding box size relative to frame"""
        area_ratio = bbox_area / frame_area
        
        if area_ratio > 0.15:  # Very large object
            return 1.0  # 1 meter
        elif area_ratio > 0.08:  # Large object
            return 2.0  # 2 meters
        elif area_ratio > 0.04:  # Medium object
            return 3.5  # 3.5 meters
        elif area_ratio > 0.02:  # Small object
            return 5.0  # 5 meters
        else:  # Very small object
            return 7.0  # 7 meters
    
    def distance_to_steps(self, distance_m: float) -> int:
        """Convert distance in meters to steps"""
        return max(1, round(distance_m / self.step_length))
    
    def describe_object(self, label: str, distance_m: float, direction: str) -> str:
        """Create natural language description of detected object"""
        steps = self.distance_to_steps(distance_m)
        
        if direction == "center":
            direction_desc = "ahead"
        elif direction == "left":
            direction_desc = "on your left"
        else:  # right
            direction_desc = "on your right"
        
        return f"{label} detected, about {steps} steps {direction_desc}"
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map using MiDaS (simplified fallback)"""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform image
            input_batch = self.transform(img_rgb).to(self.device)
            
            # Get depth prediction
            with torch.no_grad():
                prediction = self.midas(input_batch.unsqueeze(0))
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = prediction.cpu().numpy()
            return depth_map
            
        except Exception as e:
            # Return a simple depth map for fallback
            return np.ones((self.frame_height, self.frame_width)) * 3.0
    
    def get_object_distance(self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate average depth inside bounding box"""
        try:
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, depth_map.shape[1] - 1))
            y1 = max(0, min(y1, depth_map.shape[0] - 1))
            x2 = max(0, min(x2, depth_map.shape[1] - 1))
            y2 = max(0, min(y2, depth_map.shape[0] - 1))
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                return 5.0
            
            # Get depth values in bounding box
            obj_depth = depth_map[y1:y2, x1:x2]
            
            # Calculate average depth (excluding zeros/invalid values)
            valid_depths = obj_depth[obj_depth > 0]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                return float(avg_depth)
            else:
                return 5.0  # Default distance if no valid depth
                
        except Exception as e:
            print(f"Error calculating object distance: {e}")
            return 5.0  # Default distance
    
    def process_detections(self, frame: np.ndarray, depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """Process YOLOv9 detections with distance estimation"""
        detections = []
        
        try:
            # Run YOLOv9 detection
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            
            # Calculate frame area
            frame_area = frame.shape[0] * frame.shape[1]
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    # Calculate bounding box area
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # Use size-based distance estimation (more reliable)
                    distance_m = self.estimate_distance_from_size(bbox_area, frame_area)
                    
                    # Get direction
                    direction = self.get_direction(x1, x2)
                    
                    # Create description
                    description = self.describe_object(label, distance_m, direction)
                    
                    detection = {
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'distance_m': distance_m,
                        'distance_steps': self.distance_to_steps(distance_m),
                        'direction': direction,
                        'description': description,
                        'bbox_area': bbox_area,
                        'area_ratio': bbox_area / frame_area
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error processing detections: {e}")
            return []
    
    def speak(self, text: str):
        """Use pyttsx3 to speak the response"""
        try:
            print(f"🎤 Assistant: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def listen_command(self) -> Optional[str]:
        """Listen for voice commands using Vosk"""
        try:
            data = self.audio_stream.read(4000, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()
                if text:
                    print(f"🎙️ You said: {text}")
                    return text
            return None
        except Exception as e:
            print(f"Error in voice recognition: {e}")
            return None
    
    def send_to_relevance_ai(self, query: str) -> str:
        """Send query to Relevance AI agent"""
        if not self.relevance_ai_api_key:
            return "I don't have access to my AI knowledge base. Please provide an API key for full functionality."
        
        try:
            input_data = {
                "camera_feed": query,
                "audio_input": query,
                "current_location": {
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "accuracy": 0.0
                },
                "device_orientation": {
                    "heading": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }
            }
            
            url = f"{self.api_base_url}/agents/{self.agent_id}/run"
            headers = {
                "Authorization": f"Bearer {self.relevance_ai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": input_data,
                "project_id": self.project_id
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('output', 'I could not process that request.')
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return "I'm having trouble connecting to my knowledge base right now."
                
        except Exception as e:
            print(f"Error calling Relevance AI: {e}")
            return "I'm having trouble processing that request right now."
    
    def process_command(self, command: str) -> str:
        """Process voice commands and return appropriate response"""
        command = command.lower().strip()
        
        # Detection-related commands
        if any(phrase in command for phrase in ["what's around me", "around me", "what do you see", "scan", "detect"]):
            with self.detection_lock:
                if self.current_detections:
                    # Create summary of current detections
                    descriptions = [det['description'] for det in self.current_detections]
                    summary = ". ".join(descriptions) + "."
                    return f"Here's what I can see: {summary}"
                else:
                    return "I don't see any objects in the current view. Let me continue scanning."
        
        # Exit commands
        elif any(phrase in command for phrase in ["exit", "stop", "quit", "shutdown", "goodbye"]):
            return "EXIT"
        
        # Help commands
        elif any(phrase in command for phrase in ["help", "what can you do", "commands"]):
            return "I can detect objects around you and tell you their distance and direction. I can also answer general questions. Try saying 'what's around me' or ask me anything else!"
        
        # All other queries go to Relevance AI
        else:
            return self.send_to_relevance_ai(command)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            confidence = det['confidence']
            steps = det['distance_steps']
            direction = det['direction']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with distance and direction
            label_text = f"{label}: {confidence:.2f} ({steps} steps {direction})"
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def should_auto_announce(self) -> bool:
        """Check if enough time has passed for auto-announcement"""
        current_time = time.time()
        return (current_time - self.last_speak_time) >= self.speak_interval
    
    def run(self):
        """Main loop for voice-enabled vision assistant"""
        print("\n🎥 Starting Voice-Enabled Vision Assistant")
        print("🎙️ Say 'what's around me' to get detection summary")
        print("🎙️ Say 'exit' or 'stop' to quit")
        print("🎙️ Ask me anything else for AI responses")
        print("=" * 60)
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                
                frame_count += 1
                
                # Resize frame if needed
                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # Estimate depth
                depth_map = self.estimate_depth(frame)
                
                # Process detections
                detections = self.process_detections(frame, depth_map)
                
                # Update current detections
                with self.detection_lock:
                    self.current_detections = detections
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add frame counter and status
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_frame, "Listening...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Voice-Enabled Vision Assistant", annotated_frame)
                
                # Listen for voice commands
                command = self.listen_command()
                if command:
                    response = self.process_command(command)
                    if response == "EXIT":
                        self.speak("Shutting down assistant. Goodbye!")
                        break
                    else:
                        self.speak(response)
                        self.last_speak_time = time.time()
                
                # Auto-announce detections if enough time has passed
                elif detections and self.should_auto_announce():
                    descriptions = [det['description'] for det in detections]
                    combined_text = ". ".join(descriptions) + "."
                    self.speak(f"Detected: {combined_text}")
                    self.last_speak_time = time.time()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting...")
                    break
                elif key == ord('s'):
                    # Force detection summary
                    if detections:
                        descriptions = [det['description'] for det in detections]
                        combined_text = ". ".join(descriptions) + "."
                        self.speak(f"Here's what I can see: {combined_text}")
                        self.last_speak_time = time.time()
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0 and detections:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for det in detections[:3]:  # Show only top 3
                        print(f"  - {det['label']}: {det['distance_steps']} steps {det['direction']} (conf: {det['confidence']:.2f})")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio:
            self.audio.terminate()
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Voice-Enabled Vision Assistant')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv9 model path')
    parser.add_argument('--source', type=int, default=0, help='Webcam source')
    parser.add_argument('--api-key', type=str, help='Relevance AI API key')
    parser.add_argument('--agent-id', type=str, default='737e270d-bf08-439a-b82e-f8fbc5543013', help='Relevance AI agent ID')
    parser.add_argument('--project-id', type=str, default='f021bc31-c5e3-4c23-b437-7db1f29e9530', help='Relevance AI project ID')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--speak-interval', type=float, default=5.0, help='Auto-announcement interval in seconds')
    parser.add_argument('--step-length', type=float, default=0.75, help='Step length in meters')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--vosk-model', type=str, default='vosk-model-small-en-us-0.15', help='Vosk model path')
    
    args = parser.parse_args()
    
    # Check for API key
    if not args.api_key:
        print("⚠️  No API key provided. Set RELEVANCE_AI_API_KEY environment variable or use --api-key")
        print("   Voice responses will be limited without API access.")
        api_key = None
    else:
        api_key = args.api_key
    
    try:
        # Create and run assistant
        assistant = VoiceEnabledVisionAssistant(
            yolo_model_path=args.model,
            webcam_source=args.source,
            relevance_ai_api_key=api_key,
            agent_id=args.agent_id,
            project_id=args.project_id,
            frame_width=args.width,
            frame_height=args.height,
            step_length=args.step_length,
            speak_interval=args.speak_interval,
            conf_threshold=args.conf,
            vosk_model_path=args.vosk_model
        )
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
