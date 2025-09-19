#!/usr/bin/env python3
"""
YOLOv9 + Relevance AI Voice Assistant Integration
Combines real-time object detection with voice assistant responses.
"""

import cv2
import json
import time
import base64
import requests
import argparse
from ultralytics import YOLO
import pyttsx3
import threading
from typing import List, Dict, Any
import io
from PIL import Image


class YOLOv9RelevanceAssistant:
    def __init__(self, 
                 model_path='yolov8n.pt',
                 webcam_source=0,
                 relevance_ai_api_key=None,
                 agent_id='737e270d-bf08-439a-b82e-f8fbc5543013',
                 project_id='f021bc31-c5e3-4c23-b437-7db1f29e9530',
                 conf_threshold=0.25,
                 speak_interval=3.0):
        """
        Initialize the YOLOv9 + Relevance AI Voice Assistant
        
        Args:
            model_path: Path to YOLOv9 model file
            webcam_source: Webcam source (0 for default camera)
            relevance_ai_api_key: Your Relevance AI API key
            agent_id: Your Relevance AI agent ID
            project_id: Your Relevance AI project ID
            conf_threshold: Confidence threshold for detections
            speak_interval: Time interval between voice responses (seconds)
        """
        self.model_path = model_path
        self.webcam_source = webcam_source
        self.conf_threshold = conf_threshold
        self.speak_interval = speak_interval
        
        # Relevance AI configuration
        self.relevance_ai_api_key = relevance_ai_api_key
        self.agent_id = agent_id
        self.project_id = project_id
        self.api_base_url = "https://api.relevanceai.com/v1"
        
        # Initialize components
        self.model = None
        self.cap = None
        self.tts_engine = None
        self.last_speak_time = 0
        self.running = False
        
        # Detection history for context
        self.detection_history = []
        self.max_history = 10
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("Initializing YOLOv9 + Relevance AI Voice Assistant...")
        
        # Load YOLOv9 model
        print(f"Loading YOLOv9 model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("✅ YOLOv9 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLOv9 model: {e}")
            raise
        
        # Initialize webcam
        print(f"Initializing webcam source: {self.webcam_source}")
        self.cap = cv2.VideoCapture(self.webcam_source)
        if not self.cap.isOpened():
            raise Exception(f"Could not open webcam {self.webcam_source}")
        print("✅ Webcam initialized successfully!")
        
        # Initialize text-to-speech
        print("Initializing text-to-speech engine...")
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            print("✅ Text-to-speech engine initialized!")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            raise
        
        # Check Relevance AI API key
        if not self.relevance_ai_api_key:
            print("⚠️  Warning: No Relevance AI API key provided. Voice responses will be limited.")
        else:
            print("✅ Relevance AI API key configured!")
        
        print("🎉 All components initialized successfully!")
    
    def get_detections(self, frame) -> List[Dict[str, Any]]:
        """
        Extract detected objects from frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            List of detection dictionaries with label, confidence, and bounding box
        """
        try:
            # Run YOLOv9 detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            result = results[0]
            
            detections = []
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract detection data
                    conf = float(box.conf.item())
                    cls = int(box.cls.item())
                    label = self.model.names[cls]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate center position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Determine position relative to frame center
                    frame_center_x = frame.shape[1] / 2
                    frame_center_y = frame.shape[0] / 2
                    
                    # Calculate relative position
                    rel_x = (center_x - frame_center_x) / frame_center_x
                    rel_y = (center_y - frame_center_y) / frame_center_y
                    
                    # Determine direction
                    if abs(rel_x) < 0.3:  # Center
                        direction = "center"
                    elif rel_x > 0:  # Right side
                        direction = "right"
                    else:  # Left side
                        direction = "left"
                    
                    # Determine distance based on bounding box size
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = bbox_area / frame_area
                    
                    if area_ratio > 0.1:
                        distance = "close"
                    elif area_ratio > 0.05:
                        distance = "medium"
                    else:
                        distance = "far"
                    
                    detection = {
                        'label': label,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'direction': direction,
                        'distance': distance,
                        'area_ratio': area_ratio
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def format_detections_for_ai(self, detections: List[Dict[str, Any]]) -> str:
        """
        Format detections as structured text for Relevance AI
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Formatted string describing detected objects
        """
        if not detections:
            return "No objects detected in the current view."
        
        descriptions = []
        for det in detections:
            label = det['label']
            conf = det['confidence']
            direction = det['direction']
            distance = det['distance']
            
            # Create natural language description
            desc = f"{label} detected with {conf:.2f} confidence {distance} {direction}"
            descriptions.append(desc)
        
        return ". ".join(descriptions) + "."
    
    def send_to_relevance_ai(self, detection_text: str, frame=None) -> str:
        """
        Send detection data to Relevance AI agent
        
        Args:
            detection_text: Formatted detection description
            frame: Optional frame for image analysis
            
        Returns:
            AI response text
        """
        if not self.relevance_ai_api_key:
            # Fallback response without API
            return f"I can see: {detection_text}"
        
        try:
            # Prepare input data
            input_data = {
                "camera_feed": detection_text,
                "audio_input": "What do you see around me?",
                "current_location": {
                    "latitude": 0.0,  # You can add GPS coordinates here
                    "longitude": 0.0,
                    "accuracy": 0.0
                },
                "device_orientation": {
                    "heading": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }
            }
            
            # If frame is provided, encode it as base64
            if frame is not None:
                try:
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    input_data["camera_feed"] = frame_base64
                except Exception as e:
                    print(f"Warning: Could not encode frame: {e}")
            
            # Prepare API request
            url = f"{self.api_base_url}/agents/{self.agent_id}/run"
            headers = {
                "Authorization": f"Bearer {self.relevance_ai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": input_data,
                "project_id": self.project_id
            }
            
            # Make API request
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('output', 'No response from AI agent')
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return f"I can see: {detection_text}"
                
        except Exception as e:
            print(f"Error calling Relevance AI: {e}")
            return f"I can see: {detection_text}"
    
    def speak_response(self, text: str):
        """
        Use pyttsx3 to speak the response
        
        Args:
            text: Text to speak
        """
        try:
            # Create a new TTS engine instance for each speech to avoid threading issues
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.8)
            
            # Run TTS in a separate thread to avoid blocking
            def speak():
                tts_engine.say(text)
                tts_engine.runAndWait()
                tts_engine.stop()
            
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def should_speak(self) -> bool:
        """Check if enough time has passed to speak again"""
        current_time = time.time()
        return (current_time - self.last_speak_time) >= self.speak_interval
    
    def run(self):
        """Main loop for webcam detection and voice assistance"""
        print("\n🎥 Starting YOLOv9 + Relevance AI Voice Assistant")
        print("Press 'q' to quit, 's' to speak immediately")
        print("=" * 50)
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                
                frame_count += 1
                
                # Get detections
                detections = self.get_detections(frame)
                
                # Draw bounding boxes on frame
                annotated_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = det['label']
                    conf = det['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{label}: {conf:.2f}"
                    cv2.putText(annotated_frame, label_text, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("YOLOv9 + Relevance AI Assistant", annotated_frame)
                
                # Process detections and speak if needed
                if detections and self.should_speak():
                    # Format detections
                    detection_text = self.format_detections_for_ai(detections)
                    
                    # Send to Relevance AI
                    ai_response = self.send_to_relevance_ai(detection_text, frame)
                    
                    # Speak response
                    print(f"🎤 Speaking: {ai_response}")
                    self.speak_response(ai_response)
                    
                    # Update last speak time
                    self.last_speak_time = time.time()
                    
                    # Store in history
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'detections': detections,
                        'ai_response': ai_response
                    })
                    
                    # Keep only recent history
                    if len(self.detection_history) > self.max_history:
                        self.detection_history.pop(0)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting...")
                    break
                elif key == ord('s'):
                    # Force immediate speech
                    if detections:
                        detection_text = self.format_detections_for_ai(detections)
                        ai_response = self.send_to_relevance_ai(detection_text, frame)
                        print(f"🎤 Speaking (manual): {ai_response}")
                        self.speak_response(ai_response)
                        self.last_speak_time = time.time()
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0 and detections:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for det in detections:
                        print(f"  - {det['label']}: {det['confidence']:.2f} ({det['distance']} {det['direction']})")
        
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
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv9 + Relevance AI Voice Assistant')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv9 model path')
    parser.add_argument('--source', type=int, default=0, help='Webcam source')
    parser.add_argument('--api-key', type=str, help='Relevance AI API key')
    parser.add_argument('--agent-id', type=str, default='737e270d-bf08-439a-b82e-f8fbc5543013', help='Relevance AI agent ID')
    parser.add_argument('--project-id', type=str, default='f021bc31-c5e3-4c23-b437-7db1f29e9530', help='Relevance AI project ID')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--speak-interval', type=float, default=3.0, help='Speaking interval in seconds')
    
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
        assistant = YOLOv9RelevanceAssistant(
            model_path=args.model,
            webcam_source=args.source,
            relevance_ai_api_key=api_key,
            agent_id=args.agent_id,
            project_id=args.project_id,
            conf_threshold=args.conf,
            speak_interval=args.speak_interval
        )
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
