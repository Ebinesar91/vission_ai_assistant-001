#!/usr/bin/env python3
"""
YOLOv9 + Simple Distance Estimation + Relevance AI Voice Assistant
Simplified version with basic distance estimation based on bounding box size.
"""

import cv2
import numpy as np
import pyttsx3
import time
import requests
import json
from ultralytics import YOLO
from typing import List, Dict, Any
import argparse


class SimpleVisionDistanceAssistant:
    def __init__(self, 
                 yolo_model_path='yolov8n.pt',
                 webcam_source=0,
                 relevance_ai_api_key=None,
                 agent_id='737e270d-bf08-439a-b82e-f8fbc5543013',
                 project_id='f021bc31-c5e3-4c23-b437-7db1f29e9530',
                 frame_width=640,
                 frame_height=480,
                 step_length=0.75,
                 speak_interval=3.0,
                 conf_threshold=0.25):
        """
        Initialize the Simple Vision with Distance Assistant
        """
        self.yolo_model_path = yolo_model_path
        self.webcam_source = webcam_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.step_length = step_length
        self.speak_interval = speak_interval
        self.conf_threshold = conf_threshold
        
        # Relevance AI configuration
        self.relevance_ai_api_key = relevance_ai_api_key
        self.agent_id = agent_id
        self.project_id = project_id
        self.api_base_url = "https://api.relevanceai.com/v1"
        
        # Initialize components
        self.yolo_model = None
        self.cap = None
        self.tts_engine = None
        self.last_speak_time = 0
        self.running = False
        
        # Detection history
        self.detection_history = []
        self.max_history = 5
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("Initializing Simple Vision with Distance Assistant...")
        
        # Load YOLOv9 model
        print(f"Loading YOLOv9 model: {self.yolo_model_path}")
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLOv9 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLOv9 model: {e}")
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
        
        # Check Relevance AI API key
        if not self.relevance_ai_api_key:
            print("⚠️  Warning: No Relevance AI API key provided. Voice responses will be limited.")
        else:
            print("✅ Relevance AI API key configured!")
        
        print("🎉 All components initialized successfully!")
    
    def get_direction(self, x1: int, x2: int) -> str:
        """
        Determine direction based on bounding box center
        
        Args:
            x1, x2: Bounding box x coordinates
            
        Returns:
            Direction: 'left', 'center', or 'right'
        """
        center_x = (x1 + x2) // 2
        if center_x < self.frame_width // 3:
            return "left"
        elif center_x < (2 * self.frame_width) // 3:
            return "center"
        else:
            return "right"
    
    def estimate_distance_from_size(self, bbox_area: float, frame_area: float) -> float:
        """
        Estimate distance based on bounding box size relative to frame
        
        Args:
            bbox_area: Area of bounding box
            frame_area: Total frame area
            
        Returns:
            Estimated distance in meters
        """
        # Calculate area ratio
        area_ratio = bbox_area / frame_area
        
        # Estimate distance based on area ratio
        # Larger objects (higher ratio) are closer
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
        """
        Convert distance in meters to steps
        
        Args:
            distance_m: Distance in meters
            
        Returns:
            Number of steps
        """
        return max(1, round(distance_m / self.step_length))
    
    def describe_object(self, label: str, distance_m: float, direction: str) -> str:
        """
        Create natural language description of detected object
        
        Args:
            label: Object label
            distance_m: Distance in meters
            direction: Direction (left/center/right)
            
        Returns:
            Formatted description string
        """
        steps = self.distance_to_steps(distance_m)
        
        if direction == "center":
            direction_desc = "ahead"
        elif direction == "left":
            direction_desc = "on your left"
        else:  # right
            direction_desc = "on your right"
        
        return f"{label} detected, about {steps} steps {direction_desc}"
    
    def send_to_relevance_ai(self, detection_text: str) -> str:
        """
        Send detection data to Relevance AI agent
        
        Args:
            detection_text: Formatted detection description
            
        Returns:
            AI response text
        """
        if not self.relevance_ai_api_key:
            return detection_text
        
        try:
            input_data = {
                "camera_feed": detection_text,
                "audio_input": "What do you see around me?",
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
                return result.get('output', detection_text)
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return detection_text
                
        except Exception as e:
            print(f"Error calling Relevance AI: {e}")
            return detection_text
    
    def speak_response(self, text: str):
        """
        Use pyttsx3 to speak the response
        
        Args:
            text: Text to speak
        """
        try:
            print(f"🎤 Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def should_speak(self) -> bool:
        """Check if enough time has passed to speak again"""
        current_time = time.time()
        return (current_time - self.last_speak_time) >= self.speak_interval
    
    def process_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process YOLOv9 detections with simple distance estimation
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries with distance and direction
        """
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
                    
                    # Estimate distance based on size
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
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
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
    
    def run(self):
        """Main loop for vision with distance estimation"""
        print("\n🎥 Starting Simple Vision with Distance Assistant")
        print("Press 'q' to quit, 's' to speak immediately")
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
                
                # Process detections
                detections = self.process_detections(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Simple Vision AI - Distance & Direction", annotated_frame)
                
                # Process detections and speak if needed
                if detections and self.should_speak():
                    # Create combined description
                    descriptions = [det['description'] for det in detections]
                    combined_text = ". ".join(descriptions) + "."
                    
                    # Send to Relevance AI
                    ai_response = self.send_to_relevance_ai(combined_text)
                    
                    # Speak response
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
                        descriptions = [det['description'] for det in detections]
                        combined_text = ". ".join(descriptions) + "."
                        ai_response = self.send_to_relevance_ai(combined_text)
                        self.speak_response(ai_response)
                        self.last_speak_time = time.time()
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0 and detections:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for det in detections[:3]:  # Show only top 3
                        print(f"  - {det['label']}: {det['distance_steps']} steps {det['direction']} (conf: {det['confidence']:.2f}, area: {det['area_ratio']:.3f})")
        
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
    parser = argparse.ArgumentParser(description='Simple Vision with Distance Assistant')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv9 model path')
    parser.add_argument('--source', type=int, default=0, help='Webcam source')
    parser.add_argument('--api-key', type=str, help='Relevance AI API key')
    parser.add_argument('--agent-id', type=str, default='737e270d-bf08-439a-b82e-f8fbc5543013', help='Relevance AI agent ID')
    parser.add_argument('--project-id', type=str, default='f021bc31-c5e3-4c23-b437-7db1f29e9530', help='Relevance AI project ID')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--speak-interval', type=float, default=3.0, help='Speaking interval in seconds')
    parser.add_argument('--step-length', type=float, default=0.75, help='Step length in meters')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    
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
        assistant = SimpleVisionDistanceAssistant(
            yolo_model_path=args.model,
            webcam_source=args.source,
            relevance_ai_api_key=api_key,
            agent_id=args.agent_id,
            project_id=args.project_id,
            frame_width=args.width,
            frame_height=args.height,
            step_length=args.step_length,
            speak_interval=args.speak_interval,
            conf_threshold=args.conf
        )
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
