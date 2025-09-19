#!/usr/bin/env python3
"""
Vision AI Assistant with Navigation Support
Integrates YOLOv9 detection with OpenRouteService navigation
"""

import cv2
import time
import argparse
from ultralytics import YOLO
from navigation import NavigationAssistant
from voice_output import VoiceOutput


class VisionNavigationAssistant:
    """Complete Vision AI Assistant with navigation capabilities"""
    
    def __init__(self, 
                 yolo_model_path='yolov8n.pt',
                 ors_api_key=None,
                 webcam_source=0,
                 frame_width=640,
                 frame_height=480,
                 conf_threshold=0.25):
        """
        Initialize the Vision Navigation Assistant
        
        Args:
            yolo_model_path: Path to YOLO model
            ors_api_key: OpenRouteService API key
            webcam_source: Webcam source number
            frame_width: Frame width
            frame_height: Frame height
            conf_threshold: Detection confidence threshold
        """
        self.yolo_model_path = yolo_model_path
        self.ors_api_key = ors_api_key
        self.webcam_source = webcam_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.conf_threshold = conf_threshold
        
        # Initialize components
        self.yolo_model = None
        self.cap = None
        self.voice = None
        self.navigation = None
        
        # State management
        self.running = False
        self.current_detections = []
        self.navigation_mode = False
        self.current_route = None
        self.route_step = 0
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("Initializing Vision Navigation Assistant...")
        
        # Load YOLO model
        print(f"Loading YOLO model: {self.yolo_model_path}")
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
        
        # Initialize voice output
        print("Initializing voice output...")
        try:
            self.voice = VoiceOutput(rate=150, volume=0.8)
            print("✅ Voice output initialized!")
        except Exception as e:
            print(f"❌ Error initializing voice: {e}")
            raise
        
        # Initialize navigation
        print("Initializing navigation...")
        try:
            self.navigation = NavigationAssistant(self.ors_api_key or "YOUR_ORS_API_KEY")
            print("✅ Navigation initialized!")
        except Exception as e:
            print(f"❌ Error initializing navigation: {e}")
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
        
        print("🎉 All components initialized successfully!")
    
    def process_detections(self, frame):
        """Process YOLO detections on frame"""
        try:
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'label': label,
                        'confidence': confidence
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error processing detections: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def handle_voice_command(self, command):
        """Handle voice commands"""
        command = command.lower().strip()
        
        if "what's around me" in command or "around me" in command:
            # Detection summary
            if self.current_detections:
                descriptions = [f"{det['label']} detected" for det in self.current_detections]
                summary = ". ".join(descriptions) + "."
                self.voice.speak(f"Here's what I can see: {summary}")
            else:
                self.voice.speak("I don't see any objects in the current view.")
        
        elif "navigate" in command or "directions" in command:
            # Start navigation mode
            self.voice.speak("Navigation mode activated. Please provide start and destination coordinates.")
            self.navigation_mode = True
        
        elif "stop navigation" in command or "exit navigation" in command:
            # Stop navigation mode
            self.navigation_mode = False
            self.current_route = None
            self.route_step = 0
            self.voice.speak("Navigation mode deactivated.")
        
        elif "next step" in command and self.current_route:
            # Next navigation step
            self.speak_next_navigation_step()
        
        elif "route summary" in command and self.current_route:
            # Route summary
            self.speak_route_summary()
        
        elif "exit" in command or "stop" in command:
            return "EXIT"
        
        else:
            # General response
            self.voice.speak("I can help you with object detection and navigation. Try saying 'what's around me' or 'navigate'.")
        
        return None
    
    def set_navigation_route(self, start_coords, end_coords):
        """Set navigation route"""
        try:
            route = self.navigation.get_directions(start_coords, end_coords)
            
            if "error" in route:
                self.voice.speak(f"Navigation error: {route['error']}")
                return False
            
            self.current_route = route
            self.route_step = 0
            self.navigation_mode = True
            
            # Speak route summary
            self.speak_route_summary()
            return True
            
        except Exception as e:
            self.voice.speak(f"Error setting navigation route: {e}")
            return False
    
    def speak_route_summary(self):
        """Speak route summary"""
        if not self.current_route:
            return
        
        route = self.current_route
        summary = (f"Route found. Distance: {route['distance_formatted']}, "
                  f"about {route['steps']} steps. "
                  f"Estimated time: {route['duration_formatted']}")
        
        self.voice.speak(summary)
    
    def speak_next_navigation_step(self):
        """Speak next navigation step"""
        if not self.current_route or not self.current_route.get("instructions"):
            self.voice.speak("No navigation route available.")
            return
        
        instructions = self.current_route["instructions"]
        
        if self.route_step >= len(instructions):
            self.voice.speak("You have reached your destination!")
            return
        
        instruction = instructions[self.route_step]
        step_text = f"Step {self.route_step + 1}: {instruction['instruction']}"
        
        if instruction.get("name"):
            step_text += f" on {instruction['name']}"
        
        self.voice.speak(step_text)
        self.route_step += 1
    
    def run(self):
        """Main loop for the vision navigation assistant"""
        print("\n🎥 Starting Vision Navigation Assistant")
        print("🎙️ Voice Commands:")
        print("  - 'What's around me' - Get detection summary")
        print("  - 'Navigate' - Start navigation mode")
        print("  - 'Next step' - Get next navigation instruction")
        print("  - 'Route summary' - Get route overview")
        print("  - 'Stop navigation' - Exit navigation mode")
        print("  - 'Exit' - Quit application")
        print("=" * 60)
        
        self.running = True
        frame_count = 0
        last_speak_time = 0
        speak_interval = 5.0
        
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
                self.current_detections = detections
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Add status information
                status_text = "Vision Mode"
                if self.navigation_mode:
                    status_text = f"Navigation Mode (Step {self.route_step + 1})"
                
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, status_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit, 'n' for navigation", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Vision Navigation Assistant", frame)
                
                # Auto-announce detections
                current_time = time.time()
                if detections and (current_time - last_speak_time) >= speak_interval:
                    descriptions = [f"{det['label']} detected" for det in detections]
                    summary = ". ".join(descriptions) + "."
                    self.voice.speak(f"Detected: {summary}")
                    last_speak_time = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    # Manual navigation trigger
                    self.voice.speak("Navigation mode activated. Please provide coordinates.")
                    self.navigation_mode = True
                elif key == ord('s'):
                    # Next navigation step
                    if self.navigation_mode and self.current_route:
                        self.speak_next_navigation_step()
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0 and detections:
                    print(f"Frame {frame_count}: {len(detections)} objects detected")
                    for det in detections[:3]:
                        print(f"  - {det['label']}: {det['confidence']:.2f}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        if self.voice:
            self.voice.cleanup()
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Vision Navigation Assistant')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--source', type=int, default=0, help='Webcam source')
    parser.add_argument('--ors-api-key', type=str, help='OpenRouteService API key')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    
    args = parser.parse_args()
    
    try:
        # Create and run assistant
        assistant = VisionNavigationAssistant(
            yolo_model_path=args.model,
            ors_api_key=args.ors_api_key,
            webcam_source=args.source,
            frame_width=args.width,
            frame_height=args.height,
            conf_threshold=args.conf
        )
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
