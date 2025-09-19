#!/usr/bin/env python3
"""
Example: Integrating voice_output.py with Vision AI
Shows how to use the voice output module in your detection pipeline
"""

import cv2
import time
from ultralytics import YOLO
from voice_output import VoiceOutput, speak, speak_detection, speak_ai_response


def example_vision_with_voice():
    """Example of Vision AI with voice output integration"""
    
    print("Vision AI with Voice Output - Example")
    print("=" * 50)
    
    # Initialize voice output
    voice = VoiceOutput(rate=150, volume=0.8, enable_console_output=True)
    
    if not voice.is_available():
        print("❌ Voice engine not available!")
        return
    
    print("✅ Voice engine ready!")
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded!")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return
    
    print("✅ Webcam opened!")
    print("Press 'q' to quit, 's' to speak detection summary")
    
    frame_count = 0
    last_speak_time = 0
    speak_interval = 5.0  # Speak every 5 seconds
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Run detection
            results = model(frame, conf=0.25, verbose=False)
            
            # Process detections
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = r.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    # Calculate distance (simplified heuristic)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = bbox_area / frame_area
                    
                    # Estimate distance based on size
                    if area_ratio > 0.15:
                        steps = 1
                    elif area_ratio > 0.08:
                        steps = 2
                    elif area_ratio > 0.04:
                        steps = 3
                    elif area_ratio > 0.02:
                        steps = 5
                    else:
                        steps = 7
                    
                    # Determine direction
                    center_x = (x1 + x2) // 2
                    if center_x < 640 // 3:
                        direction = "left"
                    elif center_x < (2 * 640) // 3:
                        direction = "center"
                    else:
                        direction = "right"
                    
                    # Store detection
                    detection = {
                        'label': label,
                        'steps': steps,
                        'direction': direction,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {steps} steps {direction}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Auto-announce detections
            current_time = time.time()
            if detections and (current_time - last_speak_time) >= speak_interval:
                voice.speak_multiple_detections(detections)
                last_speak_time = current_time
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to speak, 'q' to quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Vision AI with Voice", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manual detection summary
                if detections:
                    voice.speak_multiple_detections(detections)
                else:
                    voice.speak("No objects detected in current view.")
            
            frame_count += 1
            
            # Print detection info every 30 frames
            if frame_count % 30 == 0 and detections:
                print(f"Frame {frame_count}: {len(detections)} objects detected")
                for det in detections[:3]:  # Show only top 3
                    print(f"  - {det['label']}: {det['steps']} steps {det['direction']} (conf: {det['confidence']:.2f})")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        voice.cleanup()
        print("✅ Cleanup completed")


def example_voice_commands():
    """Example of voice command processing"""
    
    print("\nVoice Command Processing - Example")
    print("=" * 40)
    
    # Initialize voice
    voice = VoiceOutput()
    
    if not voice.is_available():
        print("❌ Voice engine not available!")
        return
    
    # Simulate different types of responses
    print("Testing different voice responses...")
    
    # Detection response
    voice.speak_detection("person", 3, "ahead")
    time.sleep(1)
    
    # Multiple detections
    detections = [
        {"label": "person", "steps": 1, "direction": "ahead"},
        {"label": "chair", "steps": 4, "direction": "left"},
        {"label": "laptop", "steps": 6, "direction": "right"}
    ]
    voice.speak_multiple_detections(detections)
    time.sleep(1)
    
    # AI response
    voice.speak_ai_response("I can see several objects around you. There's a person ahead, a chair on your left, and a laptop on your right.")
    time.sleep(1)
    
    # Status message
    voice.speak_status("Vision AI assistant is ready and monitoring your surroundings.")
    time.sleep(1)
    
    # Error message
    voice.speak_error("Camera connection lost. Please check your webcam.")
    
    print("✅ Voice command examples completed!")


if __name__ == "__main__":
    print("Voice Output Module - Integration Examples")
    print("=" * 60)
    
    # Test basic voice functionality
    example_voice_commands()
    
    # Ask user if they want to run the full vision example
    print("\n" + "=" * 60)
    response = input("Run full Vision AI with voice example? (y/n): ").lower().strip()
    
    if response == 'y':
        example_vision_with_voice()
    else:
        print("Voice output module ready for integration!")
        print("\nTo use in your project:")
        print("1. Import: from voice_output import speak, speak_detection")
        print("2. Call: speak('Your message here')")
        print("3. Call: speak_detection('person', 3, 'ahead')")
