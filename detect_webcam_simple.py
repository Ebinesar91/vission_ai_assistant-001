#!/usr/bin/env python3
"""
YOLOv9 Webcam Detection Script - Simple Version
A lightweight script for real-time object detection using webcam feed.
Uses ultralytics library for easier setup.
"""

import cv2
import argparse
from ultralytics import YOLO


def run_webcam_detection(
        weights='yolov8n.pt',  # model path
        source=0,  # webcam source (0 for default camera)
        conf_thres=0.25,  # confidence threshold
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
):
    """
    Run YOLOv9 object detection on webcam feed using ultralytics.
    """
    print(f"Loading YOLOv9 model: {weights}")
    print("This may take a moment on first run as the model downloads...")
    
    try:
        # Load YOLOv9 model (will download automatically if not found)
        model = YOLO(weights)
        print(f"Model loaded successfully!")
        print(f"Model info: {model.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your internet connection and try again.")
        return

    # Open webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {source}")
        return

    print(f"Webcam opened successfully!")
    print(f"Press 'q' to quit")
    print(f"Confidence threshold: {conf_thres}")

    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break

            frame_count += 1
            
            # Run YOLOv9 detection
            results = model(frame, conf=conf_thres, verbose=False)
            
            # Get the first result
            result = results[0]
            
            # Draw bounding boxes and labels
            annotated_frame = result.plot(
                line_width=line_thickness,
                labels=not hide_labels,
                conf=not hide_conf
            )
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("YOLOv9 Webcam Detection", annotated_frame)
            
            # Print detection info every 30 frames
            if frame_count % 30 == 0 and len(result.boxes) > 0:
                detections = []
                for box in result.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = model.names[cls]
                    detections.append(f"{label}: {conf:.2f}")
                print(f"Frame {frame_count}: {', '.join(detections)}")
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"Detection completed. Processed {frame_count} frames.")


def main():
    """Main function to run webcam detection."""
    parser = argparse.ArgumentParser(description='YOLOv9 Webcam Detection - Simple Version')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--source', type=int, default=0, help='webcam source (0 for default camera)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    opt = parser.parse_args()
    
    print(f"YOLOv9 Webcam Detection - Simple Version")
    print(f"Arguments: {vars(opt)}")
    
    run_webcam_detection(**vars(opt))


if __name__ == "__main__":
    main()
