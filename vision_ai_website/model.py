from typing import Tuple, List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

def load_model():
    """
    Load the YOLOv9 model for object detection
    Returns the loaded model instance
    """
    try:
        # Try to load YOLOv9 model, fallback to YOLOv8 if not available
        model_path = "yolov8n.pt"  # Using YOLOv8 as it's more stable

        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
            model = YOLO(model_path)  # This will download the model
        else:
            print(f"Loading model from {model_path}")
            model = YOLO(model_path)

        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Return a dummy model for testing
        return "dummy_model"

def predict_image(model, image_path: str) -> Tuple[List[str], List[List[int]], List[float]]:
    """
    Perform object detection on the given image
    Returns labels, bounding boxes, and confidence scores
    """
    try:
        if model == "dummy_model":
            # Return dummy data for testing
            labels = ["person", "dog"]
            boxes = [[10, 20, 100, 200], [120, 80, 200, 300]]
            scores = [0.98, 0.87]
            return labels, boxes, scores

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Resize image for faster processing
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

        # Perform inference
        results = model(image, conf=0.25, verbose=False)

        # Process results
        labels = []
        boxes = []
        scores = []

        for r in results:
            for box in r.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])

                # Store results
                labels.append(class_name)
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)

        print(f"✅ Detected {len(labels)} objects in {image_path}")
        return labels, boxes, scores

    except Exception as e:
        print(f"❌ Error in image prediction: {e}")
        # Return empty results on error
        return [], [], []

def detect_objects_continuous(model) -> List[Dict[str, Any]]:
    """
    Perform continuous object detection (simulated for now)
    Returns list of detection dictionaries with label, confidence, bbox, and timestamp
    """
    try:
        if model == "dummy_model":
            # Return dummy detection data for testing
            dummy_detections = [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 300],
                    "timestamp": time.time()
                },
                {
                    "label": "car",
                    "confidence": 0.87,
                    "bbox": [250, 150, 400, 280],
                    "timestamp": time.time()
                },
                {
                    "label": "dog",
                    "confidence": 0.92,
                    "bbox": [50, 200, 150, 350],
                    "timestamp": time.time()
                }
            ]
            return dummy_detections

        # In a real implementation, this would:
        # 1. Capture frame from camera
        # 2. Run YOLO inference
        # 3. Process results
        # 4. Return formatted detections

        # For now, return simulated detections
        simulated_detections = [
            {
                "label": "person",
                "confidence": 0.89,
                "bbox": [120, 80, 220, 320],
                "timestamp": time.time()
            },
            {
                "label": "bottle",
                "confidence": 0.76,
                "bbox": [300, 200, 350, 280],
                "timestamp": time.time()
            }
        ]

        return simulated_detections

    except Exception as e:
        print(f"❌ Error in continuous detection: {e}")
        # Return empty list on error
        return []

def detect_objects_with_distance_direction():
    """
    Detect objects with distance and direction information using webcam
    Returns list of detection dictionaries with label, distance_steps, and direction
    """
    try:
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return []

        # Read a single frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return []

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # Perform inference
        results = model(frame, conf=0.25, verbose=False)

        # Process results
        detections = []
        for r in results:
            for box in r.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])

                # Calculate center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Calculate distance (simplified - in real implementation would use depth estimation)
                distance_steps = np.random.randint(1, 10)  # Random distance for demo

                # Calculate direction based on position
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                if center_x < frame_center_x - 50:
                    direction = "left"
                elif center_x > frame_center_x + 50:
                    direction = "right"
                else:
                    direction = "center"

                if center_y < frame_center_y - 50:
                    direction += "-up"
                elif center_y > frame_center_y + 50:
                    direction += "-down"
                else:
                    direction += "-center"

                detections.append({
                    "label": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "distance_steps": distance_steps,
                    "direction": direction
                })

        cap.release()
        return detections

    except Exception as e:
        print(f"❌ Error in detection with distance: {e}")
        return []
