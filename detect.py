import cv2
import numpy as np
import random
from ultralytics import YOLO
import os


class ColorMapper:
    """Manages color assignment for different object classes"""
    
    def __init__(self):
        # Predefined colors for common COCO classes
        self.class_colors = {
            'person': (0, 255, 0),        # Green
            'bicycle': (255, 0, 0),       # Blue
            'car': (0, 0, 255),           # Red
            'motorcycle': (255, 255, 0),  # Cyan
            'airplane': (255, 0, 255),    # Magenta
            'bus': (0, 255, 255),         # Yellow
            'train': (128, 0, 128),       # Purple
            'truck': (255, 165, 0),       # Orange
            'boat': (0, 128, 255),        # Light Blue
            'traffic light': (128, 255, 0), # Lime
            'fire hydrant': (255, 192, 203), # Pink
            'stop sign': (255, 20, 147),  # Deep Pink
            'parking meter': (0, 191, 255), # Deep Sky Blue
            'bench': (255, 69, 0),        # Red Orange
            'bird': (50, 205, 50),        # Lime Green
            'cat': (255, 105, 180),       # Hot Pink
            'dog': (255, 140, 0),         # Dark Orange
            'horse': (139, 69, 19),       # Saddle Brown
            'sheep': (255, 250, 240),     # Floral White
            'cow': (160, 82, 45),         # Saddle Brown
            'elephant': (105, 105, 105),  # Dim Gray
            'bear': (139, 0, 0),          # Dark Red
            'zebra': (255, 255, 255),     # White
            'giraffe': (255, 215, 0),     # Gold
            'backpack': (0, 100, 0),      # Dark Green
            'umbrella': (70, 130, 180),   # Steel Blue
            'handbag': (255, 0, 0),       # Red
            'tie': (0, 0, 139),           # Dark Blue
            'suitcase': (139, 139, 0),    # Dark Yellow
            'frisbee': (255, 20, 147),    # Deep Pink
            'skis': (0, 191, 255),        # Deep Sky Blue
            'snowboard': (255, 69, 0),    # Red Orange
            'sports ball': (50, 205, 50), # Lime Green
            'kite': (255, 105, 180),      # Hot Pink
            'baseball bat': (255, 140, 0), # Dark Orange
            'baseball glove': (139, 69, 19), # Saddle Brown
            'skateboard': (255, 250, 240), # Floral White
            'surfboard': (160, 82, 45),   # Saddle Brown
            'tennis racket': (105, 105, 105), # Dim Gray
            'bottle': (139, 0, 0),        # Dark Red
            'wine glass': (255, 255, 255), # White
            'cup': (255, 215, 0),         # Gold
            'fork': (0, 100, 0),          # Dark Green
            'knife': (70, 130, 180),      # Steel Blue
            'spoon': (255, 0, 0),         # Red
            'bowl': (0, 0, 139),          # Dark Blue
            'banana': (255, 255, 0),      # Yellow
            'apple': (255, 0, 0),         # Red
            'sandwich': (255, 165, 0),    # Orange
            'orange': (255, 165, 0),      # Orange
            'broccoli': (0, 128, 0),      # Green
            'carrot': (255, 140, 0),      # Dark Orange
            'hot dog': (255, 69, 0),      # Red Orange
            'pizza': (255, 20, 147),      # Deep Pink
            'donut': (255, 192, 203),     # Pink
            'cake': (255, 20, 147),       # Deep Pink
            'chair': (255, 0, 0),         # Red
            'couch': (139, 0, 0),         # Dark Red
            'potted plant': (0, 128, 0),  # Green
            'bed': (160, 82, 45),         # Saddle Brown
            'dining table': (139, 69, 19), # Saddle Brown
            'toilet': (255, 255, 255),    # White
            'tv': (0, 0, 0),              # Black
            'laptop': (0, 0, 255),        # Blue
            'mouse': (128, 128, 128),     # Gray
            'remote': (64, 64, 64),       # Dark Gray
            'keyboard': (192, 192, 192),  # Silver
            'cell phone': (255, 255, 0),  # Yellow
            'microwave': (255, 165, 0),   # Orange
            'oven': (139, 69, 19),        # Saddle Brown
            'toaster': (160, 82, 45),     # Saddle Brown
            'sink': (192, 192, 192),      # Silver
            'refrigerator': (255, 255, 255), # White
            'book': (139, 69, 19),        # Saddle Brown
            'clock': (255, 255, 0),       # Yellow
            'vase': (255, 20, 147),       # Deep Pink
            'scissors': (192, 192, 192),  # Silver
            'teddy bear': (255, 192, 203), # Pink
            'hair drier': (192, 192, 192), # Silver
            'toothbrush': (255, 255, 255) # White
        }
        
        # Set for tracking which classes we've seen
        self.seen_classes = set()
        
        # Generate additional random colors for new classes
        self.random_colors = self._generate_random_colors(100)
        self.color_index = 0
    
    def _generate_random_colors(self, count):
        """Generate random BGR colors"""
        colors = []
        for _ in range(count):
            # Generate bright, distinct colors
            b = random.randint(50, 255)
            g = random.randint(50, 255)
            r = random.randint(50, 255)
            colors.append((b, g, r))
        return colors
    
    def get_color(self, class_name):
        """Get color for a class, assigning new random color if needed"""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        
        # If class not seen before, assign a random color
        if class_name not in self.seen_classes:
            self.seen_classes.add(class_name)
            if self.color_index < len(self.random_colors):
                color = self.random_colors[self.color_index]
                self.color_index += 1
                print(f"🎨 Assigned new color to class '{class_name}': {color}")
                return color
            else:
                # Generate a new random color if we run out
                b = random.randint(50, 255)
                g = random.randint(50, 255)
                r = random.randint(50, 255)
                color = (b, g, r)
                print(f"🎨 Generated new random color for class '{class_name}': {color}")
                return color
        
        # Return a default color if something goes wrong
        return (128, 128, 128)  # Gray


def draw_detection_with_color(frame, detection, color, thickness=2):
    """Draw a single detection with specified color"""
    x1, y1, x2, y2 = detection['bbox']
    label = detection['label']
    confidence = detection['confidence']
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), 
                  (x1 + text_width, y1), color, -1)
    
    # Draw text
    cv2.putText(frame, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
    
    return frame


# Define model path
model_name = "yolov8n.pt"  # Using yolov8n for better compatibility

print("YOLOv9 Object Detection with Per-Class Colors")
print("=" * 50)

# Check if model file exists, if not, attempt to download
if not os.path.exists(model_name):
    print(f"Downloading {model_name}... This may take a moment.")
    try:
        YOLO(model_name)  # This will download the model
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print(f"Please check your internet connection and try again.")
        exit()

# Load YOLOv9 model
print(f"Loading YOLO model: {model_name}")
model = YOLO(model_name)
print("✅ Model loaded successfully!")

# Initialize color mapper
color_mapper = ColorMapper()
print("✅ Color mapper initialized!")

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

print("✅ Webcam opened successfully!")
print("Press 'q' to quit, 'c' to show class colors")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # Run YOLO detection
        results = model(frame, conf=0.25, verbose=False)

        # Process detections
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

        # Draw detections with per-class colors
        for detection in detections:
            class_name = detection['label']
            color = color_mapper.get_color(class_name)
            frame = draw_detection_with_color(frame, detection, color)

        # Add frame counter and instructions
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'c' for colors", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show video
        cv2.imshow("YOLOv9 Object Detection - Per-Class Colors", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Show class color information
            class_info = {}
            for class_name in color_mapper.seen_classes:
                color = color_mapper.get_color(class_name)
                class_info[class_name] = color
            print("\n🎨 Class Colors:")
            print("-" * 40)
            for class_name, color in class_info.items():
                print(f"{class_name:20} -> RGB{color}")
            print("-" * 40)
        
        frame_count += 1
        
        # Print detection info every 30 frames
        if frame_count % 30 == 0 and detections:
            print(f"Frame {frame_count}: {len(detections)} objects detected")
            for det in detections[:3]:  # Show only top 3
                color = color_mapper.get_color(det['label'])
                print(f"  - {det['label']}: {det['confidence']:.2f} (color: {color})")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Show final class color summary
    print("\n🎨 Final Class Color Summary:")
    print("=" * 50)
    class_info = {}
    for class_name in color_mapper.seen_classes:
        color = color_mapper.get_color(class_name)
        class_info[class_name] = color
    for class_name, color in class_info.items():
        print(f"{class_name:20} -> RGB{color}")
    
    print("\n✅ Detection completed!")
