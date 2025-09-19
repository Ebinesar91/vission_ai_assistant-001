import cv2
from ultralytics import YOLO
import os

# Define model path
model_name = "yolov9n.pt"

# Check if model file exists, if not, attempt to download
if not os.path.exists(model_name):
    print(f"Downloading {model_name}... This may take a moment.")
    try:
        # The YOLO constructor itself attempts to download if not found locally
        YOLO(model_name).export(format='torchscript') # This forces a download if the model isn't found and then converts it
        # However, the YOLO() constructor typically downloads it to the ultralytics cache directory.
        # To ensure it's in the current directory, we'll try to move it or just rely on the default behavior
        # which usually makes it accessible once downloaded to the cache.
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print(f"Please manually download {model_name} from https://github.com/ultralytics/yolov9/releases/download/v0.1.0/yolov9n.pt and place it in the same directory as detect.py")
        exit()

# Load YOLOv9 model
model = YOLO(model_name)  # lightweight pretrained model

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing on CPU
    # You can adjust the interpolation method (e.g., cv2.INTER_LINEAR, cv2.INTER_AREA) for different quality/speed tradeoffs.
    # A common practice for downsampling is INTER_AREA, and for upsampling is INTER_CUBIC or INTER_LINEAR.
    # For real-time applications, a smaller dimension (e.g., 640, 480) is often used.
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # Run YOLOv9 detection
    results = model(frame)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("YOLOv9 Object Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
