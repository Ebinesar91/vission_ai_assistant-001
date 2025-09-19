"""
Configuration file for YOLOv9 + Relevance AI Voice Assistant
"""

# YOLOv9 Model Configuration
MODEL_PATH = "yolov8n.pt"  # Path to YOLOv9 model file
WEBCAM_SOURCE = 0  # Webcam source (0 for default camera)
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold

# Relevance AI Configuration
RELEVANCE_AI_API_KEY = None  # Set your API key here or use environment variable
AGENT_ID = "737e270d-bf08-439a-b82e-f8fbc5543013"
PROJECT_ID = "f021bc31-c5e3-4c23-b437-7db1f29e9530"

# Voice Assistant Configuration
SPEAK_INTERVAL = 3.0  # Time between voice responses (seconds)
TTS_RATE = 150  # Speech rate (words per minute)
TTS_VOLUME = 0.8  # Speech volume (0.0 to 1.0)

# Detection Configuration
MAX_DETECTION_HISTORY = 10  # Number of recent detections to keep
DETECTION_LOG_INTERVAL = 30  # Frames between detection logging
