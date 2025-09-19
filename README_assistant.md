# YOLOv9 + Relevance AI Voice Assistant

A real-time object detection system integrated with a voice assistant powered by Relevance AI. This system uses YOLOv9 to detect objects from your webcam feed and provides natural language descriptions through voice output.

## Features

- 🎥 **Real-time Object Detection**: Uses YOLOv9 to detect objects from webcam feed
- 🗣️ **Voice Output**: Speaks detection results using text-to-speech
- 🤖 **AI Integration**: Connects to your Relevance AI agent for intelligent responses
- 📍 **Spatial Awareness**: Describes object positions (left, right, center, close, far)
- ⚡ **Real-time Processing**: Continuous detection and voice feedback

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_assistant.txt
```

### 2. Set Up Relevance AI (Optional)

If you want to use your Relevance AI agent:

1. Get your API key from [Relevance AI](https://relevanceai.com)
2. Set it as an environment variable:
   ```bash
   set RELEVANCE_AI_API_KEY=your_api_key_here
   ```
   Or pass it as a command line argument:
   ```bash
   python assistant_integration.py --api-key your_api_key_here
   ```

### 3. Run the Assistant

```bash
python assistant_integration.py
```

## Usage

### Basic Usage
```bash
# Run with default settings
python assistant_integration.py

# Run with custom settings
python assistant_integration.py --conf 0.3 --speak-interval 5 --source 1
```

### Command Line Options

- `--model`: Path to YOLOv9 model file (default: yolov8n.pt)
- `--source`: Webcam source number (default: 0)
- `--api-key`: Relevance AI API key
- `--agent-id`: Relevance AI agent ID (default: your agent ID)
- `--project-id`: Relevance AI project ID (default: your project ID)
- `--conf`: Detection confidence threshold (default: 0.25)
- `--speak-interval`: Time between voice responses in seconds (default: 3.0)

### Controls

- **'q'**: Quit the application
- **'s'**: Force immediate voice response

## How It Works

1. **Object Detection**: YOLOv9 continuously analyzes webcam frames
2. **Detection Processing**: Extracts object labels, confidence scores, and positions
3. **AI Processing**: Sends detection data to your Relevance AI agent
4. **Voice Output**: Speaks the AI's response using text-to-speech
5. **Visual Display**: Shows bounding boxes and labels on screen

## Detection Format

The system detects objects and describes them with:
- **Object type** (person, chair, laptop, etc.)
- **Confidence score** (0.0 to 1.0)
- **Position** (left, right, center)
- **Distance** (close, medium, far)

Example output: "Person detected with 0.87 confidence close center"

## Configuration

Edit `config.py` to customize:
- Model path and webcam source
- Relevance AI credentials
- Voice settings (rate, volume, interval)
- Detection parameters

## Troubleshooting

### Common Issues

1. **Webcam not found**: Try different source numbers (0, 1, 2...)
2. **No voice output**: Check if speakers are working and pyttsx3 is installed
3. **API errors**: Verify your Relevance AI API key and agent ID
4. **Performance issues**: Lower confidence threshold or increase speak interval

### Dependencies

- Python 3.8+
- OpenCV with GUI support
- PyTorch
- Ultralytics
- pyttsx3
- requests
- pillow

## File Structure

```
├── assistant_integration.py    # Main integration script
├── detect_webcam_simple.py     # Simple webcam detection
├── config.py                   # Configuration file
├── requirements_assistant.txt  # Python dependencies
├── README_assistant.md         # This file
└── yolov8n.pt                 # YOLOv9 model file
```

## API Integration

The system integrates with your Relevance AI agent using:
- **Agent ID**: 737e270d-bf08-439a-b82e-f8fbc5543013
- **Project ID**: f021bc31-c5e3-4c23-b437-7db1f29e9530
- **Actions**: Image description and text-to-speech

Your agent is configured for visual assistance and can:
- Process camera feed data
- Generate natural language descriptions
- Provide navigation assistance
- Handle voice commands

## License

This project uses the YOLOv9 model and integrates with Relevance AI services.
