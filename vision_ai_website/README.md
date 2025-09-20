# Vision AI Assistant Website

A modern Vision AI Assistant website built with React frontend and FastAPI backend that provides object detection, AI assistant chat, and navigation features.

## 🚀 Features

- **🎯 Object Detection**: Detect objects with distance and direction information
- **🤖 AI Assistant**: Chat with an AI assistant for questions and assistance
- **🧭 Navigation**: Get navigation information with distance, steps, and ETA
- **📱 Responsive Design**: Clean, modern UI that works on all devices
- **⚡ Real-time**: Fast API responses with loading states

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **OpenAI API** - AI assistant functionality
- **CORS enabled** - For frontend integration

### Frontend
- **React** - Modern JavaScript library
- **Axios** - HTTP client for API calls
- **CSS3** - Modern styling with gradients and animations

## 📁 Project Structure

```
vision_ai_website/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── frontend/               # React frontend
│   ├── package.json       # Node.js dependencies
│   ├── public/
│   │   └── index.html     # HTML template
│   └── src/
│       ├── api.js         # API functions
│       ├── App.js         # Main React component
│       ├── App.css        # Styling
│       └── index.js       # React entry point
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to project directory
cd vision_ai_website

# Install Python dependencies
pip install -r requirements.txt

# Set your OpenAI API key (optional, for AI assistant)
export OPENAI_API_KEY="your-api-key-here"

# Run the FastAPI backend
python -m uvicorn main:app --reload --port 8000
```

The backend will start at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the React development server
npm start
```

The frontend will start at `http://localhost:3000`

## 🔧 API Endpoints

### GET /detect
Returns detected objects with distance and direction information.

**Response:**
```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 300],
      "distance": "2.5m",
      "direction": "front-left"
    }
  ],
  "message": "Detected 3 objects"
}
```

### POST /voice
Accepts user query and returns AI assistant response.

**Request:**
```json
{
  "query": "What's around me?"
}
```

**Response:**
```json
{
  "response": "I can see several objects including people, cars, and animals..."
}
```

### POST /navigate
Calculates navigation from start to end coordinates.

**Request:**
```json
{
  "start": [40.7128, -74.0060],
  "end": [40.7589, -73.9851]
}
```

**Response:**
```json
{
  "distance_m": 8500,
  "duration_min": 17,
  "steps": "1. Head north for 2833m\n2. Turn right..."
}
```

## 🎨 Customization

### Adding New Object Detection Models
1. Update the `detect_objects()` function in `main.py`
2. Replace the mock data with actual YOLO model inference
3. Add model configuration options

### Styling
- Modify `frontend/src/App.css` for UI changes
- Colors, fonts, and layout can be customized
- Responsive design breakpoints included

### API Configuration
- Update `frontend/src/api.js` to change backend URL
- Add authentication headers if needed
- Configure request/response interceptors

## 🔒 Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

## 🐛 Troubleshooting

### Backend Issues
- Ensure Python 3.8+ is installed
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Verify port 8000 is not in use

### Frontend Issues
- Ensure Node.js 14+ is installed
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### CORS Issues
- Backend has CORS enabled for all origins
- If issues persist, check browser console for errors

## 📱 Mobile Support

The website is fully responsive and works on:
- Mobile phones (portrait and landscape)
- Tablets
- Desktop computers
- All modern browsers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the browser console for errors
3. Ensure both backend and frontend are running
4. Check API endpoints are accessible

---

**Happy coding! 🚀**
