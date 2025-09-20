from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import openai
import os
from typing import List, Optional
import random

# Initialize FastAPI app
app = FastAPI(title="Vision AI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VoiceQuery(BaseModel):
    query: str

class NavigationRequest(BaseModel):
    start: List[float]
    end: List[float]

class DetectionResponse(BaseModel):
    detections: List[dict]
    message: str

class VoiceResponse(BaseModel):
    response: str

class NavigationResponse(BaseModel):
    distance_m: float
    duration_min: int
    steps: str

# Mock object detection function (replace with actual YOLO model)
def detect_objects():
    """Mock object detection - replace with actual YOLOv9 model"""
    mock_detections = [
        {
            "label": "person",
            "confidence": 0.85,
            "bbox": [100, 150, 200, 300],
            "distance": "2.5m",
            "direction": "front-left"
        },
        {
            "label": "car",
            "confidence": 0.92,
            "bbox": [300, 200, 450, 280],
            "distance": "5.2m",
            "direction": "front-right"
        },
        {
            "label": "dog",
            "confidence": 0.78,
            "bbox": [50, 350, 120, 400],
            "distance": "1.8m",
            "direction": "left"
        }
    ]
    return mock_detections

# Mock navigation function
def calculate_navigation(start: List[float], end: List[float]):
    """Mock navigation calculation"""
    # Calculate simple distance
    distance = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5 * 111000  # Rough meters
    duration = int(distance / 1000 * 2)  # Assume 30km/h average speed

    steps = f"1. Head north for {distance/3:.0f}m\n2. Turn right and continue for {distance/3:.0f}m\n3. Turn left and continue for {distance/3:.0f}m\n4. You have arrived at your destination"

    return {
        "distance_m": distance,
        "duration_min": duration,
        "steps": steps
    }

# API Endpoints
@app.get("/detect", response_model=DetectionResponse)
async def detect():
    """Detect objects and return with distance and direction"""
    try:
        detections = detect_objects()
        return DetectionResponse(
            detections=detections,
            message=f"Detected {len(detections)} objects"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice", response_model=VoiceResponse)
async def voice_assistant(query: VoiceQuery):
    """Process voice query and return AI assistant response"""
    try:
        # Initialize OpenAI client (you'll need to set OPENAI_API_KEY)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Vision AI assistant. You can help users understand what objects are detected around them and provide assistance with navigation and general queries."},
                {"role": "user", "content": query.query}
            ]
        )

        ai_response = response.choices[0].message.content
        return VoiceResponse(response=ai_response)

    except Exception as e:
        # Fallback response if OpenAI is not available
        fallback_responses = {
            "what's around me": "I can see several objects including people, cars, and animals. Would you like me to describe them in more detail?",
            "help": "I can help you detect objects around you, provide navigation assistance, or answer general questions. What would you like to do?",
            "navigation": "I can help you navigate to your destination. Please provide your starting point and destination coordinates.",
            "default": "I'm here to help you with object detection, navigation, and general assistance. What would you like to know?"
        }

        response_text = fallback_responses.get(
            query.query.lower(),
            fallback_responses["default"]
        )

        return VoiceResponse(response=response_text)

@app.post("/navigate", response_model=NavigationResponse)
async def navigate(request: NavigationRequest):
    """Calculate navigation from start to end coordinates"""
    try:
        navigation_data = calculate_navigation(request.start, request.end)
        return NavigationResponse(**navigation_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vision AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "GET /detect - Detect objects with distance and direction",
            "voice": "POST /voice - AI assistant voice queries",
            "navigate": "POST /navigate - Navigation assistance"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
