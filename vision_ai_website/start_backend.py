#!/usr/bin/env python3
"""
Startup script for Vision AI Backend
Run this to start the FastAPI backend server
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI backend server"""
    print("🚀 Starting Vision AI Backend...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/live")
    print("🎯 Detection control: POST http://localhost:8000/detect/control")
    print("📊 Status: GET http://localhost:8000/detect/status")
    print("🌐 Frontend: http://localhost:3000")
    print()

    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
