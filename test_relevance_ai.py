#!/usr/bin/env python3
"""
Test script to verify Relevance AI connection
"""

import requests
import json

def test_relevance_ai_connection(api_key, agent_id, project_id):
    """Test connection to Relevance AI agent"""
    
    url = f"https://api.relevanceai.com/v1/agents/{agent_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test input
    input_data = {
        "camera_feed": "person detected, about 3 steps ahead",
        "audio_input": "what's around me",
        "current_location": {
            "latitude": 0.0,
            "longitude": 0.0,
            "accuracy": 0.0
        },
        "device_orientation": {
            "heading": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }
    }
    
    payload = {
        "input": input_data,
        "project_id": project_id
    }
    
    try:
        print("Testing Relevance AI connection...")
        print(f"Agent ID: {agent_id}")
        print(f"Project ID: {project_id}")
        print(f"API URL: {url}")
        print("-" * 50)
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Connection successful!")
            print(f"Response: {result.get('output', 'No output received')}")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Your agent configuration
    AGENT_ID = "737e270d-bf08-439a-b82e-f8fbc5543013"
    PROJECT_ID = "f021bc31-c5e3-4c23-b437-7db1f29e9530"
    
    print("Relevance AI Connection Test")
    print("=" * 50)
    
    # Get API key from user
    api_key = input("Enter your Relevance AI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        exit(1)
    
    # Test connection
    success = test_relevance_ai_connection(api_key, AGENT_ID, PROJECT_ID)
    
    if success:
        print("\n🎉 Your API key is working! You can now run:")
        print(f"python assistant.py --api-key {api_key}")
    else:
        print("\n❌ Please check your API key and try again.")
