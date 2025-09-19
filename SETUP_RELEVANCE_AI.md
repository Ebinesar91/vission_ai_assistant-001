# Setup Your Relevance AI Voice Assistant

## 🎯 Quick Setup Guide

### Step 1: Get Your API Key
1. Go to your Relevance AI dashboard: https://app.relevanceai.com/
2. Navigate to your agent: `737e270d-bf08-439a-b82e-f8fbc5543013`
3. Look for **API Settings** or **Account Settings**
4. Copy your **API Key**

### Step 2: Test Your Connection
```bash
python test_relevance_ai.py
```
Enter your API key when prompted.

### Step 3: Run Voice Assistant with Your Agent
```bash
python assistant.py --api-key YOUR_API_KEY_HERE
```

## 🔧 Alternative: Set Environment Variable
```bash
set RELEVANCE_AI_API_KEY=your_api_key_here
python assistant.py
```

## 🎮 What You'll Get

### With Your API Key:
- **"What's around me"** → Your agent responds with detection summary
- **"Tell me a joke"** → Your agent responds with a joke
- **"What's the weather?"** → Your agent responds with weather info
- **Any question** → Your agent provides intelligent responses

### Without API Key:
- **"What's around me"** → Basic detection summary
- **Other questions** → "I don't have access to my AI knowledge base"

## 🎯 Your Agent Configuration
- **Agent ID**: `737e270d-bf08-439a-b82e-f8fbc5543013`
- **Project ID**: `f021bc31-c5e3-4c23-b437-7db1f29e9530`
- **Purpose**: Vista, the Vision Assistant for visually impaired users

## 🚀 Ready to Test!

1. **Get your API key** from the Relevance AI dashboard
2. **Run the test script**: `python test_relevance_ai.py`
3. **Start your voice assistant**: `python assistant.py --api-key YOUR_KEY`

Your voice assistant will then be fully integrated with your Relevance AI agent! 🎉
