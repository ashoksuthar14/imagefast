#!/usr/bin/env python3
"""
Simple script to run the Citizen Report Agent FastAPI server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables!")
        print("   Please set your Google API key:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")
        print("   Or create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    print("🚀 Starting Citizen Report Agent API Server...")
    print("📋 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Interactive API: http://localhost:8000/redoc")
    print("\n✨ Ready to process civic reports!")
    
    uvicorn.run(
        "main:app",  # Change this to match your filename (without .py)
        host="127.0.0.1",  # Use localhost for Windows
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )