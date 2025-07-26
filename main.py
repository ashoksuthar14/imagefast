import os
import mimetypes
import json
import csv
import random
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# === Configure the API client ===
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBLxX9yhoh0fHssrZCPqGHXj6EVab0BJx4")
genai.configure(api_key=api_key)

# === BANGALORE LOCATIONS ===
BANGALORE_LOCATIONS = {
    "Banashankari": {"latitude": 12.9254533, "longitude": 77.546757},
    "Basavanagudi": {"latitude": 12.9405997, "longitude": 77.5737633},
    "Bellandur": {"latitude": 12.9304278, "longitude": 77.678404},
    "Central Bangalore": {"latitude": 12.9485134, "longitude": 77.6765184},
    "Electronic City": {"latitude": 12.8452145, "longitude": 77.6601695},
    "Hebbal": {"latitude": 13.0353557, "longitude": 77.5987874},
    "HSR Layout": {"latitude": 12.9121181, "longitude": 77.6445548},
    "Indiranagar": {"latitude": 12.9783692, "longitude": 77.6408356},
    "Jayanagar": {"latitude": 12.9308107, "longitude": 77.5838577},
    "Kengeri": {"latitude": 12.8996676, "longitude": 77.4826837},
    "Koramangala": {"latitude": 12.9352403, "longitude": 77.624532},
    "MG Road": {"latitude": 12.9746905, "longitude": 77.6094613},
    "Malleswaram": {"latitude": 13.0055113, "longitude": 77.5692358},
    "Marathahalli": {"latitude": 12.956924, "longitude": 77.701127},
    "Rajajinagar": {"latitude": 12.9981732, "longitude": 77.5530446},
    "Vijayanagar": {"latitude": 12.975596, "longitude": 77.5353881},
    "Whitefield": {"latitude": 12.9698196, "longitude": 77.7499721}
}

# === EVENT CATEGORIES ===
EVENT_CATEGORIES = {
    "infrastructure": ["fallen tree", "broken streetlight", "pothole", "broken sidewalk", "damaged road", "construction"],
    "water_issues": ["water-logged street", "flooding", "drainage problem", "water leak", "sewer overflow", "water"],
    "safety": ["traffic accident", "dangerous condition", "fire hazard", "safety concern", "accident"],
    "environmental": ["garbage", "illegal dumping", "air pollution", "noise pollution", "waste", "litter"],
    "public_facilities": ["broken bench", "damaged playground", "public toilet issue", "park maintenance"],
    "traffic": ["traffic signal issue", "road closure", "construction", "parking problem", "traffic jam"]
}

# === FastAPI App ===
app = FastAPI(
    title="Citizen Report Agent API",
    description="AI-powered civic issue reporting system for Bangalore",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CitizenReportService:
    def __init__(self, json_file="citizen_reports.json", csv_file="citizen_reports.csv"):
        self.json_file = json_file
        self.csv_file = csv_file
        self.reports = self._load_existing_reports()
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
        
    def _load_existing_reports(self):
        """Load existing reports from JSON file."""
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading existing reports: {str(e)}")
            return []
    
    def _save_reports(self):
        """Save reports to both JSON and CSV files."""
        try:
            # Save JSON
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.reports, f, indent=2, ensure_ascii=False)
            
            # Save CSV
            if self.reports:
                df_data = []
                for report in self.reports:
                    row = {
                        'id': report['id'],
                        'image_path': report['image_path'],
                        'location': report['location'],
                        'latitude': report['coordinates']['latitude'],
                        'longitude': report['coordinates']['longitude'],
                        'summary': report['summary'],
                        'priority': report['priority'],
                        'sentiment': report['sentiment'],
                        'category': report['category'],
                        'media_type': report['media_type'],
                        'timestamp': report['timestamp'],
                        'user_description': report.get('user_description', ''),
                        'full_description': report['description'][:500] + '...' if len(report['description']) > 500 else report['description']
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(self.csv_file, index=False, encoding='utf-8')
                
            return True
        except Exception as e:
            print(f"Error saving reports: {str(e)}")
            return False
    
    def _get_random_location(self):
        """Get a random Bangalore location."""
        location_name = random.choice(list(BANGALORE_LOCATIONS.keys()))
        coords = BANGALORE_LOCATIONS[location_name]
        return location_name, coords["latitude"], coords["longitude"]
    
    async def analyze_media(self, file_path: str, user_description: Optional[str] = None):
        """Analyze media file and create report."""
        try:
            # Determine if it's an image or video
            mime_type, _ = mimetypes.guess_type(file_path)
            is_video = mime_type and mime_type.startswith('video/')
            
            # Get random Bangalore location
            location_name, latitude, longitude = self._get_random_location()
            
            if is_video:
                analysis = await self._analyze_video(file_path)
            else:
                analysis = await self._analyze_image(file_path)
            
            if "error" in analysis:
                return analysis
            
            # Extract summary and sentiment from the full description
            summary_data = await self._extract_summary_and_sentiment(analysis.get("description", ""))
            
            # Create the report structure
            report = {
                "id": len(self.reports) + 1,
                "image_path": file_path,
                "location": location_name,
                "coordinates": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "description": analysis.get("description", ""),
                "summary": summary_data.get("summary", "No summary available"),
                "priority": self._determine_priority(analysis.get("description", "")),
                "sentiment": summary_data.get("sentiment", "neutral"),
                "category": self._categorize_event(analysis.get("description", "")),
                "user_description": user_description,
                "media_type": "video" if is_video else "image",
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to reports and save
            self.reports.append(report)
            if self._save_reports():
                return report
            else:
                return {"error": "Failed to save report"}
            
        except Exception as e:
            return {"error": f"Error during media processing: {str(e)}"}
    
    async def _analyze_image(self, image_path: str):
        """Analyze an image using Gemini."""
        try:
            image = Image.open(image_path)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = """
            Analyze this image thoroughly and provide:
            1. A detailed description of what you see in the image
            2. Identify any infrastructure issues, safety concerns, public problems, or civic issues
            3. Assess the severity level and urgency
            4. Any recommendations for addressing the issue
            
            Be specific about locations, objects, conditions, and potential impacts on citizens.
            """
            
            response = model.generate_content([image, prompt])
            return {"description": response.text}
            
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}
    
    async def _analyze_video(self, video_path: str):
        """Analyze a video using Gemini."""
        try:
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = """
            Analyze this video thoroughly and provide:
            1. A detailed description of what you see in the video
            2. Identify any infrastructure issues, safety concerns, public problems, or civic issues
            3. Assess the severity level and urgency
            4. Any recommendations for addressing the issue
            
            Be specific about locations, objects, conditions, and potential impacts on citizens.
            """
            
            video_part = {
                "mime_type": "video/mp4",
                "data": video_data
            }
            
            response = model.generate_content([video_part, prompt])
            return {"description": response.text}
            
        except Exception as e:
            return {"error": f"Video analysis failed: {str(e)}"}
    
    async def _extract_summary_and_sentiment(self, description: str):
        """Extract summary and sentiment using Gemini."""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = f"""
            Based on this description of a civic/infrastructure issue:
            "{description}"
            
            Please provide:
            1. A one-line summary (maximum 15 words) that captures the main issue
            2. Sentiment analysis (positive, negative, neutral, urgent)
            
            Format your response as:
            Summary: [one line summary]
            Sentiment: [sentiment]
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the response
            summary = "Issue reported"
            sentiment = "neutral"
            
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('Summary:'):
                    summary = line.replace('Summary:', '').strip()
                elif line.startswith('Sentiment:'):
                    sentiment = line.replace('Sentiment:', '').strip().lower()
            
            return {"summary": summary, "sentiment": sentiment}
            
        except Exception as e:
            return {"summary": "Issue reported", "sentiment": "neutral"}
    
    def _categorize_event(self, description: str):
        """Categorize the event based on description."""
        description_lower = description.lower()
        
        for category, keywords in EVENT_CATEGORIES.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        return "other"
    
    def _determine_priority(self, description: str):
        """Determine priority based on description content."""
        description_lower = description.lower()
        
        critical_keywords = ["emergency", "fire", "explosion", "collapse", "death", "critical", "urgent", "danger"]
        high_priority = ["accident", "dangerous", "flooding", "safety", "hazard", "broken", "severe"]
        medium_priority = ["damaged", "issue", "problem", "maintenance", "concern", "repair needed"]
        
        for keyword in critical_keywords:
            if keyword in description_lower:
                return "critical"
        
        for keyword in high_priority:
            if keyword in description_lower:
                return "high"
        
        for keyword in medium_priority:
            if keyword in description_lower:
                return "medium"
        
        return "low"
    
    def get_statistics(self):
        """Get statistics about all reports."""
        if not self.reports:
            return {"message": "No reports available."}
        
        stats = {
            "total_reports": len(self.reports),
            "by_category": {},
            "by_priority": {},
            "by_location": {},
            "by_sentiment": {},
            "by_media_type": {}
        }
        
        for report in self.reports:
            category = report.get("category", "unknown")
            priority = report.get("priority", "unknown")
            location = report.get("location", "unknown")
            sentiment = report.get("sentiment", "unknown")
            media_type = report.get("media_type", "unknown")
            
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            stats["by_location"][location] = stats["by_location"].get(location, 0) + 1
            stats["by_sentiment"][sentiment] = stats["by_sentiment"].get(sentiment, 0) + 1
            stats["by_media_type"][media_type] = stats["by_media_type"].get(media_type, 0) + 1
        
        return stats

# Initialize the service
service = CitizenReportService()

# === API ENDPOINTS ===

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Citizen Report Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-report",
            "reports": "/reports",
            "statistics": "/statistics",
            "download_json": "/download/json",
            "download_csv": "/download/csv"
        }
    }

@app.post("/upload-report")
async def upload_report(
    file: UploadFile = File(...),
    user_description: Optional[str] = Form(None)
):
    """Upload and analyze a media file (image/video)."""
    try:
        # Validate file type
        if not file.content_type.startswith(('image/', 'video/')):
            raise HTTPException(status_code=400, detail="Only image and video files are allowed")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(service.upload_dir, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze the media
        result = await service.analyze_media(file_path, user_description)
        
        if "error" in result:
            # Clean up file if analysis failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": "Report created successfully",
            "report": result,
            "total_reports": len(service.reports)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/reports")
async def get_reports(limit: int = 50, offset: int = 0):
    """Get all reports with pagination."""
    total = len(service.reports)
    reports = service.reports[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "reports": reports
    }

@app.get("/reports/{report_id}")
async def get_report(report_id: int):
    """Get a specific report by ID."""
    report = next((r for r in service.reports if r["id"] == report_id), None)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@app.get("/statistics")
async def get_statistics():
    """Get statistics about all reports."""
    return service.get_statistics()

@app.get("/download/json")
async def download_json():
    """Download reports as JSON file."""
    if not os.path.exists(service.json_file):
        raise HTTPException(status_code=404, detail="No reports available")
    
    return FileResponse(
        service.json_file,
        media_type='application/json',
        filename=f"citizen_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

@app.get("/download/csv")
async def download_csv():
    """Download reports as CSV file."""
    if not os.path.exists(service.csv_file):
        raise HTTPException(status_code=404, detail="No reports available")
    
    return FileResponse(
        service.csv_file,
        media_type='text/csv',
        filename=f"citizen_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.delete("/reports/{report_id}")
async def delete_report(report_id: int):
    """Delete a specific report."""
    report_index = next((i for i, r in enumerate(service.reports) if r["id"] == report_id), None)
    if report_index is None:
        raise HTTPException(status_code=404, detail="Report not found")
    
    deleted_report = service.reports.pop(report_index)
    
    # Clean up the uploaded file
    if os.path.exists(deleted_report["image_path"]):
        os.remove(deleted_report["image_path"])
    
    # Save updated reports
    service._save_reports()
    
    return {"success": True, "message": f"Report {report_id} deleted successfully"}

@app.get("/locations")
async def get_locations():
    """Get all available Bangalore locations."""
    return {"locations": BANGALORE_LOCATIONS}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_reports": len(service.reports),
        "files_exist": {
            "json": os.path.exists(service.json_file),
            "csv": os.path.exists(service.csv_file)
        }
    }

# === Run the server ===
if __name__ == "__main__":
    print("🚀 Starting Citizen Report Agent API...")
    print("📊 Available endpoints:")
    print("  - POST /upload-report - Upload image/video for analysis")
    print("  - GET /reports - Get all reports")
    print("  - GET /statistics - Get report statistics")
    print("  - GET /download/csv - Download CSV file")
    print("  - GET /download/json - Download JSON file")
    print("\n💾 Files will be saved as:")
    print(f"  - JSON: {service.json_file}")
    print(f"  - CSV: {service.csv_file}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)