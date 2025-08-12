from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
from datetime import datetime

app = FastAPI()  # ensure the variable is literally named "app"

# Create uploads directory if it doesn't exist
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# static files (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    idx = Path("static/index2.html")
    return FileResponse(idx) if idx.exists() else {"msg": "Hello from FastAPI"}

@app.post("/upload_frame")
async def upload_frame(
    image: UploadFile = File(...),
    timestamp: str = Form(None),
    frame_number: str = Form(None),
    pitch: str = Form(None),
    roll: str = Form(None)
):
    """Handle camera frame uploads with sensor data"""
    try:
        # Generate filename with timestamp
        now = datetime.now()
        filename = f"frame_{frame_number}_{timestamp}.jpg"
        file_path = uploads_dir / filename
        
        # Save the uploaded image
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Log the upload with sensor data
        print(f"Uploaded: {filename}")
        if pitch and roll:
            print(f"  Sensor data - Pitch: {pitch}°, Roll: {roll}°")
        
        return {
            "success": True,
            "filename": filename,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "sensor_data": {
                "pitch": pitch,
                "roll": roll
            }
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}
