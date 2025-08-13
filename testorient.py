from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
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

def detect_golf_ball(_image_bytes: bytes):
    # Reverted: no model inference; always return empty list
    return []

@app.post("/upload_frame")
async def upload_frame(
    image: UploadFile = File(...),
    timestamp: str = Form(None),
    frame_number: str = Form(None),
    pitch: str = Form(None),
    roll: str = Form(None)
):
    """Handle camera frame uploads with sensor data and golf ball detection"""
    try:
        # Read image bytes
        data = await image.read()
        
        # Generate filename with timestamp
        filename = f"frame_{frame_number}_{timestamp}.jpg"
        file_path = uploads_dir / filename
        
        # Save the uploaded image
        with open(file_path, "wb") as f:
            f.write(data)

        # Detect golf balls in the image
        detections = detect_golf_ball(data)
        
        # Log the upload with sensor data and detections
        print(f"Uploaded: {filename}")
        if pitch and roll:
            print(f"  Pitch: {pitch}° Roll: {roll}°")
        if detections:
            for i, d in enumerate(detections):
                print(f"  Ball {i+1}: center=({d['center'][0]:.1f},{d['center'][1]:.1f})")
        else:
            print("  No golf balls detected")

        return {
            "success": True,
            "filename": filename,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "sensor_data": {"pitch": pitch, "roll": roll},
            "golf_balls": detections
        }
    except Exception as e:
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/upload_frame")
async def upload_frame(
    image: UploadFile = File(...),
    timestamp: str = Form(None),
    frame_number: str = Form(None),
    pitch: str = Form(None),
    roll: str = Form(None)
):
    """Handle camera frame uploads with sensor data and golf ball detection"""
    try:
        # Read image bytes
        data = await image.read()
        
        # Generate filename with timestamp
        filename = f"frame_{frame_number}_{timestamp}.jpg"
        file_path = uploads_dir / filename
        
        # Save the uploaded image
        with open(file_path, "wb") as f:
            f.write(data)

        # Detect golf balls in the image
        detections = detect_golf_ball(data)
        
        # Log the upload with sensor data and detections
        print(f"Uploaded: {filename}")
        if pitch and roll:
            print(f"  Pitch: {pitch}° Roll: {roll}°")
        if detections:
            for i, d in enumerate(detections):
                print(f"  Ball {i+1}: center=({d['center'][0]:.1f},{d['center'][1]:.1f}) "
                      f"diam={d['diameter_px']:.1f}px conf={d['confidence']:.2f}")
        else:
            print("  No golf balls detected")
        return {
            "success": True,
            "filename": filename,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "sensor_data": {"pitch": pitch, "roll": roll},
            "golf_balls": detections
        }
    except Exception as e:
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}
