from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
import torch

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

# ---- YOLOv7 (tiny) setup ----
_YOLOV7_MODEL = None
_YOLOV7_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_YOLOV7_REPO = "WongKinYiu/yolov7"
_YOLOV7_WEIGHTS = "yolov7-tiny.pt"  # smaller & faster

def _load_yolov7():
    global _YOLOV7_MODEL
    if _YOLOV7_MODEL is None:
        try:
            _YOLOV7_MODEL = torch.hub.load(_YOLOV7_REPO, 'custom', _YOLOV7_WEIGHTS, trust_repo=True)
            _YOLOV7_MODEL.eval().to(_YOLOV7_DEVICE)
            print("YOLOv7 model loaded.")
        except Exception as e:
            print(f"YOLOv7 load failed: {e}")
            _YOLOV7_MODEL = False  # sentinel to avoid repeated attempts

def detect_golf_ball(_image_bytes: bytes):
    # Replace placeholder with YOLOv7 sports ball detection (class 32)
    _load_yolov7()
    if not _YOLOV7_MODEL:
        return []
    try:
        img = Image.open(io.BytesIO(_image_bytes)).convert("RGB")
        # YOLOv7 hub model accepts PIL directly
        with torch.no_grad():
            results = _YOLOV7_MODEL(img, size=640)
        # results.xyxy[0]: [x1,y1,x2,y2,conf,cls]
        if not results or not hasattr(results, "xyxy"):
            return []
        det = results.xyxy[0].cpu()
        out = []
        for *box, conf, cls in det.tolist():
            cls = int(cls)
            if cls != 32:  # COCO 'sports ball'
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            diameter = min(w, h)
            out.append({
                "method": "yolov7",
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(cx), float(cy)],
                "radius_px": float(diameter / 2),
                "diameter_px": float(diameter),
                "class_id": cls
            })
        # Optional: sort by confidence
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out
    except Exception as e:
        print(f"Detection error: {e}")
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
