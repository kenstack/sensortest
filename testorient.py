from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
from datetime import datetime
import cv2
import numpy as np
import io
from PIL import Image

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

def detect_golf_ball(image_bytes):
    """
    Fast golf ball detection using OpenCV (fully open source).
    Uses multiple methods: HoughCircles, contour detection, and blob detection.
    Returns: list of detections with bbox, confidence, and size info
    """
    try:
        # Convert bytes to OpenCV image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detections = []
        
        # Method 1: HoughCircles detection (fast, good for round objects)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=120
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validate circle by checking contrast and roundness
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Check brightness (golf balls are usually white/bright)
                inside_mean = cv2.mean(gray, mask=mask)[0]
                
                # Check edge contrast
                edge_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(edge_mask, (x, y), r, 255, 2)  # Ring around circle
                edge_mean = cv2.mean(gray, mask=edge_mask)[0]
                
                contrast_score = inside_mean / (edge_mean + 1e-6)
                brightness_score = inside_mean / 255.0
                
                # Golf ball validation heuristics
                if (contrast_score > 1.1 and brightness_score > 0.4) or brightness_score > 0.7:
                    confidence = min(0.9, (contrast_score - 1.0) * 0.5 + brightness_score * 0.5)
                    
                    detections.append({
                        'method': 'hough_circle',
                        'confidence': float(confidence),
                        'bbox': [float(x-r), float(y-r), float(x+r), float(y+r)],
                        'center': [float(x), float(y)],
                        'radius_px': float(r),
                        'diameter_px': float(r * 2)
                    })
        
        # Method 2: Blob detection for circular white objects
        if len(detections) < 2:  # Only if we haven't found many balls yet
            # Setup SimpleBlobDetector parameters
            params = cv2.SimpleBlobDetector_Params()
            
            # Filter by Area
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 15000
            
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.6
            
            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.7
            
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.4
            
            # Create detector
            detector = cv2.SimpleBlobDetector_create(params)
            
            # Detect blobs
            keypoints = detector.detect(gray)
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2)
                
                # Check if this detection overlaps with existing ones
                overlap = False
                for existing in detections:
                    ex, ey = existing['center']
                    if np.sqrt((x - ex)**2 + (y - ey)**2) < max(r, existing['radius_px']) * 0.8:
                        overlap = True
                        break
                
                if not overlap:
                    confidence = min(0.8, kp.response * 2.0)  # Blob detector response
                    
                    detections.append({
                        'method': 'blob_detector',
                        'confidence': float(confidence),
                        'bbox': [float(x-r), float(y-r), float(x+r), float(y+r)],
                        'center': [float(x), float(y)],
                        'radius_px': float(r),
                        'diameter_px': float(r * 2)
                    })
        
        # Method 3: Contour-based detection for backup
        if len(detections) == 0:
            # Apply Gaussian blur and threshold
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Reasonable size range
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.6:  # Fairly circular
                            # Get bounding circle
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            x, y, radius = int(x), int(y), int(radius)
                            
                            confidence = min(0.7, circularity * 0.8)
                            
                            detections.append({
                                'method': 'contour',
                                'confidence': float(confidence),
                                'bbox': [float(x-radius), float(y-radius), float(x+radius), float(y+radius)],
                                'center': [float(x), float(y)],
                                'radius_px': float(radius),
                                'diameter_px': float(radius * 2)
                            })
        
        # Sort by confidence and return top detections
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:3]  # Return max 3 best detections
        
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
        image_bytes = await image.read()
        
        # Generate filename with timestamp
        now = datetime.now()
        filename = f"frame_{frame_number}_{timestamp}.jpg"
        file_path = uploads_dir / filename
        
        # Save the uploaded image
        with open(file_path, "wb") as buffer:
            buffer.write(image_bytes)
        
        # Detect golf balls in the image
        golf_balls = detect_golf_ball(image_bytes)
        
        # Log the upload with sensor data and detections
        print(f"Uploaded: {filename}")
        if pitch and roll:
            print(f"  Sensor data - Pitch: {pitch}°, Roll: {roll}°")
        if golf_balls:
            print(f"  Detected {len(golf_balls)} golf ball(s):")
            for i, ball in enumerate(golf_balls):
                print(f"    Ball {i+1}: center=({ball['center'][0]:.1f}, {ball['center'][1]:.1f}), "
                      f"diameter={ball['diameter_px']:.1f}px, conf={ball['confidence']:.3f} ({ball['method']})")
        else:
            print("  No golf balls detected")
        
        return {
            "success": True,
            "filename": filename,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "sensor_data": {
                "pitch": pitch,
                "roll": roll
            },
            "golf_balls": golf_balls
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}
