import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import asyncio
import json
import base64
from pathlib import Path
import threading
from collections import defaultdict
import time
import os
import shutil
from datetime import datetime
import logging
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create necessary directories
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# Get the absolute path to the frontend directory
FRONTEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "frontend"
logger.info(f"Frontend directory: {FRONTEND_DIR}")

# Serve frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Available YOLO models
YOLO_MODELS = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt"
}

# Load default YOLO model (medium for better accuracy)
model_path = f"yolov8_weights/{YOLO_MODELS['medium']}"
logger.info(f"Loading YOLO model from: {model_path}")
model = YOLO(model_path)
current_model_name = "medium"  # Track current model name
logger.info("YOLO model loaded successfully")

# Load person classification model
try:
    person_classifier = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    person_classifier.eval()
    logger.info("Person classification model loaded successfully")
except Exception as e:
    logger.error(f"Error loading person classification model: {e}")
    person_classifier = None

# Define person classes
PERSON_CLASSES = [
    'child', 'teenager', 'young_adult', 'adult', 'elderly',
    'male', 'female',
    'casual', 'formal', 'business',
    'standing', 'sitting', 'walking', 'running'
]

# Image transformation for classification
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_person(frame, bbox):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None
        
        # Convert to PIL Image
        person_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        
        # Transform image
        img_tensor = transform(person_img).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = person_classifier(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top predictions
        top_prob, top_catid = torch.topk(probabilities, 3)
        
        # Map predictions to our classes (this is a simplified mapping)
        predictions = {}
        for prob, catid in zip(top_prob, top_catid):
            if prob > 0.3:  # Only include predictions with confidence > 30%
                predictions[PERSON_CLASSES[catid % len(PERSON_CLASSES)]] = float(prob)
        
        return predictions
    except Exception as e:
        logger.error(f"Error in person classification: {e}")
        return None

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.current_session = None
        self.session_counter = 0
        self.active_tracks = {}  # Track active objects by ID

    def create_session(self, source_type="webcam", source_id=0):
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        self.sessions[session_id] = {
            "start_time": datetime.now(),
            "end_time": None,
            "objects_detected": defaultdict(int),
            "total_frames": 0,
            "video_source": f"{source_type}:{source_id}",
            "model_used": current_model_name,
            "active_tracks": {},
            "track_history": defaultdict(list)
        }
        self.current_session = session_id
        return session_id

    def end_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]["end_time"] = datetime.now()
            return self.sessions[session_id]
        return None

    def update_session(self, session_id, objects_detected, active_tracks):
        if session_id in self.sessions:
            self.sessions[session_id]["total_frames"] += 1
            self.sessions[session_id]["active_tracks"] = active_tracks
            
            # Update track history
            for track_id, track_info in active_tracks.items():
                self.sessions[session_id]["track_history"][track_id].append({
                    "frame": self.sessions[session_id]["total_frames"],
                    "class": track_info["class"],
                    "confidence": track_info["confidence"]
                })

    def get_session_summary(self, session_id):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            duration = (session["end_time"] or datetime.now()) - session["start_time"]
            
            # Calculate unique objects by class
            unique_objects = defaultdict(set)
            for track_id, history in session["track_history"].items():
                if history:  # If track has any history
                    class_name = history[0]["class"]  # Get the class from first detection
                    unique_objects[class_name].add(track_id)
            
            return {
                "session_id": session_id,
                "duration": str(duration),
                "total_frames": session["total_frames"],
                "objects_detected": dict(session["objects_detected"]),
                "unique_objects": {k: len(v) for k, v in unique_objects.items()},
                "video_source": session["video_source"],
                "model_used": session["model_used"],
                "active_tracks": session["active_tracks"]
            }
        return None

    def get_all_sessions(self):
        return [self.get_session_summary(session_id) for session_id in self.sessions]

session_manager = SessionManager()

class ThreadedCamera:
    def __init__(self, src=0):
        logger.info(f"Initializing camera with source: {src}")
        self.src = src
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.initialize_camera()
        self.thread.start()
        logger.info("Camera thread started")

    def initialize_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera source: {self.src}")
            logger.info("Camera opened successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            raise

    def update(self):
        consecutive_failures = 0
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("Camera not initialized, attempting to reinitialize...")
                    self.initialize_camera()
                    continue

                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera (attempt {consecutive_failures})")
                    if consecutive_failures >= 5:
                        logger.error("Too many consecutive failures, reinitializing camera...")
                        self.initialize_camera()
                        consecutive_failures = 0
                time.sleep(0.01)  # Small delay to prevent CPU overload
            except Exception as e:
                logger.error(f"Error in camera update loop: {e}")
                time.sleep(1)  # Wait before retrying

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def release(self):
        logger.info("Releasing camera resources")
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

class Tracker:
    def __init__(self, video_source=0):
        logger.info(f"Initializing tracker with video source: {video_source}")
        self.model = model
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=int)
        self.class_names = self.model.model.names
        self.track_id_count = defaultdict(int)
        self.live_counts = defaultdict(int)
        self.video_source = video_source
        self._init_capture(video_source)
        self.frame_skip = 2
        self.frame_count = 0
        self.last_frame_time = 0
        self.target_fps = 15
        self.is_playing = True
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.5
        self.active_tracks = {}
        logger.info("Tracker initialized successfully")

    def _init_capture(self, video_source):
        try:
            if str(video_source).isdigit():
                video_source = int(video_source)
                logger.info(f"Initializing webcam source: {video_source}")
                self.cam = ThreadedCamera(video_source)
                self.is_threaded = True
            else:
                logger.info(f"Opening video source: {video_source}")
                self.cap = cv2.VideoCapture(video_source)
                if not self.cap.isOpened():
                    raise Exception(f"Failed to open video source: {video_source}")
                self.is_threaded = False
        except Exception as e:
            logger.error(f"Error initializing video source: {e}")
            logger.info("Falling back to webcam...")
            self.cam = ThreadedCamera(0)
            self.is_threaded = True

    def set_source(self, video_source):
        try:
            if hasattr(self, 'cam'):
                self.cam.release()
            if hasattr(self, 'cap'):
                self.cap.release()
            self.video_source = video_source
            self.track_id_count.clear()
            self.live_counts.clear()
            self.active_tracks.clear()
            self._init_capture(video_source)
            return True
        except Exception as e:
            logger.error(f"Error setting video source: {e}")
            return False

    def set_model(self, model_name):
        global current_model_name
        try:
            if model_name in YOLO_MODELS:
                self.model = YOLO(f"yolov8_weights/{YOLO_MODELS[model_name]}")
                self.class_names = self.model.model.names
                current_model_name = model_name
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = max(0.1, min(0.9, threshold))

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        return self.is_playing

    async def get_tracked_frame(self):
        if not self.is_playing:
            return None, None, self.active_tracks

        current_time = time.time()
        if current_time - self.last_frame_time < 1.0 / self.target_fps:
            return None, None, self.active_tracks

        try:
            if self.is_threaded:
                frame = self.cam.read()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video source")
                    return None, None, self.active_tracks

            if frame is None:
                logger.warning("Received null frame")
                return None, None, self.active_tracks

            self.frame_count += 1
            if self.frame_count % (self.frame_skip + 1) != 0:
                return None, None, self.active_tracks

            frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO inference with default tracker
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=None,
                agnostic_nms=True,
                max_det=50,
                tracker="bytetrack.yaml"
            )[0]
            
            self.live_counts = defaultdict(int)
            self.active_tracks = {}
            
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    if conf < self.confidence_threshold:
                        continue
                        
                    class_name = self.class_names[cls]
                    self.live_counts[class_name] += 1
                    
                    if track_id is not None:
                        self.track_id_count[track_id] += 1
                        color = tuple(self.colors[track_id % 100].tolist())
                        
                        # Classify person if detected
                        person_info = {}
                        if class_name == 'person' and person_classifier is not None:
                            person_info = classify_person(frame, (x1, y1, x2, y2))
                        
                        label = f"{class_name} {track_id} {conf:.2f}"
                        if person_info:
                            label += f" ({', '.join([f'{k}: {v:.2f}' for k, v in person_info.items()])})"
                        
                        self.active_tracks[track_id] = {
                            "class": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "person_info": person_info
                        }
                    else:
                        color = (0, 255, 0)
                        label = f"{class_name} {conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            self.last_frame_time = current_time
            return frame, dict(self.live_counts), self.active_tracks
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None, None, self.active_tracks

    def get_analytics(self):
        return {
            "total_unique_ids": len(self.track_id_count),
            "track_appearance_counts": dict(self.track_id_count),
            "active_tracks": self.active_tracks
        }

    def release(self):
        if hasattr(self, 'cam'):
            self.cam.release()
        if hasattr(self, 'cap'):
            self.cap.release()

tracker = Tracker(0)

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = FRONTEND_DIR / "index.html"
    logger.info(f"Serving index.html from: {index_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index.html not found at {index_path}")
    return FileResponse(index_path)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = VIDEOS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        session_id = session_manager.create_session("video", str(file_path))
        session_manager.sessions[session_id]["video_source"] = str(file_path)
        
        success = tracker.set_source(str(file_path))
        if not success:
            return JSONResponse(
                status_code=400,
                content={"message": "Failed to set video source"}
            )
        
        return {"message": "Video uploaded successfully", "session_id": session_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error uploading video: {str(e)}"}
        )

@app.post("/webcam")
async def switch_to_webcam(webcam_id: int = Form(0)):
    try:
        session_id = session_manager.create_session("webcam", webcam_id)
        success = tracker.set_source(webcam_id)
        if not success:
            return JSONResponse(
                status_code=400,
                content={"message": "Failed to switch to webcam"}
            )
        return {"message": "Switched to webcam successfully", "session_id": session_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error switching to webcam: {str(e)}"}
        )

@app.post("/model")
async def change_model(model_name: str = Form(...)):
    try:
        success = tracker.set_model(model_name)
        if not success:
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid model name"}
            )
        return {"message": f"Model changed to {model_name} successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error changing model: {str(e)}"}
        )

@app.post("/confidence")
async def set_confidence(threshold: float = Form(...)):
    try:
        tracker.set_confidence_threshold(threshold)
        return {"message": f"Confidence threshold set to {threshold}"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error setting confidence threshold: {str(e)}"}
        )

@app.post("/control")
async def control_playback(action: str = Form(...)):
    if action == "toggle":
        is_playing = tracker.toggle_playback()
        if not is_playing and session_manager.current_session:
            # Get session summary when pausing
            summary = session_manager.get_session_summary(session_manager.current_session)
            return {
                "is_playing": is_playing,
                "summary": summary
            }
        return {"is_playing": is_playing}
    elif action == "stop":
        # Stop the inference
        tracker.is_playing = False
        if session_manager.current_session:
            # End the current session and get final summary
            summary = session_manager.end_session(session_manager.current_session)
            return {
                "is_playing": False,
                "summary": summary
            }
        return {"is_playing": False}
    return {"error": "Invalid action"}

@app.get("/sessions")
async def get_sessions():
    return session_manager.get_all_sessions()

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    return session_manager.get_session_summary(session_id)

@app.get("/models")
async def get_available_models():
    return {"models": list(YOLO_MODELS.keys())}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection request")
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            frame, live_counts, active_tracks = await tracker.get_tracked_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            if session_manager.current_session:
                session_manager.update_session(session_manager.current_session, live_counts, active_tracks)
            
            try:
                _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                data = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                
                message = {
                    "frame": data,
                    "live_counts": live_counts,
                    "analytics": tracker.get_analytics(),
                    "is_playing": tracker.is_playing,
                    "current_model": current_model_name,
                    "active_tracks": active_tracks
                }
                
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
            
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket: {e}")

@app.on_event("shutdown")
def shutdown_event():
    tracker.release()
    if session_manager.current_session:
        session_manager.end_session(session_manager.current_session)
    analytics = tracker.get_analytics()
    logger.info("\nSession Analytics\n================")
    logger.info(f"Total unique IDs tracked: {analytics['total_unique_ids']}")
    logger.info("Track appearance counts:")
    for tid, count in analytics['track_appearance_counts'].items():
        logger.info(f" - ID {tid}: {count} frames")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 