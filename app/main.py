import os
import cv2
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from ultralytics import YOLO

app = FastAPI(title="Pastor Guardián - Sistema de Alertas de Zorros")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Templates setup
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load YOLO model
# Use trained model if available, otherwise fallback to pre-trained yolo11n
trained_model_path = r"d:\SOLEDAD\runs\detector_zorros_final\weights\best.pt"
if os.path.exists(trained_model_path):
    print(f"Cargando modelo personalizado entrenado: {trained_model_path}")
    model = YOLO(trained_model_path)
else:
    print("Modelo entrenado no encontrado. Usando yolo11n pre-entrenado para demostración.")
    model = YOLO("yolo11n.pt")

# Camera processing loop
class VideoCamera:
    def __init__(self):
        self.cap = None
        self.is_simulation = False
        self.sim_images = []
        self.sim_index = 0
        self.active_users = 0
        
        # Pre-load simulation image list
        dataset_train = r"d:\SOLEDAD\dataset_yolo_consolidado\train\images"
        if os.path.exists(dataset_train):
            self.sim_images = [os.path.join(dataset_train, f) for f in os.listdir(dataset_train) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def start(self):
        self.active_users += 1
        print(f"Usuario WebSocket conectado. Clientes activos: {self.active_users}", flush=True)
        if self.cap is None:
            print("Iniciando recurso de cámara...", flush=True)
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("No se detectó cámara física. Usando simulación con imágenes del dataset...", flush=True)
                self.is_simulation = True
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
            else:
                self.is_simulation = False

    def stop(self):
        self.active_users = max(0, self.active_users - 1)
        print(f"Usuario WebSocket desconectado. Clientes activos: {self.active_users}", flush=True)
        if self.active_users == 0:
            print("Cerrando recurso de cámara (no hay clientes activos)...", flush=True)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_simulation = False

    def get_frame(self):
        if self.is_simulation:
            if not self.sim_images:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No images found in dataset", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return frame
            
            img_path = self.sim_images[self.sim_index]
            frame = cv2.imread(img_path)
            self.sim_index = (self.sim_index + 1) % len(self.sim_images)
            return frame
        else:
            if self.cap is not None:
                success, frame = self.cap.read()
                if success:
                    return frame
            
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Error de camara", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

camera = VideoCamera()

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        camera.start()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        camera.stop()

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

async def generate_frames():
    # Grace period: Wait up to 3 seconds for the dashboard WebSocket connection to establish
    for _ in range(30):
        if camera.active_users > 0:
            break
        await asyncio.sleep(0.1)
    
    if camera.active_users == 0:
        print("No hay usuarios WebSocket activos tras el periodo de gracia. Cancelando transmisión de video.", flush=True)
        return

    while True:
        # If the tab is closed, camera.active_users drops to 0 immediately
        if camera.active_users == 0:
            print("Cerrando la transmisión del feed porque no hay clientes activos.", flush=True)
            break

        frame = camera.get_frame()
        
        # Run YOLO inference
        # Conf threshold: 0.3 for detections
        results = model(frame, conf=0.3, verbose=False)
        detections = []
        zorro_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Check if it is a zorro
                is_alert_target = False
                if label == "zorro" or label == "fox":
                    is_alert_target = True
                    zorro_detected = True
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if is_alert_target else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detections.append({
                    "class": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2 - x1, y2 - y1]
                })

        # Broadcast detection message over WebSocket
        if zorro_detected:
            await manager.broadcast(json.dumps({"alert": True, "message": "¡Zorro Detectado!", "detections": detections}))
        else:
            await manager.broadcast(json.dumps({"alert": False, "message": "Normal", "detections": detections}))

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Keep frame rate reasonable
        await asyncio.sleep(0.08)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
