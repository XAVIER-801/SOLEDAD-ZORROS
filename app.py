import os
import cv2
import json
import asyncio
import base64
import io
import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI(
    title="Pastor Guardián - Sistema de Alertas de Zorros",
    description="Sistema de detección de zorros en tiempo real para protección de rebaños",
    version="2.0.0",
)

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "detections.db"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
STATIC_DIR.mkdir(exist_ok=True)

# ─── Modelo unificado (5 clases) ───
CLASS_NAMES = ["zorro", "oveja", "gallina", "cuy", "conejo"]
ALERT_CLASS_ID = 0  # zorro = alerta

UNIFIED_MODEL_PATH = BASE_DIR / "output" / "unificado" / "best.pt"
model = None
model_name = None

if UNIFIED_MODEL_PATH.exists():
    print(f"Cargando modelo unificado: {UNIFIED_MODEL_PATH}")
    model = YOLO(str(UNIFIED_MODEL_PATH))
    model_name = "unificado"
else:
    print("Modelo unificado no encontrado. Usando yolo11n.pt (solo demo).")
    model = YOLO("yolo11n.pt")
    model_name = "pretrained (demo)"

# ─── Configuración de detección ───
CLASS_EMOJIS = {0: "🦊", 1: "🐑", 2: "🐔", 3: "🐹", 4: "🐇"}

class DetectionConfig:
    def __init__(self):
        self.conf_threshold = 0.25
        self.alert_conf_threshold = 0.7
        self.iou_threshold = 0.45
        self.alert_class_id = 0  # zorro
        self.enable_sound_alert = True
        self.recording_enabled = True
        self.max_history = 1000

    def to_dict(self):
        return {
            "conf_threshold": self.conf_threshold,
            "alert_conf_threshold": self.alert_conf_threshold,
            "iou_threshold": self.iou_threshold,
            "alert_class": CLASS_NAMES[self.alert_class_id],
            "classes": CLASS_NAMES,
            "enable_sound_alert": self.enable_sound_alert,
            "recording_enabled": self.recording_enabled,
            "max_history": self.max_history,
        }


config = DetectionConfig()

# ─── Base de datos SQLite para historial ───
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            animal TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            alert INTEGER NOT NULL DEFAULT 0,
            image_base64 TEXT,
            bbox TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS config_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ─── VideoCamera mejorada ───
class VideoCamera:
    def __init__(self):
        self.cap = None
        self.camera_id = 0
        self.is_simulation = False
        self.sim_images = []
        self.sim_index = 0
        self.active_users = 0
        self.resolution = (640, 480)
        self.fps = 12

        self._load_sim_images()

    def _load_sim_images(self):
        unified_train = BASE_DIR / "dataset_unificado" / "train" / "images"
        if unified_train.exists():
            self.sim_images = [
                str(f) for f in unified_train.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        if not self.sim_images:
            for animal in ["ZORROS", "OVEJAS", "GALLINAS", "CUYES", "CONEJOS"]:
                img_dir = BASE_DIR / animal / "NORMAL" / "train"
                if img_dir.exists():
                    self.sim_images.extend([
                        str(f) for f in img_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ])
        random.shuffle(self.sim_images)

    def start(self):
        self.active_users += 1
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.is_simulation = True
                if self.cap:
                    self.cap.release()
                self.cap = None
            else:
                self.is_simulation = False

    def stop(self):
        self.active_users = max(0, self.active_users - 1)
        if self.active_users == 0:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_simulation = False

    def get_frame(self):
        if self.is_simulation:
            if not self.sim_images:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No hay imagenes disponibles", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return frame
            img_path = self.sim_images[self.sim_index % len(self.sim_images)]
            self.sim_index += 1
            frame = cv2.imread(img_path)
            if frame is None:
                return self.get_frame()
            if self.resolution and self.resolution != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, self.resolution)
            return frame

        if self.cap:
            success, frame = self.cap.read()
            if success:
                if self.resolution and self.resolution != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, self.resolution)
                return frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Error de camara", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame


camera = VideoCamera()

# ─── Connection Manager ───
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        camera.start()

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
        camera.stop()

    async def broadcast(self, message: str):
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()

# ─── Procesamiento de video ───
async def generate_frames():
    camera.start()
    try:
        while True:
            if camera.active_users == 0:
                await asyncio.sleep(0.1)
                continue

            frame = camera.get_frame()
            results = model(frame, conf=config.conf_threshold, verbose=False)
            detections = []
            alert_detected = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                    conf = float(box.conf[0])

                    is_alert = (cls_id == config.alert_class_id and conf >= config.alert_conf_threshold)
                    if is_alert:
                        alert_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 0, 255) if is_alert else (0, 255, 0)
                    emoji = CLASS_EMOJIS.get(cls_id, "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    text = f"{emoji} {label} {conf:.2f}"
                    if is_alert:
                        text = f"🔴 {text}"
                    cv2.putText(frame, text, (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    detections.append({
                        "class": label,
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "alert": is_alert,
                    })

            msg = json.dumps({
                "alert": alert_detected,
                "detections": detections,
            })
            await manager.broadcast(msg)

            if alert_detected and config.recording_enabled:
                _save_detection(detections, frame)

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            await asyncio.sleep(1.0 / camera.fps)
    finally:
        camera.stop()


def _save_detection(detections, frame):
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        for d in detections:
            if d["alert"]:
                c.execute(
                    "INSERT INTO detections (id, timestamp, animal, label, confidence, alert, image_base64, bbox) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        datetime.now().isoformat(),
                        CLASS_NAMES[d.get("class_id", 0)] if d.get("class_id", 0) < len(CLASS_NAMES) else "unknown",
                        d["class"],
                        d["confidence"],
                        img_b64,
                        json.dumps(d["bbox"]),
                    ),
                )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error guardando detección: {e}")


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_name": model_name or "Ninguno",
            "model_classes": CLASS_NAMES,
            "config": config.to_dict(),
            "class_emojis": CLASS_EMOJIS,
            "camera_mode": "simulacion" if camera.is_simulation else "fisica",
        },
    )


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    return templates.TemplateResponse(
        request,
        "config.html",
        {
            "model_name": model_name,
            "config": config.to_dict(),
            "class_emojis": CLASS_EMOJIS,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(
        "SELECT id, timestamp, animal, label, confidence, alert FROM detections ORDER BY timestamp DESC LIMIT 100"
    )
    rows = c.fetchall()
    conn.close()
    detections_list = [
        {
            "id": r[0],
            "timestamp": r[1],
            "animal": r[2],
            "label": r[3],
            "confidence": r[4],
            "alert": bool(r[5]),
        }
        for r in rows
    ]
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "detections": detections_list,
        },
    )


# ─── API ───

@app.get("/api/stats")
async def get_stats():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detections WHERE alert = 1")
    alerts = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT timestamp) FROM detections")
    sessions = c.fetchone()[0]
    conn.close()
    return {
        "total_detections": total,
        "total_alerts": alerts,
        "detection_sessions": sessions,
        "model": model_name or "pretrained",
        "classes": CLASS_NAMES,
        "alert_class": CLASS_NAMES[ALERT_CLASS_ID],
        "camera_mode": "simulacion" if camera.is_simulation else "fisica",
    }


@app.post("/api/switch_model")
async def switch_model():
    return {"success": False, "error": "Modelo unificado — no requiere cambio"}


@app.post("/api/update_config")
async def update_config(data: dict = None):
    if data:
        if "conf_threshold" in data:
            config.conf_threshold = float(data["conf_threshold"])
        if "alert_conf_threshold" in data:
            config.alert_conf_threshold = float(data["alert_conf_threshold"])
        if "iou_threshold" in data:
            config.iou_threshold = float(data["iou_threshold"])
        if "enable_sound_alert" in data:
            config.enable_sound_alert = bool(data["enable_sound_alert"])
        if "recording_enabled" in data:
            config.recording_enabled = bool(data["recording_enabled"])
    return {"success": True, "config": config.to_dict()}


@app.get("/api/history")
async def get_history(limit: int = 50, offset: int = 0, alert_only: bool = False):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    if alert_only:
        c.execute(
            "SELECT id, timestamp, animal, label, confidence, alert FROM detections WHERE alert = 1 ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    else:
        c.execute(
            "SELECT id, timestamp, animal, label, confidence, alert FROM detections ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "animal": r[2],
            "label": r[3],
            "confidence": r[4],
            "alert": bool(r[5]),
        }
        for r in rows
    ]


@app.get("/api/detection_image/{detection_id}")
async def get_detection_image(detection_id: str):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT image_base64 FROM detections WHERE id = ?", (detection_id,))
    row = c.fetchone()
    conn.close()
    if row and row[0]:
        img_bytes = base64.b64decode(row[0])
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    return JSONResponse({"error": "No encontrada"}, status_code=404)


@app.post("/detect_frame")
async def detect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Imagen inválida"}, status_code=400)

    results = model(img, conf=config.conf_threshold, verbose=False)
    detections = []
    alert_detected = False
    H, W = img.shape[:2]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            conf = float(box.conf[0])
            is_alert = (cls_id == config.alert_class_id and conf >= config.alert_conf_threshold)
            if is_alert:
                alert_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class": label,
                "class_id": cls_id,
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "alert": is_alert,
            })

    return JSONResponse({
        "alert": alert_detected,
        "detections": detections,
        "width": W,
        "height": H,
    })


@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Imagen inválida"}, status_code=400)

    results = model(img, conf=config.conf_threshold, verbose=False)
    detections = []
    alert_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            conf = float(box.conf[0])
            is_alert = (cls_id == config.alert_class_id and conf >= config.alert_conf_threshold)
            if is_alert:
                alert_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            emoji = CLASS_EMOJIS.get(cls_id, "")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            text = f"{emoji} {label} {conf:.2f}"
            if is_alert:
                text = f"🔴 {text}"
            cv2.putText(img, text, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detections.append({
                "class": label,
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "alert": is_alert,
            })

    _, buffer = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "alert": alert_detected,
        "detections": detections,
        "image": img_b64,
    })


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "ping":
                    await ws.send_text(json.dumps({"status": "ok"}))
                elif action == "client_frame" and "image" in msg:
                    # Decodificar el frame del cliente
                    img_bytes = base64.b64decode(msg["image"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Procesar con YOLO
                        results = model(frame, conf=config.conf_threshold, verbose=False)
                        detections = []
                        alert_detected = False
                        
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                                conf = float(box.conf[0])
                                
                                is_alert = (cls_id == config.alert_class_id and conf >= config.alert_conf_threshold)
                                if is_alert:
                                    alert_detected = True
                                    
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                color = (0, 0, 255) if is_alert else (0, 255, 0)
                                emoji = CLASS_EMOJIS.get(cls_id, "")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                text = f"{emoji} {label} {conf:.2f}"
                                if is_alert:
                                    text = f"🔴 {text}"
                                cv2.putText(frame, text, (x1, max(y1 - 10, 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                detections.append({
                                    "class": label,
                                    "class_id": cls_id,
                                    "confidence": conf,
                                    "alert": is_alert
                                })
                        
                        # Si hay alerta, guardar en base de datos si está configurado
                        if alert_detected and config.recording_enabled:
                            _, buffer = cv2.imencode(".jpg", frame)
                            img_b64 = base64.b64encode(buffer).decode("utf-8")
                            asyncio.create_task(save_detection_async(detections, frame))
                        
                        # Codificar de vuelta a JPG en base64 para el cliente
                        _, buffer = cv2.imencode(".jpg", frame)
                        frame_b64 = base64.b64encode(buffer).decode("utf-8")
                        
                        # Enviar respuesta al cliente
                        await ws.send_text(json.dumps({
                            "alert": alert_detected,
                            "detections": detections,
                            "image": frame_b64
                        }))
            except Exception as e:
                print(f"Error procesando frame del cliente: {e}")
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Pastor Guardián - Sistema de Alertas")
    print(f"Modelo activo: {model_name}")
    print(f"Clases: {CLASS_NAMES}")
    print(f"Modo cámara: {'simulación' if camera.is_simulation else 'física'}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
