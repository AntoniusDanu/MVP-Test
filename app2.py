import os
import time
import cv2
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
import threading

# --- Setup ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models ---
yolo_model = YOLO("best.pt")
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# --- Runtime States ---
state = {
    "pit_log": ["Empty"] * 5,
    "pit_time": [None] * 5,
    "summary": [],
    "log": [],
    "image_queue": [],
    "simulation_running": False,
    "force_stop": False,
    "last_process_time": 0,
    "finished": False
}

# --- Helpers ---
def detect_plate(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Gagal Membaca"
        image = cv2.resize(image, (640, 640))
        results = yolo_model(image)
        if not results or len(results[0].boxes) == 0:
            return "Tidak Terbaca"
        boxes = results[0].boxes.xyxy.numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            return "Tidak Terbaca"
        ocr_results = ocr_model.ocr(plate_img, cls=True)
        return ocr_results[0][0][1][0] if ocr_results and ocr_results[0] else "Tidak Terbaca"
    except Exception as e:
        return f"Error: {str(e)}"

def process_image():
    if not state["simulation_running"] or state["force_stop"]:
        return
    if not state["image_queue"]:
        print("[PROCESS] Queue kosong")
        return

    image = state["image_queue"].pop(0)
    path = os.path.join(UPLOAD_DIR, image)
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        return

    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    timestamp_str = now.strftime("%H:%M:%S")
    plate = detect_plate(path)
    state["log"].append(f"[{timestamp_str}] Diproses: {image} ➜ {plate}")

    i = process_image.counter % 5
    process_image.counter += 1

    if state["pit_log"][i] != "Empty":
        masuk = state["pit_time"][i]
        durasi = (now - masuk).total_seconds() if masuk else 0
        hours = int(durasi // 3600)
        minutes = int((durasi % 3600) // 60)
        seconds = int(durasi % 60)
        durasi_str = f"{hours:02}:{minutes:02}:{seconds:02}"

        out_log = f"{timestamp_str} - OUT: {state['pit_log'][i]} (Durasi: {durasi_str})"

        state["summary"].append(f"PIT {i+1}: {out_log}")
        state["log"].append(f"[{timestamp_str}] PIT {i+1} digantikan")

    state["pit_log"][i] = plate
    state["pit_time"][i] = now
    state["log"].append(f"[{timestamp_str}] PIT {i+1} ⬅ {plate}")
    print(f"[PROCESS] PIT {i+1} => {plate}")

process_image.counter = 0

# --- Background Loop ---
def auto_loop():
    while True:
        if state["simulation_running"] and not state["force_stop"]:
            print("[LOOP] Simulasi aktif...")
            now = time.time()
            if (now - state["last_process_time"]) >= 120:
                print("[LOOP] Memanggil process_image()")
                process_image()
                state["last_process_time"] = now
        time.sleep(1)

# --- API Endpoints ---
@app.get("/")
def serve_frontend():
    if os.path.exists("frontend.html"):
        with open("frontend.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return {"message": "ALPR Backend is running"}

@app.get("/state")
def get_state():
    pit_status = []
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    for i in range(5):
        if state["pit_log"][i] != "Empty" and state["pit_time"][i]:
            elapsed = (now - state["pit_time"][i]).total_seconds()

            # Format ke HH:MM:SS
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            durasi_str = f"{hours:02}:{minutes:02}:{seconds:02}"

            pit_status.append(f"{state['pit_log'][i]} ({durasi_str})")
        else:
            pit_status.append("Empty")

    return JSONResponse({
        "pit_log": pit_status,
        "summary": state["summary"][-10:],
        "log": state["log"][-20:],
        "finished": state["finished"]
    })

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        filename = file.filename
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
            f.write(await file.read())
        state["image_queue"].append(filename)
        print(f"[UPLOAD] {filename} ditambahkan ke queue")
        state["log"].append(f"[{datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%H:%M:%S')}] Upload: {filename}")
    return {"status": "uploaded"}

@app.post("/start")
async def start_simulasi():
    state["simulation_running"] = True
    state["force_stop"] = False
    state["finished"] = False
    state["last_process_time"] = time.time() - 120
    state["image_queue"].sort()
    print(f"[START] Simulasi dimulai. Queue: {state['image_queue']}")
    state["log"].append(f"[{datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%H:%M:%S')}] Simulasi dimulai")
    return {"status": "started"}

@app.post("/stop")
async def stop_simulasi():
    state["simulation_running"] = False
    state["force_stop"] = True
    state["log"].append(f"[{datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%H:%M:%S')}] Simulasi dihentikan")
    return {"status": "stopped"}

@app.post("/reset")
async def reset_simulasi():
    state["pit_log"] = ["Empty"] * 5
    state["pit_time"] = [None] * 5
    state["summary"] = []
    state["log"] = []
    state["image_queue"] = []
    state["simulation_running"] = False
    state["force_stop"] = True
    state["finished"] = False
    process_image.counter = 0
    for filename in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, filename))
        except:
            pass
    return {"status": "reset"}

threading.Thread(target=auto_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
