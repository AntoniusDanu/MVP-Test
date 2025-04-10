import os
import time
import cv2
from datetime import datetime
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
        print("[PROCESS] Simulasi tidak berjalan atau dihentikan")  # ⬅️ log debug
        return
    if state["image_queue"]:
        image = state["image_queue"].pop(0)
        path = os.path.join(UPLOAD_DIR, image)
        print(f"[PROCESS] Memproses gambar: {image}")  # ⬅️ log debug
        if not os.path.exists(path):
            print(f"[ERROR] File tidak ditemukan: {path}")  # ⬅️ log debug
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        state["log"].append(f"[{timestamp}] Upload diproses: {image}")
        plate = detect_plate(path)
        for i in range(5):
            if state["pit_log"][i] == "Empty":
                state["pit_log"][i] = f"{timestamp} - {plate}"
                state["summary"].append(f"PIT {i+1}: {timestamp} - {plate}")
                state["log"].append(f"[{timestamp}] PIT {i+1} ⬅ {plate}")
                print(f"[PROCESS] PIT {i+1} => {plate}")  # ⬅️ log debug
                return
    else:
        print("[PROCESS] Queue kosong")  # ⬅️ log debug


# --- Background Loop ---
def auto_loop():
    while True:
        if state["simulation_running"] and not state["force_stop"]:
            print("[LOOP] Simulasi aktif...")  # ⬅️ log debug
            if all(slot != "Empty" for slot in state["pit_log"]):
                state["simulation_running"] = False
                state["finished"] = True
                state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Semua PIT terisi. Simulasi selesai.")
                print("[LOOP] Semua PIT sudah terisi, simulasi dihentikan")  # ⬅️ log debug
                continue

            now = time.time()
            if (now - state["last_process_time"]) >= 120:
                print("[LOOP] Memanggil process_image()")  # ⬅️ log debug
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
    return JSONResponse({
        "pit_log": state["pit_log"],
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
        print(f"[UPLOAD] {filename} ditambahkan ke queue")  # ⬅️ log debug
        state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Upload: {filename}")
    return {"status": "uploaded"}


@app.post("/start")
async def start_simulasi():
    state["simulation_running"] = True
    state["force_stop"] = False
    state["finished"] = False
    state["last_process_time"] = time.time() - 120  # agar langsung jalan
    state["image_queue"].sort()
    print(f"[START] Simulasi dimulai. Queue: {state['image_queue']}")  # ⬅️ log debug
    state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Simulasi dimulai")
    return {"status": "started"}


@app.post("/stop")
async def stop_simulasi():
    state["simulation_running"] = False
    state["force_stop"] = True
    state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Simulasi dihentikan")
    return {"status": "stopped"}

@app.post("/reset")
async def reset_simulasi():
    state["pit_log"] = ["Empty"] * 5
    state["summary"] = []
    state["log"] = []
    state["image_queue"] = []
    state["simulation_running"] = False
    state["force_stop"] = True
    state["finished"] = False
    for filename in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, filename))
        except:
            pass
    return {"status": "reset"}
    
# Mulai background thread auto_loop
threading.Thread(target=auto_loop, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
