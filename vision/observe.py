"""
Vision sidecar for talking_tio.

Runs the laptop camera at ~1Hz, extracts a rich set of signals (presence,
face activity, hand gestures, face recognition against vision/images/) and
writes a rolling JSON history to ./visual_context.json.

Output schema (matches the local-talking-llm reference for compatibility):

    {
      "visual_history": [
        {
          "timestamp": "ISO-8601",
          "detections": [
            {
              "object": "person",
              "person_data": {
                "name": "Rajneesh" | "Unknown",
                "face": {"smiling": bool, "talking": bool, "head_direction": "left|center|right"},
                "left_hand":  null | {fingers, open, fist, middle_finger, present: true},
                "right_hand": null | {...},
                "head_bbox":  [x1,y1,x2,y2] | null
              },
              "confidence": 0.79,
              "bbox": [x1,y1,x2,y2]
            },
            {"object": "book", "confidence": 0.44, "bbox": [...]}
          ]
        },
        ... (last 5 frames)
      ]
    }

Components:
  - YOLOv8n      → general object presence (COCO classes)
  - FaceLandmarker (mediapipe Tasks) → face mesh → talking / head_dir / smile
  - HandLandmarker (mediapipe Tasks) → 21-point hands → per-finger extension
  - OpenCV Haar  → face bbox + histogram-based name recognition against
                    images/<name>.{jpg,png}

Run: source vision/.venv/bin/activate && python vision/observe.py
"""

import json
import math
import os
import signal
import sys
import tempfile
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO


# ====== paths ======

VISION_DIR = Path(__file__).parent
OUTPUT_PATH = VISION_DIR / "visual_context.json"
IMAGES_DIR = VISION_DIR / "images"

MODELS_DIR = VISION_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# ====== tunables ======

DETECTION_INTERVAL_S = 1.0
HISTORY_LEN = 5
MOUTH_HISTORY = 10
MOUTH_TALKING_THRESHOLD = 3  # consecutive open-mouth frames to count as talking
FACE_RECOG_MIN_SCORE = 0.5   # histogram correlation threshold for known face

# Frame is resized to 512x512 for inference — keeps YOLO/MediaPipe fast.
FRAME_SIZE = 512


# ====== model downloading ======

def _download(url: str, dest: Path, min_bytes: int = 100_000):
    if dest.exists() and dest.stat().st_size > min_bytes:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} → {dest} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"  done ({dest.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)


def ensure_models():
    _download(YOLO_MODEL_URL, YOLO_MODEL_PATH, min_bytes=1_000_000)
    _download(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH, min_bytes=1_000_000)
    _download(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH, min_bytes=1_000_000)


# ====== face recognition (OpenCV Haar + histogram) ======

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_known_faces(images_dir: Path):
    """Read each image in `images_dir`, extract the largest face, compute its
    grayscale histogram. The filename (without extension) is the person's name."""
    known = []
    if not images_dir.exists():
        print(f"images dir {images_dir} not found — skipping face recognition", flush=True)
        return known
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ✗ could not read {path.name}", flush=True)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            print(f"  ✗ no face detected in {path.name}", flush=True)
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_img = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        known.append({
            "name": path.stem,
            "histogram": hist,
        })
        print(f"  ✓ loaded face: {path.stem}", flush=True)
    print(f"Loaded {len(known)} known face(s)", flush=True)
    return known


def recognize_faces(frame, known):
    """Detect faces in frame with OpenCV, compare against known histograms."""
    out = []
    if not known:
        return out
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        best_name, best_score = None, -1.0
        for k in known:
            score = cv2.compareHist(hist, k["histogram"], cv2.HISTCMP_CORREL)
            if score > best_score:
                best_name, best_score = k["name"], float(score)
        name = best_name if best_score > FACE_RECOG_MIN_SCORE else "Unknown"
        out.append({
            "name": name,
            "head_bbox": [int(x), int(y), int(x + w), int(y + h)],
            "score": round(best_score, 3),
        })
    return out


# ====== face-mesh signals ======

def _d2(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def head_direction(face_lm):
    """left | center | right from nose-vs-eye-mid."""
    nose = face_lm[1]
    left = face_lm[33]
    right = face_lm[263]
    if nose.x < left.x:
        return "left"
    if nose.x > right.x:
        return "right"
    return "center"


def mouth_open(face_lm):
    mouth_h = _d2(face_lm[13], face_lm[14])
    face_w = _d2(face_lm[234], face_lm[454])
    return (mouth_h / face_w) > 0.03 if face_w else False


def smiling(face_lm):
    mouth_w = _d2(face_lm[61], face_lm[291])
    face_w = _d2(face_lm[234], face_lm[454])
    return (mouth_w / face_w) > 0.38 if face_w else False


# ====== hand signals ======

WRIST = 0
FINGERS = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}


def _d3(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def finger_extended(hand_lm, tip, pip):
    """True if the tip is farther from the wrist than the PIP joint — proxy for extended."""
    return _d3(hand_lm[tip], hand_lm[WRIST]) > _d3(hand_lm[pip], hand_lm[WRIST])


def analyze_hand(hand_lm):
    fingers = {n: finger_extended(hand_lm, tip, pip) for n, (tip, pip) in FINGERS.items()}
    return {
        "present": True,
        "fingers": fingers,
        "open": sum(fingers.values()) >= 4,
        "fist": sum(fingers.values()) == 0,
        "middle_finger": (
            fingers["middle"]
            and not fingers["index"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ),
    }


# ====== json output ======

def atomic_write_json(path: Path, payload: dict):
    fd, tmp_path = tempfile.mkstemp(prefix=".visual_context.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_history():
    if not OUTPUT_PATH.exists():
        return {"visual_history": []}
    try:
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        if "visual_history" not in data:
            data = {"visual_history": []}
        return data
    except Exception:
        return {"visual_history": []}


def append_frame(detections):
    data = load_history()
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "detections": detections,
    }
    data["visual_history"].append(entry)
    data["visual_history"] = data["visual_history"][-HISTORY_LEN:]
    atomic_write_json(OUTPUT_PATH, data)


# ====== main loop ======

def main():
    ensure_models()
    known_faces = load_known_faces(IMAGES_DIR)

    print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...", flush=True)
    yolo = YOLO(str(YOLO_MODEL_PATH))

    print(f"Loading FaceLandmarker from {FACE_LANDMARKER_PATH} ...", flush=True)
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
    )

    print(f"Loading HandLandmarker from {HAND_LANDMARKER_PATH} ...", flush=True)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
    )

    print("Opening camera...", flush=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera 0", file=sys.stderr)
        sys.exit(1)

    print(f"Vision sidecar live. Writing to {OUTPUT_PATH}. Ctrl-C to stop.", flush=True)

    mouth_hist = deque(maxlen=MOUTH_HISTORY)
    last_emit = 0.0
    stopping = {"flag": False}

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopping["flag"]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_emit < DETECTION_INTERVAL_S:
                continue
            last_emit = now

            # Resize for consistent fast inference.
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # 1. YOLO — object presence.
            yolo_detections = []
            yolo_results = yolo.predict(frame, verbose=False, imgsz=FRAME_SIZE, conf=0.3)
            for r in yolo_results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    yolo_detections.append({
                        "object": yolo.names[cls],
                        "confidence": round(conf, 2),
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    })

            # 2. Face recognition (OpenCV Haar + histogram).
            recognized = recognize_faces(frame, known_faces)

            # 3. Face mesh & hand landmarks.
            face_result = face_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            # Aggregate face activity from the first detected face (single-person
            # assumption; refine later if multi-person becomes important).
            face_activity = None
            if face_result.face_landmarks:
                f = face_result.face_landmarks[0]
                mouth_hist.append(mouth_open(f))
                talking = sum(1 for v in mouth_hist if v) >= MOUTH_TALKING_THRESHOLD
                face_activity = {
                    "smiling": bool(smiling(f)),
                    "talking": bool(talking),
                    "head_direction": head_direction(f),
                }

            # Split hands by handedness label from the model.
            left_hand_data = None
            right_hand_data = None
            for i, hand_lm in enumerate(hand_result.hand_landmarks):
                # handedness[i] is a list of Category; the top one carries the label.
                label = hand_result.handedness[i][0].category_name  # "Left" or "Right"
                # Camera-flipped: MediaPipe reports the hand from the SUBJECT's
                # perspective by default. For a selfie cam the user's left hand
                # appears on the right of the image — we still record it as "left".
                data = analyze_hand(hand_lm)
                if label.lower() == "left":
                    left_hand_data = data
                else:
                    right_hand_data = data

            # Build people entries. If we recognized one or more faces, each one
            # gets the (shared) face_activity + hand data. If MediaPipe sees a
            # face but Haar didn't recognize anyone, emit a single "Unknown".
            people_data = []
            for rec in recognized:
                people_data.append({
                    "name": rec["name"],
                    "face": face_activity,
                    "left_hand": left_hand_data,
                    "right_hand": right_hand_data,
                    "head_bbox": rec["head_bbox"],
                })

            if not people_data and (face_activity or left_hand_data or right_hand_data):
                people_data.append({
                    "name": "Unknown",
                    "face": face_activity,
                    "left_hand": left_hand_data,
                    "right_hand": right_hand_data,
                    "head_bbox": None,
                })

            # 4. Merge YOLO + people_data into the final detections array.
            # Strategy: replace YOLO "person" rows with detailed people_data
            # entries; keep all non-person YOLO objects as-is.
            detections = []
            yolo_person_used = False
            for det in yolo_detections:
                if det["object"] == "person":
                    if people_data and not yolo_person_used:
                        for p in people_data:
                            detections.append({
                                "object": "person",
                                "person_data": p,
                                "confidence": det["confidence"],
                                "bbox": det["bbox"],
                            })
                        yolo_person_used = True
                    elif not people_data:
                        detections.append({
                            "object": "person",
                            "person_data": None,
                            "confidence": det["confidence"],
                            "bbox": det["bbox"],
                        })
                else:
                    detections.append(det)

            # If MediaPipe detected face/hands but YOLO didn't fire on "person"
            # this frame, still surface the people_data entries.
            if people_data and not yolo_person_used:
                for p in people_data:
                    bbox = p.get("head_bbox") or [0, 0, 0, 0]
                    detections.append({
                        "object": "person",
                        "person_data": p,
                        "confidence": 0.30,
                        "bbox": bbox,
                    })

            try:
                append_frame(detections)
            except Exception as e:
                print(f"write failed: {e}", file=sys.stderr)
                continue

            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(detections)} detection(s): "
                  f"{', '.join(d['object'] for d in detections) or 'none'}",
                  flush=True)

    finally:
        cap.release()
        face_landmarker.close()
        hand_landmarker.close()
        print("vision sidecar shutting down.", flush=True)


if __name__ == "__main__":
    main()
