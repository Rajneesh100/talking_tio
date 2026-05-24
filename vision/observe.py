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


# ====== drawing helpers (live preview window) ======

# MediaPipe's 21-point hand topology, connections between landmark indices.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]


def gesture_label(hand_data):
    """One-line summary of a hand's pose for the on-frame overlay."""
    if hand_data is None:
        return ""
    if hand_data["middle_finger"]:
        return "🖕 middle finger"
    if hand_data["fist"]:
        return "fist"
    if hand_data["open"]:
        return "open"
    extended = [n for n, ext in hand_data["fingers"].items() if ext]
    if extended:
        return "+".join(extended)
    return "—"


def draw_yolo_boxes(frame, yolo_detections):
    for det in yolo_detections:
        if det["object"] == "person":
            continue  # face box covers it; avoid double-boxing
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
        label = f"{det['object']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)


def draw_faces(frame, recognized, face_activity):
    for rec in recognized:
        x1, y1, x2, y2 = rec["head_bbox"]
        color = (50, 205, 50) if rec["name"] != "Unknown" else (60, 60, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name_label = rec["name"]
        if rec.get("score") is not None:
            name_label += f" ({rec['score']:.2f})"
        cv2.putText(frame, name_label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if face_activity:
            face_line = []
            if face_activity["talking"]:
                face_line.append("talking")
            face_line.append(face_activity["head_direction"])
            if face_activity["smiling"]:
                face_line.append("smile")
            cv2.putText(frame, " · ".join(face_line), (x1, min(y2 + 18, frame.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_hand(frame, hand_lm, label_side, hand_data):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
    # Connections (light green)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (150, 230, 150), 2)
    # Landmarks (red dots; tips bigger)
    tip_ids = {4, 8, 12, 16, 20}
    for i, (x, y) in enumerate(pts):
        radius = 5 if i in tip_ids else 3
        cv2.circle(frame, (x, y), radius, (0, 0, 220), -1)
    # State label near wrist
    label = f"{label_side}: {gesture_label(hand_data)}"
    cv2.putText(frame, label, (pts[0][0] - 30, pts[0][1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)


def draw_status_bar(frame, snapshot_count, fps):
    """Top status strip with timestamp and how many detections last got written."""
    h, w = frame.shape[:2]
    bar = frame[:24].copy()
    cv2.rectangle(bar, (0, 0), (w, 24), (30, 30, 30), -1)
    frame[:24] = cv2.addWeighted(frame[:24], 0.4, bar, 0.6, 0)
    text = f"{datetime.now().strftime('%H:%M:%S')}  detections: {snapshot_count}  fps: {fps:.1f}  [q] quit"
    cv2.putText(frame, text, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


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

    show_window = os.environ.get("VISION_DISPLAY", "true").lower() not in {"false", "0", "no"}
    WINDOW_NAME = "Tio Vision"
    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 800, 800)

    print(f"Vision sidecar live. Writing to {OUTPUT_PATH}. "
          f"{'Showing preview (q to quit).' if show_window else 'Headless (VISION_DISPLAY=false).'}",
          flush=True)

    mouth_hist = deque(maxlen=MOUTH_HISTORY)
    last_emit = 0.0
    last_fps_t = time.time()
    fps_count = 0
    fps = 0.0
    stopping = {"flag": False}

    # Cache of the most recent detection results. Drawn on every frame so the
    # window stays smooth even though detection only refreshes at 1Hz.
    cache = {
        "yolo": [],
        "recognized": [],
        "face_activity": None,
        "hands": [],  # list of (label, hand_lm, hand_data)
        "detection_count": 0,
    }

    def _stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopping["flag"]:
            ok, raw_frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Resize once; same frame is reused for inference and display.
            frame = cv2.resize(raw_frame, (FRAME_SIZE, FRAME_SIZE))
            now = time.time()

            # ── 1Hz: heavy work (inference + JSON write) ──
            if now - last_emit >= DETECTION_INTERVAL_S:
                last_emit = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # YOLO — object presence.
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

                # Face recognition (OpenCV Haar + histogram).
                recognized = recognize_faces(frame, known_faces)

                # Face mesh & hand landmarks.
                face_result = face_landmarker.detect(mp_image)
                hand_result = hand_landmarker.detect(mp_image)

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

                # Per-hand classification + cache for drawing.
                left_hand_data = None
                right_hand_data = None
                hands_for_draw = []
                for i, hand_lm in enumerate(hand_result.hand_landmarks):
                    label = hand_result.handedness[i][0].category_name  # "Left" | "Right"
                    data = analyze_hand(hand_lm)
                    hands_for_draw.append((label, hand_lm, data))
                    if label.lower() == "left":
                        left_hand_data = data
                    else:
                        right_hand_data = data

                # Build people_data the same way as before.
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

                # Merge into final detections.
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

                # Refresh the draw cache with the new detection batch.
                cache["yolo"] = yolo_detections
                cache["recognized"] = recognized
                cache["face_activity"] = face_activity
                cache["hands"] = hands_for_draw
                cache["detection_count"] = len(detections)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(detections)} detection(s): "
                      f"{', '.join(d['object'] for d in detections) or 'none'}",
                      flush=True)

            # ── every frame: draw + show ──
            if show_window:
                # FPS bookkeeping.
                fps_count += 1
                if now - last_fps_t >= 1.0:
                    fps = fps_count / (now - last_fps_t)
                    fps_count = 0
                    last_fps_t = now

                annotated = frame.copy()
                draw_yolo_boxes(annotated, cache["yolo"])
                draw_faces(annotated, cache["recognized"], cache["face_activity"])
                for label_side, hand_lm, hand_data in cache["hands"]:
                    draw_hand(annotated, hand_lm, label_side, hand_data)
                draw_status_bar(annotated, cache["detection_count"], fps)

                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stopping["flag"] = True
            else:
                # No window — small sleep so we don't busy-spin between detections.
                time.sleep(0.05)

    finally:
        cap.release()
        face_landmarker.close()
        hand_landmarker.close()
        if show_window:
            cv2.destroyAllWindows()
        print("vision sidecar shutting down.", flush=True)


if __name__ == "__main__":
    main()
