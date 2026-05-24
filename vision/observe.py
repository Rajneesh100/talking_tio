"""
Vision sidecar for talking_tio.

Runs the laptop camera at ~1Hz, extracts a small set of signals useful for
deciding whether ambient speech is directed at Angela, and writes the result
to ./visual_context.json (atomic temp+rename).

Signals produced per frame:
  - people: list of [{talking, head_direction, smiling}]
  - other_objects: COCO labels (e.g. "cell phone", "book") near a detected person
  - summary: one-line human-readable string the Go side injects into the prompt

The Go agent reads the JSON whenever it builds a user-turn prompt. If the file
is missing or stale (> 5s), the agent simply omits the visual block — vision
is a strict enhancement, not a hard dependency.

Run: source vision/.venv/bin/activate && python vision/observe.py
"""

import json
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
from ultralytics import YOLO

VISION_DIR = Path(__file__).parent
OUTPUT_PATH = VISION_DIR / "visual_context.json"

# Pin the model inside the repo so the user's HF / ultralytics cache being
# evicted doesn't force a redownload. Mirror of the stt/models/ pattern.
MODELS_DIR = VISION_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"


def ensure_yolo_model():
    """Download yolov8n.pt to vision/models/ if it's not already there.
    Idempotent — no-op when the file is present."""
    if YOLO_MODEL_PATH.exists() and YOLO_MODEL_PATH.stat().st_size > 1_000_000:
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {YOLO_MODEL_PATH.name} → {YOLO_MODEL_PATH} ...", flush=True)
    tmp = YOLO_MODEL_PATH.with_suffix(".pt.partial")
    try:
        urllib.request.urlretrieve(YOLO_MODEL_URL, tmp)
        os.replace(tmp, YOLO_MODEL_PATH)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    size_mb = YOLO_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  done ({size_mb:.1f} MB)", flush=True)

DETECTION_INTERVAL_S = 1.0
MOUTH_HISTORY = 10
MOUTH_TALKING_THRESHOLD = 3  # consecutive open-mouth frames to count as talking

# COCO classes worth surfacing to the agent (filtered from YOLO output).
RELEVANT_OBJECTS = {
    "person",
    "cell phone",
    "laptop",
    "tv",
    "remote",
    "book",
    "cup",
    "wine glass",
    "bottle",
}


def head_direction(face_landmarks, frame_w):
    """Classify head as 'left' | 'center' | 'right' from nose vs eye centers."""
    if not face_landmarks:
        return "unknown"
    # MediaPipe face mesh landmark indices: 1=nose tip, 33=left eye, 263=right eye.
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    dx = nose.x - eye_mid_x
    if dx < -0.025:
        return "left"
    if dx > 0.025:
        return "right"
    return "center"


def mouth_open(face_landmarks):
    """Returns True if the gap between upper and lower lip is > 1.5% of face height."""
    if not face_landmarks:
        return False
    # 13 = upper inner lip, 14 = lower inner lip.
    upper = face_landmarks.landmark[13]
    lower = face_landmarks.landmark[14]
    return abs(upper.y - lower.y) > 0.015


def smiling(face_landmarks):
    """Crude: horizontal mouth corners wider than vertical opening."""
    if not face_landmarks:
        return False
    left = face_landmarks.landmark[61]
    right = face_landmarks.landmark[291]
    top = face_landmarks.landmark[13]
    bot = face_landmarks.landmark[14]
    horiz = abs(left.x - right.x)
    vert = abs(top.y - bot.y)
    return horiz > 0.07 and horiz > vert * 4.0


def build_summary(people, objects):
    """One-line human-readable summary the Go side injects directly into the prompt."""
    if not people:
        if objects:
            return f"no one in frame; objects: {', '.join(objects)}"
        return "no one in frame"

    pieces = [f"{len(people)} person" if len(people) == 1 else f"{len(people)} people"]

    if len(people) == 1:
        p = people[0]
        if p["talking"]:
            pieces.append("talking")
        if p["head_direction"] == "center":
            pieces.append("looking at camera")
        elif p["head_direction"] in ("left", "right"):
            pieces.append(f"looking {p['head_direction']}")
        if p["smiling"]:
            pieces.append("smiling")

    if objects:
        pieces.append(f"nearby: {', '.join(objects)}")

    return ", ".join(pieces)


def atomic_write_json(path: Path, payload: dict):
    """Write to a temp file then rename, so consumers never read a half-written file."""
    fd, tmp_path = tempfile.mkstemp(prefix=".visual_context.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main():
    ensure_yolo_model()
    print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...", flush=True)
    yolo = YOLO(str(YOLO_MODEL_PATH))

    print("Loading MediaPipe face mesh...", flush=True)
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    print("Opening camera...", flush=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera 0", file=sys.stderr)
        sys.exit(1)

    print(f"Vision sidecar live. Writing to {OUTPUT_PATH}. Ctrl-C to stop.", flush=True)

    mouth_hist = deque(maxlen=MOUTH_HISTORY)
    last_emit = 0.0

    # Handle Ctrl-C cleanly.
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

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO: object presence
            yolo_results = yolo.predict(frame, verbose=False, imgsz=512, conf=0.4)
            objects_found = set()
            person_count_yolo = 0
            for r in yolo_results:
                for c, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    label = yolo.names[int(c)]
                    if label == "person":
                        person_count_yolo += 1
                    elif label in RELEVANT_OBJECTS:
                        objects_found.add(label)

            # MediaPipe: face details
            face_results = face_mesh.process(rgb)
            people = []
            if face_results.multi_face_landmarks:
                for face in face_results.multi_face_landmarks:
                    is_open = mouth_open(face)
                    mouth_hist.append(is_open)
                    talking = sum(1 for x in mouth_hist if x) >= MOUTH_TALKING_THRESHOLD
                    people.append({
                        "talking": bool(talking),
                        "head_direction": head_direction(face, w),
                        "smiling": bool(smiling(face)),
                    })

            # If YOLO saw a person but face mesh didn't, still record one entry.
            if person_count_yolo > 0 and not people:
                for _ in range(person_count_yolo):
                    people.append({
                        "talking": False,
                        "head_direction": "unknown",
                        "smiling": False,
                    })

            objects_list = sorted(objects_found - {"person"})

            payload = {
                "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "people": people,
                "other_objects": objects_list,
                "summary": build_summary(people, objects_list),
            }

            try:
                atomic_write_json(OUTPUT_PATH, payload)
            except Exception as e:
                print(f"write failed: {e}", file=sys.stderr)
                continue

            # Tiny stdout heartbeat so the user can see it's alive.
            print(f"[{payload['ts']}] {payload['summary']}", flush=True)

    finally:
        cap.release()
        face_mesh.close()
        print("vision sidecar shutting down.", flush=True)


if __name__ == "__main__":
    main()
