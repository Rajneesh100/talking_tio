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
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

VISION_DIR = Path(__file__).parent
OUTPUT_PATH = VISION_DIR / "visual_context.json"

# Pin all models inside the repo so the user's HF / ultralytics / mediapipe
# caches being evicted doesn't force redownloads. Mirror of stt/models/.
MODELS_DIR = VISION_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _download(url: str, dest: Path, min_bytes: int = 100_000):
    """Idempotent download with atomic rename."""
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
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  done ({size_mb:.1f} MB)", flush=True)


def ensure_yolo_model():
    _download(YOLO_MODEL_URL, YOLO_MODEL_PATH, min_bytes=1_000_000)


def ensure_face_landmarker():
    _download(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH, min_bytes=1_000_000)

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


# Tasks API: result.face_landmarks is List[List[NormalizedLandmark]] —
# one inner list per detected face, each landmark has .x .y .z in [0, 1].
# Indices match the legacy face_mesh module (468-point topology).

def head_direction(face_landmarks):
    """Classify head as 'left' | 'center' | 'right' from nose vs eye centers."""
    if not face_landmarks:
        return "unknown"
    nose = face_landmarks[1]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
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
    upper = face_landmarks[13]  # upper inner lip
    lower = face_landmarks[14]  # lower inner lip
    return abs(upper.y - lower.y) > 0.015


def smiling(face_landmarks):
    """Crude: horizontal mouth corners wider than vertical opening."""
    if not face_landmarks:
        return False
    left = face_landmarks[61]
    right = face_landmarks[291]
    top = face_landmarks[13]
    bot = face_landmarks[14]
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
    ensure_face_landmarker()

    print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...", flush=True)
    yolo = YOLO(str(YOLO_MODEL_PATH))

    print(f"Loading MediaPipe FaceLandmarker from {FACE_LANDMARKER_PATH} ...", flush=True)
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
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

            # MediaPipe Tasks API: detect on an mp.Image wrapping the RGB frame.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_results = face_landmarker.detect(mp_image)
            people = []
            for face in face_results.face_landmarks:
                is_open = mouth_open(face)
                mouth_hist.append(is_open)
                talking = sum(1 for x in mouth_hist if x) >= MOUTH_TALKING_THRESHOLD
                people.append({
                    "talking": bool(talking),
                    "head_direction": head_direction(face),
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
        face_landmarker.close()
        print("vision sidecar shutting down.", flush=True)


if __name__ == "__main__":
    main()
