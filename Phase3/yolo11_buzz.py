import cv2
import numpy as np
import logging
import subprocess
import os
from pathlib import Path
from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================
# SILENCE YOLO LOGS
# ============================
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ============================
# CONFIG
# ============================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "actual2.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"
TEMP_VIDEO_PATH = OUTPUT_DIR / "actual2_processed.avi"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "actual2_buzzer.mp4"
BUZZER_PATH = PROJECT_ROOT / "buzzer.mp3"

DETECTION_MODEL_PATH = SCRIPT_DIR / "yolo11n.pt"
POSE_MODEL_PATH = SCRIPT_DIR / "yolo11n-pose.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not DETECTION_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {DETECTION_MODEL_PATH}")
if not POSE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {POSE_MODEL_PATH}")
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")
if not BUZZER_PATH.exists():
    raise FileNotFoundError(f"Buzzer audio not found: {BUZZER_PATH}")

CONF_THRESH = 0.4
IOU_THRESH = 0.5
PROXIMITY_IOU = 0.25
FRAME_PERSISTENCE = 5

# ============================
# UTILITY FUNCTION
# ============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)

# ============================
# LOAD MODELS
# ============================
det_model = YOLO(str(DETECTION_MODEL_PATH))
pose_model = YOLO(str(POSE_MODEL_PATH))

det_model.to(DEVICE)
pose_model.to(DEVICE)

# ============================
# VIDEO IO
# ============================
cap = cv2.VideoCapture(str(VIDEO_PATH))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Resolution:", w, h)
print("FPS:", fps)

out = cv2.VideoWriter(
    str(TEMP_VIDEO_PATH),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (w, h)
)

if not out.isOpened():
    raise RuntimeError("VideoWriter failed to open.")

# ============================
# STATE
# ============================
cheat_counter = 0
cheat_intervals = []
cheat_active = False
cheat_start_time = 0
frame_index = 0

# ============================
# MAIN LOOP
# ============================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_index / fps
    frame_index += 1

    cheating = False
    boxes = []

    results = det_model.track(
        frame,
        persist=True,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

    pose_results = pose_model(frame, conf=0.3, verbose=False)

    for r in pose_results:
        if r.keypoints is None:
            continue

        kpts = r.keypoints.xy.cpu().numpy()

        for person in kpts:
            nose = person[0]
            left_shoulder = person[5]
            right_shoulder = person[6]

            if np.any(nose <= 0):
                continue

            shoulder_mid = (left_shoulder + right_shoulder) / 2
            nose_offset = nose[0] - shoulder_mid[0]
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

            if shoulder_width > 0:
                normalized_offset = abs(nose_offset) / shoulder_width
                if normalized_offset > 0.35:
                    cheating = True

    if len(boxes) == 2:
        if iou(boxes[0], boxes[1]) > PROXIMITY_IOU:
            cheating = True

    if cheating:
        cheat_counter += 1
    else:
        cheat_counter = max(0, cheat_counter - 1)

    is_cheating_now = cheat_counter >= FRAME_PERSISTENCE

    if is_cheating_now and not cheat_active:
        cheat_active = True
        cheat_start_time = current_time

    if not is_cheating_now and cheat_active:
        cheat_active = False
        cheat_intervals.append((cheat_start_time, current_time))

    if is_cheating_now:
        cv2.putText(frame,
                    "CHEATING DETECTED",
                    (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    4)

    out.write(frame)

if cheat_active:
    cheat_intervals.append((cheat_start_time, frame_index / fps))

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
print("Cheat intervals:", cheat_intervals)

# ============================
# BUILD FFMPEG FILTER COMPLEX
# ============================

if not cheat_intervals:
    print("No cheating detected. Saving video without audio.")
    os.rename(str(TEMP_VIDEO_PATH), str(FINAL_OUTPUT_PATH))
else:
    filter_parts = []
    for i, (start, end) in enumerate(cheat_intervals):
        duration = end - start
        filter_parts.append(
            f"[1:a]atrim=0:{duration},asetpts=PTS-STARTPTS,adelay={int(start*1000)}|{int(start*1000)}[a{i}]"
        )

    amix_inputs = "".join([f"[a{i}]" for i in range(len(cheat_intervals))])
    filter_complex = ";".join(filter_parts)
    filter_complex += f";{amix_inputs}amix=inputs={len(cheat_intervals)}:dropout_transition=0[aout]"

    cmd = [
    "ffmpeg",
    "-y",
    "-i", str(TEMP_VIDEO_PATH),
    "-i", str(BUZZER_PATH),
    "-filter_complex", filter_complex,
    "-map", "0:v",
    "-map", "[aout]",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-c:a", "aac",
    "-movflags", "+faststart",
    str(FINAL_OUTPUT_PATH)
    ]

    print("Embedding audio using FFmpeg...")
    subprocess.run(cmd, check=True)

    os.remove(str(TEMP_VIDEO_PATH))

print("Final video saved:", FINAL_OUTPUT_PATH)