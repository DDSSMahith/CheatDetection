import cv2
import time
import json
import os
import sys
import numpy as np
import mediapipe as mp

from collections import deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from filters import EMAFilter, MedianFilter
from head_pose import estimate_head_pose
from calibaration import UserCalibrator
from eye_analysis import eye_aspect_ratio, pupil_vertical_offset
from tracker import FaceTracker
from detectors.face_absence import FaceAbsenceDetector

# =========================
# Config
# =========================

OUT_DIR = "flagged_clips"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 20
WINDOW_SEC = 1.0
WINDOW_SIZE = int(WINDOW_SEC * FPS)

LONG_WINDOW_SEC = 5.0
LONG_WINDOW_SIZE = int(LONG_WINDOW_SEC * FPS)

SHIFT_YAW_THRESHOLD = 5.0
SHIFT_ACTIVE_RATIO = 0.6

YAW_TURN_THRESHOLD = 8.0
YAW_SUSTAIN_SEC = 0.25
YAW_SUSTAIN_FRAMES = int(YAW_SUSTAIN_SEC * FPS)

PITCH_THRESHOLD = 12.0

PHONE_SUSTAIN_FRAMES = int(0.4 * FPS)

EAR_CLOSED = 0.20
PUPIL_LOW = 0.65

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

mp_face = mp.solutions.face_mesh

# MediaPipe landmark indices
NOSE_IDX = 1

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()

    yaw_buf = deque(maxlen=WINDOW_SIZE)
    yaw_long_buf = deque(maxlen=LONG_WINDOW_SIZE)
    frame_buf = deque(maxlen=int(5 * FPS))

    yaw_ema = EMAFilter(alpha=0.3)
    pitch_ema = EMAFilter(alpha=0.3)
    yaw_median = MedianFilter(window_size=5)

    user_calibrators = {}
    phone_like_counts = {}

    flag_count = 0

    face_absence_detector = FaceAbsenceDetector(max_missing_sec=3.0)

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            face_count = 0
            face_detected = False

            if result.multi_face_landmarks:
                detections = []

                for face_lms in result.multi_face_landmarks:
                    xs = [lm.x * w for lm in face_lms.landmark]
                    ys = [lm.y * h for lm in face_lms.landmark]
                    detections.append([
                        int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)),
                        0.99
                    ])

                face_count = len(detections)
                face_detected = True

                tracks = tracker.update(detections)[:3]
                active_ids = set(int(t[4]) for t in tracks)

                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track)
                    cx, cy = box_center((x1, y1, x2, y2))

                    best_idx = None
                    best_dist = 1e9

                    for i, face_lms in enumerate(result.multi_face_landmarks):
                        fx = int(face_lms.landmark[NOSE_IDX].x * w)
                        fy = int(face_lms.landmark[NOSE_IDX].y * h)
                        d = (fx - cx) ** 2 + (fy - cy) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_idx = i

                    if best_idx is None:
                        continue

                    lm = result.multi_face_landmarks[best_idx].landmark

                    # ---- HEAD POSE ----
                    yaw, pitch, _ = estimate_head_pose(lm, w, h)

                    yaw_s = yaw_median.update(yaw_ema.update(yaw))
                    pitch_s = pitch_ema.update(pitch)

                    yaw_abs = abs(yaw_s)

                    # ---- CALIBRATION ----
                    if track_id not in user_calibrators:
                        user_calibrators[track_id] = UserCalibrator(
                            duration_sec=5,
                            fps=FPS,
                            k=2.5
                        )

                    calibrator = user_calibrators[track_id]
                    calibrator.update(yaw_abs, pitch_s)

                    yaw_thr, pitch_thr = calibrator.get_thresholds(
                        YAW_TURN_THRESHOLD,
                        PITCH_THRESHOLD
                    )

                    # ---- EYE ANALYSIS ----
                    ear = (
                        eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                        eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                    ) / 2.0

                    pupil_v = (
                        pupil_vertical_offset(lm, LEFT_EYE, w, h) +
                        pupil_vertical_offset(lm, RIGHT_EYE, w, h)
                    ) / 2.0

                    head_down = pitch_s > pitch_thr
                    eye_suspicious = (ear < EAR_CLOSED) or (pupil_v > PUPIL_LOW)

                    phone_like_counts.setdefault(track_id, 0)

                    if head_down and eye_suspicious:
                        phone_like_counts[track_id] += 1
                    else:
                        phone_like_counts[track_id] = 0

                    flagged = False
                    reason = None

                    if yaw_abs > yaw_thr:
                        flagged = True
                        reason = "head_turn"

                    if phone_like_counts[track_id] >= PHONE_SUSTAIN_FRAMES:
                        flagged = True
                        reason = "phone_like_behavior"

                    if flagged:
                        ts = int(time.time())
                        video_path = os.path.join(
                            OUT_DIR, f"flag_{ts}_{track_id}.mp4"
                        )

                        writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
                        for f in list(frame_buf):
                            writer.write(f)
                        writer.release()

                        meta = {
                            "time": ts,
                            "track_id": track_id,
                            "reason": reason,
                            "file": video_path
                        }
                        with open(os.path.join(OUT_DIR, "flags.jsonl"), "a") as f:
                            f.write(json.dumps(meta) + "\n")

                    # ---- UI ----
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2
                    )

                    if reason == "phone_like_behavior":
                        cv2.putText(
                            frame, "PHONE-LIKE BEHAVIOR",
                            (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2
                        )

                    status = "CALIBRATING" if not calibrator.calibrated else "CALIBRATED"
                    cv2.putText(
                        frame, status,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if not calibrator.calibrated else (0, 255, 0),
                        2
                    )

                # ---- FACE ABSENCE ----
                face_absence_events = face_absence_detector.update(active_ids)
            else:
                face_absence_events = face_absence_detector.update(set())

                for ev in face_absence_events:
                    ts = ev["time"]
                    tid = ev["track_id"]

                    video_path = os.path.join(
                        OUT_DIR, f"flag_{ts}_{tid}_face_absent.mp4"
                    )

                    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
                    for f in list(frame_buf):
                        writer.write(f)
                    writer.release()

                    ev["video"] = video_path

                    with open(os.path.join(OUT_DIR, "flags.jsonl"), "a") as f:
                        f.write(json.dumps(ev) + "\n")


            frame_buf.append(frame.copy())

            cv2.putText(
                frame, f"Faces: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if face_count == 1 else (0, 0, 255),
                2
            )

            cv2.imshow("Anomaly Detector (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
