# CheatDetection_ALP2

## 1) Project Goal

This project detects potential cheating behavior in exam/classroom videos.
The implementation evolved in stages:

- **Phase2**: MediaPipe-based face/eye/head analysis with calibration and tracking.
- **Phase3**: YOLO11n-based multi-person detection + pose analysis for robust classroom scenarios.
- **Divide&Conquror**: Split-screen workaround to process students independently, then merge outputs.

---

## 2) Repository Sections and File Roles

### 2.1 Phase2 (`Phase2/`)

- `main.py`:
	- Main runtime pipeline for face landmarks, head pose, gaze/eye cues, per-person calibration, and event logging.
	- Uses `FaceTracker` + `FaceAbsenceDetector` and writes flagged clips/metadata.
- `calibaration.py`:
	- Per-user threshold calibration (yaw/pitch adaptive thresholds).
- `eye_analysis.py`:
	- Eye Aspect Ratio (EAR) and pupil vertical offset heuristics.
- `filters.py`:
	- Temporal smoothing with EMA and median filters.
- `head_pose.py`:
	- Pose estimation via `cv2.solvePnP` from FaceMesh landmarks.
- `tracker.py` + `sort.py`:
	- SORT-based ID tracking using Kalman filter + IOU association.

### 2.2 Phase3 (`Phase3/`)

- `yolo11_buzz.py`:
	- Uses `yolo11n.pt` (person detection/tracking) and `yolo11n-pose.pt` (pose keypoints).
	- Detects suspicious behavior using:
		- Nose vs shoulder-center offset (head turn proxy).
		- Inter-person proximity (IOU overlap).
	- Generates output video and overlays buzzer audio in detected intervals via FFmpeg.
- `yolo11n.pt`, `yolo11n-pose.pt`:
	- YOLO11 nano weights used for lightweight inference.

### 2.3 Divide&Conquror (`Divide&Conquror/`)

- `manual_crop_left.py`: crops left half of classroom video.
- `manual_crop_right.py`: crops right half with stable dimensions.
- `left_gaze_detection.py`: per-half gaze pipeline using MediaPipe iris/eye landmarks.
- `video_merger.py`: merges processed half-videos side-by-side.
- `tracker.py` + `sort.py`: tracking utilities reused here.

---

## 3) Phase2 vs Phase3: Approach Difference

### 3.1 Detection Backbone

- **Phase2**: Landmark-first approach (`mediapipe`) for face mesh + custom gaze/head heuristics.
- **Phase3**: Detector-first approach (`ultralytics` YOLO11n) for robust multi-person bounding boxes and tracking.

### 3.2 Multi-Person Reliability

- **Phase2 challenge**:
	- Even with `max_num_faces`, real classroom stability drops (ID consistency, occlusion, landmark confusion) when multiple students are present.
- **Phase3 improvement**:
	- YOLO11n gives stronger multi-person person boxes + tracking persistence and better behavior localization in crowded frames.

### 3.3 Behavior Signals

- **Phase2**:
	- Yaw/pitch thresholds (adaptive calibration), EAR, pupil vertical offset, face absence events.
- **Phase3**:
	- Person tracking + pose geometry (nose-shoulder relation) + pairwise overlap (IOU proximity) + temporal persistence.

### 3.4 Why We Switched

- MediaPipe is excellent for detailed landmarks, but classroom-scale multi-person consistency was a bottleneck.
- We switched to **YOLO11 nano (`yolo11n`)** to generate person bounding boxes and track multiple students more reliably under occlusion and movement.

---

## 4) Key Problems Faced and Lessons

- Landmark jitter and identity switching in multi-person scenes.
- Occlusions and head-turn extremes reducing FaceMesh confidence.
- Need for per-person temporal stability to avoid false positives.
- Video codec/dimension mismatches during split/merge pipelines.
- Requirement of FFmpeg as an external dependency for buzzer audio embedding.

---

## 5) Divide-and-Conquer Multi-Student Pipeline (2nd Last Subsection)

This section explains how we handled multiple students by dividing the frame, processing each side independently, then merging results.

### 5.1 Current 2-Student Flow

1. Run `manual_crop_left.py` to produce left-half video.
2. Run `manual_crop_right.py` to produce right-half video.
3. Run gaze analysis on each half (using `left_gaze_detection.py` logic per segment).
4. Merge both processed videos using `video_merger.py`.

### 5.2 Why This Helps

- Each sub-video has fewer faces and less interference.
- Gaze estimation becomes more stable because each model run focuses on one student region.
- Easier debugging and per-student output inspection.

### 5.3 Extending Same Idea to 5–6 People

- Split the original frame into a grid/strips (for example, 3x2 tiles).
- Run per-tile detection/gaze analysis independently.
- Keep per-tile track IDs and timestamps.
- Reassemble tiles into a final composite video (or keep per-student streams).

**Practical note:** for 5–6 people, YOLO-based person detection should be used first to auto-crop dynamic ROIs, then optional gaze/head modules can run per ROI.

---

## 6) Environment and Requirements

### 6.1 Python Packages

Install dependencies from:

```bash
pip install -r requirements.txt
```

### 6.2 External Tools

- Install **FFmpeg** and ensure `ffmpeg` is available in PATH (required by `Phase3/yolo11_buzz.py`).

### 6.3 Typical Run Order

- **Phase2**: run `Phase2/main.py`
- **Phase3**: run `Phase3/yolo11_buzz.py`
- **Divide&Conquror**:
	- `manual_crop_left.py`
	- `manual_crop_right.py`
	- gaze processing scripts
	- `video_merger.py`
