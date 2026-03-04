import cv2
import mediapipe as mp
import numpy as np

video_path = "outputs\\gaze\\cropped_footage_RightHalfScreen.avi"
output_path = "outputs\\gaze\\right_gaze_output.mp4"

# ==============================
# LOAD VIDEO
# ==============================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open input video.")
    exit()

fps = int(round(cap.get(cv2.CAP_PROP_FPS)))  # FORCE INT
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Force divisible by 4
width -= width % 4
height -= height % 4

print("Resolution:", width, height)
print("FPS:", fps)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("VideoWriter failed.")
    cap.release()
    exit()

# ==============================
# MEDIAPIPE SETUP
# ==============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def iris_center(landmarks, indices):
    pts = np.array([[landmarks[i].x * width,
                     landmarks[i].y * height] for i in indices])
    return pts.mean(axis=0)

def eye_center(landmarks, indices):
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    return np.array([
        (p1.x + p2.x) / 2 * width,
        (p1.y + p2.y) / 2 * height
    ])

# ==============================
# PROCESS LOOP
# ==============================

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enforce consistent dimension
    frame = frame[0:height, 0:width]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_text = "NO FACE"

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        li = iris_center(lm, LEFT_IRIS)
        ri = iris_center(lm, RIGHT_IRIS)

        le = eye_center(lm, LEFT_EYE)
        re = eye_center(lm, RIGHT_EYE)

        dx = ((li[0]-le[0]) + (ri[0]-re[0])) / 2
        dy = ((li[1]-le[1]) + (ri[1]-re[1])) / 2

        
        # LEFT EYE
        left_corner_left = lm[33]
        left_corner_right = lm[133]

        lx1 = left_corner_left.x * width
        lx2 = left_corner_right.x * width
        left_eye_width = abs(lx2 - lx1)

        left_iris_x = li[0]

        left_ratio = (left_iris_x - min(lx1, lx2)) / (left_eye_width + 1e-6)

        # RIGHT EYE
        right_corner_left = lm[362]
        right_corner_right = lm[263]

        rx1 = right_corner_left.x * width
        rx2 = right_corner_right.x * width
        right_eye_width = abs(rx2 - rx1)

        right_iris_x = ri[0]

        right_ratio = (right_iris_x - min(rx1, rx2)) / (right_eye_width + 1e-6)

        # Average both eyes
        avg_ratio = (left_ratio + right_ratio) / 2

        # ---- DECISION ----

        if avg_ratio < 0.30:
            gaze_text = "LEFT"
        elif avg_ratio > 0.70:
            gaze_text = "RIGHT"
        elif dy > 8:   # keep your vertical check
            gaze_text = "DOWN"
        else:
            gaze_text = "FORWARD"

    cv2.putText(frame, gaze_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
face_mesh.close()

print("Done.")
print("Frames written:", frame_count)