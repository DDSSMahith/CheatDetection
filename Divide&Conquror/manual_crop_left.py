import cv2
import os

# ==============================
# CONFIG
# ==============================
video_path = "data\\actual2.mp4"        # <-- change to your path
output_path = "outputs\\gaze\\cropped_footage_HalfScreen.mp4"  # <-- change to your desired output path
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop LEFT half
    crop = frame[:, :width // 2]

    out.write(crop)

cap.release()
out.release()

print("Left half cropped successfully.")