import cv2
import os

video_path = "data\\actual2.mp4"
output_path = "outputs\\gaze\\cropped_footage_right_full.avi"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open input video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Original:", width, "x", height)

# Force dimensions divisible by 4 (very important)
width = width - (width % 4)
height = height - (height % 4)

half_width = width // 2

print("Adjusted:", width, "x", height)
print("Half width:", half_width)

# Use MJPG (very stable)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (half_width, height))

if not out.isOpened():
    print("VideoWriter failed to open!")
    cap.release()
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame to adjusted size first
    frame = frame[0:height, 0:width]

    # Right half
    crop = frame[:, half_width:width]

    out.write(crop)
    frame_count += 1

cap.release()
out.release()

print("Done.")
print("Frames written:", frame_count)
print("Saved to:", output_path)