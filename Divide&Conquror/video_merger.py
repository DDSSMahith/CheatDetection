import cv2

left_path = "outputs\\gaze\\left_gaze_output.mp4"
right_path = "outputs\\gaze\\right_gaze_output.mp4"
output_path = "outputs\\gaze\\final_output.mp4"

cap_left = cv2.VideoCapture(left_path)
cap_right = cv2.VideoCapture(right_path)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error opening one of the videos.")
    exit()

fps = int(cap_left.get(cv2.CAP_PROP_FPS))

width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ensure same height
target_height = min(height_left, height_right)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width_left + width_right, target_height)
)

while True:
    ret1, frame1 = cap_left.read()
    ret2, frame2 = cap_right.read()

    if not ret1 or not ret2:
        break

    # Resize both to same height
    frame1 = cv2.resize(frame1, (width_left, target_height))
    frame2 = cv2.resize(frame2, (width_right, target_height))

    combined = cv2.hconcat([frame1, frame2])

    out.write(combined)

cap_left.release()
cap_right.release()
out.release()

print("Videos merged successfully.")