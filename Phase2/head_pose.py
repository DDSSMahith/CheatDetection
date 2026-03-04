# head_pose.py
import cv2
import numpy as np

# Selected MediaPipe FaceMesh landmark indices
# (stable + symmetric)
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 263,
    "right_eye_outer": 33,
    "left_mouth": 287,
    "right_mouth": 57
}

# Generic 3D face model (mm, approximate)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [0.0, -63.6, -12.5],   # Chin
    [-43.3, 32.7, -26.0],  # Left eye outer
    [43.3, 32.7, -26.0],   # Right eye outer
    [-28.9, -28.9, -24.1], # Left mouth
    [28.9, -28.9, -24.1],  # Right mouth
], dtype=np.float64)


def estimate_head_pose(landmarks, img_w, img_h):
    """
    Returns yaw, pitch, roll in degrees using SolvePnP
    """

    # 2D image points
    face_2d = []
    for key in LANDMARK_IDS.values():
        lm = landmarks[key]
        face_2d.append([lm.x * img_w, lm.y * img_h])

    face_2d = np.array(face_2d, dtype=np.float64)

    # Camera internals
    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL,
        face_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # Convert rotation matrix to Euler angles
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw   = np.arctan2(-rot_mat[2, 0], sy)
        roll  = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        yaw   = np.arctan2(-rot_mat[2, 0], sy)
        roll  = 0

    return (
        np.degrees(yaw),
        np.degrees(pitch),
        np.degrees(roll)
    )
