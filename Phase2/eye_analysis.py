# eye_analysis.py
import numpy as np

# MediaPipe eye landmark indices (left eye)
LEFT_EYE = [33, 160, 158, 133, 153, 144]  
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Computes EAR for one eye
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))

    p1, p2, p3, p4, p5, p6 = pts

    # vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # horizontal distance
    h_dist = np.linalg.norm(p1 - p4)

    ear = (v1 + v2) / (2.0 * h_dist + 1e-6)
    return ear


def pupil_vertical_offset(landmarks, eye_indices, w, h):
    """
    Estimates pupil position relative to eye box (0 = top, 1 = bottom)
    """
    eye_pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        eye_pts.append([lm.x * w, lm.y * h])

    eye_pts = np.array(eye_pts)
    ymin, ymax = eye_pts[:, 1].min(), eye_pts[:, 1].max()

    # Approx pupil center using eye landmark centroid
    pupil_y = eye_pts[:, 1].mean()

    return (pupil_y - ymin) / (ymax - ymin + 1e-6)
