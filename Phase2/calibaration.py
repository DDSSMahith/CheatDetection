# calibration.py
import numpy as np
from collections import deque
import time


class UserCalibrator:
    def __init__(self, duration_sec=5, fps=20, k=2.5):
        self.max_frames = int(duration_sec * fps)
        self.k = k

        self.yaw_vals = deque(maxlen=self.max_frames)
        self.pitch_vals = deque(maxlen=self.max_frames)

        self.start_time = time.time()
        self.calibrated = False

        self.yaw_threshold = None
        self.pitch_threshold = None

    def update(self, yaw, pitch):
        if self.calibrated:
            return

        self.yaw_vals.append(yaw)
        self.pitch_vals.append(pitch)

        if len(self.yaw_vals) >= self.max_frames:
            self._compute_thresholds()

    def _compute_thresholds(self):
        yaw_arr = np.array(self.yaw_vals)
        pitch_arr = np.array(self.pitch_vals)

        yaw_mean, yaw_std = yaw_arr.mean(), yaw_arr.std()
        pitch_mean, pitch_std = pitch_arr.mean(), pitch_arr.std()

        self.yaw_threshold = yaw_mean + self.k * yaw_std
        self.pitch_threshold = pitch_mean + self.k * pitch_std

        self.calibrated = True

    def get_thresholds(self, default_yaw, default_pitch):
        if not self.calibrated:
            return default_yaw, default_pitch
        return self.yaw_threshold, self.pitch_threshold
