# tracker.py
import numpy as np
from sort import Sort


class FaceTracker:
    def __init__(self, max_age=10, min_hits=3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits)

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, score]
        returns: list of [x1, y1, x2, y2, track_id]
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets)

        return tracks
