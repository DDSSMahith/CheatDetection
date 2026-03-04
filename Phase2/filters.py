# filters.py
import numpy as np
from collections import deque


class EMAFilter:
    """
    Causal exponential moving average filter
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class MedianFilter:
    """
    Small sliding-window median filter (for yaw spikes)
    """
    def __init__(self, window_size=5):
        self.buf = deque(maxlen=window_size)

    def update(self, x):
        self.buf.append(x)
        return float(np.median(self.buf))
