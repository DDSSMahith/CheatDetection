import time

class FaceAbsenceDetector:
    def __init__(self, max_missing_sec=3.0):
        self.max_missing_sec = max_missing_sec
        self.states = {}

    def update(self, active_track_ids):
        """
        active_track_ids: set[int]
        Returns: list of absence events (dict)
        """
        now = time.time()
        events = []

        # Initialize unseen IDs
        for tid in active_track_ids:
            if tid not in self.states:
                self.states[tid] = {
                    "last_seen": now,
                    "missing_since": None,
                    "is_missing": False
                }

        # Update states
        for tid, state in self.states.items():
            if tid in active_track_ids:
                # Face is visible
                if state["is_missing"]:
                    print(f"[ID {tid}] FACE REAPPEARED")

                state["last_seen"] = now
                state["missing_since"] = None
                state["is_missing"] = False

            else:
                # Face is missing
                if not state["is_missing"]:
                    state["missing_since"] = now
                    state["is_missing"] = True
                    print(f"[ID {tid}] FACE MISSING started")

                missing_dur = now - state["missing_since"]

                if missing_dur >= self.max_missing_sec:
                    event = {
                        "track_id": tid,
                        "event": "face_absence",
                        "level": "red",
                        "duration_sec": round(missing_dur, 2),
                        "time": int(now)
                    }
                    events.append(event)

                    print(
                        f"[ID {tid}] FACE ABSENT ≥ {self.max_missing_sec}s "
                        f"({missing_dur:.2f}s) → RED FLAG"
                    )

                    # Reset timer so it can fire AGAIN if absence continues
                    state["missing_since"] = now

        return events
