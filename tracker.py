# ============================================================
#  tracker.py  —  DeepSORT Vehicle Tracker Wrapper
# ============================================================

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class VehicleTracker:
    """
    Thin wrapper around deep_sort_realtime.

    Input  → list of ( (x1,y1,x2,y2), confidence, class_id )
    Output → list of ( x1, y1, x2, y2, track_id )
    """

    def __init__(self, max_age: int = 30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Parameters
        ----------
        detections : list of ((x1,y1,x2,y2), conf, cls_id)
        frame      : current BGR frame (used by DeepSORT re-id)

        Returns
        -------
        list of (x1, y1, x2, y2, track_id)  — confirmed tracks only
        """
        if not detections:
            # Still must tick the tracker so ages increment
            self.tracker.update_tracks([], frame=frame)
            return []

        # deep_sort_realtime expects: ([left, top, width, height], conf, class)
        ds_input = []
        for (x1, y1, x2, y2), conf, cls_id in detections:
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], float(conf), str(cls_id)))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results.append((x1, y1, x2, y2, track.track_id))

        return results
