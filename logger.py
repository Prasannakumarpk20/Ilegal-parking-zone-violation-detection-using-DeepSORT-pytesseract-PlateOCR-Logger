# ============================================================
#  logger.py  —  Violation Logger (CSV + snapshots)
# ============================================================
#
#  Each vehicle is logged AT MOST ONCE (keyed by DeepSORT track_id).
#  The log entry contains:
#    • Timestamp          — wall-clock time of the violation
#    • Track_ID           — DeepSORT's persistent vehicle ID
#    • Plate_Number       — OCR result (or "UNKNOWN")
#    • Duration_s         — seconds the vehicle spent in the zone
#    • Snapshot           — filename of the saved frame crop
# ============================================================

import csv
import os
from datetime import datetime

import cv2
import numpy as np


class ViolationLogger:
    """
    Writes violations to a CSV file and saves a snapshot image
    for every unique vehicle that violates the parking zone.
    """

    _CSV_HEADER = [
        "Timestamp", "Track_ID", "Plate_Number",
        "Duration_In_Zone_s", "Snapshot_File"
    ]

    def __init__(self, csv_path: str, snapshots_dir: str):
        self.csv_path      = csv_path
        self.snapshots_dir = snapshots_dir
        self._logged: set  = set()   # track IDs already logged

        # Create output directories
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)

        # Write CSV header if this is a new file
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(self._CSV_HEADER)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def is_logged(self, track_id: int) -> bool:
        return track_id in self._logged

    def log(self, track_id: int, plate: str,
            duration: float, frame: np.ndarray) -> None:
        """
        Log a violation.  Silently skips if track_id was already logged.

        Parameters
        ----------
        track_id : DeepSORT track ID
        plate    : OCR plate text (may be empty → stored as "UNKNOWN")
        duration : seconds the vehicle was inside the zone
        frame    : BGR frame at the moment of violation (saved as snapshot)
        """
        if track_id in self._logged:
            return

        self._logged.add(track_id)
        timestamp   = datetime.now()
        plate_str   = plate.strip() if plate else "UNKNOWN"
        ts_str      = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        file_ts     = timestamp.strftime("%Y%m%d_%H%M%S")
        snap_name   = f"violation_ID{track_id}_{file_ts}.jpg"
        snap_path   = os.path.join(self.snapshots_dir, snap_name)

        # Save snapshot
        cv2.imwrite(snap_path, frame)

        # Append CSV row
        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [ts_str, track_id, plate_str, f"{duration:.2f}", snap_name]
            )

        print(
            f"  🚨 VIOLATION  |  ID: {track_id:>4}  "
            f"|  Plate: {plate_str:<10}  "
            f"|  In-zone: {duration:.1f}s  "
            f"|  {snap_name}"
        )

    @property
    def total(self) -> int:
        """Number of unique violations logged so far."""
        return len(self._logged)
