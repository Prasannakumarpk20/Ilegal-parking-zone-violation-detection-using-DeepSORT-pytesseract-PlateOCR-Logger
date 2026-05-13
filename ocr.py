# ============================================================
#  ocr.py  —  License Plate OCR using pytesseract
# ============================================================
#
#  Strategy:
#   1. Crop the bottom 40% of the vehicle bounding box
#      (that's where plates usually live)
#   2. Upscale + denoise + threshold the crop
#   3. Run pytesseract in single-word mode (alphanumeric only)
#   4. Return cleaned text (≥ 3 chars) or empty string
# ============================================================

import re
import cv2
import numpy as np
import pytesseract

from config import TESSERACT_CMD


# Apply custom tesseract path if set in config
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class PlateOCR:
    """
    Attempts to read a license plate from a vehicle bounding box crop.
    Results are cached per track_id to avoid redundant computation.
    """

    # Tesseract config: OEM 3 = best available engine, PSM 8 = single word
    _TESS_CFG = (
        r"--oem 3 --psm 8 "
        r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    def __init__(self):
        self._cache: dict[int, str] = {}   # track_id → best plate text

    # ------------------------------------------------------------------ #
    #  Image preprocessing                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop to a clean binary image for Tesseract."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2× upscale — Tesseract works much better at higher resolution
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Bilateral filter — removes noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Otsu binarisation — automatically picks the best threshold
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def read_plate(self, frame: np.ndarray, bbox: tuple,
                   track_id: int | None = None) -> str:
        """
        Parameters
        ----------
        frame    : full BGR video frame
        bbox     : (x1, y1, x2, y2) of the detected vehicle
        track_id : if provided, result is cached so we only OCR once per vehicle

        Returns
        -------
        Cleaned plate string (e.g. "ABC1234") or "" if nothing found.
        """
        # Return cached result if we already have a plate for this vehicle
        if track_id is not None and self._cache.get(track_id):
            return self._cache[track_id]

        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]

        # Focus on lower 40 % of the bounding box (typical plate region)
        plate_y1 = max(0, int(y1 + (y2 - y1) * 0.60))
        plate_y2 = min(H, y2)
        plate_x1 = max(0, x1)
        plate_x2 = min(W, x2)

        crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
        if crop.size == 0:
            return ""

        processed = self._preprocess(crop)

        try:
            raw = pytesseract.image_to_string(processed, config=self._TESS_CFG)
            text = re.sub(r"[^A-Z0-9]", "", raw.upper()).strip()
        except Exception:
            text = ""

        # Only accept results with ≥ 3 characters (avoids noise)
        if len(text) >= 3:
            if track_id is not None:
                self._cache[track_id] = text
            return text

        return ""

    def get_cached(self, track_id: int) -> str:
        """Return previously found plate for a track (empty string if none)."""
        return self._cache.get(track_id, "")
