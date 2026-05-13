# ============================================================
#  main.py  —  Illegal Parking Zone Violation Detection
# ============================================================
#
#  Pipeline:
#    1. Open first frame → user draws illegal parking zone
#    2. Load YOLOv8 (vehicle detection) + DeepSORT (tracking)
#    3. For every frame:
#         • Detect vehicles → track with DeepSORT
#         • Check if vehicle centroid is inside the zone
#         • Time how long it stays → log violation after threshold
#         • Run pytesseract OCR to read licence plate
#    4. Write annotated video + violations.csv + snapshot images
#
#  Usage:
#    python main.py                    # uses VIDEO_PATH from config.py
#    python main.py my_video.mp4       # override video at runtime
# ============================================================

import os
import sys
import cv2
import numpy as np

from ultralytics import YOLO

from config import (
    VIDEO_PATH, OUTPUT_PATH, VIOLATIONS_CSV, SNAPSHOTS_DIR,
    YOLO_MODEL, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD,
    DEEPSORT_MAX_AGE, PARK_DURATION_THRESHOLD, FPS_DEFAULT,
    ZONE_COLOR, ZONE_ALPHA, TRACK_COLOR, ORANGE_COLOR,
    VIOLATION_COLOR, TEXT_COLOR, HUD_BG_COLOR
)
from zone_drawer import ZoneDrawer
from tracker    import VehicleTracker
from ocr        import PlateOCR
from logger     import ViolationLogger


# ------------------------------------------------------------------ #
#  Geometry helpers                                                    #
# ------------------------------------------------------------------ #

def centroid(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def in_zone(point: tuple, polygon: np.ndarray) -> bool:
    """True if point lies inside or on the edge of the polygon."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# ------------------------------------------------------------------ #
#  Drawing helpers                                                     #
# ------------------------------------------------------------------ #

def draw_zone_overlay(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Semi-transparent red zone + border + label."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], ZONE_COLOR)
    cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)
    cv2.polylines(frame, [polygon], True, (0, 0, 200), 2)
    label_pt = tuple(polygon[0].tolist())
    cv2.putText(frame, "NO PARKING ZONE", label_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def draw_vehicle(frame: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int,
                 track_id: int, plate: str,
                 in_restricted: bool, is_violation: bool,
                 duration: float) -> None:
    """Draw bounding box, label, centroid dot, and duration bar."""

    if is_violation:
        color = VIOLATION_COLOR    # red
    elif in_restricted:
        color = ORANGE_COLOR       # orange — in zone, timer counting
    else:
        color = TRACK_COLOR        # green — normal

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Centroid dot
    cx, cy = centroid(x1, y1, x2, y2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # Label bar (above box)
    label = f"ID:{track_id}"
    if plate:
        label += f"  [{plate}]"
    if is_violation:
        label += "  !! VIOLATION"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = x1, max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
    cv2.putText(frame, label, (lx + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Duration bar (below box) — only shown when inside zone
    if in_restricted:
        bar_text = f"In zone: {duration:.1f}s / {PARK_DURATION_THRESHOLD}s"
        cv2.putText(frame, bar_text, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)


def draw_hud(frame: np.ndarray,
             frame_num: int, total: int, fps: float,
             n_tracks: int, n_in_zone: int,
             n_violations: int) -> None:
    """Top-left heads-up display panel."""
    elapsed = frame_num / fps
    pct     = (frame_num / total * 100) if total > 0 else 0.0

    lines = [
        f"Frame : {frame_num:>5} / {total}  ({pct:4.1f}%)",
        f"Time  : {int(elapsed // 60):02d}:{elapsed % 60:05.2f}",
        f"Tracks: {n_tracks}  |  In zone: {n_in_zone}",
        f"Violations: {n_violations}",
    ]

    pad_x, pad_y = 12, 12
    line_h = 24
    box_w  = 320
    box_h  = pad_y * 2 + line_h * len(lines)

    bg = frame.copy()
    cv2.rectangle(bg, (pad_x - 4, pad_y - 4),
                  (pad_x + box_w, pad_y + box_h), HUD_BG_COLOR, -1)
    cv2.addWeighted(bg, 0.60, frame, 0.40, 0, frame)

    for i, line in enumerate(lines):
        y = pad_y + line_h * (i + 1)
        cv2.putText(frame, line, (pad_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, TEXT_COLOR, 1)


# ------------------------------------------------------------------ #
#  Main pipeline                                                       #
# ------------------------------------------------------------------ #

def main() -> None:
    print("\n" + "=" * 60)
    print("  🚗  ILLEGAL PARKING VIOLATION DETECTION SYSTEM")
    print("       YOLOv8  +  DeepSORT  +  pytesseract")
    print("=" * 60)

    # ── Resolve video path ────────────────────────────────────────
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    if not os.path.exists(video_path):
        print(f"\n❌  Video not found: {video_path}")
        print(f"    Usage:  python main.py <video_file.mp4>")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌  Cannot open video!")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  📹  {video_path}")
    print(f"       {width}×{height}  @  {fps:.1f} fps  —  {total_frames} frames")

    # ── Step 1: Draw zone ─────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        print("❌  Cannot read first frame!")
        sys.exit(1)

    zone_polygon = ZoneDrawer(first_frame).draw()
    if zone_polygon is None or len(zone_polygon) < 3:
        print("❌  No valid zone drawn.  Exiting.")
        sys.exit(1)

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Step 2: Load models ───────────────────────────────────────
    print("  ⚙   Loading YOLOv8 …")
    model   = YOLO(YOLO_MODEL)          # downloads weights on first run
    print("  ✅  YOLO ready.")

    tracker = VehicleTracker(max_age=DEEPSORT_MAX_AGE)
    ocr     = PlateOCR()
    vlogger = ViolationLogger(VIOLATIONS_CSV, SNAPSHOTS_DIR)

    # ── Step 3: Video writer ──────────────────────────────────────
    os.makedirs("violations", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # ── Step 4: Per-track state ───────────────────────────────────
    zone_entry: dict[int, int]  = {}   # track_id → frame number when entered zone
    frame_num = 0

    print(f"\n  🎬  Processing …  (press Q to stop early)")
    print(f"       Violation threshold: {PARK_DURATION_THRESHOLD}s in zone\n")

    cv2.namedWindow("Parking Violation Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parking Violation Detection", 1280, 720)

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        display = frame.copy()

        # Draw illegal zone
        draw_zone_overlay(display, zone_polygon)

        # ── YOLOv8 detection ─────────────────────────────────────
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        raw_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_dets.append(((x1, y1, x2, y2), conf, cls_id))

        # ── DeepSORT tracking ─────────────────────────────────────
        tracks = tracker.update(raw_dets, frame)

        active_ids   = set()
        ids_in_zone  = set()

        for (x1, y1, x2, y2, tid) in tracks:
            active_ids.add(tid)
            cx, cy       = centroid(x1, y1, x2, y2)
            inside        = in_zone((cx, cy), zone_polygon)

            # ── Zone entry / exit bookkeeping ─────────────────────
            if inside:
                ids_in_zone.add(tid)
                if tid not in zone_entry:
                    zone_entry[tid] = frame_num
                    print(f"  🚗  Vehicle #{tid} entered zone  (frame {frame_num})")
            else:
                if tid in zone_entry:
                    del zone_entry[tid]   # left the zone, reset timer

            # ── Duration inside zone ──────────────────────────────
            frames_inside    = (frame_num - zone_entry[tid]) if inside else 0
            duration_seconds = frames_inside / fps
            is_violation     = inside and duration_seconds >= PARK_DURATION_THRESHOLD

            # ── OCR — try every 30 frames while vehicle is in zone ─
            if inside and frame_num % 30 == 0:
                ocr.read_plate(frame, (x1, y1, x2, y2), track_id=tid)

            plate_text = ocr.get_cached(tid)

            # ── Log violation (once per vehicle) ──────────────────
            if is_violation and not vlogger.is_logged(tid):
                vlogger.log(tid, plate_text, duration_seconds, display)

            # ── Draw this vehicle ─────────────────────────────────
            draw_vehicle(display, x1, y1, x2, y2, tid,
                         plate_text, inside, is_violation, duration_seconds)

        # Clean up entries for tracks that disappeared
        for gone_id in list(zone_entry.keys()):
            if gone_id not in active_ids:
                del zone_entry[gone_id]

        # ── HUD overlay ───────────────────────────────────────────
        draw_hud(display, frame_num, total_frames, fps,
                 len(tracks), len(ids_in_zone), vlogger.total)

        writer.write(display)
        cv2.imshow("Parking Violation Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  ⏹  User stopped early.")
            break

    # ── Teardown ──────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  ✅  DONE")
    print(f"      Total violations : {vlogger.total}")
    print(f"      Annotated video  : {OUTPUT_PATH}")
    print(f"      Violations CSV   : {VIOLATIONS_CSV}")
    print(f"      Snapshots        : {SNAPSHOTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
