# ============================================================
#  config.py  —  Central Configuration
#  Edit these settings to match your setup
# ============================================================

# ------ Input / Output -------------------------------------------------------
VIDEO_PATH       = "sample.mp4"              # Default input video
OUTPUT_PATH      = "violations/output.mp4"   # Annotated output video
VIOLATIONS_CSV   = "violations/violations.csv"
SNAPSHOTS_DIR    = "violations/snapshots"

# ------ YOLOv8 ---------------------------------------------------------------
YOLO_MODEL       = "yolov8n.pt"   # nano model (fast); swap to yolov8s.pt for better accuracy
# COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES  = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.4

# ------ DeepSORT -------------------------------------------------------------
DEEPSORT_MAX_AGE = 30   # frames to keep a lost track alive

# ------ Violation Logic ------------------------------------------------------
# How many seconds a vehicle must stay in the zone before it's a violation
PARK_DURATION_THRESHOLD = 3.0   # seconds

FPS_DEFAULT = 25   # fallback if video FPS metadata is missing

# ------ Display Colors (BGR) -------------------------------------------------
ZONE_COLOR      = (0, 0, 255)    # Red fill for illegal zone
ZONE_ALPHA      = 0.30           # Zone transparency
TRACK_COLOR     = (0, 220, 0)    # Green box — normal vehicles
ORANGE_COLOR    = (0, 165, 255)  # Orange — in zone but not yet violation
VIOLATION_COLOR = (0, 0, 255)    # Red box — confirmed violation
TEXT_COLOR      = (255, 255, 255)
HUD_BG_COLOR    = (20, 20, 20)

# ------ Tesseract (optional override) ----------------------------------------
# If tesseract is not on your PATH, set the full path here, e.g.:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD   = None   # None = use system PATH
