# ============================================================
#  zone_drawer.py  —  Interactive Illegal Zone Drawing Tool
# ============================================================
#
#  Controls:
#    LEFT CLICK   → Add a polygon point
#    RIGHT CLICK  → Undo last point
#    ENTER or C   → Confirm & close the zone
#    R            → Reset (clear all points)
#    Q            → Quit without saving
# ============================================================

import cv2
import numpy as np


class ZoneDrawer:
    """
    Opens an OpenCV window on the first video frame and lets the user
    draw a closed polygon by clicking.  Returns the polygon as a
    numpy array of (x, y) integer points.
    """

    WINDOW = "📍 Draw Illegal Parking Zone  [LEFT=add  RIGHT=undo  ENTER=confirm  R=reset]"

    def __init__(self, frame: np.ndarray):
        self.original = frame.copy()
        self.points   = []
        self.done     = False

    # ------------------------------------------------------------------ #
    #  Mouse callback                                                      #
    # ------------------------------------------------------------------ #
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    # ------------------------------------------------------------------ #
    #  Render helpers                                                      #
    # ------------------------------------------------------------------ #
    def _render(self) -> np.ndarray:
        display = self.original.copy()

        # Semi-transparent instruction banner
        banner = display.copy()
        cv2.rectangle(banner, (0, 0), (display.shape[1], 50), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.7, display, 0.3, 0, display)
        cv2.putText(display,
                    "LEFT CLICK = add point | RIGHT CLICK = undo | ENTER = confirm | R = reset",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if not self.points:
            return display

        pts = np.array(self.points, dtype=np.int32)

        # Filled polygon preview (semi-transparent red)
        if len(self.points) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)

        # Polygon edges
        for i in range(1, len(self.points)):
            cv2.line(display, self.points[i - 1], self.points[i], (0, 255, 0), 2)
        if len(self.points) >= 3:
            cv2.line(display, self.points[-1], self.points[0], (0, 255, 0), 1)

        # Point markers
        for pt in self.points:
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.circle(display, pt, 6, (255, 255, 255), 1)

        # Point counter
        counter_text = f"Points: {len(self.points)}"
        (tw, th), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (10, 58), (10 + tw + 8, 58 + th + 8), (20, 20, 20), -1)
        cv2.putText(display, counter_text, (14, 58 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        return display

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def draw(self) -> np.ndarray | None:
        """
        Blocks until the user confirms or quits.
        Returns np.ndarray of shape (N, 2) or None if aborted.
        """
        print("\n" + "=" * 60)
        print("  STEP 1 — DRAW YOUR ILLEGAL PARKING ZONE")
        print("=" * 60)
        print("  LEFT CLICK  → Add polygon point")
        print("  RIGHT CLICK → Undo last point")
        print("  ENTER / C   → Confirm zone  (need ≥ 3 points)")
        print("  R           → Reset all points")
        print("  Q           → Quit")
        print("=" * 60 + "\n")

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1280, 720)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        while not self.done:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord('c')):          # Enter or C → confirm
                if len(self.points) >= 3:
                    self.done = True
                else:
                    print("  ⚠  Need at least 3 points to define a zone!")
            elif key == ord('r'):              # R → reset
                self.points.clear()
                print("  ↩  Zone reset.")
            elif key == ord('q'):              # Q → abort
                cv2.destroyWindow(self.WINDOW)
                return None

        cv2.destroyWindow(self.WINDOW)
        polygon = np.array(self.points, dtype=np.int32)
        print(f"  ✅ Zone confirmed — {len(polygon)} points.\n")
        return polygon
