# ============================================================
#  zone_drawer.py  —  Interactive Illegal Zone Drawing Tool
# ============================================================
#
#  Controls:
#    LEFT CLICK   → Add a polygon point
#    RIGHT CLICK  → Undo last point
#    ENTER or C   → Confirm & close the zone (need >= 3 pts)
#    R            → Reset (clear all points)
#    Q            → Quit without saving
# ============================================================

import cv2
import numpy as np


class ZoneDrawer:
    """
    Opens a resizable, maximisable OpenCV window on the first video frame
    and lets the user draw a closed polygon by clicking.
    Returns the polygon as a numpy array of (x, y) integer points.
    """

    # Plain ASCII window name — emoji in titles breaks mouse callbacks on Windows
    WINDOW = "Draw Illegal Parking Zone"

    def __init__(self, frame: np.ndarray):
        # Resize frame to a sensible display size if it is very large or small
        h, w = frame.shape[:2]
        # Target display: fit inside 1280×720 while keeping aspect ratio
        scale = min(1280 / w, 720 / h, 1.0)   # never upscale
        if scale < 1.0:
            dw, dh = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

        self.original  = frame.copy()
        self.disp_h, self.disp_w = frame.shape[:2]
        self.points    = []          # list of (x, y) tuples
        self.mouse_pos = (0, 0)      # live cursor position for preview line
        self.done      = False

    # ------------------------------------------------------------------ #
    #  Mouse callback                                                      #
    # ------------------------------------------------------------------ #
    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
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
        W, H = self.disp_w, self.disp_h

        # ── Instruction banner ────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (W, 44), (15, 15, 15), -1)
        cv2.putText(
            display,
            "LEFT CLICK=add  |  RIGHT CLICK=undo  |  ENTER=confirm  |  R=reset  |  Q=quit",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1,
            cv2.LINE_AA,
        )

        if not self.points:
            # Hint when no points yet
            cv2.putText(
                display,
                "Click on the frame to start marking the illegal parking zone corners",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                cv2.LINE_AA,
            )
            return display

        pts = np.array(self.points, dtype=np.int32)

        # ── Filled polygon preview (semi-transparent red) ─────────────
        if len(self.points) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 220))
            cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)

        # ── Polygon edges ─────────────────────────────────────────────
        for i in range(1, len(self.points)):
            cv2.line(display, self.points[i - 1], self.points[i],
                     (50, 220, 50), 2, cv2.LINE_AA)

        # Closing edge preview (last point → cursor)
        cv2.line(display, self.points[-1], self.mouse_pos,
                 (0, 200, 255), 1, cv2.LINE_AA)

        # Dashed closing edge (last → first) when ≥ 3 points
        if len(self.points) >= 3:
            cv2.line(display, self.points[-1], self.points[0],
                     (50, 220, 50), 1, cv2.LINE_AA)

        # ── Point markers ─────────────────────────────────────────────
        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 7, (50, 220, 50), -1)
            cv2.circle(display, pt, 7, (255, 255, 255), 1)
            # Number each point
            cv2.putText(display, str(i + 1), (pt[0] + 9, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1,
                        cv2.LINE_AA)

        # Cursor crosshair
        mx, my = self.mouse_pos
        cv2.line(display, (mx - 12, my), (mx + 12, my), (0, 200, 255), 1)
        cv2.line(display, (mx, my - 12), (mx, my + 12), (0, 200, 255), 1)

        # ── Status bar ────────────────────────────────────────────────
        n = len(self.points)
        if n < 3:
            status = f"Points: {n}  |  Need at least 3 to confirm"
            col = (0, 200, 255)
        else:
            status = f"Points: {n}  |  Press ENTER to confirm zone"
            col = (50, 220, 50)

        cv2.rectangle(display, (0, H - 32), (W, H), (15, 15, 15), -1)
        cv2.putText(display, status, (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)

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
        print("  ENTER / C   → Confirm zone  (need >= 3 points)")
        print("  R           → Reset all points")
        print("  Q           → Quit")
        print("=" * 60 + "\n")

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.disp_w, self.disp_h)
        # Allow user to resize / maximise freely
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_ASPECT_RATIO,
                              cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        while not self.done:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKey(16) & 0xFF   # ~60 fps refresh

            if key in (13, ord('c'), ord('C')):    # Enter or C → confirm
                if len(self.points) >= 3:
                    self.done = True
                else:
                    print("  ⚠  Need at least 3 points to define a zone!")

            elif key in (ord('r'), ord('R')):      # R → reset
                self.points.clear()
                print("  ↩  Zone reset.")

            elif key in (ord('q'), ord('Q')):      # Q → abort
                cv2.destroyWindow(self.WINDOW)
                return None

            elif key == 27:                        # ESC → abort
                cv2.destroyWindow(self.WINDOW)
                return None

        cv2.destroyWindow(self.WINDOW)
        polygon = np.array(self.points, dtype=np.int32)
        print(f"  ✅ Zone confirmed — {len(polygon)} points.\n")
        return polygon
