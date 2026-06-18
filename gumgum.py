"""
Gum-Gum — webcam hand-stretch effect (One Piece / Luffy inspired).

Two hands: a *grabber* hand pinches one of the *body* hand's fingertips and
pulls. The real finger photo is deformed like elastic via As-Rigid-As-Possible
2D mesh deformation (Igarashi et al. 2005, see arap.py) — rooted at the live
MCP knuckle, dragged to the grabber pinch — then snaps back with a punchy
under-damped spring, like Luffy's Gomu Gomu.

Enhanced features:
  - All 5 fingers (including thumb) can be grabbed and stretched
  - Face stretching: grab nose, cheeks, chin, forehead with full 478-point mesh
  - Improved precision via adaptive confidence, Kalman-like smoothing, and
    multi-landmark finger width estimation
  - Full MediaPipe landmark utilization (21 hand points, 478 face points)

Controls (no on-screen text; the camera feed stays clean):
    ESC / q   quit              SPACE  mirror on/off
    h         toggle hand guides (skeleton + pinch markers)
    r         swap body/grabber roles      s  toggle skin silhouette
    , / .     specular gain  -/+           j / k  rubber thinning  -/+
    l / o     snap overshoot -/+           i  toggle gap inpaint
    m         finger-mask overlay          g  ARAP mesh wireframe
    f         toggle face-stretch mode (grab facial features)
    d         toggle debug overlay (landmarks, pinch distance, state)

Run:
    python gumgum.py                   # native window
    python gumgum.py --serve           # MJPEG stream on localhost:$PORT
    python gumgum.py --selftest        # headless math check, no camera
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

import arap  # ARAP 2D mesh deformation engine (Igarashi 2005)

# This build of mediapipe (0.10.35 on this platform) ships only the modern
# Tasks API, not the legacy mp.solutions.* graphs — so we use HandLandmarker /
# FaceLandmarker with downloaded .task bundles.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = None


# ---------------------------------------------------------------------------
# Tuning — these constants control the *feel*. Adjust here.
# ---------------------------------------------------------------------------
class Config:
    CAM_INDEX = 0
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720

    HAND_MODEL = "models/hand_landmarker.task"
    FACE_MODEL = "models/face_landmarker.task"

    # --- Detection confidence (higher = more precise, fewer false positives) ---
    HAND_DETECTION_CONF = 0.7
    HAND_TRACKING_CONF = 0.6
    FACE_DETECTION_CONF = 0.7
    FACE_TRACKING_CONF = 0.6

    # Pinch hysteresis (pixels between thumb tip and finger tip).
    PINCH_ENGAGE = 34.0   # must get closer than this to start a stretch
    PINCH_RELEASE = 52.0  # must open wider than this to let go

    # Live pinch-point smoothing — adaptive Kalman-like EMA.
    # Base smoothing factor (0..1). Higher = snappier/jitterier.
    TIP_SMOOTHING = 0.55
    # When velocity is high, increase smoothing for responsiveness.
    TIP_SMOOTHING_HIGH_VEL = 0.8
    VEL_THRESHOLD = 40.0  # px/frame velocity threshold for adaptive smoothing

    # Rubber appearance.
    BASE_WIDTH = 34.0       # half-thickness at the anchor when relaxed (px)
    TIP_WIDTH_FRAC = 0.35   # tip half-thickness as fraction of anchor width
    MIN_WIDTH = 4.0         # never thinner than this (px)
    # Volume conservation: thickness shrinks as length grows.
    THINNING_LENGTH = 520.0
    # Sideways bulge of the bezier midpoint, as a fraction of length, clamped.
    BULGE_FRAC = 0.16
    BULGE_MAX = 70.0
    BEZIER_SAMPLES = 28     # polygon resolution along the tube

    RUBBER_COLOR = (60, 70, 235)      # BGR — Luffy red-ish
    RUBBER_SHADE = (35, 40, 150)      # darker outline for depth
    HIGHLIGHT_COLOR = (180, 190, 255) # light tube highlight
    ANCHOR_COLOR = (40, 220, 255)     # yellow-ish anchor marker

    # Snap-back animation — punchy under-damped spring (fast: 250-350ms).
    SNAP_DURATION = 0.30    # seconds for the recoil
    SNAP_OVERSHOOT = 1.9    # ease-out-back tension (used by demo / legacy curve)
    SNAP_WOBBLE = 2.4       # spring oscillations across recoil (secondary wobble)

    # --- ARAP mesh deformation ---
    ANCHOR_RADIUS = 0.9     # anchor-handle radius around the MCP knuckle (*fw)
    USE_SILHOUETTE = True   # toggle 's' — fall back to the raw capsule if off
    SKIN_DCR = 22           # YCrCb Cr tolerance around the finger's median
    SKIN_DCB = 22           # YCrCb Cb tolerance
    SILHOUETTE_ERODE = 2    # px inward erosion so the edge sits inside the skin

    # --- Rubber feel: middle-band dynamics ---
    RUBBER_THIN_AMT = 0.15
    HIGHLIGHT_GAIN = 0.2
    BASE_PROTECT = 1.2
    TIP_PROTECT = 1.2
    ELONG_BLUR = 0.0
    ELONG_THRESH = 1.6
    RUBBER_THIN_STEP = 0.03
    OVERSHOOT_STEP = 0.15

    # --- Displacement-field warp ---
    SIGMA_SCALE = 1.3
    ALONG_GAMMA = 1.6
    MAX_STRETCH_PX = None
    SIGMA_SCALE_STEP = 0.1

    # --- Finger mask + masked composite ---
    MASK_WIDTH_SCALE = 0.95
    MASK_DILATE = 5
    MASK_FEATHER = 9
    MASK_TAPER_LENGTH = 220.0
    MASK_TIP_MIN = 0.12
    INPAINT_GAP = True
    USE_SKIN_REFINE = False

    # --- Constructed limb ---
    STRIP_HEIGHT = 13
    TUBE_WIDTH_SCALE = 1.0
    TIP_PATCH_SCALE = 1.55
    TAPER_FRAC = 0.18
    N_JOINTS = 2.5
    UNDULATION_AMP = 0.12
    CREASE_OPACITY = 0.35
    HIGHLIGHT_OFFSET = 0.30
    COLOR_MATCH = 0.5
    LIMB_CURVATURE = 0.12
    LIMB_FEATHER = 7
    LIMB_SHADING = True
    MAX_STRIPS = 280
    TUBE_WIDTH_STEP = 0.08

    # Two-hand grab: grabber pinch must close within this of a body fingertip.
    ATTACH_RADIUS = 50.0

    # Default handedness roles (on the mirrored feed). Toggle with 'r'.
    BODY_HAND = "Left"     # this hand gets stretched
    GRABBER_HAND = "Right" # this hand pinches & pulls

    # --- Face stretch mode ---
    # Grabbable face regions with their landmark indices (MediaPipe 478-point mesh).
    # Each region is (name, [landmark_indices], grab_radius_px).
    FACE_REGIONS = {
        "nose_tip": ([1, 2, 4, 5, 6], 45.0),
        "left_cheek": ([50, 101, 116, 117, 118, 119, 120, 121, 122, 123], 60.0),
        "right_cheek": ([280, 330, 345, 346, 347, 348, 349, 350, 351, 352], 60.0),
        "chin": ([152, 175, 176, 148, 149, 150, 151, 377, 378, 379, 397, 400], 55.0),
        "forehead": ([10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338], 55.0),
        "left_eyebrow": ([63, 66, 70, 105, 107], 40.0),
        "right_eyebrow": ([293, 296, 300, 334, 336], 40.0),
        "upper_lip": ([0, 13, 14, 17, 37, 39, 40, 61, 185, 267, 269, 270, 291, 409], 35.0),
        "lower_lip": ([14, 17, 84, 87, 88, 91, 95, 146, 178, 181, 314, 317, 318, 321, 324, 375, 402, 405], 35.0),
    }
    FACE_MESH_BOUNDARY_PTS = 64   # boundary resampling for face region mesh
    FACE_ANCHOR_RADIUS = 1.2      # anchor radius multiplier for face mesh


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ease_out_back(t, overshoot=Config.SNAP_OVERSHOOT):
    """Overshoots past 1.0 then settles back — the rubber recoil curve."""
    c1 = overshoot
    c3 = c1 + 1.0
    u = t - 1.0
    return 1.0 + c3 * u * u * u + c1 * u * u


def ease_spring(t, overshoot=Config.SNAP_OVERSHOOT, wobble=Config.SNAP_WOBBLE):
    """
    Under-damped spring: shoots PAST 1.0 (tip snaps past the anchor), a single
    smaller secondary wobble, then settles.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    damp = 3.0 / max(0.3, overshoot)
    return 1.0 - math.exp(-damp * t) * math.cos(wobble * math.pi * t)


def quad_bezier(p0, p1, p2, t):
    """Quadratic bezier point at parameter t (p1 is the control point)."""
    mt = 1.0 - t
    a = mt * mt
    b = 2.0 * mt * t
    c = t * t
    return (a * p0[0] + b * p1[0] + c * p2[0],
            a * p0[1] + b * p1[1] + c * p2[1])


def perpendicular(dx, dy):
    """Unit vector perpendicular to (dx, dy); (0,0) if degenerate."""
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 0.0)
    return (-dy / length, dx / length)


# ---------------------------------------------------------------------------
# Adaptive smoothing (Kalman-like EMA)
# ---------------------------------------------------------------------------
class AdaptiveSmoother:
    """
    Adaptive exponential moving average that responds faster to rapid movement
    and smooths more aggressively during slow/stationary phases.
    """
    def __init__(self, cfg=Config):
        self.cfg = cfg
        self.pos = None
        self.vel = (0.0, 0.0)

    def update(self, raw_point):
        if raw_point is None:
            return self.pos
        if self.pos is None:
            self.pos = raw_point
            return self.pos
        # Estimate velocity
        vx = raw_point[0] - self.pos[0]
        vy = raw_point[1] - self.pos[1]
        speed = math.hypot(vx, vy)
        # Adaptive smoothing: faster movement -> higher alpha (more responsive)
        if speed > self.cfg.VEL_THRESHOLD:
            alpha = self.cfg.TIP_SMOOTHING_HIGH_VEL
        else:
            # Interpolate between base and high-vel smoothing
            t = speed / self.cfg.VEL_THRESHOLD
            alpha = self.cfg.TIP_SMOOTHING + (
                self.cfg.TIP_SMOOTHING_HIGH_VEL - self.cfg.TIP_SMOOTHING) * t
        self.pos = (self.pos[0] + (raw_point[0] - self.pos[0]) * alpha,
                    self.pos[1] + (raw_point[1] - self.pos[1]) * alpha)
        self.vel = (vx, vy)
        return self.pos

    def reset(self):
        self.pos = None
        self.vel = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Rubber renderer
# ---------------------------------------------------------------------------
def draw_rubber(frame, anchor, tip, cfg=Config):
    """
    Render a tapering, bulging rubber tube from anchor to tip.
    """
    length = dist(anchor, tip)
    dx, dy = tip[0] - anchor[0], tip[1] - anchor[1]
    px, py = perpendicular(dx, dy)

    bulge = min(length * cfg.BULGE_FRAC, cfg.BULGE_MAX)
    mid = lerp(anchor, tip, 0.5)
    ctrl = (mid[0] + px * bulge, mid[1] + py * bulge)

    thinning = 1.0 / (1.0 + length / cfg.THINNING_LENGTH)
    w_anchor = max(cfg.BASE_WIDTH * thinning, cfg.MIN_WIDTH)
    w_tip = max(w_anchor * cfg.TIP_WIDTH_FRAC, cfg.MIN_WIDTH)

    n = cfg.BEZIER_SAMPLES
    left, right, centers = [], [], []
    prev = quad_bezier(anchor, ctrl, tip, 0.0)
    for i in range(n + 1):
        t = i / n
        pt = quad_bezier(anchor, ctrl, tip, t)
        tdx, tdy = pt[0] - prev[0], pt[1] - prev[1]
        if i == 0:
            nxt = quad_bezier(anchor, ctrl, tip, 1.0 / n)
            tdx, tdy = nxt[0] - pt[0], nxt[1] - pt[1]
        lpx, lpy = perpendicular(tdx, tdy)
        w = w_anchor + (w_tip - w_anchor) * t
        left.append((pt[0] + lpx * w, pt[1] + lpy * w))
        right.append((pt[0] - lpx * w, pt[1] - lpy * w))
        centers.append(pt)
        prev = pt

    poly = np.array(left + right[::-1], dtype=np.int32)

    cv2.fillPoly(frame, [poly], cfg.RUBBER_SHADE, lineType=cv2.LINE_AA)
    cv2.fillPoly(frame, [poly], cfg.RUBBER_COLOR, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [poly], True, cfg.RUBBER_SHADE, 2, cv2.LINE_AA)

    # Highlight line
    hl = []
    for i, c in enumerate(centers):
        t = i / n
        w = (w_anchor + (w_tip - w_anchor) * t) * 0.35
        hl.append((int(c[0] + px * w), int(c[1] + py * w)))
    if len(hl) >= 2:
        thick = max(1, int(w_tip * 0.5))
        cv2.polylines(frame, [np.array(hl, np.int32)], False,
                      cfg.HIGHLIGHT_COLOR, thick, cv2.LINE_AA)

    # Rounded cap at the tip.
    cv2.circle(frame, (int(tip[0]), int(tip[1])), int(w_tip),
               cfg.RUBBER_COLOR, -1, cv2.LINE_AA)
    return centers


def build_finger_mask(h, w, body_px, grabbed_tip, finger_width, stretch_len,
                      cfg=Config):
    """
    Paint a single-channel uint8 mask of the grabbed finger from its landmark
    bone chain.
    """
    mask = np.zeros((h, w), np.uint8)
    chain = FINGER_CHAINS.get(grabbed_tip)
    if chain is None or body_px is None:
        return mask

    base_r = max(3.0, finger_width * cfg.MASK_WIDTH_SCALE * 0.5)
    tip_taper = max(cfg.MASK_TIP_MIN,
                    1.0 / (1.0 + stretch_len / cfg.MASK_TAPER_LENGTH))
    n = len(chain)
    pts = [(int(body_px[i][0]), int(body_px[i][1])) for i in chain]
    radii = [base_r * (1.0 + (tip_taper - 1.0) * (k / (n - 1)))
             for k in range(n)]

    for p, r in zip(pts, radii):
        cv2.circle(mask, p, max(1, int(round(r))), 255, -1, cv2.LINE_AA)
    for i in range(n - 1):
        th = max(1, int(round(radii[i] + radii[i + 1])))
        cv2.line(mask, pts[i], pts[i + 1], 255, th, cv2.LINE_AA)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (cfg.MASK_DILATE, cfg.MASK_DILATE))
    return cv2.dilate(mask, k)


def build_face_region_mask(h, w, face_px, region_indices, cfg=Config):
    """
    Build a mask for a face region using the convex hull of its landmark points.
    Dilated to cover the surrounding skin area for natural deformation.
    """
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([(int(face_px[i][0]), int(face_px[i][1]))
                    for i in region_indices if i < len(face_px)], dtype=np.int32)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    # Dilate to include surrounding skin
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, k)
    return mask


def silhouette_mask(frame, capsule, cfg=Config):
    """
    Tighten the fat landmark capsule to the REAL finger outline.
    """
    if capsule is None or not capsule.any():
        return capsule
    core = cv2.erode(capsule, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (9, 9)))
    if not core.any():
        core = capsule
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr0 = float(np.median(ycc[:, :, 1][core > 0]))
    cb0 = float(np.median(ycc[:, :, 2][core > 0]))
    skin = (((np.abs(ycc[:, :, 1].astype(np.int16) - cr0) <= cfg.SKIN_DCR) &
             (np.abs(ycc[:, :, 2].astype(np.int16) - cb0) <= cfg.SKIN_DCB))
            .astype(np.uint8) * 255)
    tight = cv2.bitwise_and(skin, capsule)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, el)
    tight = cv2.morphologyEx(tight, cv2.MORPH_OPEN, el)
    cnts, _ = cv2.findContours(tight, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return capsule
    out = np.zeros_like(capsule)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    if cfg.SILHOUETTE_ERODE > 0:
        e = cfg.SILHOUETTE_ERODE * 2 + 1
        out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    if out.sum() < 0.22 * capsule.sum():
        return capsule
    return out


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
IDLE, STRETCHING, SNAPPING = "idle", "stretching", "snapping"


class StretchFX:
    def __init__(self, cfg=Config, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.state = IDLE
        self.pinched = False
        self.anchor = None
        self.tip = None
        self.snap_from = None
        self.snap_start = 0.0

    def _update_pinch_latch(self, pinch_dist):
        if not self.pinched and pinch_dist <= self.cfg.PINCH_ENGAGE:
            self.pinched = True
        elif self.pinched and pinch_dist >= self.cfg.PINCH_RELEASE:
            self.pinched = False
        return self.pinched

    def update(self, pinch_point, pinch_dist, anchor_override=None):
        now = self.clock()
        has_hand = pinch_point is not None
        pinched = self._update_pinch_latch(pinch_dist) if has_hand else False

        if has_hand:
            if self.tip is None:
                self.tip = pinch_point
            else:
                s = self.cfg.TIP_SMOOTHING
                self.tip = (self.tip[0] + (pinch_point[0] - self.tip[0]) * s,
                            self.tip[1] + (pinch_point[1] - self.tip[1]) * s)

        if self.state == IDLE:
            if pinched and has_hand:
                self.anchor = anchor_override or pinch_point
                self.tip = pinch_point
                self.state = STRETCHING

        elif self.state == STRETCHING:
            if not pinched or not has_hand:
                self.snap_from = self.tip if self.tip else self.anchor
                self.snap_start = now
                self.state = SNAPPING

        elif self.state == SNAPPING:
            elapsed = now - self.snap_start
            if elapsed >= self.cfg.SNAP_DURATION:
                self.state = IDLE
                self.anchor = None
            elif pinched and has_hand:
                self.anchor = anchor_override or self.anchor or pinch_point
                self.tip = pinch_point
                self.state = STRETCHING

    def render_point(self):
        if self.state == STRETCHING:
            return self.tip
        if self.state == SNAPPING:
            t = (self.clock() - self.snap_start) / self.cfg.SNAP_DURATION
            t = max(0.0, min(1.0, t))
            e = ease_out_back(t)
            return lerp(self.snap_from, self.anchor, e)
        return None

    def draw(self, frame):
        if self.state in (STRETCHING, SNAPPING) and self.anchor:
            tip = self.render_point()
            if tip is not None:
                draw_rubber(frame, self.anchor, tip, self.cfg)
                cv2.circle(frame, (int(self.anchor[0]), int(self.anchor[1])),
                           7, self.cfg.ANCHOR_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(self.anchor[0]), int(self.anchor[1])),
                           7, (0, 0, 0), 1, cv2.LINE_AA)


class GrabFX:
    """
    Two-hand grab mechanic. A *grabber* hand pinches near one of the *body*
    hand's fingertips (all 5 fingers including thumb) to attach; the rubber
    then runs from that finger's live MCP/CMC knuckle to the live grabber
    pinch point.
    """

    def __init__(self, cfg=Config, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.state = IDLE
        self.pinched = False
        self.grabbed_tip = None
        self.tip = None
        self.snap_from = None
        self.snap_start = 0.0
        self.last_knuckle = None
        self.last_fingertip = None
        self.overshoot = cfg.SNAP_OVERSHOOT
        self.wobble = cfg.SNAP_WOBBLE
        self.smoother = AdaptiveSmoother(cfg)

    def _latch(self, pinch_dist):
        if not self.pinched and pinch_dist <= self.cfg.PINCH_ENGAGE:
            self.pinched = True
        elif self.pinched and pinch_dist >= self.cfg.PINCH_RELEASE:
            self.pinched = False
        return self.pinched

    def _nearest_body_tip(self, point, body_px):
        """Nearest body fingertip (all 5 fingers) within ATTACH_RADIUS."""
        best, best_d = None, self.cfg.ATTACH_RADIUS
        for tip in ALL_FINGER_TIPS:
            d = dist(point, body_px[tip])
            if d < best_d:
                best, best_d = tip, d
        return best

    def update(self, grab_pinch, grab_dist, body_px):
        now = self.clock()
        has_grab = grab_pinch is not None
        pinched = self._latch(grab_dist) if has_grab else False

        if has_grab:
            self.tip = self.smoother.update(grab_pinch)
        else:
            self.smoother.reset()

        # Cache live body positions for the grabbed finger.
        if body_px is not None and self.grabbed_tip is not None:
            self.last_knuckle = body_px[TIP_TO_MCP[self.grabbed_tip]]
            self.last_fingertip = body_px[self.grabbed_tip]

        if self.state == IDLE:
            self.grabbed_tip = None
            if pinched and has_grab and body_px is not None:
                tip = self._nearest_body_tip(grab_pinch, body_px)
                if tip is not None:
                    self.grabbed_tip = tip
                    self.tip = self.smoother.update(grab_pinch)
                    self.last_knuckle = body_px[TIP_TO_MCP[tip]]
                    self.last_fingertip = body_px[tip]
                    self.state = STRETCHING

        elif self.state == STRETCHING:
            if not pinched or not has_grab or body_px is None:
                self.snap_from = self.tip or self.last_knuckle
                self.snap_start = now
                self.state = SNAPPING

        elif self.state == SNAPPING:
            if now - self.snap_start >= self.cfg.SNAP_DURATION:
                self.state = IDLE
                self.grabbed_tip = None
            elif pinched and has_grab and body_px is not None:
                tip = self._nearest_body_tip(grab_pinch, body_px)
                if tip is not None:
                    self.grabbed_tip = tip
                    self.tip = self.smoother.update(grab_pinch)
                    self.state = STRETCHING

    def anchor_point(self, body_px):
        if self.grabbed_tip is None:
            return None
        if body_px is not None:
            return body_px[TIP_TO_MCP[self.grabbed_tip]]
        return self.last_knuckle

    def render_tip(self, body_px):
        if self.state == STRETCHING:
            return self.tip
        if self.state == SNAPPING:
            t = max(0.0, min(1.0,
                    (self.clock() - self.snap_start) / self.cfg.SNAP_DURATION))
            target = (body_px[self.grabbed_tip]
                      if (body_px is not None and self.grabbed_tip is not None)
                      else self.last_fingertip)
            if self.snap_from is None or target is None:
                return None
            return lerp(self.snap_from, target,
                        ease_spring(t, self.overshoot, self.wobble))
        return None

    def draw(self, frame, body_px):
        if self.state not in (STRETCHING, SNAPPING):
            return
        anchor = self.anchor_point(body_px)
        tip = self.render_tip(body_px)
        if anchor is None or tip is None:
            return
        draw_rubber(frame, anchor, tip, self.cfg)
        a = (int(anchor[0]), int(anchor[1]))
        cv2.circle(frame, a, 7, self.cfg.ANCHOR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, a, 7, (0, 0, 0), 1, cv2.LINE_AA)


class FaceGrabFX:
    """
    Face-stretch mechanic. The grabber hand pinches near a facial feature
    (nose, cheeks, chin, forehead, eyebrows, lips) and pulls it with ARAP
    deformation on the 478-point face mesh.
    """

    def __init__(self, cfg=Config, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.state = IDLE
        self.pinched = False
        self.grabbed_region = None  # name of the grabbed face region
        self.grabbed_center = None  # center point of the grabbed region
        self.tip = None
        self.snap_from = None
        self.snap_start = 0.0
        self.overshoot = cfg.SNAP_OVERSHOOT
        self.wobble = cfg.SNAP_WOBBLE
        self.smoother = AdaptiveSmoother(cfg)
        self.mesh_cache = None

    def _latch(self, pinch_dist):
        if not self.pinched and pinch_dist <= self.cfg.PINCH_ENGAGE:
            self.pinched = True
        elif self.pinched and pinch_dist >= self.cfg.PINCH_RELEASE:
            self.pinched = False
        return self.pinched

    def _nearest_face_region(self, point, face_px):
        """Find the nearest grabbable face region within its grab radius."""
        best_name, best_center, best_d = None, None, float("inf")
        for name, (indices, radius) in self.cfg.FACE_REGIONS.items():
            valid_indices = [i for i in indices if i < len(face_px)]
            if not valid_indices:
                continue
            # Region center = mean of its landmarks
            cx = sum(face_px[i][0] for i in valid_indices) / len(valid_indices)
            cy = sum(face_px[i][1] for i in valid_indices) / len(valid_indices)
            center = (cx, cy)
            d = dist(point, center)
            if d < radius and d < best_d:
                best_name, best_center, best_d = name, center, d
        return best_name, best_center

    def update(self, grab_pinch, grab_dist, face_px):
        now = self.clock()
        has_grab = grab_pinch is not None
        has_face = face_px is not None and len(face_px) > 0
        pinched = self._latch(grab_dist) if has_grab else False

        if has_grab:
            self.tip = self.smoother.update(grab_pinch)
        else:
            self.smoother.reset()

        if self.state == IDLE:
            self.grabbed_region = None
            self.mesh_cache = None
            if pinched and has_grab and has_face:
                name, center = self._nearest_face_region(grab_pinch, face_px)
                if name is not None:
                    self.grabbed_region = name
                    self.grabbed_center = center
                    self.tip = self.smoother.update(grab_pinch)
                    self.state = STRETCHING

        elif self.state == STRETCHING:
            # Update grabbed center from live face landmarks
            if has_face and self.grabbed_region is not None:
                indices = self.cfg.FACE_REGIONS[self.grabbed_region][0]
                valid = [i for i in indices if i < len(face_px)]
                if valid:
                    cx = sum(face_px[i][0] for i in valid) / len(valid)
                    cy = sum(face_px[i][1] for i in valid) / len(valid)
                    self.grabbed_center = (cx, cy)
            if not pinched or not has_grab:
                self.snap_from = self.tip or self.grabbed_center
                self.snap_start = now
                self.state = SNAPPING

        elif self.state == SNAPPING:
            if now - self.snap_start >= self.cfg.SNAP_DURATION:
                self.state = IDLE
                self.grabbed_region = None
                self.mesh_cache = None
            elif pinched and has_grab and has_face:
                name, center = self._nearest_face_region(grab_pinch, face_px)
                if name is not None:
                    self.grabbed_region = name
                    self.grabbed_center = center
                    self.tip = self.smoother.update(grab_pinch)
                    self.state = STRETCHING

    def render_tip(self):
        if self.state == STRETCHING:
            return self.tip
        if self.state == SNAPPING:
            t = max(0.0, min(1.0,
                    (self.clock() - self.snap_start) / self.cfg.SNAP_DURATION))
            if self.snap_from is None or self.grabbed_center is None:
                return None
            return lerp(self.snap_from, self.grabbed_center,
                        ease_spring(t, self.overshoot, self.wobble))
        return None

    def draw(self, frame, face_px):
        """Draw the rubber tube and perform ARAP face deformation."""
        if self.state not in (STRETCHING, SNAPPING):
            return
        if self.grabbed_center is None:
            return
        tip = self.render_tip()
        if tip is None:
            return

        # Draw rubber tube from face anchor to pull point
        draw_rubber(frame, self.grabbed_center, tip, self.cfg)
        a = (int(self.grabbed_center[0]), int(self.grabbed_center[1]))
        cv2.circle(frame, a, 7, self.cfg.ANCHOR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, a, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # ARAP face deformation
        if face_px is not None and self.grabbed_region is not None:
            h, w = frame.shape[:2]
            indices = self.cfg.FACE_REGIONS[self.grabbed_region][0]
            valid = [i for i in indices if i < len(face_px)]
            if len(valid) >= 3:
                region_mask = build_face_region_mask(h, w, face_px, valid, self.cfg)
                if region_mask.any():
                    # Build or reuse mesh
                    if self.mesh_cache is None or self.mesh_cache.get("region") != self.grabbed_region:
                        mesh = arap.build_finger_mesh(region_mask,
                                                      n_boundary=self.cfg.FACE_MESH_BOUNDARY_PTS)
                        if mesh is not None:
                            rest, tris = mesh
                            # Anchor: vertices far from the grabbed center
                            # Tip: vertex nearest to grabbed center
                            center_arr = np.array([self.grabbed_center[0],
                                                   self.grabbed_center[1]])
                            dists = np.hypot(rest[:, 0] - center_arr[0],
                                             rest[:, 1] - center_arr[1])
                            tip_v = int(dists.argmin())
                            # Anchors: boundary vertices far from center
                            median_d = np.median(dists)
                            anchors = [int(i) for i in np.where(dists > median_d * 1.3)[0]
                                       if i != tip_v]
                            if not anchors:
                                anchors = [int(dists.argmax())]
                            try:
                                solver = arap.ARAPSolver(rest, tris, anchors + [tip_v])
                                tex = np.dstack([frame, region_mask]).copy()
                                free = np.ones(len(rest), bool)
                                free[anchors + [tip_v]] = False
                                self.mesh_cache = {
                                    "region": self.grabbed_region,
                                    "solver": solver, "rest": rest,
                                    "tris": tris, "tex": tex,
                                    "anchors": anchors, "tipv": tip_v,
                                    "free": free, "center0": self.grabbed_center
                                }
                            except Exception:
                                self.mesh_cache = None

                    # Solve and render deformation
                    if self.mesh_cache is not None:
                        mc = self.mesh_cache
                        targets = {i: (mc["rest"][i][0], mc["rest"][i][1])
                                   for i in mc["anchors"]}
                        targets[mc["tipv"]] = (tip[0], tip[1])
                        try:
                            deformed = mc["solver"].solve(targets)
                            arap.render_mesh(mc["tex"], mc["rest"], deformed,
                                             mc["tris"], frame, self.cfg.LIMB_FEATHER,
                                             0.0, None, None, 0.0)
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# Hand / pinch extraction
# ---------------------------------------------------------------------------
# All 5 fingertips (including thumb)
ALL_FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky (non-thumb)
THUMB_TIP = 4

# Fingertip -> its MCP/CMC knuckle (the rubber roots here while grabbed).
TIP_TO_MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}

# Bone chains knuckle->tip per finger, used to paint the finger mask.
FINGER_CHAINS = {
    4: [1, 2, 3, 4],       # thumb: CMC -> MCP -> IP -> TIP
    8: [5, 6, 7, 8],       # index
    12: [9, 10, 11, 12],   # middle
    16: [13, 14, 15, 16],  # ring
    20: [17, 18, 19, 20],  # pinky
}

# Standard 21-point hand topology.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
)


def to_px(landmarks, w, h):
    """Normalized landmarks -> list of (x, y) pixel tuples."""
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def finger_width_px(body_px, grabbed_tip):
    """
    Estimate finger width using multiple landmarks for better precision.
    Uses the perpendicular distance between adjacent finger bones and
    the MCP-knuckle spacing.
    """
    if body_px is None:
        return 40.0
    mcps = [5, 9, 13, 17]
    gaps = [dist(body_px[mcps[i]], body_px[mcps[i + 1]])
            for i in range(len(mcps) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 40.0

    # For thumb, use a slightly wider estimate
    if grabbed_tip == 4:
        return avg_gap * 1.2

    # For other fingers, also consider the bone length ratio for better estimate
    chain = FINGER_CHAINS.get(grabbed_tip, [])
    if len(chain) >= 2:
        bone_len = dist(body_px[chain[0]], body_px[chain[1]])
        # Finger width is typically ~40-50% of the first bone length
        width_from_bone = bone_len * 0.45
        # Blend both estimates
        return (avg_gap * 0.6 + width_from_bone * 0.4)

    return avg_gap


def draw_hand_skeleton(frame, pts, color=(0, 200, 0), hide_tip=None):
    """Draw a hand from pixel-space points."""
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        if hide_tip is not None and (a == hide_tip or b == hide_tip):
            continue
        cv2.line(frame, ipts[a], ipts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(ipts):
        if i == hide_tip:
            continue
        col = (0, 0, 255) if i in ALL_FINGER_TIPS else (0, 255, 255)
        cv2.circle(frame, p, 4, col, -1, cv2.LINE_AA)


def pinch_from_landmarks(landmarks, w, h, any_finger):
    """
    Return (pinch_point, pinch_dist, finger_idx) in pixel space.
    Enhanced: uses all 5 fingers when any_finger is True.
    """
    thumb = (landmarks[THUMB_TIP].x * w, landmarks[THUMB_TIP].y * h)
    if any_finger:
        best_idx, best_d, best_pt = 8, float("inf"), None
        for idx in ALL_FINGER_TIPS:
            if idx == THUMB_TIP:
                continue
            pt = (landmarks[idx].x * w, landmarks[idx].y * h)
            d = dist(thumb, pt)
            if d < best_d:
                best_idx, best_d, best_pt = idx, d, pt
        finger = best_pt
        finger_idx = best_idx
    else:
        finger = (landmarks[8].x * w, landmarks[8].y * h)
        finger_idx = 8
    d = dist(thumb, finger)
    mid = lerp(thumb, finger, 0.5)
    return mid, d, finger_idx


def draw_face_landmarks_debug(frame, face_px, regions, cfg=Config):
    """Draw face landmark points and grabbable regions for debug mode."""
    # Draw all face landmarks as tiny dots
    for i, (x, y) in enumerate(face_px):
        cv2.circle(frame, (int(x), int(y)), 1, (100, 100, 100), -1)

    # Highlight grabbable regions
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255),
              (255, 128, 0)]
    for ci, (name, (indices, radius)) in enumerate(cfg.FACE_REGIONS.items()):
        color = colors[ci % len(colors)]
        valid = [i for i in indices if i < len(face_px)]
        if not valid:
            continue
        cx = sum(face_px[i][0] for i in valid) / len(valid)
        cy = sum(face_px[i][1] for i in valid) / len(valid)
        cv2.circle(frame, (int(cx), int(cy)), int(radius * 0.3), color, 2, cv2.LINE_AA)
        for i in valid:
            cv2.circle(frame, (int(face_px[i][0]), int(face_px[i][1])),
                       3, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Self-test (no camera) — verifies the math/state behaves.
# ---------------------------------------------------------------------------
def selftest():
    ok = True

    # Easing overshoots then settles.
    assert ease_out_back(0.0) == 0.0
    assert abs(ease_out_back(1.0) - 1.0) < 1e-9
    peak = max(ease_out_back(t / 100) for t in range(101))
    assert peak > 1.0, "ease_out_back should overshoot past 1.0"
    print(f"  ease_out_back: endpoints OK, peak overshoot = {peak:.3f}")

    # Bezier endpoints land on anchor/tip.
    a, c, t2 = (0, 0), (50, 80), (100, 0)
    assert quad_bezier(a, c, t2, 0.0) == a
    assert quad_bezier(a, c, t2, 1.0) == t2
    print("  quad_bezier: endpoints OK")

    # Renderer doesn't crash and stays in-bounds-ish.
    frame = np.zeros((480, 640, 3), np.uint8)
    centers = draw_rubber(frame, (100, 240), (520, 200))
    assert len(centers) == Config.BEZIER_SAMPLES + 1
    assert frame.sum() > 0, "rubber should have drawn pixels"
    print(f"  draw_rubber: drew {len(centers)} samples, frame non-empty")

    # Volume conservation
    short = np.zeros((480, 640, 3), np.uint8)
    long_ = np.zeros((480, 640, 3), np.uint8)
    draw_rubber(short, (300, 240), (360, 240))
    draw_rubber(long_, (60, 240), (600, 240))
    short_density = short.sum() / 60.0
    long_density = long_.sum() / 540.0
    assert long_density < short_density, "longer tube should be thinner/unit"
    print(f"  volume conservation: short {short_density:.0f} > "
          f"long {long_density:.0f} per px")

    # State machine: idle -> stretching -> snapping -> idle
    clk = {"t": 0.0}
    fx = StretchFX(clock=lambda: clk["t"])
    fx.update((100, 100), 100.0)
    assert fx.state == IDLE
    fx.update((200, 200), 10.0)
    assert fx.state == STRETCHING, fx.state
    assert fx.anchor == (200, 200)
    fx.update((400, 300), 12.0)
    assert fx.state == STRETCHING
    fx.update((400, 300), 100.0)
    assert fx.state == SNAPPING, fx.state
    clk["t"] = Config.SNAP_DURATION + 0.01
    fx.update(None, float("inf"))
    assert fx.state == IDLE, fx.state
    assert fx.anchor is None
    print("  state machine: idle->stretching->snapping->idle OK")

    # Hysteresis
    fx2 = StretchFX()
    assert fx2._update_pinch_latch(100) is False
    assert fx2._update_pinch_latch(30) is True
    assert fx2._update_pinch_latch(45) is True
    assert fx2._update_pinch_latch(60) is False
    print("  hysteresis: dead-band latch OK")

    # Adaptive smoother
    sm = AdaptiveSmoother()
    p1 = sm.update((100, 100))
    assert p1 == (100, 100), "first point should be exact"
    p2 = sm.update((200, 200))
    assert p2[0] > 100 and p2[0] < 200, "smoothed should be between"
    print(f"  adaptive smoother: {p1} -> {p2} OK")

    # Finger width estimation (with thumb)
    fake_px = [(i * 30, i * 20) for i in range(21)]
    fw_thumb = finger_width_px(fake_px, 4)
    fw_index = finger_width_px(fake_px, 8)
    assert fw_thumb > 0 and fw_index > 0
    print(f"  finger_width: thumb={fw_thumb:.1f} index={fw_index:.1f} OK")

    print("\nself-test PASSED" if ok else "\nself-test FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _make_hand_landmarker(cfg):
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=cfg.HAND_DETECTION_CONF,
        min_tracking_confidence=cfg.HAND_TRACKING_CONF)
    return mp_vision.HandLandmarker.create_from_options(opts)


def _make_face_landmarker(cfg):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=cfg.FACE_DETECTION_CONF,
        min_face_tracking_confidence=cfg.FACE_TRACKING_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def run(cfg=Config, sink=None, stop_event=None):
    """
    Live webcam loop. If `sink` is given, each rendered BGR frame is passed to
    sink(frame) instead of being shown in an OpenCV window.
    """
    if mp is None:
        print("mediapipe is not installed. pip install mediapipe", file=sys.stderr)
        return 1
    if not os.path.exists(cfg.HAND_MODEL):
        print(f"Missing hand model: {cfg.HAND_MODEL}\n"
              "Download with:\n  curl -sSL -o models/hand_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task",
              file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(cfg.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_HEIGHT)
    ok_probe, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok_probe:
        cap.release()
        print("No usable webcam. Falling back to synthetic demo.",
              file=sys.stderr)
        return demo_loop(cfg, sink=sink, stop_event=stop_event)

    hands = _make_hand_landmarker(cfg)
    face = None  # lazy-loaded when face mode is activated
    face_mode = False  # 'f' toggles face-stretch mode

    grab = GrabFX(cfg)
    face_grab = FaceGrabFX(cfg)
    mirror = True
    show_guides = False
    show_debug = False   # 'd' — debug overlay with all landmarks
    swap_roles = False
    inpaint_gap = cfg.INPAINT_GAP
    show_mask = False
    show_mesh = False
    use_silhouette = cfg.USE_SILHOUETTE
    thin_amt = cfg.RUBBER_THIN_AMT
    hl_gain = cfg.HIGHLIGHT_GAIN
    mesh_cache = None
    body_color = (0, 200, 0)
    grab_color = (0, 170, 255)

    frame_idx = 0

    print("Gum-Gum (Enhanced): All fingers + face stretch. "
          "h=guides | f=face mode | d=debug | r=swap | SPACE=mirror | "
          ", .=spec | j k=thin | l o=overshoot | i=inpaint | g=mesh | ESC=quit")

    while stop_event is None or not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000.0 / 30.0)
        frame_idx += 1

        results = hands.detect_for_video(mp_image, ts_ms)

        # --- Face detection (when face mode is active) ---
        face_px = None
        if face_mode:
            if face is None:
                if os.path.exists(cfg.FACE_MODEL):
                    face = _make_face_landmarker(cfg)
                else:
                    print(f"Face model not found: {cfg.FACE_MODEL}", file=sys.stderr)
                    face_mode = False
            if face is not None:
                face_results = face.detect_for_video(mp_image, ts_ms)
                if (face_results.face_landmarks and
                        len(face_results.face_landmarks) > 0):
                    face_px = to_px(face_results.face_landmarks[0], w, h)

        # --- Assign body / grabber by handedness ---
        handed = getattr(results, "handedness", None) or []
        labeled = [((handed[i][0].category_name
                     if i < len(handed) and handed[i] else None), lms)
                   for i, lms in enumerate(results.hand_landmarks or [])]
        body_label = cfg.GRABBER_HAND if swap_roles else cfg.BODY_HAND
        grab_label = cfg.BODY_HAND if swap_roles else cfg.GRABBER_HAND
        body_lms = grabber_lms = None
        for label, lms in labeled:
            if label == body_label and body_lms is None:
                body_lms = lms
            elif grabber_lms is None:
                grabber_lms = lms
        if len(labeled) == 2 and (body_lms is None or grabber_lms is None):
            l, r = sorted(labeled, key=lambda kv: kv[1][0].x)
            body_lms, grabber_lms = ((r[1], l[1]) if swap_roles else (l[1], r[1]))

        body_px = to_px(body_lms, w, h) if body_lms else None
        grabber_px = to_px(grabber_lms, w, h) if grabber_lms else None

        # --- Grabber pinch (thumb tip 4 to index tip 8).
        grab_pinch, grab_dist = None, float("inf")
        if grabber_px is not None:
            grab_pinch = lerp(grabber_px[4], grabber_px[8], 0.5)
            grab_dist = dist(grabber_px[4], grabber_px[8])

        # --- Face stretch or hand stretch ---
        if face_mode and face_px is not None:
            face_grab.update(grab_pinch, grab_dist, face_px)
            face_grab.draw(frame, face_px)
            # Don't update hand grab in face mode
            grab.state = IDLE
            grab.grabbed_tip = None
        else:
            face_grab.state = IDLE
            face_grab.grabbed_region = None
            grab.update(grab_pinch, grab_dist, body_px)

            # Drop cached mesh when the grab ends
            if grab.state == IDLE:
                mesh_cache = None

            # --- ARAP render: deform the REAL finger crop ---
            if (grab.state in (STRETCHING, SNAPPING) and body_px is not None
                    and grab.grabbed_tip is not None):
                mcp = TIP_TO_MCP[grab.grabbed_tip]
                live_knuckle = body_px[mcp]
                origin_tip = body_px[grab.grabbed_tip]
                pull_target = grab.render_tip(body_px)
                fw = finger_width_px(body_px, grab.grabbed_tip)

                if mesh_cache is None or mesh_cache.get("for") != grab.grabbed_tip:
                    fmask = build_finger_mask(h, w, body_px, grab.grabbed_tip, fw,
                                              0.0, cfg)
                    sil = (silhouette_mask(frame, fmask, cfg) if use_silhouette
                           else fmask)
                    mesh = arap.build_finger_mesh(sil)
                    mesh_cache = None
                    if mesh is not None:
                        rest, tris = mesh
                        anchors, tipv = arap.pick_handles(
                            rest, live_knuckle, origin_tip, fw * cfg.ANCHOR_RADIUS)
                        try:
                            solver = arap.ARAPSolver(rest, tris, anchors + [tipv])
                        except Exception:
                            solver = None
                        if solver is not None:
                            free = np.ones(len(rest), bool)
                            free[anchors + [tipv]] = False
                            tex = np.dstack([frame, sil]).copy()
                            mesh_cache = {"for": grab.grabbed_tip, "solver": solver,
                                          "rest": rest, "tris": tris,
                                          "tex": tex, "anchors": anchors,
                                          "tipv": tipv, "free": free,
                                          "knuckle0": live_knuckle}

                if mesh_cache is not None and pull_target is not None:
                    mc = mesh_cache
                    tdx = live_knuckle[0] - mc["knuckle0"][0]
                    tdy = live_knuckle[1] - mc["knuckle0"][1]
                    targets = {i: (mc["rest"][i][0] + tdx, mc["rest"][i][1] + tdy)
                               for i in mc["anchors"]}
                    targets[mc["tipv"]] = (pull_target[0], pull_target[1])
                    deformed = mc["solver"].solve(targets)

                    natural = max(1.0, dist(mc["knuckle0"], body_px[grab.grabbed_tip]))
                    amt = max(0.0, min(2.0,
                              dist(live_knuckle, pull_target) / natural - 1.0))
                    deformed = arap.thin_toward_axis(deformed, live_knuckle,
                                                     pull_target, mc["free"],
                                                     thin_amt * amt)
                    if inpaint_gap:
                        cur = build_finger_mask(h, w, body_px, grab.grabbed_tip,
                                                fw, 0.0, cfg)
                        x, y, ww, hh = cv2.boundingRect(cur)
                        if ww > 0 and hh > 0:
                            sub = frame[y:y + hh, x:x + ww]
                            frame[y:y + hh, x:x + ww] = cv2.inpaint(
                                sub, (cur[y:y+hh, x:x+ww] > 10).astype(np.uint8),
                                3, cv2.INPAINT_TELEA)
                    arap.render_mesh(mc["tex"], mc["rest"], deformed, mc["tris"],
                                     frame, cfg.LIMB_FEATHER, hl_gain * amt,
                                     live_knuckle, pull_target, cfg.HIGHLIGHT_OFFSET,
                                     finger_width=fw)
                    if show_mesh:
                        for t in mc["tris"]:
                            cv2.polylines(frame, [deformed[t].astype(np.int32)],
                                          True, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Mask visualization ---
        if show_mask and body_px is not None:
            vis_tip = grab.grabbed_tip if grab.grabbed_tip is not None else 8
            vmask = build_finger_mask(h, w, body_px, vis_tip,
                                      finger_width_px(body_px, vis_tip), 0.0, cfg)
            tint = np.zeros_like(frame)
            tint[:, :, 1] = vmask
            frame[:] = cv2.addWeighted(frame, 1.0, tint, 0.5, 0)

        # --- Hand-tracking guides ---
        if show_guides:
            if body_px is not None:
                hide = (grab.grabbed_tip
                        if grab.state in (STRETCHING, SNAPPING) else None)
                draw_hand_skeleton(frame, body_px, body_color, hide_tip=hide)
            if grabber_px is not None:
                draw_hand_skeleton(frame, grabber_px, grab_color)
            if grab.state == IDLE and grab_pinch is not None and body_px:
                near = grab._nearest_body_tip(grab_pinch, body_px)
                if near is not None:
                    p = body_px[near]
                    col = (0, 255, 0) if grab.pinched else (0, 200, 255)
                    cv2.circle(frame, (int(p[0]), int(p[1])), 16, col, 2,
                               cv2.LINE_AA)
            if grab_pinch is not None:
                col = (0, 255, 0) if grab.pinched else (0, 165, 255)
                cv2.circle(frame, (int(grab_pinch[0]), int(grab_pinch[1])),
                           8, col, 2, cv2.LINE_AA)

        # --- Debug overlay: face landmarks + regions ---
        if show_debug and face_px is not None:
            draw_face_landmarks_debug(frame, face_px, cfg.FACE_REGIONS, cfg)

        if sink is not None:
            sink(frame)
            continue

        cv2.imshow("Gum-Gum", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            swap_roles = not swap_roles
        elif key == ord('h'):
            show_guides = not show_guides
        elif key == ord('f'):
            face_mode = not face_mode
            if face_mode:
                print("[mode] Face stretch ON — grab nose/cheeks/chin/forehead/lips")
            else:
                print("[mode] Face stretch OFF — hand stretch mode")
        elif key == ord('d'):
            show_debug = not show_debug
            if show_debug and not face_mode:
                face_mode = True  # auto-enable face detection for debug
                print("[debug] Face detection auto-enabled for debug view")
        elif key == ord(' '):
            mirror = not mirror
        elif key == ord(','):
            hl_gain = max(0.0, hl_gain - 0.1)
        elif key == ord('.'):
            hl_gain = min(2.0, hl_gain + 0.1)
        elif key == ord('i'):
            inpaint_gap = not inpaint_gap
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('g'):
            show_mesh = not show_mesh
        elif key == ord('s'):
            use_silhouette = not use_silhouette
            mesh_cache = None
        elif key == ord('j'):
            thin_amt = max(0.0, thin_amt - cfg.RUBBER_THIN_STEP)
        elif key == ord('k'):
            thin_amt = min(0.6, thin_amt + cfg.RUBBER_THIN_STEP)
        elif key == ord('l'):
            grab.overshoot = max(1.0, grab.overshoot - cfg.OVERSHOOT_STEP)
        elif key == ord('o'):
            grab.overshoot = min(3.0, grab.overshoot + cfg.OVERSHOOT_STEP)

    hands.close()
    if face:
        face.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def demo_loop(cfg=Config, sink=None, stop_event=None):
    """
    Headless, camera-free demo. Drives the *real* StretchFX + rubber renderer
    with a synthetic pinch that periodically grabs, stretches, and releases.
    """
    w, h = 960, 540
    max_frames = int(os.environ.get("GUMGUM_DEMO_FRAMES", "0"))
    fx = StretchFX(cfg)
    anchor = (w * 0.32, h * 0.5)
    dest = "HTTP stream" if sink else "demo_frame.png"
    print(f"Gum-Gum synthetic demo running ({w}x{h}) -> {dest}. Ctrl-C to stop.",
          flush=True)

    i = 0
    while (max_frames == 0 or i < max_frames) and (
            stop_event is None or not stop_event.is_set()):
        t = i / 30.0
        phase = t % 4.0
        holding = phase < 2.6
        reach = math.sin(min(phase, 2.6) / 2.6 * math.pi) if holding else 0.0
        tip = (anchor[0] + reach * (w * 0.5),
               anchor[1] - reach * (h * 0.22) + math.sin(t * 3) * 12)
        pinch_dist = 12.0 if holding else 90.0
        fx.update(tip, pinch_dist, anchor_override=anchor if holding else None)

        frame = np.full((h, w, 3), 24, np.uint8)
        frame[:] = np.linspace(18, 46, h, dtype=np.uint8)[:, None, None]
        fx.draw(frame)
        cv2.putText(frame, f"Gum-Gum (synthetic demo)  state={fx.state}",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 2, cv2.LINE_AA)

        if sink is not None:
            sink(frame)
        elif i % 15 == 0:
            cv2.imwrite("demo_frame.png", frame)
        if i % 30 == 0:
            print(f"[demo] frame={i} state={fx.state} "
                  f"tip=({tip[0]:.0f},{tip[1]:.0f})", flush=True)
        i += 1
        time.sleep(1.0 / 30.0)
    return 0


_PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>Gum-Gum</title><style>
html,body{margin:0;height:100%;background:#0d0d10;display:flex;
align-items:center;justify-content:center;font-family:system-ui,sans-serif}
img{max-width:100%;max-height:100vh;border-radius:10px;
box-shadow:0 8px 40px rgba(0,0,0,.6)}
.tag{position:fixed;top:12px;left:14px;color:#aab;font-size:13px;opacity:.8}
</style></head><body>
<div class="tag">Gum-Gum &mdash; live MJPEG stream</div>
<img src="/stream" alt="Gum-Gum stream"></body></html>"""


def serve(cfg=Config):
    """
    Serve the rendered effect over HTTP as an MJPEG stream.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(os.environ.get("PORT") or "3000")
    host = os.environ.get("HOST", "0.0.0.0")
    shared = {"jpeg": None}
    lock = threading.Lock()
    stop = threading.Event()

    def sink(frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with lock:
                shared["jpeg"] = buf.tobytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_):
            pass

        def _latest(self):
            with lock:
                return shared["jpeg"]

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_PAGE)))
                self.end_headers()
                self.wfile.write(_PAGE)
            elif self.path == "/frame.jpg":
                j = self._latest()
                if not j:
                    self.send_response(503); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers()
                self.wfile.write(j)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while not stop.is_set():
                        j = self._latest()
                        if j:
                            self.wfile.write(b"--frame\r\n"
                                             b"Content-Type: image/jpeg\r\n"
                                             b"Content-Length: " +
                                             str(len(j)).encode() + b"\r\n\r\n")
                            self.wfile.write(j)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 30.0)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404); self.end_headers()

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Gum-Gum serving on http://localhost:{port}  "
          f"(open / for the viewer, /stream for raw MJPEG). Ctrl-C to stop.",
          flush=True)
    try:
        run(cfg, sink=sink, stop_event=stop)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[serve] capture error: {exc!r}", file=sys.stderr)
    finally:
        stop.set()
        httpd.shutdown()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Gum-Gum hand-stretch webcam FX (Enhanced)")
    ap.add_argument("--selftest", action="store_true",
                    help="run headless math/state checks (no camera)")
    ap.add_argument("--demo", action="store_true",
                    help="headless camera-free demo loop (renders to "
                         "demo_frame.png); works without a webcam or display")
    ap.add_argument("--serve", action="store_true",
                    help="serve the effect over HTTP (MJPEG) on $PORT or 3000 "
                         "so it shows in a browser / preview pane")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.serve:
        return serve()
    if args.demo:
        return demo_loop()
    return run()


if __name__ == "__main__":
    sys.exit(main())
