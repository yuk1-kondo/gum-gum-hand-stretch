"""
Gum-Gum — webcam hand-stretch effect (One Piece / Luffy inspired).

Two hands: a *grabber* hand pinches one of the *body* hand's fingertips and
pulls. The real finger photo is deformed like elastic via As-Rigid-As-Possible
2D mesh deformation (Igarashi et al. 2005, see arap.py) — rooted at the live
MCP knuckle, dragged to the grabber pinch — then snaps back with a punchy
under-damped spring, like Luffy's Gomu Gomu.

Controls (no on-screen text; the camera feed stays clean):
    ESC / q   quit              SPACE  mirror on/off
    h         toggle hand guides (skeleton + pinch markers)
    r         swap body/grabber roles      s  toggle skin silhouette
    , / .     specular gain  -/+           j / k  rubber thinning  -/+
    l / o     snap overshoot -/+           i  toggle gap inpaint
    m         finger-mask overlay          g  ARAP mesh wireframe

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

    # Pinch hysteresis (pixels between thumb tip and finger tip).
    PINCH_ENGAGE = 38.0   # must get closer than this to start a stretch
    PINCH_RELEASE = 55.0  # must open wider than this to let go

    # Live pinch-point smoothing (exponential moving average, 0..1).
    # Higher = snappier/jitterier, lower = smoother/laggier.
    TIP_SMOOTHING = 0.5

    # Rubber appearance.
    BASE_WIDTH = 34.0       # half-thickness at the anchor when relaxed (px)
    TIP_WIDTH_FRAC = 0.35   # tip half-thickness as fraction of anchor width
    MIN_WIDTH = 4.0         # never thinner than this (px)
    # Volume conservation: thickness shrinks as length grows.
    # width *= 1 / (1 + length / THINNING_LENGTH)
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
    # Tight silhouette: the landmark capsule is wider than the real finger, so
    # adaptively trim it to skin (YCrCb sampled from the finger's own core) so
    # the mesh hugs the finger outline, not a fat rectangle of finger+wall.
    USE_SILHOUETTE = True   # toggle 's' — fall back to the raw capsule if off
    SKIN_DCR = 22           # YCrCb Cr tolerance around the finger's median
    SKIN_DCB = 22           # YCrCb Cb tolerance
    SILHOUETTE_ERODE = 2    # px inward erosion so the edge sits inside the skin

    # --- Branding: One Piece logo overlay (toggle 'b') ---
    LOGO_PATH = "assets/onepiece_logo.png"
    LOGO_CORNER = "tl"      # tl / tr / bl / br
    LOGO_WIDTH_FRAC = 0.15  # logo width as a fraction of the frame width
    LOGO_MARGIN = 18        # px inset from the frame edges
    LOGO_OPACITY = 0.92

    # --- Rubber feel: middle-band dynamics (base/tip are protected/photographic) ---
    RUBBER_THIN_AMT = 0.15  # middle narrows up to this * clamp(scale-1,0,2)
    HIGHLIGHT_GAIN = 0.2    # specular sheen — subtle; capped well below white
    BASE_PROTECT = 1.2      # protected base length = this * finger_width
    TIP_PROTECT = 1.2       # protected tip length  = this * finger_width
    ELONG_BLUR = 0.0        # axis motion-blur on the middle at high stretch (0=off)
    ELONG_THRESH = 1.6      # middle_scale above which elongation blur ramps in
    RUBBER_THIN_STEP = 0.03 # 'j'/'k' live nudge
    OVERSHOOT_STEP = 0.15   # 'l'/'o' live nudge

    # --- Displacement-field warp (pixel stretch of the real finger) ---
    # Gaussian perpendicular falloff width = SIGMA_SCALE * (finger_width / 2).
    # Bigger = wider band of pixels dragged; smaller = only the finger core.
    SIGMA_SCALE = 1.3
    # along_weight = clip(t, 0, 1) ** ALONG_GAMMA, t = position along the finger
    # axis (0 at knuckle, 1 at tip). >1 keeps the base more rooted.
    ALONG_GAMMA = 1.6
    # No hard clamp — the masked finger is a cutout, so it can stretch far into
    # empty space without dragging background. None = unbounded.
    MAX_STRETCH_PX = None
    SIGMA_SCALE_STEP = 0.1  # '[' / ']' nudge amount at runtime

    # --- Finger mask + masked composite (only finger pixels move) ---
    MASK_WIDTH_SCALE = 0.95   # mask thickness relative to measured finger width
    MASK_DILATE = 5           # px kernel; expand mask to fully cover the finger
    MASK_FEATHER = 9          # odd; gaussian blur on warped-mask alpha (soft seam)
    MASK_TAPER_LENGTH = 220.0 # stretch px at which the tip thins to ~half
    MASK_TIP_MIN = 0.12       # floor on tip thickness factor (never vanish)
    INPAINT_GAP = True        # inpaint the vacated finger area before compositing
    USE_SKIN_REFINE = False   # YCrCb skin trim (toggle 's' if edges look blocky)

    # --- Constructed limb: tiled shaft + real fingertip (reads as a finger) ---
    STRIP_HEIGHT = 13         # taller sample => real skin texture tiles through
    TUBE_WIDTH_SCALE = 1.0    # tube width relative to measured finger width
    TIP_PATCH_SCALE = 1.55    # solid tip size — large enough to include the nail
    TAPER_FRAC = 0.18         # gentle base->tip width reduction (layered on top)
    N_JOINTS = 2.5            # knuckle bulges over the visible length (2-3)
    UNDULATION_AMP = 0.12     # half-width modulation amount at the knuckles
    CREASE_OPACITY = 0.35     # darkness of skin-fold arcs at the joints
    HIGHLIGHT_OFFSET = 0.30   # highlight band offset from center (skin, not plastic)
    COLOR_MATCH = 0.5         # tint shaft toward local hand lighting (0=off)
    LIMB_CURVATURE = 0.12     # spline perpendicular bow (0 = straight line)
    LIMB_FEATHER = 7          # odd; gaussian blur on limb alpha for soft edges
    LIMB_SHADING = True       # offset highlight + edge darkening
    MAX_STRIPS = 280          # safety cap on tile count
    TUBE_WIDTH_STEP = 0.08    # ',' / '.' runtime nudge

    # Two-hand grab: grabber pinch must close within this of a body fingertip.
    ATTACH_RADIUS = 50.0

    # Default handedness roles (on the mirrored feed). Toggle with 'r'.
    BODY_HAND = "Left"     # this hand gets stretched
    GRABBER_HAND = "Right" # this hand pinches & pulls

    # Face-cheek mode: cheek landmarks in the Face Mesh (left & right).
    CHEEK_LANDMARKS = (50, 280)
    CHEEK_GRAB_RADIUS = 90.0  # px; pinch must engage within this of a cheek


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
    smaller secondary wobble, then settles. `overshoot` lowers the damping (more
    punch), `wobble` sets the oscillation count across the recoil.
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
# Rubber renderer
# ---------------------------------------------------------------------------
def draw_rubber(frame, anchor, tip, cfg=Config):
    """
    Render a tapering, bulging rubber tube from anchor to tip.

    - Quadratic bezier with a perpendicular bulge in the middle.
    - Thinner toward the tip (taper) and thinner overall as it lengthens
      (volume conservation), which sells the elasticity.
    - A lighter highlight line over the core gives it a tube look.
    """
    length = dist(anchor, tip)
    dx, dy = tip[0] - anchor[0], tip[1] - anchor[1]
    px, py = perpendicular(dx, dy)

    # Mid control point pushed sideways for the bulge.
    bulge = min(length * cfg.BULGE_FRAC, cfg.BULGE_MAX)
    mid = lerp(anchor, tip, 0.5)
    ctrl = (mid[0] + px * bulge, mid[1] + py * bulge)

    # Volume conservation: longer => thinner.
    thinning = 1.0 / (1.0 + length / cfg.THINNING_LENGTH)
    w_anchor = max(cfg.BASE_WIDTH * thinning, cfg.MIN_WIDTH)
    w_tip = max(w_anchor * cfg.TIP_WIDTH_FRAC, cfg.MIN_WIDTH)

    n = cfg.BEZIER_SAMPLES
    left, right, centers = [], [], []
    prev = quad_bezier(anchor, ctrl, tip, 0.0)
    for i in range(n + 1):
        t = i / n
        pt = quad_bezier(anchor, ctrl, tip, t)
        # Tangent via finite difference -> local perpendicular.
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

    # Soft drop shadow, fill, then a darker outline for depth.
    cv2.fillPoly(frame, [poly], cfg.RUBBER_SHADE, lineType=cv2.LINE_AA)
    inset = poly.copy()
    cv2.fillPoly(frame, [poly], cfg.RUBBER_COLOR, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [poly], True, cfg.RUBBER_SHADE, 2, cv2.LINE_AA)

    # Highlight line riding just above the core centerline.
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
    bone chain. Thickness ∝ finger width, tapering toward the tip — and the
    taper deepens with stretch length so a long pull thins like real rubber.
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
        th = max(1, int(round(radii[i] + radii[i + 1])))   # ~ avg diameter
        cv2.line(mask, pts[i], pts[i + 1], 255, th, cv2.LINE_AA)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (cfg.MASK_DILATE, cfg.MASK_DILATE))
    return cv2.dilate(mask, k)


def silhouette_mask(frame, capsule, cfg=Config):
    """
    Tighten the fat landmark `capsule` to the REAL finger outline. The capsule's
    eroded core is guaranteed-finger, so we sample its median YCrCb skin chroma
    and keep only capsule pixels matching that skin — discarding the band of
    wall/background that sits inside the capsule beside the finger. Returns a
    tight, slightly-eroded mask, or the capsule unchanged if segmentation fails.
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
    if out.sum() < 0.22 * capsule.sum():      # segmentation failed -> fall back
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
        self.pinched = False          # hysteresis latch
        self.anchor = None
        self.tip = None               # smoothed live pinch point
        self.snap_from = None
        self.snap_start = 0.0

    def _update_pinch_latch(self, pinch_dist):
        """Hysteresis: engage below ENGAGE, release above RELEASE."""
        if not self.pinched and pinch_dist <= self.cfg.PINCH_ENGAGE:
            self.pinched = True
        elif self.pinched and pinch_dist >= self.cfg.PINCH_RELEASE:
            self.pinched = False
        return self.pinched

    def update(self, pinch_point, pinch_dist, anchor_override=None):
        """
        Advance the state machine for one frame.

        pinch_point   : live (x, y) of the pinch, or None if no hand.
        pinch_dist    : thumb<->finger distance in px (inf if no hand).
        anchor_override: optional (x,y) to freeze as anchor (cheek mode).
        """
        now = self.clock()
        has_hand = pinch_point is not None
        pinched = self._update_pinch_latch(pinch_dist) if has_hand else False

        # Smooth the live tip to kill jitter.
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
                # Begin recoil from wherever the tip currently is.
                self.snap_from = self.tip if self.tip else self.anchor
                self.snap_start = now
                self.state = SNAPPING

        elif self.state == SNAPPING:
            elapsed = now - self.snap_start
            if elapsed >= self.cfg.SNAP_DURATION:
                self.state = IDLE
                self.anchor = None
            elif pinched and has_hand:
                # Re-grab mid-recoil.
                self.anchor = anchor_override or self.anchor or pinch_point
                self.tip = pinch_point
                self.state = STRETCHING

    def render_point(self):
        """Where the tube tip should be drawn this frame, or None."""
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
    Two-hand grab mechanic. A *grabber* hand pinches (thumb tip 4 to index tip
    8) near one of the *body* hand's fingertips to attach; the rubber then runs
    from that finger's live MCP knuckle to the live grabber pinch point. On
    release it snaps back to the body fingertip with an elastic overshoot.

    Both endpoints are real landmarks updated every frame, so the rubber stays
    rooted in the hand and follows it. State: idle -> stretching -> snapping.
    """

    def __init__(self, cfg=Config, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.state = IDLE
        self.pinched = False        # grabber pinch hysteresis latch
        self.grabbed_tip = None     # body fingertip idx (8/12/16/20) once attached
        self.tip = None             # smoothed live grabber pinch point
        self.snap_from = None
        self.snap_start = 0.0
        self.last_knuckle = None    # last seen body knuckle (px), for dropout
        self.last_fingertip = None
        self.overshoot = cfg.SNAP_OVERSHOOT   # live-tunable recoil punch
        self.wobble = cfg.SNAP_WOBBLE

    def _latch(self, pinch_dist):
        if not self.pinched and pinch_dist <= self.cfg.PINCH_ENGAGE:
            self.pinched = True
        elif self.pinched and pinch_dist >= self.cfg.PINCH_RELEASE:
            self.pinched = False
        return self.pinched

    def _nearest_body_tip(self, point, body_px):
        """Nearest body fingertip to point within ATTACH_RADIUS, else None."""
        best, best_d = None, self.cfg.ATTACH_RADIUS
        for tip in FINGER_TIPS:
            d = dist(point, body_px[tip])
            if d < best_d:
                best, best_d = tip, d
        return best

    def update(self, grab_pinch, grab_dist, body_px):
        """
        grab_pinch : grabber pinch midpoint (x,y) or None if no grabber hand.
        grab_dist  : grabber thumb<->index distance px (inf if none).
        body_px    : list of 21 (x,y) for the body hand, or None if not seen.
        """
        now = self.clock()
        has_grab = grab_pinch is not None
        pinched = self._latch(grab_dist) if has_grab else False

        if has_grab:  # smooth the grabber tip to kill jitter
            if self.tip is None:
                self.tip = grab_pinch
            else:
                s = self.cfg.TIP_SMOOTHING
                self.tip = (self.tip[0] + (grab_pinch[0] - self.tip[0]) * s,
                            self.tip[1] + (grab_pinch[1] - self.tip[1]) * s)

        # Cache live body positions for the grabbed finger (dropout fallback).
        if body_px is not None and self.grabbed_tip is not None:
            self.last_knuckle = body_px[TIP_TO_MCP[self.grabbed_tip]]
            self.last_fingertip = body_px[self.grabbed_tip]

        if self.state == IDLE:
            self.grabbed_tip = None
            if pinched and has_grab and body_px is not None:
                tip = self._nearest_body_tip(grab_pinch, body_px)
                if tip is not None:
                    self.grabbed_tip = tip
                    self.tip = grab_pinch
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
                if tip is not None:  # re-grab mid-recoil
                    self.grabbed_tip = tip
                    self.tip = grab_pinch
                    self.state = STRETCHING

    def anchor_point(self, body_px):
        """Live knuckle of the grabbed finger (or last known)."""
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
            # Under-damped spring: snaps PAST the rest point, wobbles, settles.
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


# ---------------------------------------------------------------------------
# Hand / pinch extraction
# ---------------------------------------------------------------------------
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky
THUMB_TIP = 4
# Fingertip -> its MCP knuckle (the rubber roots here while grabbed).
TIP_TO_MCP = {8: 5, 12: 9, 16: 13, 20: 17}
# Bone chains knuckle->tip per finger, used to paint the finger mask.
FINGER_CHAINS = {
    8: [5, 6, 7, 8], 12: [9, 10, 11, 12],
    16: [13, 14, 15, 16], 20: [17, 18, 19, 20], 4: [2, 3, 4],
}

# Standard 21-point hand topology (Tasks API gives no built-in drawer).
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
    """Estimate finger width from the mean MCP-knuckle spacing of the hand."""
    if body_px is None:
        return 40.0
    mcps = [5, 9, 13, 17]
    gaps = [dist(body_px[mcps[i]], body_px[mcps[i + 1]])
            for i in range(len(mcps) - 1)]
    return sum(gaps) / len(gaps) if gaps else 40.0


def load_logo(cfg=Config):
    """Load the branding logo (BGRA) scaled to nothing yet; None if missing."""
    path = cfg.LOGO_PATH
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        print(f"Logo not found: {path}", file=sys.stderr)
        return None
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is not None and logo.ndim == 3 and logo.shape[2] == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)  # add opaque alpha
    return logo


def scale_logo(logo, frame_w, cfg=Config):
    """Resize the logo so its width is LOGO_WIDTH_FRAC of the frame."""
    if logo is None:
        return None
    tw = max(16, int(frame_w * cfg.LOGO_WIDTH_FRAC))
    th = max(1, int(logo.shape[0] * tw / logo.shape[1]))
    return cv2.resize(logo, (tw, th), interpolation=cv2.INTER_AREA)


def overlay_logo(frame, logo, cfg=Config):
    """Alpha-composite the BGRA logo into the configured corner of the frame."""
    if logo is None:
        return
    h, w = frame.shape[:2]
    lh, lw = logo.shape[:2]
    m, corner = cfg.LOGO_MARGIN, cfg.LOGO_CORNER
    x = m if "l" in corner else w - lw - m
    y = m if "t" in corner else h - lh - m
    x, y = max(0, x), max(0, y)
    x1, y1 = min(w, x + lw), min(h, y + lh)
    cw, ch = x1 - x, y1 - y
    if cw <= 0 or ch <= 0:
        return
    a = (logo[:ch, :cw, 3:4].astype(np.float32) / 255.0) * cfg.LOGO_OPACITY
    rgb = logo[:ch, :cw, :3].astype(np.float32)
    roi = frame[y:y1, x:x1].astype(np.float32)
    frame[y:y1, x:x1] = (roi * (1.0 - a) + rgb * a).astype(np.uint8)


def draw_hand_skeleton(frame, pts, color=(0, 200, 0), hide_tip=None):
    """
    Draw a hand from pixel-space points. If hide_tip is a fingertip index, that
    tip's dot and its final bone are skipped — used so the grabbed fingertip
    appears replaced by the rubber.
    """
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        if hide_tip is not None and (a == hide_tip or b == hide_tip):
            continue
        cv2.line(frame, ipts[a], ipts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(ipts):
        if i == hide_tip:
            continue
        col = (0, 0, 255) if i in (THUMB_TIP,) + FINGER_TIPS else (0, 255, 255)
        cv2.circle(frame, p, 4, col, -1, cv2.LINE_AA)


def pinch_from_landmarks(landmarks, w, h, any_finger):
    """
    Return (pinch_point, pinch_dist, finger_idx) in pixel space.

    Default uses the index fingertip; any_finger picks whichever of the four
    fingertips is nearest the thumb.
    """
    thumb = (landmarks[THUMB_TIP].x * w, landmarks[THUMB_TIP].y * h)
    if any_finger:
        best_idx, best_d, best_pt = 8, float("inf"), None
        for idx in FINGER_TIPS:
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


def nearest_cheek(face_landmarks, w, h, point, cfg=Config):
    """Nearest cheek landmark to `point` within grab radius, else None."""
    best, best_d = None, float("inf")
    for idx in cfg.CHEEK_LANDMARKS:
        lm = face_landmarks[idx]
        pt = (lm.x * w, lm.y * h)
        d = dist(pt, point)
        if d < best_d:
            best, best_d = pt, d
    if best is not None and best_d <= cfg.CHEEK_GRAB_RADIUS:
        return best
    return None


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

    # Volume conservation: longer tube => thinner. Probe widths indirectly
    # by comparing drawn pixel mass per unit length (rough proxy).
    short = np.zeros((480, 640, 3), np.uint8)
    long_ = np.zeros((480, 640, 3), np.uint8)
    draw_rubber(short, (300, 240), (360, 240))
    draw_rubber(long_, (60, 240), (600, 240))
    short_density = short.sum() / 60.0
    long_density = long_.sum() / 540.0
    assert long_density < short_density, "longer tube should be thinner/unit"
    print(f"  volume conservation: short {short_density:.0f} > "
          f"long {long_density:.0f} per px")

    # State machine: idle -> stretching -> snapping -> idle, with a fake clock.
    clk = {"t": 0.0}
    fx = StretchFX(clock=lambda: clk["t"])
    fx.update((100, 100), 100.0)              # open hand, far
    assert fx.state == IDLE
    fx.update((200, 200), 10.0)               # pinch engages
    assert fx.state == STRETCHING, fx.state
    assert fx.anchor == (200, 200)
    fx.update((400, 300), 12.0)               # drag while held
    assert fx.state == STRETCHING
    fx.update((400, 300), 100.0)              # release
    assert fx.state == SNAPPING, fx.state
    clk["t"] = Config.SNAP_DURATION + 0.01    # let recoil finish
    fx.update(None, float("inf"))
    assert fx.state == IDLE, fx.state
    assert fx.anchor is None
    print("  state machine: idle->stretching->snapping->idle OK")

    # Hysteresis: between engage and release thresholds, latch holds.
    fx2 = StretchFX()
    assert fx2._update_pinch_latch(100) is False
    assert fx2._update_pinch_latch(30) is True       # below engage
    assert fx2._update_pinch_latch(45) is True        # in dead-band, holds
    assert fx2._update_pinch_latch(60) is False       # above release
    print("  hysteresis: dead-band latch OK")

    print("\nself-test PASSED" if ok else "\nself-test FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _make_hand_landmarker(cfg):
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    return mp_vision.HandLandmarker.create_from_options(opts)


def _make_face_landmarker(cfg):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def run(cfg=Config, sink=None, stop_event=None):
    """
    Live webcam loop. If `sink` is given, each rendered BGR frame is passed to
    sink(frame) instead of being shown in an OpenCV window — used by --serve to
    stream frames over HTTP. `stop_event` (threading.Event) cleanly ends the
    loop when running in a background thread.
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
    # A camera may report opened but still fail the first read (e.g. when
    # macOS denies access to a headless/background process). Probe a frame.
    ok_probe, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok_probe:
        cap.release()
        print("No usable webcam (not opened / not authorized). "
              "Falling back to synthetic demo — run in a foreground "
              "terminal with camera permission to use the live webcam.",
              file=sys.stderr)
        return demo_loop(cfg, sink=sink, stop_event=stop_event)

    hands = _make_hand_landmarker(cfg)
    face = None  # unused in the two-hand grab mechanic; kept for cleanup guard

    grab = GrabFX(cfg)
    mirror = True
    logo = load_logo(cfg)   # One Piece badge; scaled to the frame on first read
    logo_scaled = None
    show_logo = True        # 'b' — toggle the branding logo
    show_guides = False  # 'h' — hand-tracking guides (skeleton, pinch markers)
    swap_roles = False  # 'r' flips which handedness is body vs grabber
    inpaint_gap = cfg.INPAINT_GAP        # 'i' — fill vacated finger area
    show_mask = False                    # 'm' — visualize the finger mask
    show_mesh = False                    # 'g' — overlay the ARAP mesh wireframe
    use_silhouette = cfg.USE_SILHOUETTE  # 's' — tighten mesh to real skin outline
    thin_amt = cfg.RUBBER_THIN_AMT       # 'j'/'k' — volume-conservation thinning
    hl_gain = cfg.HIGHLIGHT_GAIN         # ',' / '.' — specular gain
    mesh_cache = None                    # ARAP solver + texture + mesh, per grab
    body_color = (0, 200, 0)      # green body hand
    grab_color = (0, 170, 255)    # orange grabber hand

    frame_idx = 0  # monotonic timestamp source for the VIDEO running mode

    print("Gum-Gum (ARAP): grabber pinches a body fingertip to elastically "
          "stretch the real finger. h=hand guides | r=swap | SPACE=mirror | "
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
        ts_ms = int(frame_idx * 1000.0 / 30.0)  # strictly increasing
        frame_idx += 1

        results = hands.detect_for_video(mp_image, ts_ms)

        # --- Assign body / grabber by handedness (swappable on mirrored feed).
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
        # Fallback: two hands but labels didn't split cleanly -> screen x order.
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

        grab.update(grab_pinch, grab_dist, body_px)

        # Drop cached mesh when the grab ends; the next grab rebuilds it.
        if grab.state == IDLE:
            mesh_cache = None

        # --- ARAP render: deform the REAL finger crop like elastic material.
        # One-time per grab: mask -> triangulate -> texture-map -> pre-factorize.
        # Per frame: update handles (live knuckle + grabber pinch) -> solve ->
        # texture-map each triangle into a feathered limb layer. No outline.
        if (grab.state in (STRETCHING, SNAPPING) and body_px is not None
                and grab.grabbed_tip is not None):
            mcp = TIP_TO_MCP[grab.grabbed_tip]
            live_knuckle = body_px[mcp]
            origin_tip = body_px[grab.grabbed_tip]
            pull_target = grab.render_tip(body_px)  # live pinch, or snap-back ease
            fw = finger_width_px(body_px, grab.grabbed_tip)

            if mesh_cache is None or mesh_cache.get("for") != grab.grabbed_tip:
                fmask = build_finger_mask(h, w, body_px, grab.grabbed_tip, fw,
                                          0.0, cfg)
                # Tighten the fat capsule to the real finger outline so the mesh
                # carries no surrounding background band.
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
                    except Exception as exc:  # singular config — skip this grab
                        solver = None
                    if solver is not None:
                        free = np.ones(len(rest), bool)
                        free[anchors + [tipv]] = False
                        # BGRA texture: BGR + the tight silhouette as a hard
                        # alpha, warped per triangle so background never renders.
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

                # Rubber: thin toward the axis + specular, scaled by how far
                # past natural length we are (auto-reverses during snap-back).
                natural = max(1.0, dist(mc["knuckle0"], body_px[grab.grabbed_tip]))
                amt = max(0.0, min(2.0,
                          dist(live_knuckle, pull_target) / natural - 1.0))
                deformed = arap.thin_toward_axis(deformed, live_knuckle,
                                                 pull_target, mc["free"],
                                                 thin_amt * amt)
                if inpaint_gap:               # erase the live finger underneath
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

        # --- Mask visualization (build-step-1 check): green-tint the would-be
        # mask of the grabbed finger (or the index finger when idle).
        if show_mask and body_px is not None:
            vis_tip = grab.grabbed_tip if grab.grabbed_tip is not None else 8
            vmask = build_finger_mask(h, w, body_px, vis_tip,
                                      finger_width_px(body_px, vis_tip), 0.0, cfg)
            tint = np.zeros_like(frame)
            tint[:, :, 1] = vmask
            frame[:] = cv2.addWeighted(frame, 1.0, tint, 0.5, 0)

        # --- Hand-tracking guides (toggle with 'h'). No text on the frame.
        if show_guides:
            if body_px is not None:
                hide = (grab.grabbed_tip
                        if grab.state in (STRETCHING, SNAPPING) else None)
                draw_hand_skeleton(frame, body_px, body_color, hide_tip=hide)
            if grabber_px is not None:
                draw_hand_skeleton(frame, grabber_px, grab_color)
            # Proximity highlight: nearest grabbable body fingertip in range.
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

        # --- Branding logo in the corner (drawn last, on top of everything).
        if show_logo and logo is not None:
            if logo_scaled is None:
                logo_scaled = scale_logo(logo, w, cfg)
            overlay_logo(frame, logo_scaled, cfg)

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
        elif key == ord('b'):
            show_logo = not show_logo
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
            mesh_cache = None              # rebuild with the new silhouette mode
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

    If `sink` is given each frame is passed to sink(frame) (used by --serve to
    stream over HTTP); otherwise snapshots are saved to demo_frame.png. Exists
    so the effect can run where a webcam isn't reachable — e.g. under a
    background preview process with no camera permission or display.
    Runs until killed / stop_event set. Bound it with GUMGUM_DEMO_FRAMES.
    """
    w, h = 960, 540
    max_frames = int(os.environ.get("GUMGUM_DEMO_FRAMES", "0"))  # 0 = forever
    fx = StretchFX(cfg)
    anchor = (w * 0.32, h * 0.5)
    dest = "HTTP stream" if sink else "demo_frame.png"
    print(f"Gum-Gum synthetic demo running ({w}x{h}) -> {dest}. Ctrl-C to stop.",
          flush=True)

    i = 0
    while (max_frames == 0 or i < max_frames) and (
            stop_event is None or not stop_event.is_set()):
        t = i / 30.0
        # Synthetic hand: a 4s cycle — pinch & drag out, then release & recoil.
        phase = t % 4.0
        holding = phase < 2.6
        reach = math.sin(min(phase, 2.6) / 2.6 * math.pi) if holding else 0.0
        tip = (anchor[0] + reach * (w * 0.5),
               anchor[1] - reach * (h * 0.22) + math.sin(t * 3) * 12)
        pinch_dist = 12.0 if holding else 90.0
        # Freeze the anchor exactly where the synthetic pinch starts.
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
    Serve the rendered effect over HTTP as an MJPEG stream so it shows up in a
    browser / preview pane (localhost). Uses the webcam when available, else the
    synthetic demo. Binds the PORT env var (preview-assigned) or 3000.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(os.environ.get("PORT") or "3000")
    host = os.environ.get("HOST", "127.0.0.1")
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

        def log_message(self, *_):  # silence per-request noise
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
    # The HTTP server runs on a background thread; camera capture stays on the
    # MAIN thread because macOS AVFoundation can only grab the webcam there.
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Gum-Gum serving on http://localhost:{port}  "
          f"(open / for the viewer, /stream for raw MJPEG). Ctrl-C to stop.",
          flush=True)
    try:
        # Real webcam when authorized (this thread), else synthetic demo.
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
    ap = argparse.ArgumentParser(description="Gum-Gum hand-stretch webcam FX")
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
