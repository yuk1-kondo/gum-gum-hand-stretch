# Gum-Gum — webcam hand & face stretch effect

Pinch your thumb and a fingertip together, then drag your hand away: a rubbery
limb stretches from a frozen anchor to your live pinch point. Release and it
snaps back with an elastic overshoot — Luffy's Gomu Gomu, via OpenCV + MediaPipe.

## Features

- **All 5 fingers** (including thumb) can be grabbed and stretched
- **Face stretching**: grab nose, cheeks, chin, forehead, eyebrows, or lips and pull
- **Improved precision**: adaptive Kalman-like smoothing, multi-landmark finger width estimation, higher detection confidence
- **Full MediaPipe utilization**: 21 hand landmarks + 478 face mesh landmarks
- **ARAP deformation**: As-Rigid-As-Possible mesh warping for realistic elastic deformation
- **Snap-back animation**: under-damped spring physics with configurable overshoot

## Setup

```bash
pip install -r requirements.txt

# model bundles (already downloaded if you cloned with models/):
mkdir -p models
curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -sSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> This build of MediaPipe ships the modern **Tasks API** (not the legacy
> `mp.solutions`), so the app uses `HandLandmarker` / `FaceLandmarker` with the
> `.task` model bundles above.

## Run

```bash
python gumgum.py            # live webcam (grant camera permission on macOS)
python gumgum.py --serve    # MJPEG stream on localhost:$PORT
python gumgum.py --selftest # headless math/state check, no camera
python gumgum.py --demo     # headless camera-free demo loop
```

## Controls

| Key   | Action |
|-------|--------|
| ESC/q | quit |
| f     | toggle face-stretch mode (grab nose/cheeks/chin/forehead/lips) |
| d     | toggle debug overlay (all landmarks, face regions) |
| h     | toggle hand guides (skeleton + pinch markers) |
| r     | swap body/grabber hand roles |
| SPACE | mirror on/off |
| s     | toggle skin silhouette refinement |
| i     | toggle gap inpaint |
| m     | finger-mask overlay |
| g     | ARAP mesh wireframe |
| , / . | specular gain -/+ |
| j / k | rubber thinning -/+ |
| l / o | snap overshoot -/+ |

## How it works

### Hand Stretch
1. Two hands are detected: a **body** hand (gets stretched) and a **grabber** hand (pinches & pulls)
2. The grabber pinches near any of the body hand's 5 fingertips
3. The finger is deformed using ARAP mesh warping rooted at the MCP/CMC knuckle
4. On release, the finger snaps back with under-damped spring physics

### Face Stretch
1. Press `f` to enter face mode — the full 478-point face mesh is activated
2. The grabber hand pinches near a facial feature (nose, cheeks, chin, etc.)
3. The face region is deformed using ARAP mesh warping
4. On release, the face snaps back elastically

### Precision Improvements
- **Adaptive smoothing**: velocity-aware EMA that's responsive during fast movement and stable during slow phases
- **Multi-landmark finger width**: blends MCP spacing with bone-length ratios for accurate finger sizing
- **Higher detection confidence**: reduced false positives with 0.7 detection / 0.6 tracking thresholds
- **Tighter pinch hysteresis**: 34px engage / 52px release for more reliable pinch detection

## How it feels (the bits that matter)

- **Pinch hysteresis** — engage <34px, release >52px, so it doesn't flicker.
- **Adaptive smoothing** — velocity-aware EMA kills jitter without adding lag.
- **Ease-out-back snap-back** — overshoots the anchor, then settles.
- **Bezier rubber** — quadratic curve with a perpendicular mid-bulge, tapered
  toward the tip, thinner as it lengthens (volume conservation), with a
  highlight line for a tube look.

All tunables live in the `Config` class at the top of `gumgum.py`.
