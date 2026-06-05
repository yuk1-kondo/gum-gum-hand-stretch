# Gum-Gum — webcam hand-stretch effect

Pinch your thumb and a fingertip together, then drag your hand away: a rubbery
limb stretches from a frozen anchor to your live pinch point. Release and it
snaps back with an elastic overshoot — Luffy's Gomu Gomu, via OpenCV + MediaPipe.

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
python gumgum.py --selftest # headless math/state check, no camera
```

| Key   | Action |
|-------|--------|
| ESC/q | quit |
| f     | toggle face-cheek anchor mode (grab a cheek, Nami-style) |
| a     | toggle any-finger pinch (nearest of index/middle/ring/pinky) |
| d     | toggle debug overlay (landmarks, pinch distance, state) |
| SPACE | mirror on/off |

## How it feels (the bits that matter)

- **Pinch hysteresis** — engage <38px, release >55px, so it doesn't flicker.
- **Ease-out-back snap-back** — overshoots the anchor, then settles.
- **Bezier rubber** — quadratic curve with a perpendicular mid-bulge, tapered
  toward the tip, thinner as it lengthens (volume conservation), with a
  highlight line for a tube look.

All tunables live in the `Config` class at the top of `gumgum.py`.
