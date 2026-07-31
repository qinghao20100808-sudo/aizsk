# ⚡ AI Real-Time Force Analysis System

> **See objects · Understand scenes · Reason about physics**
>
> An intelligent high-school physics teaching assistant built on computer vision and a physics reasoning engine.
>
> 🔗 Repository: https://github.com/qinghao20100808-sudo/aizsk
> 📄 License: MIT

![License](https://img.shields.io/badge/license-MIT-green)

**English** | [中文](./README.zh-CN.md)

---

## ✨ Key Features

- 📦 **DJI-style box-select tracking**: drag to lock a target; pure tracking mode skips YOLO entirely (saves CPU), server ready in ~1s
- 📐 **Scale-adaptive**: multi-scale template matching (25 levels, 0.5×–2.0×), the box shrinks/grows smoothly as the object moves away/toward the camera (3D→2D projection)
- 🔍 **Occlusion handling**: LK optical flow + template matching dual engine; uniformly-accelerated extrapolation while occluded, auto re-acquisition on reappearance
- 🧲 **Physics reasoning engine**: auto-detects scene, reasons gravity / normal force / tension / friction and draws force arrows
- 🎯 **Free-fall detection**: acceleration-first judgment + slow-motion trajectory replay
- 🖱️ **Click fallback**: clicking anywhere always triggers force analysis (works even when nothing is detected)
- 🎨 **Pure Canvas rendering**: no video controls, unified drawing of boxes / force arrows / trajectories

---

## Supported Scenarios

| Scenario | Forces analyzed |
|----------|-----------------|
| Static on level surface | Gravity ↓, Normal force ↑ |
| Moving on level surface | Gravity, Normal, Friction, External force |
| Inclined plane θ° | Gravity ↓, Normal ⊥ plane, Friction ∥ plane |
| Free fall | Gravity ↓ (acceleration-first detection) |
| Hanging / pendulum | Gravity ↓, Tension ↑ (auto-detected; static T=G, accelerating T>G) |

---

## Quick Start

### Web mode (recommended, browser camera)

```bash
uv sync
uv run python3 -m aizsk.main --web
# Open http://localhost:5000 in your browser
```

### Demo mode (no camera)

```bash
uv run python3 -m aizsk.main --demo incline     # block on inclined plane
uv run python3 -m aizsk.main --demo flat_push   # horizontal push
uv run python3 -m aizsk.main --demo free_fall   # free fall
uv run python3 -m aizsk.main --demo pendulum    # pendulum
```

### Windows shared-file camera mode (WSL)

WSL cannot access the Windows camera directly — start ffmpeg on the Windows side to write a shared JPEG frame:

```powershell
# PowerShell (Windows host)
Start-Process -FilePath 'C:\Users\<user>\ffmpeg\bin\ffmpeg.exe' -ArgumentList @(
    '-f', 'dshow', '-i', 'video=Webcam C110',
    '-vf', 'fps=10', '-q:v', '10', '-update', '1',
    'C:\Users\<user>\AppData\Local\Temp\live_cam_hermes.jpg'
) -WindowStyle Hidden
```

```bash
uv run python3 -m aizsk.main --win-cam
```

---

## Usage

1. **Start camera**: click the "Start Camera" button and allow browser permission
2. **Box-select target**: drag on the frame to lock an object → pure tracking mode (no YOLO)
3. **Quick analysis**: single-click anywhere → immediate force analysis (creates a virtual object when nothing is detected)
4. **Choose scene**: sidebar switches scenario (level / incline / free fall / hanging); incline angle adjustable
5. **Enter mass**: with mass → real values (e.g. G=49.0N); without → force symbols only (G / N / T)
6. **Reset**: press `R` to clear the selection / reset tracking

---

## Architecture

```
Browser camera (getUserMedia)
    ↓ downsampled to 640 width (JPEG ~60KB)
Flask /api/detect
    ├─ box-select mode → ROI tracker (LK optical flow + multi-scale template matching)  ← no YOLO
    └─ click mode → YOLO detection (yolo11n, lazy-loaded) + tracker
            ↓
    Scene understanding (hanging / contact surface / incline)
            ↓
    Physics reasoning engine (high-school physics rules)
            ↓
    Force visualization (length clamped [30,70]px, uniform line width)
            ↓
Browser Canvas rendering (boxes + force arrows + trajectories)
```

### Core Modules

| Module | Description |
|--------|-------------|
| `roi_tracker.py` | ROI tracker: multi-scale template matching (primary) + LK optical flow, uniformly-accelerated occlusion extrapolation, recovery search, scale adaptation, anti-teleport |
| `detector.py` | YOLO detection (blacklist filtering: people / furniture / appliances / vehicles / large items excluded) |
| `tracker.py` | Target tracking (EMA smoothing of center & velocity, removes detection jitter) |
| `physics.py` | Physics reasoning engine (hanging heuristic, acceleration-first judgment, dynamic tension) |
| `web_server.py` | Flask server + API (lazy-loaded YOLO; ROI mode never loads the model) |
| `verify.py` | 59-item self-test suite (synthetic frames; no camera / network needed) |

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect/start` | POST | Initialize detector / reset tracking |
| `/api/detect` | POST | Process a frame (multipart; supports `roi` / `click_x` / `click_y` / `scene` / `mass`) |
| `/api/click` | POST | Force analysis at click coordinates |
| `/api/scene` | POST | Set scenario type |
| `/api/reset` | POST | Clear selection / reset tracking |
| `/api/status` | GET | Server status |

---

## Physics Reasoning Rules

### Scene auto-detection
- **Hanging heuristic**: object bottom < 55% of frame height → hanging → tension T; otherwise → support surface → normal force N
- **Free fall**: acceleration `ay > 2` triggers immediately (no waiting for velocity smoothing)

### Mass handling
- Mass provided → real values (`G=49.0N`)
- Mass unknown → symbols only (`G` / `N` / `T` / `f`), never assume a value

### Force visualization
- Arrow length clamped to `[30, 70]px` (640-space), uniform line width
- Equal forces naturally equal length (static T=G); acceleration shows difference (T>G)

---

## Testing

```bash
uv run python3 -m aizsk.verify
# ✅ All 59 checks pass (synthetic frames + Flask test_client, ~10–20s, no YOLO load)
```

---

## Known Issues & Limitations

| Issue | Cause | Status |
|-------|-------|--------|
| Low detection rate for held/small objects | YOLO nano is limited; hand occlusion | ⚠️ Mitigated: conf threshold lowered to 0.05 |
| Keys, pens, etc. not detected | Not in the COCO 80 classes | ❌ Unresolvable |
| Detection box jitter | YOLO frame-to-frame instability | ⚠️ Mitigated: tracker EMA smoothing |
| Tracking lags / loses very fast motion (>160px/s) | 200ms detection interval | ⚠️ Partially mitigated: chase + extrapolation; faster sampling possible |
| 4K frames lag (browser camera) | Full-res JPEG upload + CPU inference | ✅ Fixed: 640 downsampling |
| Canvas render flicker | rAF vs toBlob async | ✅ Fixed |
| torchvision NMS crash | torch 2.13 + torchvision 0.28 + Python 3.14 incompatible | ✅ Fixed: patched ultralytics autobackend.py |
| NNPACK warning | CPU-only torch has no NNPACK | ✅ Harmless |
| WSL shared-camera delay | File-based frame transfer | ⚠️ ~50ms latency, 10fps cap |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep learning | PyTorch |
| Object detection | YOLOv11 (Ultralytics) |
| Computer vision | OpenCV |
| Math | NumPy |
| Web server | Flask |
| Frontend | Vanilla HTML + JS (Canvas) |
| Package manager | uv |

---

## Development Environment

Developed and run in **WSL2 (Windows Subsystem for Linux)**:

- 🖥️ Host: Windows 11 + WSL2 (Ubuntu)
- ⚡ Hardware: CPU-only (no GPU, PyTorch CPU inference)
- 📷 Camera: Windows camera (e.g. Webcam C110) bridged via **shared-file mode** — ffmpeg on Windows writes a shared JPEG that WSL reads (the `--win-cam` mode)
- 🌐 Browser: Windows-side browser opens `http://localhost:<port>` (WSL port forwarding)

---

## Acknowledgments

Built on the following open-source projects — thanks for their contributions:

| Project | Used for | Link |
|---------|----------|------|
| **PyTorch** | Deep learning framework (inference) | https://github.com/pytorch/pytorch |
| **Ultralytics YOLO** | YOLOv11 object detection | https://github.com/ultralytics/ultralytics |
| **OpenCV** | Image processing / optical flow | https://github.com/opencv/opencv |
| **NumPy** | Numerical computation | https://github.com/numpy/numpy |
| **Flask** | Web server | https://github.com/pallets/flask |
| **uv** | Python package management & running | https://github.com/astral-sh/uv |

> ⚠️ Note: this repository's code is MIT-licensed, but the **YOLO model weights (yolo11n.pt) are AGPL-3.0** (Ultralytics license). Comply with their license terms when distributing model files.

---

## Roadmap

- [x] Project scaffolding
- [x] YOLO detection (blacklist filtering + lazy loading)
- [x] Target tracking (EMA smoothing)
- [x] ROI box-select tracking (LK optical flow + multi-scale template matching)
- [x] Scale adaptation (3D→2D projection zoom tracking)
- [x] Occlusion prediction (uniformly-accelerated extrapolation + auto recovery)
- [x] Physics reasoning engine (hanging heuristic / dynamic tension / free-fall priority)
- [x] Force visualization (uniform arrow spec)
- [x] Automatic scene recognition
- [x] Self-test suite (59 items)
- [ ] SAM precise segmentation
- [ ] Depth estimation
- [ ] Object mass database (default masses for common objects)
- [ ] Multi-object interaction
