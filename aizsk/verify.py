"""
aizsk/verify.py - 项目自测脚本（ad-hoc 验证，非 pytest 套件）

运行: uv run python3 -m aizsk.verify

覆盖：
1. tracker.py     EMA 平滑（静止抖动/匀速/骤停）
2. physics.py     场景识别（悬空→拉力 T=G、向上加速→T>G、支撑面→支持力 N）
                  与 mass_unknown 标签约定
3. roi_tracker.py LK 光流框选跟踪（跟踪/遮挡外推/模板恢复/lost）
4. web_server.py  /api/detect 管线（ROI 模式、YOLO 懒加载、reset）
5. 前端 index.html 静态断言 + JS 语法（node --check）

全程不加载 YOLO（用合成帧/假检测器），CPU 上约 10-20 秒。
"""

import io
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import cv2
import numpy as np

PASSED = []


def check(name, cond, extra=""):
    assert cond, f"FAIL: {name} {extra}"
    PASSED.append(name)
    print(f"  [PASS] {name}" + (f"  ({extra})" if extra else ""))


# ================= 合成测试素材 =================
W, H = 640, 480
_rng = np.random.default_rng(7)
BG = _rng.integers(40, 215, (H, W), dtype=np.uint8)


def make_target(cx, cy, size=56):
    """棋盘格目标（强角点纹理）"""
    img = np.full((size, size), 120, np.uint8)
    for r in range(0, size, 7):
        for c in range(0, size, 7):
            img[r:r+7, c:c+7] = 30 if (r//7 + c//7) % 2 == 0 else 230
    img = (img.astype(np.int16) + _rng.integers(-15, 15, img.shape)
           ).clip(0, 255).astype(np.uint8)
    f = cv2.cvtColor(BG.copy(), cv2.COLOR_GRAY2BGR)
    x1, y1 = cx - size//2, cy - size//2
    f[y1:y1+size, x1:x1+size] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return f


def blank():
    return cv2.cvtColor(BG.copy(), cv2.COLOR_GRAY2BGR)


# ================= 1. tracker.py：EMA 平滑 =================
print("── tracker.py EMA 平滑 ──")
from aizsk.tracker import TrackedObject

t = TrackedObject(track_id=1, class_name="bottle")
_rng2 = np.random.default_rng(42)
for _ in range(20):
    n = _rng2.uniform(-3, 3)
    x1 = int(95 + n)
    t.update((x1, 95, x1 + 10, 105))
check("静止+3px噪声不误判运动", not t.is_moving(), f"speed={t.speed:.2f}")

t2 = TrackedObject(track_id=2, class_name="cup")
for i in range(1, 15):
    cx = 50 + i * 5
    t2.update((cx, 30, cx + 10, 40))
check("匀速5px/帧速度跟踪", abs(t2.current_velocity[0] - 5) < 1.5,
      f"vx={t2.current_velocity[0]:.2f}")

t3 = TrackedObject(track_id=3, class_name="book")
for i in range(1, 8):
    t3.update((50 + i*5, 30, 60 + i*5, 40))
for _ in range(8):
    t3.update((85, 30, 95, 40))
check("骤停后判定静止", not t3.is_moving(), f"vx={t3.current_velocity[0]:.2f}")

# ================= 2. physics.py：场景识别 + 标签 =================
print("── physics.py 场景识别与标签 ──")
from aizsk.physics import PhysicsEngine, SceneType, ForceType

eng = PhysicsEngine()
FRAME = (W, H)


def mk_tracked(bbox, vel=(0.0, 0.0), acc=(0.0, 0.0)):
    t = TrackedObject(track_id=1, class_name="目标")
    t.bbox_history.append(bbox)
    t.center_history.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
    t.smoothed_center = t.center_history[-1]
    t.smoothed_velocity = vel
    t.velocities.append(vel)
    t.accelerations.append(acc)
    return t


# 悬空静止（bbox 底 160 < 0.55*480=264）→ 拉力 T = G
a = eng.analyze(mk_tracked((200, 100, 260, 160)), SceneType.UNKNOWN,
                mass=0, mass_unknown=True, frame_size=FRAME)
types = [f.type for f in a.forces]
check("悬空静止 → 拉力 T（非支持力）", ForceType.TENSION in types
      and ForceType.NORMAL not in types)
g = next(f for f in a.forces if f.type == ForceType.GRAVITY)
tn = next(f for f in a.forces if f.type == ForceType.TENSION)
check("静止悬挂 T = G", abs(tn.magnitude - g.magnitude) < 1e-6,
      f"T={tn.magnitude:.2f} G={g.magnitude:.2f}")
check("悬空场景 = pendulum", a.scene_type == SceneType.PENDULUM)

# 自由落体提前判定：松手瞬间速度≈0 但加速度显著 → 立即自由落体（不等待速度积累）
a_ff = eng.analyze(mk_tracked((200, 300, 260, 360), vel=(0.0, 0.5), acc=(0.0, 4.0)),
                   SceneType.UNKNOWN, mass=0, mass_unknown=True, frame_size=FRAME)
check("松手瞬间(速度≈0)即判自由落体", a_ff.scene_type == SceneType.FREE_FALL
      and ForceType.TENSION not in [f.type for f in a_ff.forces],
      a_ff.scene_type.value)

# 向上加速 → T > G
a2 = eng.analyze(mk_tracked((200, 100, 260, 160), acc=(0.0, -3.0)),
                 SceneType.UNKNOWN, mass=2, mass_unknown=False, frame_size=FRAME)
g2 = next(f for f in a2.forces if f.type == ForceType.GRAVITY)
t2f = next(f for f in a2.forces if f.type == ForceType.TENSION)
check("向上加速 → T > G", t2f.magnitude > g2.magnitude,
      f"T={t2f.magnitude:.1f} > G={g2.magnitude:.1f}")

# 支撑面静止（bbox 底 440 > 264）→ 支持力 N
a3 = eng.analyze(mk_tracked((200, 380, 260, 440)), SceneType.UNKNOWN,
                 mass=0, mass_unknown=True, frame_size=FRAME)
types3 = [f.type for f in a3.forces]
check("支撑面静止 → 支持力 N（非拉力）", ForceType.NORMAL in types3
      and ForceType.TENSION not in types3)
check("支撑面场景 = flat", a3.scene_type == SceneType.FLAT_SURFACE)

# 无 frame_size → 旧行为
a4 = eng.analyze(mk_tracked((200, 100, 260, 160)), SceneType.UNKNOWN,
                 mass=0, mass_unknown=True, frame_size=None)
check("无帧尺寸 → 旧行为 flat/N", a4.scene_type == SceneType.FLAT_SURFACE
      and ForceType.NORMAL in [f.type for f in a4.forces])

# 手动选悬挂 → 拉力（不受启发式影响）
a5 = eng.analyze(mk_tracked((200, 380, 260, 440)), SceneType.PENDULUM,
                 mass=0, mass_unknown=True, frame_size=FRAME)
check("手动选悬挂 → 拉力 T", ForceType.TENSION in [f.type for f in a5.forces])

# mass_unknown 标签：只显示符号
a6 = eng.analyze(mk_tracked((200, 380, 260, 440)), SceneType.PENDULUM,
                 mass=0, mass_unknown=True)
check("未知质量只显示符号", [f.label for f in a6.forces] == ["G", "T"])
a7 = eng.analyze(mk_tracked((200, 380, 260, 440)), SceneType.PENDULUM,
                 mass=2, mass_unknown=False)
check("已知质量显示 N 值", [f.label for f in a7.forces] == ["G=19.6N", "T=19.6N"])

# ================= 3. roi_tracker.py：跟踪/遮挡/恢复 =================
print("── roi_tracker.py 框选跟踪 ──")
from aizsk.roi_tracker import ROITracker

SPEED, IB = 4, (122, 122, 178, 178)
tr = ROITracker()
assert tr.init(make_target(150, 150), IB)
check("init → tracking", tr.state == "tracking")
oks, errs = [], []
for i in range(1, 21):
    ok, bb = tr.update(make_target(150 + SPEED*i, 150))
    oks.append(ok)
    errs.append(abs((bb[0]+bb[2])/2 - (150 + SPEED*i)) if ok and bb else 999)
check("可见期 20 帧全跟踪", all(oks), f"ok={sum(oks)}/20")
check("中心误差 < 12px", max(errs) < 12, f"max={max(errs):.1f}px")

sts = []
for _ in range(12):
    ok, bb = tr.update(blank())
    sts.append((tr.state, ok, bb))
check("遮挡 → predicting", any(s[0] == "predicting" for s in sts))
lastp = [s[2] for s in sts if s[2] is not None][-1]
check("预测框外推右移", lastp[0] > IB[2], f"x1={lastp[0]}")

nr = 0
for i in range(16):
    ok, bb = tr.update(make_target(150 + SPEED*44 + SPEED*i, 150))
    if tr.state == "tracking" and ok:
        nr += 1
check("目标重现后恢复 tracking", nr >= 3, f"tracking={nr}/16")

# RECOVER_MARGIN=80：重现于外推路径外 70px
tr2 = ROITracker()
tr2.init(make_target(150, 150), IB)
for i in range(1, 6):
    tr2.update(make_target(150 + 4*i, 150))
tr2.update(blank()); tr2.update(blank())
ok, bb = tr2.update(make_target(240, 150))
check("外推路径外 70px 恢复(margin=80)", ok and tr2.state == "tracking")

# 持续遮挡 → lost
tr3 = ROITracker()
tr3.init(make_target(150, 150), IB)
for _ in range(4):
    tr3.update(make_target(154, 150))
for _ in range(ROITracker.MAX_LOST + 5):
    ok, bb = tr3.update(blank())
check("持续遮挡 → lost 无预测框", tr3.state == "lost" and bb is None)

# 方向修复：目标向上运动（y 递减）不得识别成向左
tr_dir = ROITracker()
tr_dir.init(make_target(200, 320), (172, 292, 228, 348))
for i in range(1, 11):
    tr_dir.update(make_target(200, 320 - 4*i))
x_offs, y_seq = [], []
for i in range(11, 16):
    ok, bb = tr_dir.update(make_target(200, 320 - 4*i))
    assert ok and bb is not None
    x_offs.append((bb[0]+bb[2])/2 - 200)
    y_seq.append(bb[1])
check("向上运动方向正确(不偏左)", max(abs(d) for d in x_offs) < 8,
      f"中心x偏移={[round(d,1) for d in x_offs]}")
check("向上运动 y 持续递减", all(y_seq[i] < y_seq[i-1] for i in range(1, len(y_seq))),
      f"y序列={y_seq}")

# 防瞬移：目标突然跳 60px（超出 MAX_STEP）→ 位移必须被拆成多帧追赶，
# 任何单帧位移不得超过 MAX_STEP（不能一次瞬移到目标处）
tr_jump = ROITracker()
tr_jump.init(make_target(150, 150), IB)
for i in range(1, 6):
    tr_jump.update(make_target(150 + 4*i, 150))
disp_all = []
for i in range(6, 12):
    ok, bb = tr_jump.update(make_target(150 + 4*5 + 60, 150))
    assert ok and bb is not None
    disp_all.append((bb[0]+bb[2])/2 - (150 + 4*5))
check("无瞬移：单帧位移 ≤ MAX_STEP",
      all(disp_all[i] - disp_all[i-1] <= ROITracker.MAX_STEP
          for i in range(1, len(disp_all))),
      f"单帧增量={[round(disp_all[i]-disp_all[i-1],1) for i in range(1,len(disp_all))]}")
check("多帧渐进追赶(累计接近目标)", max(disp_all) >= 40,
      f"最终偏移={max(disp_all):.1f}px (目标60px)")

# 匀加速外推：遮挡期预测步长应递增（自由落体式加速）
tr_acc = ROITracker()
tr_acc.init(make_target(150, 100), (122, 72, 178, 128))
for i in range(1, 8):
    tr_acc.update(make_target(150, 100 + 4*i))       # 匀速 4px/帧向下
for i in range(8, 13):
    tr_acc.update(make_target(150, 100 + 4*7 + 8*(i-7)))  # 加速 8px/帧
ok, bb0 = tr_acc.update(blank())                     # 进入预测
steps = []
prev = bb0[1] if bb0 else None
for _ in range(4):
    ok, bb = tr_acc.update(blank())
    if bb is None:
        break
    if prev is not None:
        steps.append(bb[1] - prev)
    prev = bb[1]
check("遮挡期步长递增(匀加速)", len(steps) >= 3 and steps[-1] > steps[0],
      f"步长={steps}")

# 尺度自适应（3D→2D 投影：物体沿光轴移动 → 画面尺寸变化）
BASE = _rng.integers(30, 230, (80, 80), dtype=np.uint8)  # 固定基准图案


def make_target_scale(cx, cy, size):
    img = cv2.resize(BASE, (size, size), interpolation=cv2.INTER_AREA)
    f = cv2.cvtColor(BG.copy(), cv2.COLOR_GRAY2BGR)
    x1, y1 = cx - size // 2, cy - size // 2
    f[y1:y1+size, x1:x1+size] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return f


# 缩小 56→32（远离摄像头）：bbox 尺寸跟随缩小，全程不跟丢
tr_s = ROITracker()
tr_s.init(make_target_scale(150, 150, 56), (122, 122, 178, 178))
hs = []
for sz in range(54, 30, -4):
    ok, bb = tr_s.update(make_target_scale(150, 150, sz))
    hs.append(bb[3] - bb[1] if bb else None)
check("缩小43%全程跟踪", tr_s.state == "tracking")
check("bbox 尺寸跟随缩小", hs[-1] < hs[0] and hs[-1] <= 36, f"高度={hs}")

# 放大 32→60（靠近摄像头）：bbox 尺寸跟随放大
tr_b = ROITracker()
tr_b.init(make_target_scale(150, 150, 32), (134, 134, 166, 166))
hs2 = []
for sz in range(36, 62, 4):
    ok, bb = tr_b.update(make_target_scale(150, 150, sz))
    hs2.append(bb[3] - bb[1] if bb else None)
check("放大87%全程跟踪", tr_b.state == "tracking", f"高度={hs2}")
check("bbox 尺寸跟随放大", hs2[-1] > hs2[0] and hs2[-1] >= 50, f"高度={hs2}")

# 快速移动+缩放组合：上移 30px/帧 + 放大（之前光流融合会拉低位移导致掉队）
tr_fast = ROITracker()
tr_fast.init(make_target_scale(200, 300, 40), (180, 280, 220, 320))
centers = []
for i, sz in enumerate(range(44, 62, 4)):
    ok, bb = tr_fast.update(make_target_scale(200, 300 - 30*(i+1), sz))
    centers.append(((bb[1]+bb[3])/2) if bb else None)
check("快速移动+放大全程跟踪", tr_fast.state == "tracking", f"中心y={[round(c) for c in centers]}")
check("快速移动跟踪误差<8px",
      all(abs(centers[i] - (300 - 30*(i+1))) < 8 for i in range(len(centers))),
      f"误差={[round(abs(centers[i]-(300-30*(i+1))),1) for i in range(len(centers))]}")

# 缩小→放大组合（远离后再靠近）：尺寸双向跟随，全程不丢
tr_sr = ROITracker()
tr_sr.init(make_target_scale(150, 150, 50), (125, 125, 175, 175))
hs_sr = []
for sz in [42, 34, 26]:          # 远离（缩小）
    ok, bb = tr_sr.update(make_target_scale(150, 150, sz))
    hs_sr.append(bb[3] - bb[1] if bb else None)
for sz in [38, 52]:              # 靠近（放大）
    ok, bb = tr_sr.update(make_target_scale(150, 150, sz))
    hs_sr.append(bb[3] - bb[1] if bb else None)
check("缩小→放大组合全程跟踪", tr_sr.state == "tracking", f"高度序列={hs_sr}")
check("尺寸双向跟随(先缩后放)", hs_sr[0] > hs_sr[2] and hs_sr[-1] > hs_sr[2],
      f"高度序列={hs_sr}")

# ================= 4. web_server.py：/api/detect 管线 =================
print("── web_server.py /api/detect 管线 ──")
from aizsk.web_server import WebServer

srv = WebServer()
client = srv.app.test_client()
check("启动时 YOLO 未加载(懒加载)", srv._detector_loaded is False)


def jpg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(buf.tobytes())


def post(frame, **form):
    data = {"image": (jpg(frame), "f.jpg")}
    data.update(form)
    return client.post("/api/detect", data=data,
                       content_type="multipart/form-data").get_json()


d = post(make_target(150, 150), roi="122,122,178,178", mass="0")
check("框选 → mode=roi tracking", d["mode"] == "roi"
      and d["tracking_state"] == "tracking")
check("ROI 响应含分析/速度/选中", d["analysis"] and d["velocity"]
      and d["selected_id"] == 1)
check("未知质量无 N 值", all("=" not in f["label"] for f in d["analysis"]["forces"]))

d2 = post(make_target(158, 150))
check("后续帧持续 ROI 跟踪", d2["mode"] == "roi"
      and d2["tracking_state"] == "tracking")

d3 = post(blank()); d3 = post(blank())
check("连续 2 帧遮挡 → predicting", d3["tracking_state"] == "predicting")

d4 = post(make_target(158, 150), mass="2")
check("质量2kg显示 N 值", any("=" in f["label"] for f in d4["analysis"]["forces"]))

# 悬空/支撑面识别经 ROI 管线贯通
d5 = post(make_target(150, 100), roi="122,72,178,128", mass="0")
ft5 = [f["type"] for f in d5["analysis"]["forces"]]
check("ROI 悬空 → tension", "tension" in ft5 and "normal" not in ft5, str(ft5))
d6 = post(make_target(150, 410), roi="122,382,178,438", mass="0")
ft6 = [f["type"] for f in d6["analysis"]["forces"]]
check("ROI 支撑面 → normal", "normal" in ft6 and "tension" not in ft6, str(ft6))

r = client.post("/api/reset")
check("reset 成功", r.get_json()["success"] is True)
d7 = post(make_target(158, 150))
check("reset 后 ROI 清空(回 YOLO 懒加载)", d7.get("mode") == "yolo"
      and srv._detector_loaded is True, f"mode={d7.get('mode')}")

# trail 轨迹点随速度信息返回（慢动作视觉数据）
d8 = post(make_target(150, 150), roi="122,122,178,178", mass="0")
d9 = post(make_target(158, 150), mass="0")
check("velocity 含 trail 轨迹点", d9.get("velocity") is not None
      and len(d9["velocity"].get("trail", [])) >= 1,
      f"trail点数={len((d9.get('velocity') or {}).get('trail', []))}")

# ================= 5. 前端 index.html =================
print("── 前端 index.html ──")
html = (PROJECT / "aizsk" / "static" / "index.html").read_text(encoding="utf-8")
js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
p = subprocess.run(["node", "--check", "-"], input=js.encode(),
                   capture_output=True, timeout=30)
check("JS 语法 (node --check)", p.returncode == 0, p.stderr.decode()[:200])
for name, needle in [
    ("640 降采样", "const CAP_W = 640" in js),
    ("拖拽画框", all(k in js for k in ["mousedown", "mousemove", "mouseup", "drawDragRect"])),
    ("roi 参数", "fd.append('roi', roiStr)" in js),
    ("跟踪状态着色", "遮挡预测中" in js and "已丢失" in js),
    ("间隔自适应 100/500", "let want = 500" in js and "? 100 : 100" in js),
    ("力线 clamp [30,70]", "MIN_LEN = 30, MAX_LEN = 70" in js),
    ("统一线宽", "ctx.lineWidth = 2.5 * scale" in js),
    ("重心实时外推", "extrapolatedCenter" in js and "lastVel.vx * dtFrames" in js),
    ("无拖尾轨迹", "velocity.trail" not in js and "pts[0][0]*scale" not in js),
    ("自由落体高频采样", "=== 'free_fall') ? 100 : 100" in js),
    ("像素/秒换算", "像素/秒" in js),
    ("/api/reset 调用", "fetch('/api/reset'" in js),
]:
    check(name, needle)

print(f"\n✅ 全部 {len(PASSED)} 项验证通过（ad-hoc，非测试套件）")
