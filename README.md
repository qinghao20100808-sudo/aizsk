# ⚡ AI 实时受力分析系统

> **看见物体 + 理解环境 + 推理物理规律**
>
> 基于计算机视觉和物理推理引擎的智能高中物理教学辅助工具。
>
> 🔗 项目地址：https://github.com/qinghao20100808-sudo/aizsk

---

## 效果演示

通过摄像头实时获取画面 → 点击物体 → 自动分析受力 → 实时绘制力箭头。

支持场景：
- 📦 **水平面**：静止/滑动物体 → 重力、支持力、摩擦力
- 📐 **斜面**：物体在斜面上 → 重力分解、支持力、摩擦力
- 🎯 **自由落体**：抛体运动 → 重力、空气阻力
- 🔗 **悬挂**：单摆/静止悬挂 → 重力、拉力

---

## 快速启动

```bash
# 1. 安装依赖
uv sync

# 2. 启动
uv run python -m aizsk.main

# 3. 浏览器打开
# http://localhost:5000
```

### 其他启动方式

```bash
# 无摄像头模式（调试前端）
uv run python -m aizsk.main --no-cam

# 指定端口
uv run python -m aizsk.main --port 8080

# 指定摄像头
uv run python -m aizsk.main --camera 1
```

---

## 操作说明

1. **打开页面**：浏览器访问 `http://localhost:5000`
2. **点击物体**：点击视频画面中的物体
3. **自动分析**：系统检测物体 → 跟踪运动 → 推理受力
4. **手动调整**：在侧栏选择场景类型（水平面/斜面/自由落体/悬挂）
5. **斜面角度**：选择"斜面"后，用滑块调整角度
6. **重置**：按 `R` 键

---

## 系统架构

```
摄像头视频流
    ↓
OpenCV 图像处理
    ↓
YOLO 目标检测（Ultralytics）
    ↓
ByteTrack 目标跟踪
    ↓
场景理解（接触面/斜面检测）
    ↓
物理推理引擎（高中物理规则）
    ↓
Flask Web 服务器
    ↓
浏览器前端（HTML + JavaScript）
```

---

## 项目结构

```
aizsk/
├── pyproject.toml        # 项目配置
├── aizsk/
│   ├── main.py           # 入口
│   ├── detector.py       # YOLO 检测
│   ├── tracker.py        # 目标跟踪
│   ├── physics.py        # 物理推理引擎 ★
│   ├── scene.py          # 场景理解
│   ├── visualizer.py     # 受力可视化
│   ├── web_server.py     # Flask 服务器
│   └── static/
│       └── index.html    # 前端页面
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 目标检测 | YOLOv11 (Ultralytics) |
| 计算机视觉 | OpenCV |
| 数学计算 | NumPy |
| Web 服务器 | Flask |
| 前端 | 原生 HTML + JS |
| 包管理 | uv |

---

## 开源项目致谢

本项目基于以下开源项目构建，感谢它们的贡献：

| 项目 | 用途 | 链接 |
|------|------|------|
| **PyTorch** | 深度学习框架（模型推理） | https://github.com/pytorch/pytorch |
| **Ultralytics YOLO** | YOLOv11 目标检测模型 | https://github.com/ultralytics/ultralytics |
| **OpenCV** | 图像处理 / 光流跟踪 | https://github.com/opencv/opencv |
| **NumPy** | 数值计算 | https://github.com/numpy/numpy |
| **Flask** | Web 服务器 | https://github.com/pallets/flask |
| **uv** | Python 包管理与运行 | https://github.com/astral-sh/uv |

---

## 物理推理规则

### 水平面静止
- **重力** ↓：G = mg
- **支持力** ↑：N = G

### 水平面运动
- **重力** ↓：G = mg
- **支持力** ↑：N = G
- **摩擦力** ←/→：f = μN（与运动方向相反）
- **外力** →：根据加速度推算

### 斜面（角度 θ）
- **重力** ↓：G = mg
- **重力分解**：G_∥ = mg sinθ（沿斜面）
- **重力分解**：G_⊥ = mg cosθ（垂直斜面）
- **支持力**：N = G_⊥（垂直斜面向上）
- **摩擦力**：f = μN（沿斜面，与运动趋势相反）

### 自由落体
- **重力** ↓：G = mg
- **空气阻力** ↑（高速时）：f = kv²

### 悬挂
- **重力** ↓：G = mg
- **拉力** ↑：T = G（静止时）

---

## 开发计划

- [x] 基础项目结构
- [x] YOLO 目标检测
- [x] 目标跟踪
- [x] 物理推理引擎
- [x] 受力可视化
- [x] Web 界面
- [ ] SAM 精确分割
- [ ] 深度估计
- [ ] 自动场景识别
- [ ] 多物体交互
