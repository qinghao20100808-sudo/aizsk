# ⚡ AI 实时受力分析系统

> **看见物体 + 理解环境 + 推理物理规律**
>
> 基于计算机视觉和物理推理引擎的智能高中物理教学辅助工具。
>
> 🔗 项目地址：https://github.com/qinghao20100808-sudo/aizsk
> 📄 开源协议：MIT License

![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ 核心特性

- 📦 **大疆式框选跟踪**：拖拽画框锁定目标，纯跟踪模式不跑 YOLO（省算力），服务器 1 秒就绪
- 📐 **尺度自适应**：多尺度模板匹配（0.5~2.0 共 25 档），物体远离/靠近摄像头时框随远近平滑缩放
- 🔍 **遮挡处理**：LK 光流 + 模板匹配双引擎，遮挡时匀加速外推预判，重现时自动恢复
- 🧲 **物理推理引擎**：自动识别场景，推理重力 / 支持力 / 拉力 / 摩擦力并绘制力箭头
- 🎯 **自由落体检测**：加速度优先判定 + 慢动作轨迹回放
- 🖱️ **点击兜底**：点击画面任意位置都会触发受力分析（检测不到也能分析）
- 🎨 **纯 Canvas 渲染**：无视频控件干扰，检测框 / 力线 / 轨迹统一绘制

---

## 支持场景

| 场景 | 力的分析 |
|------|---------|
| 水平面静止 | 重力 ↓、支持力 ↑ |
| 水平面运动 | 重力、支持力、摩擦力、外力 |
| 斜面 θ° | 重力 ↓、支持力 ⊥ 斜面、摩擦力 ∥ 斜面 |
| 自由落体 | 重力 ↓（加速度优先判定） |
| 悬挂 / 单摆 | 重力 ↓、拉力 ↑（悬空自动识别，静止 T=G、加速 T>G） |

---

## 快速启动

### Web 模式（推荐，浏览器摄像头）

```bash
uv sync
uv run python3 -m aizsk.main --web
# 浏览器打开 http://localhost:5000
```

### 演示模式（无摄像头）

```bash
uv run python3 -m aizsk.main --demo incline     # 斜面滑块
uv run python3 -m aizsk.main --demo flat_push   # 水平推动
uv run python3 -m aizsk.main --demo free_fall   # 自由落体
uv run python3 -m aizsk.main --demo pendulum    # 单摆
```

### Windows 共享摄像头模式（WSL 环境）

WSL 无法直接访问 Windows 摄像头，先启动 Windows 端 ffmpeg 写入共享帧：

```powershell
# PowerShell（Windows 端）
Start-Process -FilePath 'C:\Users\<用户>\ffmpeg\bin\ffmpeg.exe' -ArgumentList @(
    '-f', 'dshow', '-i', 'video=Webcam C110',
    '-vf', 'fps=10', '-q:v', '10', '-update', '1',
    'C:\Users\<用户>\AppData\Local\Temp\live_cam_hermes.jpg'
) -WindowStyle Hidden
```

```bash
uv run python3 -m aizsk.main --win-cam
```

---

## 操作说明

1. **启动摄像头**：点击「启动摄像头」按钮并允许浏览器授权
2. **框选目标**：在画面上拖拽画框锁定物体 → 进入纯跟踪模式（不跑 YOLO）
3. **快速分析**：单击画面任意位置 → 立即受力分析（无检测时自动创建虚拟物体）
4. **选择场景**：侧栏切换场景类型（水平面 / 斜面 / 自由落体 / 悬挂），斜面可调角度
5. **填写质量**：填入质量显示真实数值（如 G=49.0N）；留空只显示力的符号（G / N / T）
6. **重置**：按 `R` 键清除框选 / 重置跟踪

---

## 系统架构

```
浏览器摄像头 (getUserMedia)
    ↓ 640 降采样（JPEG ~60KB）
Flask /api/detect
    ├─ 框选模式 → ROI 跟踪器（LK 光流 + 多尺度模板匹配）  ← 不跑 YOLO
    └─ 点击模式 → YOLO 检测（yolo11n，懒加载）+ 跟踪器
            ↓
    场景理解（悬空 / 接触面 / 斜面检测）
            ↓
    物理推理引擎（高中物理规则）
            ↓
    力线可视化（长度统一 [30,70]px、线宽统一）
            ↓
浏览器 Canvas 绘制（检测框 + 力箭头 + 运动轨迹）
```

### 核心模块

| 模块 | 说明 |
|------|------|
| `roi_tracker.py` | ROI 框选跟踪器：多尺度模板匹配主位移 + LK 光流辅助、遮挡匀加速外推、恢复搜索、尺度自适应、防瞬移 |
| `detector.py` | YOLO 目标检测（黑名单过滤：排除人 / 家具 / 电器 / 车 / 大型物品） |
| `tracker.py` | 目标跟踪（中心点 / 速度 EMA 平滑，消除检测抖动） |
| `physics.py` | 物理推理引擎（悬空启发式、加速度优先判定、动态拉力） |
| `web_server.py` | Flask 服务器 + API（YOLO 懒加载，ROI 模式不加载模型） |
| `verify.py` | 59 项项目自测（合成帧，不依赖真实摄像头 / 网络） |

---

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/detect/start` | POST | 初始化检测器 / 重置跟踪 |
| `/api/detect` | POST | 接收帧（multipart，含 `roi` / `click_x` / `click_y` / `scene` / `mass` 参数） |
| `/api/click` | POST | 点击坐标 → 强制分析 |
| `/api/scene` | POST | 设置场景类型 |
| `/api/reset` | POST | 清除框选 / 重置跟踪 |
| `/api/status` | GET | 服务器状态 |

---

## 物理推理规则

### 场景自动识别
- **悬空判定**：物体底部 < 55% 画面高度 → 悬空 → 拉力 T；否则 → 支撑面 → 支持力 N
- **自由落体**：加速度 `ay > 2` 立即判定（不等速度平滑）

### 质量处理
- 填写质量 → 显示真实数值（`G=49.0N`）
- 质量未知 → 只显示力的符号（`G` / `N` / `T` / `f`），不显示假设的数值

### 力线可视化
- 长度统一压缩区间 `[30, 70]px`（640 检测空间），线宽统一
- 等值力自然等长（静止 T=G）；加速场景体现差异（向上加速 T>G）

---

## 测试验证

```bash
uv run python3 -m aizsk.verify
# ✅ 全部 59 项验证通过（合成帧 + Flask test_client，约 10-20 秒，不加载 YOLO）
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
| 前端 | 原生 HTML + JS（Canvas） |
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

> ⚠️ 注意：本仓库代码采用 MIT 协议，但 **YOLO 模型权重（yolo11n.pt）遵循 AGPL-3.0**（Ultralytics 许可），分发模型文件时请遵守其许可条款。

---

## 开发计划

- [x] 基础项目结构
- [x] YOLO 目标检测（黑名单过滤 + 懒加载）
- [x] 目标跟踪（EMA 平滑）
- [x] ROI 框选跟踪（LK 光流 + 多尺度模板匹配）
- [x] 尺度自适应（3D→2D 投影缩放跟随）
- [x] 遮挡预测（匀加速外推 + 自动恢复）
- [x] 物理推理引擎（悬空判定 / 动态拉力 / 自由落体优先）
- [x] 受力可视化（力线统一规范）
- [x] 自动场景识别
- [x] 项目自测（59 项）
- [ ] SAM 精确分割
- [ ] 深度估计
- [ ] 物体质量库（常见物体默认质量）
- [ ] 多物体交互
