"""
visualizer.py - 受力可视化模块

负责：
1. 在视频帧上绘制力箭头
2. 显示物体信息（标签、ID、速度等）
3. 场景标注（斜面角、接触面等）
"""

import logging
import math

import cv2
import numpy as np

from .detector import Detection, DetectionResult
from .physics import Force, ForceAnalysis, ForceType, SceneType
from .tracker import TrackedObject

logger = logging.getLogger(__name__)


class ForceVisualizer:
    """受力可视化器"""

    # 颜色方案 (BGR)
    COLORS = {
        ForceType.GRAVITY: (0, 0, 255),        # 红色
        ForceType.NORMAL: (0, 255, 0),          # 绿色
        ForceType.FRICTION: (255, 128, 0),      # 蓝色偏橙
        ForceType.TENSION: (255, 255, 0),       # 青色
        ForceType.APPLIED: (255, 0, 255),       # 紫色
        ForceType.AIR_RESISTANCE: (200, 200, 200),  # 灰色
        ForceType.NET: (0, 255, 255),           # 黄色
    }

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: 力的绘制缩放因子
        """
        self.scale = scale

    def draw_force_arrow(
        self,
        frame: np.ndarray,
        origin: tuple[int, int],
        force: Force,
        scale: float = 30.0,
    ):
        """
        绘制一个力的箭头
        Args:
            frame: 图像
            origin: 箭头起点 (x, y)
            force: 力
            scale: 像素缩放（每个单位力多少像素）
        """
        # 计算终点
        length = force.magnitude * scale * self.scale
        if length < 2:
            return  # 太短不画

        rad = math.radians(force.angle)
        # y 轴在图像中是向下的，角度中 90° 是向下
        # 但在标准物理中角度从 x 轴正方向逆时针
        # 所以图像坐标中：dx = length * cos(θ), dy = length * sin(θ)
        end_x = int(origin[0] + length * math.cos(rad))
        end_y = int(origin[1] + length * math.sin(rad))

        color = self.COLORS.get(force.type, (255, 255, 255))
        thickness = max(2, int(force.magnitude / 20))

        # 画箭头主体
        cv2.arrowedLine(
            frame,
            origin,
            (end_x, end_y),
            color,
            thickness,
            tipLength=0.3,
        )

        # 画标签
        if force.label:
            label_pos = (end_x + 10, end_y - 10)
            cv2.putText(
                frame,
                force.label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        selected: bool = False,
    ):
        """
        绘制检测框
        Args:
            frame: 图像
            detection: 检测结果
            selected: 是否被选中
        """
        x1, y1, x2, y2 = detection.bbox
        color = (0, 255, 0) if selected else (255, 255, 255)
        thickness = 2 if selected else 1

        # 画检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 画标签
        label = detection.class_name
        if detection.track_id is not None:
            label = f"#{detection.track_id} {label}"
        if detection.confidence:
            label += f" {detection.confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # 画分割掩膜（如果有）
        if detection.mask is not None:
            overlay = frame.copy()
            overlay[detection.mask] = (
                overlay[detection.mask] * 0.7 + np.array([0, 255, 0]) * 0.3
            ).astype(np.uint8)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def draw_force_analysis(
        self,
        frame: np.ndarray,
        analysis: ForceAnalysis,
        center: tuple[int, int],
        show_net: bool = True,
    ):
        """
        绘制完整的受力分析图
        Args:
            frame: 图像
            analysis: 受力分析结果
            center: 力的起点（物体中心）
            show_net: 是否显示合力
        """
        # 画所有分力
        for force in analysis.forces:
            self.draw_force_arrow(frame, center, force)

        # 画合力（虚线效果）
        if show_net:
            net = analysis.net_force
            if net.magnitude > 0.5:
                self.draw_force_arrow(frame, center, net)

        # 画场景信息
        if analysis.note:
            info_y = 30
            # 先在顶部画场景类型
            scene_label = f"场景: {analysis.scene_type.value}"
            cv2.putText(
                frame,
                scene_label,
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            # 画推理说明（多行）
            line_y = info_y + 25
            for line in analysis.note.split("。"):
                line = line.strip()
                if not line:
                    continue
                cv2.putText(
                    frame,
                    line + "。",
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                line_y += 22

    def draw_velocity_info(
        self,
        frame: np.ndarray,
        tracked: TrackedObject,
    ):
        """绘制速度/加速度信息"""
        if tracked.current_center is None:
            return

        cx, cy = tracked.current_center
        cx, cy = int(cx), int(cy)

        # 速度向量（绿色）
        vx, vy = tracked.current_velocity
        speed = math.sqrt(vx**2 + vy**2)
        if speed > 0.5:
            ve_end = (cx + int(vx * 5), cy + int(vy * 5))
            cv2.arrowedLine(
                frame, (cx, cy), ve_end, (0, 255, 0), 1, tipLength=0.3
            )

        # 加速度向量（黄色）
        ax, ay = tracked.current_acceleration
        accel = math.sqrt(ax**2 + ay**2)
        if accel > 0.5:
            ae_end = (cx + int(ax * 10), cy + int(ay * 10))
            cv2.arrowedLine(
                frame, (cx, cy), ae_end, (0, 255, 255), 1, tipLength=0.3
            )

    def draw_legend(self, frame: np.ndarray):
        """绘制图例"""
        legend_items = [
            ("重力 G", ForceType.GRAVITY),
            ("支持力 N", ForceType.NORMAL),
            ("摩擦力 f", ForceType.FRICTION),
            ("拉力 T", ForceType.TENSION),
            ("外力 F", ForceType.APPLIED),
            ("合力", ForceType.NET),
        ]

        y_start = frame.shape[0] - 30 * len(legend_items) - 10
        x = frame.shape[1] - 180

        cv2.rectangle(
            frame,
            (x - 5, y_start - 20),
            (x + 170, frame.shape[0] - 5),
            (30, 30, 30),
            -1,
        )

        for i, (label, ftype) in enumerate(legend_items):
            y = y_start + i * 30
            color = self.COLORS.get(ftype, (255, 255, 255))
            cv2.line(frame, (x, y), (x + 20, y), color, 3)
            cv2.putText(
                frame,
                label,
                (x + 25, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def draw_trajectory(
        self,
        frame: np.ndarray,
        tracked: TrackedObject,
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        """绘制运动轨迹"""
        if len(tracked.center_history) < 2:
            return

        points = [
            (int(p[0]), int(p[1]))
            for p in tracked.center_history
        ]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, points[i - 1], points[i], color, thickness)

    def draw_fps(self, frame: np.ndarray, fps: float):
        """绘制 FPS"""
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 120, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    def draw_status(self, frame: np.ndarray, status: str):
        """绘制状态信息"""
        # 底部状态栏
        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 30),
            (frame.shape[1], frame.shape[0]),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            frame,
            status,
            (10, frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def make_frame_preview(
    frame: np.ndarray,
    max_width: int = 800,
) -> np.ndarray:
    """缩放图像到合适预览大小"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame
