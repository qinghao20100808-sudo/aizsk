"""
simulation.py - 模拟场景生成器

生成虚拟物理场景用于演示，不需要摄像头也能展示完整的受力分析流程。
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .detector import Detection
from .physics import PhysicsEngine, SceneType, Force, ForceType

logger = logging.getLogger(__name__)


@dataclass
class SimObject:
    """模拟物体"""
    name: str
    x: float  # 中心 x
    y: float  # 中心 y
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    color: tuple[int, int, int] = (100, 200, 255)
    scene_type: SceneType = SceneType.INCLINE
    incline_angle: float = 0.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x1 = int(self.x - self.width / 2)
        y1 = int(self.y - self.height / 2)
        x2 = int(self.x + self.width / 2)
        y2 = int(self.y + self.height / 2)
        return (x1, y1, x2, y2)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)

    def update(self):
        """更新物理状态"""
        self.vx += self.ax
        self.vy += self.ay
        self.x += self.vx
        self.y += self.vy


class SimulationScene:
    """模拟场景"""

    SCENES = {
        "incline": {
            "name": "斜面滑块",
            "desc": "物体在 30° 斜面上滑动",
            "scene_type": SceneType.INCLINE,
            "incline_angle": 30,
        },
        "flat_push": {
            "name": "水平推动",
            "desc": "物体在水平面上被推动",
            "scene_type": SceneType.FLAT_SURFACE,
            "incline_angle": 0,
        },
        "free_fall": {
            "name": "自由落体",
            "desc": "物体自由下落",
            "scene_type": SceneType.FREE_FALL,
            "incline_angle": 0,
        },
        "pendulum": {
            "name": "单摆",
            "desc": "物体悬挂在绳子上摆动",
            "scene_type": SceneType.PENDULUM,
            "incline_angle": 0,
        },
    }

    def __init__(self, scene_id: str = "incline"):
        self.scene_id = scene_id
        config = self.SCENES.get(scene_id, self.SCENES["incline"])
        self.scene_type = config["scene_type"]
        self.incline_angle = config["incline_angle"]
        self.width = 640
        self.height = 480

        self.objects: list[SimObject] = []
        self._init_objects()

        self.frame_count = 0

    def _init_objects(self):
        """初始化场景物体"""
        self.objects.clear()

        if self.scene_id == "incline":
            # 在 30° 斜面上
            self.surface_angle = self.incline_angle
            theta = math.radians(self.surface_angle)

            # 物体在斜面顶部
            obj = SimObject(
                name="滑块",
                x=220,
                y=200,
                width=80,
                height=50,
                scene_type=SceneType.INCLINE,
                incline_angle=self.surface_angle,
                color=(100, 200, 255),
            )
            # 沿斜面加速度（慢速，便于观察和点击）
            g_parallel = 9.8 * math.sin(theta) * 0.02
            obj.ax = g_parallel * math.cos(theta)
            obj.ay = g_parallel * math.sin(theta)
            self.objects.append(obj)

        elif self.scene_id == "flat_push":
            # 水平面推动
            obj = SimObject(
                name="箱子",
                x=200,
                y=350,
                width=60,
                height=60,
                scene_type=SceneType.FLAT_SURFACE,
                color=(200, 150, 100),
            )
            obj.vx = 3.0
            self.objects.append(obj)

        elif self.scene_id == "free_fall":
            obj = SimObject(
                name="球",
                x=320,
                y=100,
                width=30,
                height=30,
                scene_type=SceneType.FREE_FALL,
                color=(255, 100, 100),
            )
            obj.vy = 4.0
            obj.ay = 0.3
            self.objects.append(obj)

        elif self.scene_id == "pendulum":
            obj = SimObject(
                name="摆球",
                x=370,
                y=180,
                width=25,
                height=25,
                scene_type=SceneType.PENDULUM,
                color=(100, 255, 100),
            )
            obj.vx = -1.5  # 摆动
            self.objects.append(obj)

    def step(self) -> list[Detection]:
        """前进一帧"""
        self.frame_count += 1
        detections = []

        for obj in self.objects:
            obj.update()

            # 边界约束
            if obj.x < 30 or obj.x > self.width - 30:
                obj.vx *= -0.5
                obj.x = max(30, min(self.width - 30, obj.x))
            if obj.y < 30 or obj.y > self.height - 30:
                obj.vy *= -0.5
                obj.y = max(30, min(self.height - 30, obj.y))

            # 如果是斜面场景，约束在斜面附近并循环
            if self.scene_id == "incline":
                # 滑到底部后反弹，不重置（方便点击）
                if obj.y > 380:
                    obj.vy = -abs(obj.vy) * 0.8
                    obj.y = 380
                    obj.vx *= 0.9
                if obj.x > 580:
                    obj.vx = -abs(obj.vx) * 0.8
                    obj.x = 580
                    obj.vy *= 0.9

            # 单摆：模拟简谐运动
            if self.scene_id == "pendulum":
                if abs(obj.x - 370) > 120:
                    obj.vx *= -0.9

            # 创建检测对象（用于显示）
            det = Detection(
                class_id=0,
                class_name=obj.name,
                confidence=0.95,
                bbox=obj.bbox,
            )
            detections.append(det)

        return detections

    def render(self, frame: np.ndarray, detections: list[Detection]):
        """渲染场景"""
        # 画背景
        if self.scene_id == "incline":
            self._draw_incline(frame)
        elif self.scene_id == "flat_push":
            self._draw_flat_surface(frame)
        elif self.scene_id == "free_fall":
            pass  # 纯背景
        elif self.scene_id == "pendulum":
            self._draw_pendulum(frame)

        # 画物体
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj.color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # 标签
            cv2.putText(
                frame,
                obj.name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                obj.color,
                2,
            )

        # 场景信息
        info = self.SCENES.get(self.scene_id, {})
        cv2.putText(
            frame,
            f"演示: {info.get('name', '')} ({info.get('desc', '')})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 255),
            2,
        )

    def _draw_incline(self, frame: np.ndarray):
        """画斜面"""
        theta = math.radians(self.incline_angle)
        length = 500
        x1, y1 = 80, 400
        x2 = int(x1 + length * math.cos(theta))
        y2 = int(y1 - length * math.sin(theta))

        cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 150), 8)

        # 地面
        cv2.line(frame, (0, y1), (640, y1), (100, 100, 100), 3)

        # 角度标注
        cx = x1 + 60
        cv2.ellipse(
            frame,
            (x1, y1),
            (80, 80),
            0,
            360 - self.incline_angle,
            360,
            (0, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"θ={self.incline_angle}°",
            (x1 + 70, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 200),
            1,
        )

    def _draw_flat_surface(self, frame: np.ndarray):
        """画水平面"""
        cv2.line(frame, (0, 410), (640, 410), (150, 150, 150), 6)
        cv2.putText(
            frame, "水平面", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )

    def _draw_pendulum(self, frame: np.ndarray):
        """画单摆"""
        if not self.objects:
            return
        pivot = (370, 80)
        obj = self.objects[0]
        cx, cy = int(obj.x), int(obj.y)

        # 绳子
        cv2.line(frame, pivot, (cx, cy), (200, 200, 100), 2)
        # 悬挂点
        cv2.circle(frame, pivot, 5, (200, 200, 100), -1)

    def get_tracked_data(self) -> dict:
        """获取当前帧的跟踪数据（模拟）"""
        data = {}
        for i, obj in enumerate(self.objects):
            data[i + 1] = {
                "center": (obj.x, obj.y),
                "velocity": (obj.vx, obj.vy),
                "acceleration": (obj.ax, obj.ay),
                "bbox": obj.bbox,
            }
        return data
