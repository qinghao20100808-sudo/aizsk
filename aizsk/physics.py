"""
physics.py - 物理推理引擎

核心功能：
1. 判断物体受力情况（重力、支持力、摩擦力、拉力）
2. 根据高中物理规则计算力的方向和大小
3. 支持场景：静止、斜面、悬挂、自由落体

物理约定：
- y 轴向下为正（图像坐标系）
- 所有力以 (magnitude, angle_degrees) 表示，角度从 x 轴正方向逆时针
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .tracker import TrackedObject

logger = logging.getLogger(__name__)

# 重力常数（像素/帧²，需要根据镜头标定）
GRAVITY_DEFAULT = 9.8  # m/s²，实际使用时需要缩放


class ForceType(Enum):
    GRAVITY = "gravity"          # 重力
    NORMAL = "normal"            # 支持力
    FRICTION = "friction"        # 摩擦力
    TENSION = "tension"          # 拉力/张力
    APPLIED = "applied"          # 外力
    AIR_RESISTANCE = "air"       # 空气阻力
    NET = "net"                  # 合力


class SceneType(Enum):
    FLAT_SURFACE = "flat"        # 水平面静止/滑动
    INCLINE = "incline"          # 斜面
    FREE_FALL = "free_fall"      # 自由落体/抛体
    PENDULUM = "pendulum"        # 悬挂/单摆
    UNKNOWN = "unknown"          # 未知


@dataclass
class Force:
    """一个力"""
    type: ForceType
    magnitude: float  # 力的大小（像素大小，用于绘图）
    angle: float      # 角度（度），从 x 轴正方向逆时针
    label: str = ""   # 显示标签
    color: tuple[int, int, int] = (255, 255, 255)  # BGR 颜色

    @property
    def components(self) -> tuple[float, float]:
        """力的分量 (fx, fy)"""
        rad = math.radians(self.angle)
        fx = self.magnitude * math.cos(rad)
        fy = self.magnitude * math.sin(rad)
        return (fx, fy)

    def scale(self, factor: float) -> "Force":
        return Force(
            type=self.type,
            magnitude=self.magnitude * factor,
            angle=self.angle,
            label=self.label,
            color=self.color,
        )


@dataclass
class ForceAnalysis:
    """一个物体的受力分析结果"""
    forces: list[Force] = field(default_factory=list)
    scene_type: SceneType = SceneType.UNKNOWN
    mass: float = 1.0  # 默认质量 1 kg
    mass_unknown: bool = True  # True=用户未填质量，不显示 N 值
    incline_angle: float = 0.0  # 斜面角度（度）
    note: str = ""  # 推理说明

    @property
    def net_force(self) -> Force:
        """计算合力"""
        fx_total = sum(f.components[0] for f in self.forces)
        fy_total = sum(f.components[1] for f in self.forces)
        magnitude = math.sqrt(fx_total**2 + fy_total**2)
        angle = math.degrees(math.atan2(fy_total, fx_total)) % 360
        return Force(
            type=ForceType.NET,
            magnitude=magnitude,
            angle=angle,
            label="合力",
            color=(0, 255, 255),
        )

    def add_force(
        self,
        ftype: ForceType,
        magnitude: float,
        angle: float,
        label: str = "",
        color: Optional[tuple[int, int, int]] = None,
    ):
        """添加一个力"""
        colors = {
            ForceType.GRAVITY: (0, 0, 255),      # 红色（BGR）
            ForceType.NORMAL: (0, 255, 0),        # 绿色
            ForceType.FRICTION: (255, 0, 0),      # 蓝色
            ForceType.TENSION: (255, 255, 0),     # 青色
            ForceType.APPLIED: (255, 0, 255),     # 紫色
            ForceType.AIR_RESISTANCE: (128, 128, 128),  # 灰色
        }
        self.forces.append(
            Force(
                type=ftype,
                magnitude=magnitude,
                angle=angle,
                label=label or ftype.value,
                color=color or colors.get(ftype, (255, 255, 255)),
            )
        )


class PhysicsEngine:
    """物理推理引擎"""

    def __init__(self):
        self.known_inclines: dict[int, float] = {}  # track_id -> angle

    def analyze(
        self,
        tracked: TrackedObject,
        scene_type: SceneType = SceneType.UNKNOWN,
        incline_angle: float = 0.0,
        mass: float = 1.0,
        mass_unknown: bool = False,
    ) -> ForceAnalysis:
        """
        分析物体受力

        Args:
            tracked: 被跟踪的物体
            scene_type: 场景类型
            incline_angle: 斜面角度（度）
            mass: 物体质量（kg），mass_unknown=True 时此值仅用于箭头比例
            mass_unknown: True=用户未填质量，不显示 N 值

        Returns:
            受力分析结果
        """
        analysis = ForceAnalysis(
            scene_type=scene_type,
            mass=mass if not mass_unknown else 0,
            mass_unknown=mass_unknown,
            incline_angle=incline_angle,
        )

        # 重力总是存在（指向 y 正方向，即向下）
        # 用 mass=1 计算比例，但标签不显示 N 值
        display_mass = mass if not mass_unknown else 1.0
        gravity_mag = display_mass * GRAVITY_DEFAULT
        # 图像中 y 向下为正，所以重力角度为 90°
        g_label = "G" if mass_unknown else f"G={gravity_mag:.1f}N"
        analysis.add_force(
            ForceType.GRAVITY,
            gravity_mag,
            90,
            label=g_label,
        )

        if scene_type == SceneType.FLAT_SURFACE:
            self._analyze_flat_surface(tracked, analysis)
        elif scene_type == SceneType.INCLINE:
            self._analyze_incline(tracked, analysis, incline_angle)
        elif scene_type == SceneType.FREE_FALL:
            self._analyze_free_fall(tracked, analysis)
        elif scene_type == SceneType.PENDULUM:
            self._analyze_pendulum(tracked, analysis)
        else:
            # 自动推断场景
            self._auto_detect_scene(tracked, analysis)

        return analysis

    def _analyze_flat_surface(
        self, tracked: TrackedObject, analysis: ForceAnalysis
    ):
        """水平面受力分析"""
        # 支持力 = 重力（方向向上）
        gravity = analysis.forces[0]
        n_label = "N" if analysis.mass_unknown else f"N={gravity.magnitude:.1f}N"
        analysis.add_force(
            ForceType.NORMAL,
            gravity.magnitude,
            270,  # 向上（y 负方向）
            label=n_label,
        )

        if tracked.is_moving():
            # 有运动→可能有摩擦力
            dir_x, dir_y = tracked.movement_direction()
            # 摩擦力与运动方向相反
            friction_angle = math.degrees(math.atan2(-dir_y, -dir_x)) % 360

            # 简单摩擦：动摩擦系数默认 0.2
            mu = 0.2
            friction_mag = mu * gravity.magnitude
            f_label = "f" if analysis.mass_unknown else f"f={friction_mag:.1f}N"
            analysis.add_force(
                ForceType.FRICTION,
                friction_mag,
                friction_angle,
                label=f_label,
            )

            # 分析加速度方向 → 可能的外力
            ax, ay = tracked.current_acceleration
            if abs(ax) > 0.1 or abs(ay) > 0.1:
                applied_angle = math.degrees(math.atan2(ay, ax)) % 360
                applied_mag = math.sqrt(ax**2 + ay**2) * 5  # 缩放
                if applied_mag > 5:
                    F_label = "F" if analysis.mass_unknown else f"F={applied_mag:.1f}N"
                    analysis.add_force(
                        ForceType.APPLIED,
                        applied_mag,
                        applied_angle,
                        label=F_label,
                    )

            analysis.note = f"物体在水平面上{'运动' if tracked.is_moving() else '静止'}"
        else:
            analysis.note = "物体在水平面上静止"
            if not any(f.type == ForceType.NORMAL for f in analysis.forces):
                analysis.add_force(
                    ForceType.NORMAL,
                    gravity.magnitude,
                    270,
                    label=f"N={gravity.magnitude:.1f}N",
                )

    def _analyze_incline(
        self, tracked: TrackedObject, analysis: ForceAnalysis, angle: float
    ):
        """斜面受力分析"""
        theta = math.radians(angle)
        gravity = analysis.forces[0]

        # 重力沿斜面分量
        parallel_mag = gravity.magnitude * math.sin(theta)
        # 重力垂直斜面分量
        perpendicular_mag = gravity.magnitude * math.cos(theta)

        # 支持力 = 重力垂直分量（垂直斜面向上）
        normal_angle = 270 - angle  # 垂直斜面向上
        n_label = "N" if analysis.mass_unknown else f"N={perpendicular_mag:.1f}N"
        analysis.add_force(
            ForceType.NORMAL,
            perpendicular_mag,
            normal_angle % 360,
            label=n_label,
        )

        # 摩擦力（沿斜面，与运动趋势相反）
        if tracked.is_moving():
            # 如果正在下滑，摩擦力沿斜面向上
            friction_angle = (angle + 180) % 360
            mu = 0.15  # 斜面动摩擦系数
            friction_mag = mu * perpendicular_mag
            f_label = "f" if analysis.mass_unknown else f"f={friction_mag:.1f}N"
            analysis.add_force(
                ForceType.FRICTION,
                friction_mag,
                friction_angle,
                label=f_label,
                color=(255, 0, 0),
            )

            # 沿斜面的合力方向（判断加速/减速）
            note_parts = [f"物体在 {angle:.0f}° 斜面上滑动。"]
            if not analysis.mass_unknown:
                note_parts.append(f"沿斜面分力 = {parallel_mag:.1f}N")
            analysis.note = " ".join(note_parts)
        else:
            # 静止在斜面上
            # 静摩擦力 = 重力沿斜面分量
            ax, ay = tracked.current_acceleration
            accel_mag = math.sqrt(ax**2 + ay**2)

            if accel_mag < 0.5:
                # 真正静止
                friction_angle = (angle + 180) % 360  # 沿斜面向上
                fs_label = "f(静)" if analysis.mass_unknown else f"f={parallel_mag:.1f}N(静)"
                analysis.add_force(
                    ForceType.FRICTION,
                    parallel_mag,
                    friction_angle,
                    label=fs_label,
                    color=(255, 0, 0),
                )
                note_static = f"物体静止在 {angle:.0f}° 斜面上。"
                if not analysis.mass_unknown:
                    note_static += f"静摩擦力 = {parallel_mag:.1f}N"
                analysis.note = note_static
            else:
                analysis.note = f"物体在 {angle:.0f}° 斜面上有加速度"

    def _analyze_free_fall(
        self, tracked: TrackedObject, analysis: ForceAnalysis
    ):
        """自由落体受力分析"""
        # 只有重力
        gravity = analysis.forces[0]
        analysis.note = "自由落体：只受重力作用"

        # 检查是否有空气阻力
        vx, vy = tracked.current_velocity
        speed = math.sqrt(vx**2 + vy**2)
        if speed > 10:
            # 空气阻力与速度相反
            drag_angle = math.degrees(math.atan2(-vy, -vx)) % 360
            drag_mag = 0.01 * speed**2  # 简单空气阻力模型
            analysis.add_force(
                ForceType.AIR_RESISTANCE,
                drag_mag,
                drag_angle,
                label=f"f_drag={drag_mag:.1f}N",
            )

    def _analyze_pendulum(
        self, tracked: TrackedObject, analysis: ForceAnalysis
    ):
        """悬挂物体受力分析"""
        gravity = analysis.forces[0]

        # 拉力沿绳子方向（从物体指向悬挂点）
        # 简化：假设悬挂点在物体正上方
        cx, cy = tracked.current_center or (0, 0)
        bbox = tracked.current_bbox
        if bbox:
            # 假设悬挂点在物体上方一定距离
            tension_angle = 270  # 竖直向上
            analysis.add_force(
                ForceType.TENSION,
                gravity.magnitude,
                tension_angle,
                label=f"T={gravity.magnitude:.1f}N",
            )

            # 如果物体有水平速度，分析单摆
            vx, vy = tracked.current_velocity
            if abs(vx) > 1:
                analysis.note = "单摆运动：拉力与重力的合力提供向心力"
            else:
                analysis.note = "静止悬挂：拉力 = 重力"

    def _auto_detect_scene(
        self, tracked: TrackedObject, analysis: ForceAnalysis
    ):
        """自动推断场景类型"""
        bbox = tracked.current_bbox
        vx, vy = tracked.current_velocity
        speed = math.sqrt(vx**2 + vy**2)

        if speed < 2:
            # 基本静止 → 假设在水平面上
            self._analyze_flat_surface(tracked, analysis)
            analysis.scene_type = SceneType.FLAT_SURFACE
            return

        # 有运动 → 看加速度方向
        ax, ay = tracked.current_acceleration

        if abs(ax) < 1 and ay > 3:
            # 主要是竖直向下加速 → 自由落体
            analysis.scene_type = SceneType.FREE_FALL
            self._analyze_free_fall(tracked, analysis)
        elif abs(ax) < 1 and ay < -1:
            # 有向上的加速度（例如被抛起）
            analysis.note = "有竖直向上的运动（可能被抛出）"
            self._analyze_free_fall(tracked, analysis)
            analysis.scene_type = SceneType.FREE_FALL
        else:
            # 有水平分量 → 可能是斜面或水平推力
            analysis.note = "物体有水平运动分量"
            analysis.scene_type = SceneType.FLAT_SURFACE
            self._analyze_flat_surface(tracked, analysis)

    def set_incline_angle(self, track_id: int, angle: float):
        """手动设置斜面角度"""
        self.known_inclines[track_id] = angle
        logger.info(f"物体 {track_id} 斜面角度设为 {angle:.1f}°")
