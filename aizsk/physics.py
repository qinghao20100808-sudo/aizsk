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
    net_smoothed: Optional[Force] = None  # 平滑合力（None=用实时矢量合成）

    @property
    def net_force(self) -> Force:
        """计算合力（有平滑合力时优先返回，避免方向乱跳）"""
        if self.net_smoothed is not None:
            return self.net_smoothed
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
        self._net_smooth: dict[int, tuple[float, float]] = {}  # track_id -> 平滑合力矢量

    def analyze(
        self,
        tracked: TrackedObject,
        scene_type: SceneType = SceneType.UNKNOWN,
        incline_angle: float = 0.0,
        mass: float = 1.0,
        mass_unknown: bool = False,
        frame_size: Optional[tuple[int, int]] = None,  # (W, H)，用于悬空判断
    ) -> ForceAnalysis:
        """
        分析物体受力

        Args:
            tracked: 被跟踪的物体
            scene_type: 场景类型
            incline_angle: 斜面角度（度）
            mass: 物体质量（kg），mass_unknown=True 时此值仅用于箭头比例
            mass_unknown: True=用户未填质量，不显示 N 值
            frame_size: 检测帧尺寸 (W, H)，自动场景推断时用于判断物体是否悬空
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
            self._auto_detect_scene(tracked, analysis, frame_size)

        # 合力方向平滑（EMA 矢量）：手持微颤/模板噪声会让加速度方向
        # 乱跳 → 合力方向乱跳（用户实测"方向乱跳"）。平滑后合力只
        # 跟随整体运动趋势；幅度过小（近匀速）设死区不显示。
        nf = analysis.net_force
        fx = nf.magnitude * math.cos(math.radians(nf.angle))
        fy = nf.magnitude * math.sin(math.radians(nf.angle))
        prev = self._net_smooth.get(tracked.track_id, (fx, fy))
        sx = 0.5 * fx + 0.5 * prev[0]
        sy = 0.5 * fy + 0.5 * prev[1]
        self._net_smooth[tracked.track_id] = (sx, sy)
        mag = math.hypot(sx, sy)
        if mag < 0.4:  # 死区：合力过小（近匀速）不显示，避免噪声方向
            mag = 0.0
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(sy, sx)) % 360
        analysis.net_smoothed = Force(
            type=ForceType.NET,
            magnitude=mag,
            angle=angle,
            label="合力",
            color=(0, 255, 255),
        )

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

            # 分析加速度方向 → 可能的外力。
            # 阈值必须低：手持加速度常为 0.2~1.0 px/帧²，阈值过高会漏掉
            # 外力 → 合力只剩摩擦力（方向与运动相反），用户会看到
            # "往左移动合力却向右"（实测 bug）
            ax, ay = tracked.current_acceleration
            if abs(ax) > 0.1 or abs(ay) > 0.1:
                applied_angle = math.degrees(math.atan2(ay, ax)) % 360
                applied_mag = math.sqrt(ax**2 + ay**2) * 5  # 缩放
                if applied_mag > 0.5:
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
            drag_label = "f_阻" if analysis.mass_unknown else f"f_阻={drag_mag:.1f}N"
            analysis.add_force(
                ForceType.AIR_RESISTANCE,
                drag_mag,
                drag_angle,
                label=drag_label,
            )

    def _analyze_pendulum(
        self, tracked: TrackedObject, analysis: ForceAnalysis
    ):
        """悬挂物体受力分析（拉力 T）
        静止悬挂：T = G（两线等长）
        向上加速：T > G（拉力线更长，体现加速状态）
        """
        gravity = analysis.forces[0]

        # 拉力沿绳子方向（从物体指向悬挂点）
        # 简化：假设悬挂点在物体正上方
        bbox = tracked.current_bbox
        if bbox:
            tension_angle = 270  # 竖直向上

            # 竖直加速度（图像 y 向下为正，ay<0 表示向上加速）
            ax, ay = tracked.current_acceleration
            vx, vy = tracked.current_velocity

            if ay < -1.0:
                # 向上加速 → 拉力大于重力（加速度按比例折算成力）
                extra = abs(ay) * 2.0
                tension_mag = gravity.magnitude + extra
                note = f"物体向上加速运动：拉力大于重力"
                if not analysis.mass_unknown:
                    note = f"物体向上加速运动：拉力 {tension_mag:.1f}N > 重力 {gravity.magnitude:.1f}N"
            elif abs(vx) > 1:
                # 有水平速度 → 单摆
                tension_mag = gravity.magnitude
                note = "单摆运动：拉力与重力的合力提供向心力"
            else:
                # 静止悬挂 → 拉力 = 重力
                tension_mag = gravity.magnitude
                note = "静止悬挂：拉力 = 重力"

            t_label = "T" if analysis.mass_unknown else f"T={tension_mag:.1f}N"
            analysis.add_force(
                ForceType.TENSION,
                tension_mag,
                tension_angle,
                label=t_label,
            )
            analysis.note = note

    def _auto_detect_scene(
        self,
        tracked: TrackedObject,
        analysis: ForceAnalysis,
        frame_size: Optional[tuple[int, int]] = None,
    ):
        """自动推断场景类型

        关键区分：物体悬空（画面中上方、下方无支撑）→ 悬挂（拉力 T）；
                 物体在支撑面上（贴近画面底部/下方有支撑）→ 水平面（支持力 N）。
        悬空判定用画面位置启发式：检测框底部距画面顶部较近 → 悬空。
        """
        bbox = tracked.current_bbox
        vx, vy = tracked.current_velocity
        speed = math.sqrt(vx**2 + vy**2)
        ax, ay = tracked.current_acceleration

        # 悬空判断：bbox 底部在画面上部区域（< 55% 高度）→ 无支撑 → 悬挂
        suspended = False
        if frame_size and bbox:
            h = frame_size[1]
            suspended = bbox[3] < h * 0.55

        if suspended:
            # 悬空：拉力场景（静止 T=G，向上加速 T>G）
            analysis.scene_type = SceneType.PENDULUM
            self._analyze_pendulum(tracked, analysis)
            return

        # 自由落体判定提前：竖直加速度显著且水平分量小 → 立即判为自由落体。
        # 松手瞬间速度≈0，靠加速度触发（不等待速度积累），反应更快。
        if abs(ax) < 2 and ay > 2:
            analysis.scene_type = SceneType.FREE_FALL
            self._analyze_free_fall(tracked, analysis)
            return

        if speed < 2:
            # 基本静止 → 在支撑面上（水平面）
            self._analyze_flat_surface(tracked, analysis)
            analysis.scene_type = SceneType.FLAT_SURFACE
            return

        # 有运动 → 看加速度方向
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
