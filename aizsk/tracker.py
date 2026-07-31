"""
tracker.py - 目标跟踪模块

负责：
1. ByteTrack/BoT-SORT 目标跟踪
2. 跨帧保持物体 ID
3. 运动趋势分析（速度、加速度估计）
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """被跟踪的物体"""
    track_id: int
    class_name: str
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    center_history: list[tuple[float, float]] = field(default_factory=list)
    velocities: list[tuple[float, float]] = field(default_factory=list)
    accelerations: list[tuple[float, float]] = field(default_factory=list)
    max_history: int = 30  # 保存近 30 帧历史
    # 平滑状态（EMA 抑制 YOLO 检测框抖动）
    smoothed_center: Optional[tuple[float, float]] = None
    smoothed_velocity: tuple[float, float] = (0.0, 0.0)
    _smooth_alpha: float = 0.4  # EMA 平滑系数（越大响应越快，越小越稳）

    def update(self, bbox: tuple[int, int, int, int]):
        """更新轨迹（带 EMA 平滑，抑制检测框抖动）"""
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)

        # 计算中心点（原始值入历史，供画轨迹）
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.center_history.append((cx, cy))
        if len(self.center_history) > self.max_history:
            self.center_history.pop(0)

        # EMA 平滑中心点
        alpha = self._smooth_alpha
        if self.smoothed_center is None:
            self.smoothed_center = (cx, cy)
            self.velocities.append((0.0, 0.0))
            return

        prev_sx, prev_sy = self.smoothed_center
        self.smoothed_center = (
            alpha * cx + (1 - alpha) * prev_sx,
            alpha * cy + (1 - alpha) * prev_sy,
        )

        # 速度 = 平滑中心差分，再 EMA 滤波（像素/帧）
        raw_vx = self.smoothed_center[0] - prev_sx
        raw_vy = self.smoothed_center[1] - prev_sy
        ovx, ovy = self.smoothed_velocity
        self.smoothed_velocity = (
            alpha * raw_vx + (1 - alpha) * ovx,
            alpha * raw_vy + (1 - alpha) * ovy,
        )
        self.velocities.append(self.smoothed_velocity)
        if len(self.velocities) > self.max_history:
            self.velocities.pop(0)

        # 加速度 = 平滑速度差分（像素/帧²）
        if len(self.velocities) >= 2:
            ax = self.velocities[-1][0] - self.velocities[-2][0]
            ay = self.velocities[-1][1] - self.velocities[-2][1]
            self.accelerations.append((ax, ay))
            if len(self.accelerations) > self.max_history:
                self.accelerations.pop(0)

    @property
    def current_bbox(self) -> Optional[tuple[int, int, int, int]]:
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def current_center(self) -> Optional[tuple[float, float]]:
        """当前中心点（平滑后，用于受力箭头绘制）"""
        return self.smoothed_center or (
            self.center_history[-1] if self.center_history else None
        )

    @property
    def current_velocity(self) -> tuple[float, float]:
        """当前速度 (vx, vy)，像素/帧（已平滑）"""
        return self.smoothed_velocity

    @property
    def current_acceleration(self) -> tuple[float, float]:
        """当前加速度 (ax, ay)，像素/帧²"""
        if self.accelerations:
            return self.accelerations[-1]
        return (0.0, 0.0)

    @property
    def speed(self) -> float:
        """速率（像素/帧）"""
        vx, vy = self.current_velocity
        return math.sqrt(vx**2 + vy**2)

    def is_moving(self, threshold: float = 3.0) -> bool:
        """判断物体是否在运动"""
        return self.speed > threshold

    def movement_direction(self) -> tuple[float, float]:
        """运动方向单位向量"""
        vx, vy = self.current_velocity
        mag = math.sqrt(vx**2 + vy**2)
        if mag < 0.01:
            return (0.0, 0.0)
        return (vx / mag, vy / mag)


class Tracker:
    """目标跟踪器 - 用 IoU 匹配实现简单跟踪"""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15):
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._lost_count: dict[int, int] = {}

    def _compute_iou(
        self,
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """计算两个检测框的 IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        更新跟踪状态
        Args:
            detections: 当前帧检测结果
        Returns:
            赋予 track_id 后的检测结果
        """
        # 对每个已有 track，找最佳匹配检测
        matched_detections = set()
        matched_tracks = set()

        # 先匹配最高 IoU 的
        for track_id, tracked in sorted(self.tracked_objects.items()):
            best_iou = self.iou_threshold
            best_det = None
            best_det_idx = -1

            tracked_bbox = tracked.current_bbox
            if tracked_bbox is None:
                continue

            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                iou = self._compute_iou(tracked_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_det_idx = idx

            if best_det is not None:
                tracked.update(best_det.bbox)
                best_det.track_id = track_id
                matched_detections.add(best_det_idx)
                matched_tracks.add(track_id)
                self._lost_count[track_id] = 0

        # 为匹配上的检测更新未匹配的 track 的 lost 计数
        for track_id in self.tracked_objects:
            if track_id not in matched_tracks:
                self._lost_count[track_id] = self._lost_count.get(track_id, 0) + 1

        # 移除丢失太久的 track
        lost_ids = [
            tid
            for tid, count in self._lost_count.items()
            if count > self.max_lost
        ]
        for tid in lost_ids:
            del self.tracked_objects[tid]
            del self._lost_count[tid]

        # 新检测创建新的 track
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                tracked = TrackedObject(
                    track_id=self.next_id,
                    class_name=det.class_name,
                )
                tracked.update(det.bbox)
                det.track_id = self.next_id
                self.tracked_objects[self.next_id] = tracked
                self._lost_count[self.next_id] = 0
                self.next_id += 1

        return detections

    def get_tracked_object(self, track_id: int) -> Optional[TrackedObject]:
        return self.tracked_objects.get(track_id)

    def get_all_tracked(self) -> list[TrackedObject]:
        return list(self.tracked_objects.values())

    def reset(self):
        """重置跟踪器"""
        self.tracked_objects.clear()
        self._lost_count.clear()
