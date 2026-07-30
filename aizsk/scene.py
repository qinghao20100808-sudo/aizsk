"""
scene.py - 场景理解模块

负责：
1. 判断物体之间的接触关系
2. 识别支撑面（桌面、斜面等）
3. 估计斜面角度
4. 判断物体是否悬挂
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np

from .detector import Detection
from .physics import SceneType

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    """场景分析器"""

    # 被认为可能是支撑面的物体类别
    SUPPORT_CLASSES = {
        "dining_table", "table",
        "desk",
        "chair",
        "couch", "sofa",
        "bed",
        "counter",
        "shelf",
        "floor",  # 虚拟类别
        "ground",
    }

    def __init__(self):
        pass

    def analyze_contact(
        self,
        obj_detection: Detection,
        all_detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> tuple[Optional[Detection], str]:
        """
        判断物体与哪些支撑面接触
        Returns: (支撑面检测, 接触关系描述)
        """
        obj_bottom = obj_detection.bbox[3]  # y2
        obj_center_x = (obj_detection.bbox[0] + obj_detection.bbox[2]) / 2

        best_support = None
        best_distance = float("inf")

        for det in all_detections:
            if det.class_name not in self.SUPPORT_CLASSES:
                continue
            if det is obj_detection:
                continue

            # 检查物体是否在该支撑面的上方附近
            support_top = det.bbox[1]  # y1
            support_left = det.bbox[0]
            support_right = det.bbox[2]

            # 物体中心在支撑面水平范围内
            if support_left <= obj_center_x <= support_right:
                # 物体底部接近支撑面顶部
                distance = abs(obj_bottom - support_top)
                if distance < best_distance and distance < frame_shape[0] * 0.3:
                    best_distance = distance
                    best_support = det

        if best_support:
            return best_support, f"放在{best_support.class_name}上"
        else:
            return None, "没有检测到接触面"

    def estimate_incline_angle(
        self,
        frame: np.ndarray,
        object_detection: Detection,
        surface_detection: Optional[Detection] = None,
    ) -> float:
        """
        估计斜面角度（度）
        通过物体在斜面上的倾斜或用户标注来估计

        Args:
            frame: 当前帧
            object_detection: 物体检测结果
            surface_detection: 支撑面检测结果

        Returns:
            角度（度），0=水平
        """
        # 如果有支撑面，通过支撑面的形状估算角度
        if surface_detection and surface_detection.mask is not None:
            mask = surface_detection.mask
            # 提取支撑面的轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 用最小外接矩形估算角度
                rect = cv2.minAreaRect(contours[0])
                angle = rect[2]  # 返回 -90~0 度
                # 转换为 0~90 度
                angle = abs(angle)
                if angle > 45:
                    angle = 90 - angle
                return angle

        # 如果没有 mask，用 bounding box 宽高比
        x1, y1, x2, y2 = object_detection.bbox
        w = x2 - x1
        h = y2 - y1
        if h > w * 0.5:
            # 物体是竖直的 → 可能是水平面
            return 0.0

        return 0.0

    def check_hanging(
        self,
        obj_detection: Detection,
        all_detections: list[Detection],
    ) -> bool:
        """
        判断物体是否悬挂（下面没有支撑物）
        """
        obj_y2 = obj_detection.bbox[3]
        obj_center_x = (obj_detection.bbox[0] + obj_detection.bbox[2]) / 2
        frame_h = 480  # 估算

        support_below = False
        for det in all_detections:
            if det is obj_detection:
                continue
            # 检测物体在目标下方
            det_y1 = det.bbox[1]
            if det_y1 > obj_y2:
                # 检查水平重叠
                det_x1, det_x2 = det.bbox[0], det.bbox[2]
                if det_x1 <= obj_center_x <= det_x2:
                    support_below = True
                    break

        # 如果物体底部接近画面底部，认为放在地上
        if obj_y2 > frame_h * 0.85:
            support_below = True

        return not support_below

    def detect_scene(
        self,
        obj_detection: Detection,
        all_detections: list[Detection],
        frame: np.ndarray,
    ) -> tuple[SceneType, float, str]:
        """
        综合判断场景类型

        Returns:
            (场景类型, 斜面角度, 场景描述)
        """
        # 检查是否悬挂
        is_hanging = self.check_hanging(obj_detection, all_detections)

        if is_hanging:
            return SceneType.PENDULUM, 0.0, "物体处于悬挂状态"

        # 检查接触面
        contact_surface, contact_desc = self.analyze_contact(
            obj_detection, all_detections, frame.shape
        )

        # 估算斜面角度
        incline_angle = self.estimate_incline_angle(
            frame, obj_detection, contact_surface
        )

        if incline_angle > 5:
            return (
                SceneType.INCLINE,
                incline_angle,
                f"物体在倾斜面上，角度约 {incline_angle:.0f}°",
            )
        else:
            return SceneType.FLAT_SURFACE, 0.0, contact_desc or "物体在水平面上"
