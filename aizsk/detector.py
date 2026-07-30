"""
detector.py - YOLO 目标检测 + SAM 分割模块

负责：
1. 加载 YOLO 模型进行实时目标检测
2. SAM 点击分割（精确物体轮廓）
3. 检测结果格式化为统一数据结构
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None  # 分割掩膜 (H, W) bool
    track_id: Optional[int] = None  # 跟踪 ID


@dataclass
class DetectionResult:
    """一帧的检测结果"""
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    timestamp: float = 0.0


class YOLODetector:
    """YOLO 目标检测器"""

    # 高中物理常见物体类别
    PHYSICS_CLASSES = {
        0: "person",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports_ball",
        33: "kite",
        34: "baseball_bat",
        35: "baseball_glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis_racket",
        39: "bottle",
        40: "wine_glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        56: "chair",
        57: "couch",
        58: "potted_plant",
        59: "bed",
        60: "dining_table",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell_phone",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy_bear",
        78: "hair_drier",
        79: "toothbrush",
    }

    # 可被视为"支撑面"的物体
    SURFACE_CLASSES = {56, 57, 60}  # chair, couch, dining_table

    def __init__(self, model_size: str = "nano"):
        """
        初始化检测器
        Args:
            model_size: 'nano', 'small', 'medium', 'large', 'xlarge'
        """
        self.model = None
        self.model_name = f"yolo11{model_size[0]}"

    def load(self) -> bool:
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
            logger.info(f"✅ YOLO 模型加载成功: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ YOLO 模型加载失败: {e}")
            return False

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> DetectionResult:
        """
        对一帧进行目标检测
        Args:
            frame: 输入图像 (H, W, 3) BGR
            conf_threshold: 置信度阈值
        Returns:
            DetectionResult 包含所有检测结果
        """
        if self.model is None:
            logger.warning("模型未加载，先调用 load()")
            return DetectionResult(frame=frame)

        results = self.model(frame, conf=conf_threshold, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = results.names[class_id] if results.names else str(class_id)

                # 获取分割掩膜（如果有）
                mask = None
                if results.masks is not None and i < len(results.masks):
                    mask = results.masks.data[i].cpu().numpy()
                    # Resize to frame dimensions
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = mask > 0.5

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        mask=mask,
                    )
                )

        return DetectionResult(frame=frame, detections=detections)

    def detect_by_click(
        self, frame: np.ndarray, click_x: int, click_y: int
    ) -> Optional[Detection]:
        """
        根据用户点击位置检测物体
        1. 先用 YOLO 检测所有物体
        2. 找到包含点击点的物体
        3. 如果有点击分割，用 SAM 获取精确轮廓

        Args:
            frame: 输入图像
            click_x, click_y: 点击坐标

        Returns:
            点击对应的 Detection，或 None
        """
        result = self.detect(frame, conf_threshold=0.3)

        # 找包含点击点的检测框
        best_detection = None
        best_area = float("inf")

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_detection = det

        return best_detection


class SAMSegmenter:
    """SAM 点击分割器（用于精确物体轮廓）"""

    def __init__(self, model_type: str = "sam2.1_hiera_tiny"):
        self.predictor = None
        self.model_type = model_type

    def load(self) -> bool:
        """加载 SAM 模型"""
        try:
            # 尝试加载 SAM2
            try:
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                logger.info("SAM2 加载成功")
            except ImportError:
                # 回退到 SAM
                from segment_anything import sam_model_registry, SamPredictor

                logger.info("SAM 加载成功")
            return True
        except Exception as e:
            logger.warning(f"SAM 不可用 (不影响基础检测): {e}")
            return False

    def segment_at_point(
        self, frame: np.ndarray, x: int, y: int
    ) -> Optional[np.ndarray]:
        """
        在给定点分割物体
        Returns: 二值掩膜 (H, W) 或 None
        """
        # TODO: 当 SAM 可用时实现
        return None
