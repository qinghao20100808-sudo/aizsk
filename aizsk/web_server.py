"""
web_server.py - Flask 网页服务器

负责：
1. 摄像头视频流（MJPEG）
2. 处理用户点击事件
3. 实时显示受力分析结果
"""

import io
import json
import logging
import math
import os
import threading
import time
from queue import Queue
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from .detector import Detection, YOLODetector
from .physics import PhysicsEngine, SceneType
from .scene import SceneAnalyzer
from .tracker import Tracker, TrackedObject
from .visualizer import ForceVisualizer
from .simulation import SimulationScene, SimObject

logger = logging.getLogger(__name__)


class CameraStream:
    """摄像头视频流 - 支持真实摄像头和 Windows 共享文件两种模式"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        use_windows_cam: bool = False,
        windows_cam_path: str = "/mnt/c/Users/34931/AppData/Local/Temp/live_cam_hermes.jpg",
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.use_windows_cam = use_windows_cam
        self.windows_cam_path = windows_cam_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.selected_track_id: Optional[int] = None
        self.selected_scene: SceneType = SceneType.UNKNOWN
        self.incline_angle: float = 0.0

        # 子系统
        self.detector = YOLODetector()
        self.tracker = Tracker()
        self.physics = PhysicsEngine()
        self.scene_analyzer = SceneAnalyzer()
        self.visualizer = ForceVisualizer(scale=0.8)

        self.model_loaded = False

    def start(self) -> bool:
        """启动摄像头"""
        if self.use_windows_cam:
            # Windows 共享文件模式 - 从 JPEG 文件读取
            if not os.path.exists(self.windows_cam_path):
                logger.error(f"Windows 摄像头文件不存在: {self.windows_cam_path}")
                logger.info("请先在 Windows 上运行 ffmpeg 摄像头采集")
                return False
            logger.info(f"📷 使用 Windows 摄像头 (文件模式): {self.windows_cam_path}")
            self.model_loaded = self.detector.load()
            self.running = True
            self._thread = threading.Thread(target=self._run_file_mode, daemon=True)
            self._thread.start()
            return True
        else:
            # 标准 OpenCV 摄像头模式
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self.model_loaded = self.detector.load()
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("✅ 摄像头已启动")
            return True

    def _read_windows_frame(self) -> Optional[np.ndarray]:
        """从 Windows 共享文件读取一帧"""
        try:
            img = cv2.imread(self.windows_cam_path)
            if img is not None:
                return cv2.resize(img, (self.width, self.height))
            return None
        except Exception:
            return None

    def _run_file_mode(self):
        """从 Windows 共享文件读取帧的运行循环"""
        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        while self.running:
            frame = self._read_windows_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # 计算 FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # YOLO 检测 + 跟踪
            if self.model_loaded:
                result = self.detector.detect(frame, conf_threshold=0.4)
                result.detections = self.tracker.update(result.detections)

                # 受力分析
                analysis_result = None
                selected_tracked = None

                if self.selected_track_id is not None:
                    tracked = self.tracker.get_tracked_object(
                        self.selected_track_id
                    )
                    if tracked:
                        selected_tracked = tracked
                        selected_det = None
                        for det in result.detections:
                            if det.track_id == self.selected_track_id:
                                selected_det = det
                                break

                        if self.selected_scene == SceneType.UNKNOWN:
                            st, angle, desc = self.scene_analyzer.detect_scene(
                                selected_det or Detection(
                                    class_id=0, class_name="object",
                                    confidence=0,
                                    bbox=tracked.current_bbox or (0, 0, 10, 10),
                                ),
                                result.detections,
                                frame,
                            )
                            self.selected_scene = st
                            self.incline_angle = angle

                        analysis_result = self.physics.analyze(
                            tracked,
                            scene_type=self.selected_scene,
                            incline_angle=self.incline_angle,
                            mass=1.0,
                        )

                # 可视化
                for det in result.detections:
                    is_selected = det.track_id == self.selected_track_id
                    self.visualizer.draw_detection(frame, det, is_selected)

                if analysis_result and selected_tracked and selected_tracked.current_center:
                    cx, cy = int(selected_tracked.current_center[0]), int(
                        selected_tracked.current_center[1]
                    )
                    self.visualizer.draw_force_analysis(
                        frame, analysis_result, (cx, cy)
                    )
                    self.visualizer.draw_velocity_info(frame, selected_tracked)
                    self.visualizer.draw_trajectory(frame, selected_tracked)

            status = "Windows 摄像头模式 - 点击物体进行受力分析"
            if self.selected_track_id is not None:
                status = f"已选中 #{self.selected_track_id} | 场景: {self.selected_scene.value}"
            self.visualizer.draw_status(frame, status)
            self.visualizer.draw_fps(frame, fps)
            self.visualizer.draw_legend(frame)

            with self.lock:
                self.current_frame = frame

            # 控制帧率 ~20 FPS
            time.sleep(0.05)

    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("摄像头已停止")

    def _run(self):
        """运行循环"""
        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            frame_count += 1

            # 计算 FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # YOLO 检测 + 跟踪
            if self.model_loaded:
                result = self.detector.detect(frame, conf_threshold=0.4)
                result.detections = self.tracker.update(result.detections)

                # 如果选中了物体，做受力分析
                analysis_result = None
                selected_tracked = None

                if self.selected_track_id is not None:
                    tracked = self.tracker.get_tracked_object(
                        self.selected_track_id
                    )
                    if tracked:
                        selected_tracked = tracked
                        selected_det = None
                        for det in result.detections:
                            if det.track_id == self.selected_track_id:
                                selected_det = det
                                break

                        # 场景分析
                        if self.selected_scene == SceneType.UNKNOWN:
                            st, angle, desc = self.scene_analyzer.detect_scene(
                                selected_det or Detection(
                                    class_id=0,
                                    class_name="object",
                                    confidence=0,
                                    bbox=tracked.current_bbox or (0, 0, 10, 10),
                                ),
                                result.detections,
                                frame,
                            )
                            self.selected_scene = st
                            self.incline_angle = angle

                        # 物理推理
                        analysis_result = self.physics.analyze(
                            tracked,
                            scene_type=self.selected_scene,
                            incline_angle=self.incline_angle,
                            mass=1.0,
                        )

                # 可视化
                for det in result.detections:
                    is_selected = (
                        det.track_id == self.selected_track_id
                    )
                    self.visualizer.draw_detection(frame, det, is_selected)

                if analysis_result and selected_tracked and selected_tracked.current_center:
                    cx, cy = int(selected_tracked.current_center[0]), int(
                        selected_tracked.current_center[1]
                    )
                    self.visualizer.draw_force_analysis(
                        frame, analysis_result, (cx, cy)
                    )
                    self.visualizer.draw_velocity_info(frame, selected_tracked)
                    self.visualizer.draw_trajectory(frame, selected_tracked)

            else:
                # 无模型模式：简单地在画面中心画十字
                cv2.putText(
                    frame,
                    "等待模型加载...",
                    (self.width // 2 - 100, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # 状态文字
            status = "点击物体进行受力分析"
            if self.selected_track_id is not None:
                status = (
                    f"已选中 #{self.selected_track_id} | "
                    f"场景: {self.selected_scene.value}"
                )
            self.visualizer.draw_status(frame, status)
            self.visualizer.draw_fps(frame, fps)
            self.visualizer.draw_legend(frame)

            with self.lock:
                self.current_frame = frame

    def get_frame(self) -> Optional[bytes]:
        """获取当前帧的 JPEG 编码"""
        with self.lock:
            if self.current_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.current_frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def select_object(self, x: int, y: int) -> dict:
        """选择点击位置的物体"""
        with self.lock:
            if self.current_frame is None:
                return {"success": False, "message": "没有画面"}

        # 对当前帧做检测
        frame = self.current_frame.copy()
        result = self.detector.detect(frame, conf_threshold=0.3)
        result.detections = self.tracker.update(result.detections)

        # 找点击位置的物体
        best_det = None
        best_area = float("inf")
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_det = det

        if best_det is not None:
            self.selected_track_id = best_det.track_id
            self.selected_scene = SceneType.UNKNOWN
            self.incline_angle = 0.0

            tracked = self.tracker.get_tracked_object(self.selected_track_id)
            return {
                "success": True,
                "track_id": best_det.track_id,
                "class_name": best_det.class_name,
                "bbox": list(best_det.bbox),
                "speed": tracked.speed if tracked else 0,
            }
        else:
            # 没点到物体，取消选择
            self.selected_track_id = None
            return {"success": False, "message": "没有检测到物体"}

    def set_scene(self, scene_type: str, incline_angle: float = 0):
        """手动设置场景类型"""
        try:
            self.selected_scene = SceneType(scene_type)
            self.incline_angle = incline_angle
            return {"success": True}
        except ValueError:
            return {"success": False, "message": f"无效场景类型: {scene_type}"}


class SimulationStream:
    """模拟视频流（无需摄像头）"""

    def __init__(self, scene_id: str = "incline", width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.running = False
        self.lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.selected_track_id: Optional[int] = None
        self.selected_scene: SceneType = SceneType.UNKNOWN
        self.incline_angle: float = 0.0

        # 子系统
        self.detector = None  # 模拟模式不需要 YOLO
        self.tracker = Tracker()
        self.physics = PhysicsEngine()
        self.scene_analyzer = SceneAnalyzer()
        self.visualizer = ForceVisualizer(scale=0.8)
        self.simulation = SimulationScene(scene_id)

        self.model_loaded = True
        self._auto_selected = False

    def start(self) -> bool:
        """启动模拟"""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"✅ 模拟场景已启动: {self.simulation.scene_id}")
        return True

    def stop(self):
        """停止模拟"""
        self.running = False
        logger.info("模拟已停止")

    def _run(self):
        """运行循环"""
        frame_count = 0
        fps_timer = time.time()
        fps = 0.0

        while self.running:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (20, 20, 40)  # 深色背景

            frame_count += 1

            # 计算 FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_timer = time.time()

            # 模拟场景更新
            sim_detections = self.simulation.step()

            # 把模拟物体转为 Detection
            detections = []
            for sim_obj in self.simulation.objects:
                det = Detection(
                    class_id=0,
                    class_name=sim_obj.name,
                    confidence=0.95,
                    bbox=sim_obj.bbox,
                )
                detections.append(det)

            # 跟踪
            detections = self.tracker.update(detections)

            # 演示模式下自动选中第一个物体
            if not self._auto_selected and detections:
                self.selected_track_id = detections[0].track_id
                self._auto_selected = True

            # 渲染场景
            self.simulation.render(frame, sim_detections)

            # 如果选中了物体，做受力分析
            analysis_result = None
            selected_tracked = None

            if self.selected_track_id is not None:
                tracked = self.tracker.get_tracked_object(
                    self.selected_track_id
                )
                if tracked:
                    selected_tracked = tracked
                    # 从模拟数据获取速度和加速度
                    sim_obj = self.simulation.objects[0] if self.simulation.objects else None
                    if sim_obj:
                        tracked.velocities.append((sim_obj.vx, sim_obj.vy))
                        if len(tracked.velocities) > tracked.max_history:
                            tracked.velocities.pop(0)

                    # 场景类型（从模拟场景获取）
                    scene_type = self.simulation.scene_type
                    incline_angle = self.simulation.incline_angle

                    if self.selected_scene != SceneType.UNKNOWN:
                        scene_type = self.selected_scene
                        incline_angle = self.incline_angle

                    # 物理推理
                    analysis_result = self.physics.analyze(
                        tracked,
                        scene_type=scene_type,
                        incline_angle=incline_angle,
                        mass=1.0,
                    )

            # 画检测框和受力分析
            for det in detections:
                is_selected = det.track_id == self.selected_track_id
                self.visualizer.draw_detection(frame, det, is_selected)

            if analysis_result and selected_tracked and selected_tracked.current_center:
                cx, cy = int(selected_tracked.current_center[0]), int(
                    selected_tracked.current_center[1]
                )
                self.visualizer.draw_force_analysis(
                    frame, analysis_result, (cx, cy)
                )
                self.visualizer.draw_velocity_info(frame, selected_tracked)
                self.visualizer.draw_trajectory(frame, selected_tracked)

            # 状态文字
            scene_name = self.selected_scene.value if self.selected_scene != SceneType.UNKNOWN else self.simulation.scene_type.value
            status = f"演示模式 | 场景: {self.simulation.SCENES.get(self.simulation.scene_id, {}).get('name', '')}"
            if self.selected_track_id is not None:
                status += f" | 已选中物体"
            self.visualizer.draw_status(frame, status)
            self.visualizer.draw_fps(frame, fps)
            self.visualizer.draw_legend(frame)

            with self.lock:
                self.current_frame = frame

            # 控制帧率
            time.sleep(0.05)  # ~20 FPS

    def get_frame(self) -> Optional[bytes]:
        """获取当前帧的 JPEG 编码"""
        with self.lock:
            if self.current_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.current_frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def select_object(self, x: int, y: int) -> dict:
        """选择点击位置的物体（模拟版）"""
        with self.lock:
            if self.current_frame is None:
                return {"success": False, "message": "没有画面"}

            # 找点击位置的物体（在锁里操作，防止被模拟线程更新）
            best_det = None
            best_area = float("inf")
            for sim_obj in self.simulation.objects:
                x1, y1, x2, y2 = sim_obj.bbox
                logger.debug(f"Object '{sim_obj.name}' at ({x1},{y1})-({x2},{y2})")
                if x1 <= x <= x2 and y1 <= y <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if area < best_area:
                        best_area = area
                        det = Detection(
                            class_id=0, class_name=sim_obj.name,
                            confidence=0.95, bbox=sim_obj.bbox,
                        )
                        best_det = det

        # 锁释放后处理
        if best_det is not None:
            # 用 tracker 生成 track_id
            dets = self.tracker.update([best_det])
            if dets:
                best_det = dets[0]

            self.selected_track_id = best_det.track_id
            return {
                "success": True,
                "track_id": best_det.track_id,
                "class_name": best_det.class_name,
                "bbox": list(best_det.bbox),
                "speed": 0,
            }
        else:
            # 如果没点到，找最近的物体
            nearest = None
            nearest_dist = float("inf")
            with self.lock:
                for sim_obj in self.simulation.objects:
                    cx = sim_obj.x
                    cy = sim_obj.y
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = sim_obj

            if nearest and nearest_dist < 200:  # 200px 内就自动选中
                det = Detection(
                    class_id=0, class_name=nearest.name,
                    confidence=0.95, bbox=nearest.bbox,
                )
                dets = self.tracker.update([det])
                if dets:
                    self.selected_track_id = dets[0].track_id
                    return {
                        "success": True,
                        "track_id": dets[0].track_id,
                        "class_name": dets[0].class_name,
                        "bbox": list(dets[0].bbox),
                        "speed": 0,
                        "note": "自动吸附到最近物体",
                    }

            self.selected_track_id = None
            # 打印调试信息
            if self.simulation.objects:
                obj = self.simulation.objects[0]
                logger.info(f"Click miss: click=({x},{y}), object bbox={obj.bbox}, center=({obj.x:.0f},{obj.y:.0f})")
            return {"success": False, "message": "没有点击到物体"}

    def set_scene(self, scene_type: str, incline_angle: float = 0):
        """手动设置场景类型"""
        try:
            self.selected_scene = SceneType(scene_type)
            self.incline_angle = incline_angle
            return {"success": True}
        except ValueError:
            return {"success": False, "message": f"无效场景类型: {scene_type}"}


class WebServer:
    """Flask Web 服务器 - 支持浏览器摄像头模式"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.camera: Optional[CameraStream] = None

        # 浏览器摄像头模式的全局状态
        self._detector = YOLODetector()
        self._detector_loaded = False
        self._tracker = Tracker()
        self._physics = PhysicsEngine()
        self._scene_analyzer = SceneAnalyzer()
        self._selected_track_id: Optional[int] = None
        self._selected_scene: SceneType = SceneType.UNKNOWN
        self._incline_angle: float = 0.0

        self.app = Flask(__name__, static_folder="static")
        self._setup_routes()

    def _setup_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return send_file("static/index.html")

        @app.route("/video_feed")
        def video_feed():
            """MJPEG 视频流"""
            def generate():
                while True:
                    frame = self.camera.get_frame() if self.camera else None
                    if frame:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                        )
                    else:
                        time.sleep(0.05)
            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/click", methods=["POST"])
        def handle_click():
            """处理点击事件"""
            data = request.get_json()
            x, y = data.get("x", 0), data.get("y", 0)
            result = self.camera.select_object(x, y) if self.camera else {
                "success": False,
                "message": "摄像头未启动",
            }
            return jsonify(result)

        @app.route("/api/scene", methods=["POST"])
        def set_scene():
            """设置场景类型"""
            data = request.get_json()
            scene_type = data.get("scene", "unknown")
            incline_angle = data.get("angle", 0.0)
            result = self.camera.set_scene(scene_type, incline_angle) if self.camera else {
                "success": False,
                "message": "摄像头未启动",
            }
            return jsonify(result)

        @app.route("/api/status")
        def get_status():
            """获取状态"""
            if self.camera is None:
                return jsonify({"running": False})
            return jsonify({
                "running": True,
                "selected": self.camera.selected_track_id,
                "scene": self.camera.selected_scene.value if self.camera.selected_scene else "unknown",
                "model_loaded": self.camera.model_loaded,
            })

        @app.route("/api/detect/start", methods=["POST"])
        def detect_start():
            """初始化检测器（浏览器摄像头模式）"""
            if not self._detector_loaded:
                self._detector_loaded = self._detector.load()
                # 重置状态
                self._tracker.reset()
                self._selected_track_id = None
                self._selected_scene = SceneType.UNKNOWN
            return jsonify({"success": self._detector_loaded, "model_loaded": self._detector_loaded})

        @app.route("/api/detect", methods=["POST"])
        def detect_frame():
            """处理前端发来的摄像头帧
            接收：multipart/form-data 包含 image (JPEG) + 可选参数
            返回：JSON 包含检测结果和受力分析
            """
            if not self._detector_loaded:
                return jsonify({"success": False, "error": "模型未加载"})

            # 读取上传的图片
            if "image" not in request.files:
                return jsonify({"success": False, "error": "缺少 image 字段"})

            file = request.files["image"]
            img_bytes = file.read()
            if not img_bytes:
                return jsonify({"success": False, "error": "空图片"})

            # 解码 JPEG
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"success": False, "error": "图片解码失败"})

            # 获取前端参数
            click_x = request.form.get("click_x", type=int)
            click_y = request.form.get("click_y", type=int)
            scene_type_str = request.form.get("scene", "")
            incline_angle = request.form.get("angle", type=float, default=0.0)
            mass_val = request.form.get("mass", type=float, default=0.0)
            mass_unknown = mass_val <= 0

            # 场景类型
            if scene_type_str:
                try:
                    self._selected_scene = SceneType(scene_type_str)
                    self._incline_angle = incline_angle
                except ValueError:
                    pass

            # YOLO 检测（低阈值识别更多物体，包括小物体）
            result = self._detector.detect(frame, conf_threshold=0.15)
            result.detections = self._tracker.update(result.detections)

            # 只保留能做受力分析的小物件（过滤家具/人/大件）
            PHYSICS_OBJECTS = {
                # 容器类
                "bottle", "wine_glass", "cup", "bowl", "vase",
                # 餐具/工具类
                "fork", "knife", "spoon", "scissors", "book", "cell_phone",
                "remote", "mouse", "keyboard", "clock",
                # 球类/运动
                "sports_ball", "frisbee", "kite", "baseball_bat",
                "baseball_glove", "tennis_racket", "skateboard",
                # 食物类
                "apple", "banana", "orange", "sandwich", "broccoli", "carrot",
                "hot_dog", "pizza", "donut", "cake",
                # 小物件
                "teddy_bear", "toothbrush", "hair_drier", "umbrella",
                "backpack", "handbag", "suitcase", "tie",
            }
            result.detections = [
                d for d in result.detections
                if d.class_name in PHYSICS_OBJECTS
            ]

            # 点击选物体
            if click_x is not None and click_y is not None:
                best_det = None
                best_area = float("inf")
                for det in result.detections:
                    x1, y1, x2, y2 = det.bbox
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < best_area:
                            best_area = area
                            best_det = det
                if best_det:
                    self._selected_track_id = best_det.track_id
                else:
                    self._selected_track_id = None

            # 受力分析
            analysis_json = None
            if self._selected_track_id is not None:
                tracked = self._tracker.get_tracked_object(self._selected_track_id)
                if tracked:
                    # 用模拟的物理推理（基于场景类型）
                    analysis = self._physics.analyze(
                        tracked,
                        scene_type=self._selected_scene,
                        incline_angle=self._incline_angle,
                        mass=mass_val if not mass_unknown else 1.0,
                        mass_unknown=mass_unknown,
                    )
                    # 序列化为 JSON
                    forces = []
                    for f in analysis.forces:
                        forces.append({
                            "type": f.type.value,
                            "magnitude": round(f.magnitude, 1),
                            "angle": round(f.angle, 1),
                            "label": f.label,
                            "color": list(f.color),
                        })
                    net = analysis.net_force
                    analysis_json = {
                        "scene": analysis.scene_type.value,
                        "note": analysis.note,
                        "forces": forces,
                        "net_force": {
                            "magnitude": round(net.magnitude, 1),
                            "angle": round(net.angle, 1),
                            "label": net.label,
                        },
                    }

            # 序列化检测结果
            detections_json = []
            for det in result.detections:
                detections_json.append({
                    "track_id": det.track_id,
                    "class_name": det.class_name,
                    "confidence": round(det.confidence, 2),
                    "bbox": list(det.bbox),
                    "selected": det.track_id == self._selected_track_id,
                })

            # 选中物体的速度信息
            velocity_info = None
            if self._selected_track_id is not None:
                tracked = self._tracker.get_tracked_object(self._selected_track_id)
                if tracked:
                    velocity_info = {
                        "vx": round(tracked.current_velocity[0], 2),
                        "vy": round(tracked.current_velocity[1], 2),
                        "speed": round(tracked.speed, 2),
                        "is_moving": tracked.is_moving(),
                    }

            return jsonify({
                "success": True,
                "detections": detections_json,
                "selected_id": self._selected_track_id,
                "analysis": analysis_json,
                "velocity": velocity_info,
                "scene": self._selected_scene.value if self._selected_scene else "unknown",
            })

    def start(self, camera: Optional[CameraStream] = None):
        """启动服务器"""
        self.camera = camera

        # 如果没有传 camera，启用浏览器摄像头模式并预加载 YOLO
        if camera is None:
            logger.info("📷 浏览器摄像头模式 - YOLO 模型预加载中...")
            self._detector_loaded = self._detector.load()
            if self._detector_loaded:
                logger.info("✅ YOLO 模型已加载，等待前端连接")
            else:
                logger.warning("⚠️ YOLO 模型加载失败")

        logger.info(f"🌐 Web 服务器启动: http://{self.host}:{self.port}")
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
