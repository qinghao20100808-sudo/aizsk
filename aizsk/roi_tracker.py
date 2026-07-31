"""
roi_tracker.py - 框选 ROI 跟踪器（大疆追车式）

核心思路：
1. 用户在画面上框选目标 → 保存 ROI 外观模板 + 提取特征点
2. 每帧位移估计采用"模板匹配为主、LK 光流为辅"：
   - 模板匹配（TM_CCOEFF_NORMED）峰值位置即真实位移，方向不会受
     背景角点/光流误导（用户反馈"往上走识别成向左"的根因修复）
   - 光流仅在模板分数一般时兜底，或与模板一致时融合
3. 目标被遮挡（模板低分且光流不可靠）→ 进入"预测模式"：
   用平滑速度外推目标位置（匀速运动模型），持续若干帧
4. 超过最大丢失帧数 → lost；目标重现 → 预测位置邻域模板搜索恢复

优点：纯 OpenCV 自带函数，零额外依赖；640 分辨率下 CPU 单帧 ~5-10ms，
比 YOLO 快两个数量级，比 opencv-contrib 的 KCF/CSRT 更快且无需换包。
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROITracker:
    """框选 ROI 跟踪器（模板匹配主位移 + LK 光流辅助），带遮挡运动预测"""

    MAX_LOST = 20            # 连续丢失超过该帧数 → lost（约 4 秒 @200ms）
    MAX_POINTS = 80          # ROI 内最大特征点数
    MIN_POINTS = 8           # 跟踪点少于该数则重采样
    INLIER_RATIO = 0.35      # 位移一致性 inlier 比例低于该值 → 光流不可靠
    INLIER_TOL = 6.0         # 与中值位移偏差超过该值（px）视为离群点
    RESAMPLE_EVERY = 15      # 每 N 帧重采样一次特征点
    MIN_BBOX_SIZE = 8        # 框最小边长
    SEARCH_RADIUS = 32       # 模板匹配搜索窗半径（覆盖 ~160px/s @200ms）
    TRACK_MIN_SCORE = 0.30   # 模板分数 ≥ 该值 → 模板做主位移
    TM_ACCEPT_LOW = 0.22     # 模板分数 ≥ 该值且光流可靠 → 光流位移
    LOW_SCORE_FRAMES = 2     # 连续低分帧数达到该值才判定遮挡（防抖动）
    TEMPLATE_UPDATE = 0.10   # 模板指数更新权重（高分会话中缓慢跟随外观渐变）
    TEMPLATE_UPDATE_MIN = 0.45  # 只有分数 ≥ 该值才更新模板（防背景污染模板）
    RECOVER_THRESH = 0.40    # 遮挡恢复：预测位置邻域模板匹配分数阈值
    RECOVER_MARGIN = 80      # 恢复搜索区域相对预测框的外扩像素（遮挡期间目标可能被带着移动）
    # 防瞬移（模板匹配在目标移出搜索窗时会返回背景峰值 → 跳变）
    MAX_STEP = 32            # 单帧最大可信位移（200ms 间隔 ≈ 160px/s，超出视为误匹配）
    STEP_ANGLE_DEG = 70      # 位移与历史速度夹角上限（超出视为误匹配）
    JUMP_FRAMES = 2          # 连续异常位移帧数达到该值 → 进入遮挡预测
    VEL_SMOOTH = 0.5         # 速度 EMA 系数（越小越平滑，预测越稳）
    ACC_SMOOTH = 0.35        # 加速度 EMA 系数（用于遮挡期匀加速外推）
    # 尺度自适应（3D→2D 投影：物体沿光轴移动时画面大小变化，固定模板会跟丢）
    # 档位细化到 ~6%：分辨率高 → 平滑滞后小 → bbox≈目标尺寸 → patch 重建
    # 几乎不含背景（粗档位时模板混入背景会导致缩小时尺度估计死锁）
    SCALES = tuple(round(0.5 + 0.0625 * i, 4) for i in range(25))  # 0.5~2.0
    MAX_SCALE_CHANGE = 0.4   # 单帧允许的最大尺度变化（快速前后移动；超出视为误匹配）
    SCALE_SMOOTH = 0.6       # 尺度 EMA 平滑系数（细粒度下可更快响应）

    def __init__(self):
        self.roi: Optional[tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)
        self.prev_gray: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None  # (N,1,2) float32
        self.last_bbox: Optional[tuple[int, int, int, int]] = None
        self.vel: tuple[float, float] = (0.0, 0.0)  # 平滑速度 px/帧
        self.acc: tuple[float, float] = (0.0, 0.0)  # 平滑加速度 px/帧²（遮挡外推用）
        self.lost_count: int = 0
        self.state: str = "idle"   # idle / tracking / predicting / lost
        self._frames_since_resample = 0
        self._ref_template: Optional[np.ndarray] = None  # 参考外观（初始清晰细节，匹配/缩放源）
        self._low_score_count: int = 0
        self._jump_count: int = 0  # 连续异常位移计数（防瞬移）
        self._smooth_scale: float = 1.0  # 平滑尺度（bbox 尺寸跟随，随远近连续缩放）

    # ---------- 状态 ----------

    def is_active(self) -> bool:
        return self.roi is not None

    def clear(self):
        self.roi = None
        self.prev_gray = None
        self.points = None
        self.last_bbox = None
        self.vel = (0.0, 0.0)
        self.acc = (0.0, 0.0)
        self.lost_count = 0
        self.state = "idle"
        self._ref_template = None
        self._low_score_count = 0
        self._jump_count = 0
        self._smooth_scale = 1.0

    # ---------- 初始化 ----------

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """框选初始化。bbox 为 (x1,y1,x2,y2)"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < self.MIN_BBOX_SIZE or y2 - y1 < self.MIN_BBOX_SIZE:
            logger.warning(f"ROI 太小: {x2-x1}x{y2-y1}")
            return False

        self.roi = (x1, y1, x2, y2)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._ref_template = self.prev_gray[y1:y2, x1:x2].copy()  # 初始清晰外观
        self._low_score_count = 0
        self._jump_count = 0
        self._smooth_scale = 1.0
        self.points = self._sample_points()
        self.last_bbox = self.roi
        self.vel = (0.0, 0.0)
        self.acc = (0.0, 0.0)
        self.lost_count = 0
        self.state = "tracking"
        self._frames_since_resample = 0
        logger.info(f"ROI 跟踪初始化: {self.roi}, 特征点 {0 if self.points is None else len(self.points)} 个")
        return True

    # ---------- 特征点 ----------

    def _sample_points(self) -> Optional[np.ndarray]:
        """在 ROI 内提取 Shi-Tomasi 角点"""
        if self.prev_gray is None or self.roi is None:
            return None
        x1, y1, x2, y2 = self.roi
        sub = self.prev_gray[y1:y2, x1:x2]
        if sub.size == 0:
            return None
        pts = cv2.goodFeaturesToTrack(
            sub, maxCorners=self.MAX_POINTS, qualityLevel=0.01,
            minDistance=8, blockSize=7,
        )
        if pts is None:
            return None
        pts = pts.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
        return pts.reshape(-1, 1, 2).astype(np.float32)

    # ---------- 每帧更新 ----------

    def update(self, frame: np.ndarray) -> tuple[bool, Optional[tuple[int, int, int, int]]]:
        """跟踪一帧。
        Returns:
            (ok, bbox): ok=True 真实跟踪到；ok=False 遮挡/丢失，
                        bbox 为运动外推的预测框（可能为 None）
        """
        if self.roi is None:
            return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 非 tracking 状态（遮挡预测中/已丢失）：先尝试在预测位置邻域
        # 用模板搜索重新捕获目标；找不到则继续运动外推
        if self.state != "tracking":
            if self._try_recover(gray):
                return True, self.last_bbox
            return self._mark_lost(gray)

        # ===== 位移估计：模板匹配为主，LK 光流为辅 =====
        # 模板匹配峰值位置就是真实位移（方向不会受背景角点/光流误导），
        # 光流用于：模板分数一般时兜底、模板完全失效时确认遮挡。
        tm = self._template_displacement(gray)
        lk = self._lk_displacement(gray)

        tm_dx, tm_dy, tm_score, tm_scale = (
            tm if tm is not None else (None, None, None, None)
        )
        lk_dx, lk_dy, lk_ratio, lk_n, lk_inliers = (
            lk if lk is not None else (None, None, None, 0, None)
        )
        lk_ok = lk is not None and lk_n >= 4 and lk_ratio >= self.INLIER_RATIO
        scale_use = 1.0  # 尺度应用值（各分支覆盖；非模板主分支不缩放）

        if tm_score is not None and tm_score >= self.TRACK_MIN_SCORE:
            # 模板可靠 → 主位移；光流一致时融合（方向仲裁：不一致则以模板为准）
            assert tm_dx is not None and tm_dy is not None and tm_scale is not None
            # 尺度钳制：单帧尺度变化（相对上一帧平滑尺度）超限视为误匹配
            # （注意 tm_scale 是相对参考外观的绝对尺度，需与 _smooth_scale 比较）
            if abs(tm_scale - self._smooth_scale) > self.MAX_SCALE_CHANGE:
                tm_scale = self._smooth_scale  # 保持上帧尺度，不采信突变
            # 尺度自适应平滑：偏差大时快速对齐（目标前后快速移动），
            # 偏差小时平滑（防抖动/档位跳变）
            diff = abs(tm_scale - 1.0)
            alpha = 0.6 if diff < 0.06 else min(0.95, 0.6 + (diff - 0.06) * 3)
            self._smooth_scale = (
                alpha * tm_scale + (1 - alpha) * self._smooth_scale
            )
            scale_use = self._smooth_scale
            if lk_ok and self._angle_ok((tm_dx, tm_dy), (lk_dx, lk_dy)):
                assert lk_dx is not None and lk_dy is not None
                # 融合前检查光流量级：快速移动时光流跟不上（位移≈0），
                # 融合会把正确的大位移拉低 → 此时纯用模板位移
                tm_mag = math.hypot(tm_dx, tm_dy)
                lk_mag = math.hypot(lk_dx, lk_dy)
                if tm_mag > 8 and lk_mag < tm_mag * 0.4:
                    dx, dy = tm_dx, tm_dy  # 光流滞后，模板为准
                else:
                    dx = 0.6 * tm_dx + 0.4 * lk_dx
                    dy = 0.6 * tm_dy + 0.4 * lk_dy
            else:
                dx, dy = tm_dx, tm_dy
            self._low_score_count = 0
            update_template = True
        elif lk_ok and (tm_score is None or tm_score >= self.TM_ACCEPT_LOW):
            # 模板分数一般/不可用，但光流一致 → 用光流位移（尺度变化场景）
            assert lk_dx is not None and lk_dy is not None
            dx, dy = lk_dx, lk_dy
            scale_use = self._smooth_scale  # 尺寸保持上帧（目标可原地缩放）
            self._low_score_count = 0
            update_template = False
        else:
            # 模板和光流都不可靠 → 疑似遮挡。
            # 过渡帧：先尝试"原地多尺度恢复"——目标可能只是尺寸突变
            # （快速前后移动，模板匹配不上但物体仍在），恢复会重新对齐尺度；
            # 找不到则保持位置/速度不变（不污染模型——否则 vel 被拉低、
            # acc 变负，遮挡外推会越来越慢甚至反向）
            self._low_score_count += 1
            if self._low_score_count >= self.LOW_SCORE_FRAMES:
                return self._mark_lost(gray)
            if self._try_recover(gray):
                return True, self.last_bbox
            return True, self.roi

        # ===== 防瞬移：位移钳制 + 方向一致性检查 =====
        # 模板匹配在目标移出搜索窗时会返回背景峰值（分数可能恰好过阈值），
        # 表现为 bbox 突然跳到远处。连续异常 → 进入遮挡预测（速度模型接管）。
        disp_mag = math.hypot(dx, dy)
        vel_mag = math.hypot(self.vel[0], self.vel[1])
        jumped = False
        if disp_mag > self.MAX_STEP:
            # 方向可信（与历史速度一致，或速度尚未建立——初始快速移动时
            # 模板峰值方向可信）→ 朝峰值方向追赶 MAX_STEP
            if (vel_mag > 3 and self._angle_ok((dx, dy), self.vel, self.STEP_ANGLE_DEG)) \
                    or vel_mag < 2:
                k = self.MAX_STEP / disp_mag
                dx, dy = dx * k, dy * k
                self._jump_count = 0
                update_template = False  # 追赶帧不更新模板（位置可能仍不准）
            else:
                jumped = True
        elif (disp_mag > 6 and vel_mag > 3
                and not self._angle_ok((dx, dy), self.vel, self.STEP_ANGLE_DEG)):
            if disp_mag < 10:
                # 小幅反向位移：模板峰值抖动/目标静止 → 不判跳变，
                # 位置微调即可（避免 vel 被抖动方向污染后连锁误判）
                dx, dy = self.vel[0] * 0.5, self.vel[1] * 0.5
                self._jump_count = 0
            else:
                jumped = True
        else:
            self._jump_count = 0

        if jumped:
            self._jump_count += 1
            if self._jump_count >= self.JUMP_FRAMES:
                # 连续异常位移 → 目标大概率遮挡/丢失，速度模型接管
                return self._mark_lost(gray)
            # 单帧异常：不跳，用历史速度小幅补位（防瞬移）；
            # 尺寸保持上帧（位移异常与尺度无关，目标可能正在原地缩放）
            dx = self.vel[0] * 0.5
            dy = self.vel[1] * 0.5
            scale_use = self._smooth_scale
            update_template = False

        # 更新 bbox（位移平移 + 平滑尺度缩放；参考外观尺寸 × 平滑尺度）
        dx_m, dy_m = int(round(dx)), int(round(dy))
        x1, y1, x2, y2 = self.roi
        th, tw = self._ref_template.shape if self._ref_template is not None else (y2 - y1, x2 - x1)
        new_w = max(self.MIN_BBOX_SIZE, int(round(tw * scale_use)))
        new_h = max(self.MIN_BBOX_SIZE, int(round(th * scale_use)))
        new_bbox = (x1 + dx_m, y1 + dy_m, x1 + dx_m + new_w, y1 + dy_m + new_h)

        # bbox 完全出画面 → 丢失
        h, w = frame.shape[:2]
        if new_bbox[2] <= 0 or new_bbox[3] <= 0 or new_bbox[0] >= w or new_bbox[1] >= h:
            return self._mark_lost(gray)

        # 平滑速度 + 加速度（EMA）。
        # 速度污染防护：极微位移（<4px）或位移与 vel 夹角 >45°（模板峰值
        # 抖动/转向，方向随机）→ vel 衰减而非混合。静止时抖动位移方向
        # 随机，夹角门槛保证 vel 无法单向累积（否则尺寸突变帧的匹配偏移
        # 会被误判为跳变）；真实匀速运动（同向）正常建立 vel。
        prev_vx, prev_vy = self.vel
        dot = prev_vx * dx_m + prev_vy * dy_m
        disp_sq = dx_m * dx_m + dy_m * dy_m
        if (abs(dx_m) + abs(dy_m) < 4
                or (prev_vx * prev_vx + prev_vy * prev_vy > 1
                    and dot <= 0.7 * math.sqrt((prev_vx * prev_vx + prev_vy * prev_vy) * disp_sq))):
            self.vel = (prev_vx * 0.5, prev_vy * 0.5)
        else:
            self.vel = (
                self.VEL_SMOOTH * dx_m + (1 - self.VEL_SMOOTH) * prev_vx,
                self.VEL_SMOOTH * dy_m + (1 - self.VEL_SMOOTH) * prev_vy,
            )
        self.acc = (
            self.ACC_SMOOTH * (self.vel[0] - prev_vx) + (1 - self.ACC_SMOOTH) * self.acc[0],
            self.ACC_SMOOTH * (self.vel[1] - prev_vy) + (1 - self.ACC_SMOOTH) * self.acc[1],
        )

        # 模板更新：参考外观 = 当前 patch 慢速混合——**仅当目标尺寸≈参考
        # 尺寸时**（tm_scale≈1，bbox 基本不含背景；缩放期间冻结 ref，
        # 避免把背景混入参考外观导致匹配失效）
        if (update_template and self._ref_template is not None
                and tm_score is not None and tm_score >= self.TEMPLATE_UPDATE_MIN
                and abs(tm_scale - 1.0) < 0.05):
            assert tm_scale is not None
            x1b, y1b, x2b, y2b = new_bbox
            patch = gray[y1b:y2b, x1b:x2b]
            if patch.size > 0:
                rth, rtw = self._ref_template.shape
                if patch.shape != (rth, rtw):
                    patch = cv2.resize(patch, (rtw, rth),
                                       interpolation=cv2.INTER_AREA)
                self._ref_template = cv2.addWeighted(
                    self._ref_template, 1.0 - self.TEMPLATE_UPDATE,
                    patch, self.TEMPLATE_UPDATE, 0,
                )

        self.roi = new_bbox
        self.last_bbox = new_bbox
        self.lost_count = 0
        self.state = "tracking"
        self._frames_since_resample += 1

        # 光流点维护：保留 inlier 点，定期重采样防止漂移
        if lk is not None and lk[4] is not None:
            inlier_pts = lk[4]
            if len(inlier_pts) >= 4:
                self.points = inlier_pts.reshape(-1, 1, 2).astype(np.float32)
        if (self._frames_since_resample >= self.RESAMPLE_EVERY
                or self.points is None or len(self.points) < self.MIN_POINTS):
            self.points = self._sample_points()
            self._frames_since_resample = 0

        self.prev_gray = gray
        return True, new_bbox

    # ---------- 位移估计 ----------

    def _match_multi_scale(
        self, gray: np.ndarray, sx1: int, sy1: int, sx2: int, sy2: int,
    ) -> Optional[tuple[float, float, float, float, tuple[int, int]]]:
        """在给定搜索区域做多尺度模板匹配。
        Returns: (dx, dy, score, scale, loc) 或 None
          dx,dy: 模板左上角相对搜索区域原点的位移
          scale: 相对参考外观的最佳尺度（1.0=不变）
          loc: 匹配位置 (x, y)（搜索区域内）

        模板 = 参考外观（初始清晰外观）按尺度缩放——内容始终是清晰细节
        的缩放版，目标缩小/放大都能匹配（避免 patch 重建模板的退化问题）。
        """
        src = self._ref_template
        if src is None:
            return None
        rth, rtw = src.shape
        best: Optional[tuple[float, float, float, float, tuple[int, int]]] = None
        for s in self.SCALES:
            ntw = max(self.MIN_BBOX_SIZE, int(round(rtw * s)))
            nth = max(self.MIN_BBOX_SIZE, int(round(rth * s)))
            if ntw > (sx2 - sx1) or nth > (sy2 - sy1):
                continue
            if abs(s - 1.0) < 1e-6:
                tmpl = src
            else:
                tmpl = cv2.resize(src, (ntw, nth),
                                  interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray[sy1:sy2, sx1:sx2],
                                    tmpl, cv2.TM_CCOEFF_NORMED)
            _mv, sc, _ml, loc = cv2.minMaxLoc(res)
            if best is None or sc > best[2]:
                best = (float(loc[0]), float(loc[1]), float(sc), s, (int(loc[0]), int(loc[1])))
        return best

    def _template_displacement(
        self, gray: np.ndarray
    ) -> Optional[tuple[float, float, float, float]]:
        """多尺度模板匹配位移：在当前 ROI 邻域搜索，返回 (dx, dy, score, scale)。
        峰值位置即真实平移（方向可靠）；scale 反映目标 3D 前后移动导致的
        画面尺寸变化（物体靠近变大 / 远离变小）。
        """
        if self._ref_template is None or self.roi is None:
            return None
        rth, rtw = self._ref_template.shape
        x1, y1, x2, y2 = self.roi
        h, w = gray.shape
        # 搜索窗 = 当前位置 ± (SEARCH_RADIUS + 模板半尺寸)：
        # 保证模板在窗内任意匹配位置都能完全覆盖（否则窗边缘处模板被截断，
        # 分数打折，快速移动帧会被误判为低分）
        r = self.SEARCH_RADIUS + max(rtw, rth) // 2
        sx1, sy1 = max(0, x1 - r), max(0, y1 - r)
        sx2, sy2 = min(w, x2 + r), min(h, y2 + r)
        if sx2 - sx1 <= self.MIN_BBOX_SIZE or sy2 - sy1 <= self.MIN_BBOX_SIZE:
            return None
        m = self._match_multi_scale(gray, sx1, sy1, sx2, sy2)
        if m is None:
            return None
        loc_x, loc_y, score, scale, _loc = m
        dx = float((sx1 + loc_x) - x1)
        dy = float((sy1 + loc_y) - y1)
        return dx, dy, score, scale

    def _lk_displacement(
        self, gray: np.ndarray
    ) -> Optional[tuple[float, float, float, int, Optional[np.ndarray]]]:
        """LK 光流位移：(dx, dy, inlier_ratio, n_good, inlier_points) 或 None"""
        prev_gray, points = self.prev_gray, self.points
        if prev_gray is None or points is None or len(points) == 0:
            return None

        # 跟踪点不足时重采样
        if len(points) < self.MIN_POINTS:
            self.points = self._sample_points()
            points = self.points
        if points is None:
            return None

        # 半速度预平移作为 LK 初始猜测（快速运动引导；方向反转时伤害减半）
        guess = points.reshape(-1, 2) + np.array(
            [self.vel[0] * 0.5, self.vel[1] * 0.5], dtype=np.float32
        )
        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, guess.reshape(-1, 1, 2),
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        good_mask = status.flatten() == 1
        n_good = int(good_mask.sum())
        if n_good < 4:
            return None

        prev_pts = points.reshape(-1, 2)[good_mask]
        good_pts = next_pts[good_mask].reshape(-1, 2)
        delta = good_pts - prev_pts

        # 位移一致性过滤（中值位移 + 离群点剔除）
        med_dx, med_dy = float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))
        dists = np.hypot(delta[:, 0] - med_dx, delta[:, 1] - med_dy)
        inliers = dists < self.INLIER_TOL
        ratio = float(inliers.sum()) / n_good
        inlier_pts = good_pts[inliers] if inliers.any() else None
        return med_dx, med_dy, ratio, n_good, inlier_pts

    @staticmethod
    def _angle_ok(a: tuple[float, float], b: tuple[float, float],
                  max_deg: float = 40.0) -> bool:
        """两个位移向量夹角是否小于 max_deg（用于模板/光流一致性仲裁）"""
        ax, ay = a
        bx, by = b
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na < 0.5 or nb < 0.5:
            return True  # 位移过小，方向无意义，视为一致
        cos_ang = (ax * bx + ay * by) / (na * nb)
        return cos_ang >= math.cos(math.radians(max_deg))

    # ---------- 遮挡处理 ----------

    def _try_recover(self, gray: np.ndarray) -> bool:
        """遮挡恢复：在预测位置邻域做模板匹配搜索。
        目标重新出现（如手移开）且外观匹配 → 重新初始化跟踪。
        """
        if self._ref_template is None or self.last_bbox is None:
            return False
        # 搜索窗尺寸基于参考外观
        rth, rtw = self._ref_template.shape
        x1, y1, x2, y2 = self.last_bbox
        h, w = gray.shape
        m = self.RECOVER_MARGIN
        sx1, sy1 = max(0, x1 - m), max(0, y1 - m)
        sx2, sy2 = min(w, x2 + m), min(h, y2 + m)
        if sx2 - sx1 < rtw or sy2 - sy1 < rth:
            return False

        m = self._match_multi_scale(gray, sx1, sy1, sx2, sy2)
        if m is None:
            return False
        loc_x, loc_y, score, scale, _loc = m

        # 距离加权分数阈值：候选位置离预测位置越远，要求分数越高
        # （防背景相似纹理被误判为目标 → 瞬移）
        cand_cx = sx1 + loc_x + rtw * scale / 2
        cand_cy = sy1 + loc_y + rth * scale / 2
        pred_cx = (x1 + x2) / 2
        pred_cy = (y1 + y2) / 2
        dist = math.hypot(cand_cx - pred_cx, cand_cy - pred_cy)
        required = self.RECOVER_THRESH + min(0.25, dist / 400.0)
        if score < required:
            return False

        # 捕获成功：以匹配位置重建跟踪（保留速度；模板/bbox 跟随命中尺度）
        bx1, by1 = sx1 + int(loc_x), sy1 + int(loc_y)
        bw = max(self.MIN_BBOX_SIZE, int(round(rtw * scale)))
        bh = max(self.MIN_BBOX_SIZE, int(round(rth * scale)))
        prev_vel = self.vel
        self.roi = (bx1, by1, bx1 + bw, by1 + bh)
        self.last_bbox = self.roi
        self.prev_gray = gray
        self._smooth_scale = scale  # 恢复时直接对齐命中尺度（模板保持参考外观）
        self.points = self._sample_points()
        self.vel = prev_vel
        self.lost_count = 0
        self.state = "tracking"
        self._low_score_count = 0
        self._jump_count = 0
        self._frames_since_resample = 0
        logger.info(f"ROI 遮挡恢复: 位置 {self.roi}, 分数 {score:.2f}, 尺度 {scale:.2f}")
        return True

    def _mark_lost(
        self, gray: Optional[np.ndarray] = None
    ) -> tuple[bool, Optional[tuple[int, int, int, int]]]:
        """特征点丢失：更新参考帧，用速度外推预测位置"""
        self.lost_count += 1
        if gray is not None:
            self.prev_gray = gray

        if self.lost_count >= self.MAX_LOST:
            self.state = "lost"
            return False, None

        self.state = "predicting"
        # 匀加速运动外推：vel += acc（自由落体/手带着动时比匀速更准），
        # bbox += vel。速度持续增长会过快，限制单帧外推步长。
        if self.last_bbox is not None:
            vx = self.vel[0] + self.acc[0]
            vy = self.vel[1] + self.acc[1]
            step = math.hypot(vx, vy)
            if step > self.MAX_STEP:
                scale = self.MAX_STEP / step
                vx, vy = vx * scale, vy * scale
            self.vel = (vx, vy)
            x1, y1, x2, y2 = self.last_bbox
            pred = (
                x1 + int(round(vx)), y1 + int(round(vy)),
                x2 + int(round(vx)), y2 + int(round(vy)),
            )
            self.last_bbox = pred
            return False, pred
        return False, None
