#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_photos.py — 将 A4 扫描件（jpg/png）中的多张照片按边界自动拆分并单独保存。

流程:
  1. 估计纸张背景色，结合色差与 Canny 边缘生成前景掩膜；
  2. 连通域分析定位每张照片，并按空白间隙拆分被粘连的区域；
  3. 在全分辨率下精修照片四角，做透视校正（类似 PS 的透视裁切）；
  4. 按阅读顺序输出为单独文件。

用法示例:
    python split_photos.py scan.jpg
    python split_photos.py scans/ -o output/
    python split_photos.py scan1.jpg scan2.png --format jpg --debug

依赖: opencv-python, numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Box = Tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """支持中文/特殊字符路径的图片读取。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, params: Optional[Sequence[int]] = None) -> bool:
    """支持中文/特殊字符路径的图片写出。"""
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img, list(params) if params else [])
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def _odd(value: int, minimum: int = 3) -> int:
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def _order_points(pts: np.ndarray) -> np.ndarray:
    """按 左上/右上/右下/左下 排序（适用于倾角小于 45° 的四边形）。"""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _line_intersection(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray):
    den = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * v2[1] - (p2[1] - p1[1]) * v2[0]) / den
    return np.array([p1[0] + t * v1[0], p1[1] + t * v1[1]], dtype=np.float32)


def _fit_side_lines(quad: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """对四边形的四条边分别拟合直线，再求交点，得到更精确的四角。"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        return quad

    lines = []
    for i in range(4):
        p1, p2 = quad[i], quad[(i + 1) % 4]
        edge = p2 - p1
        length = float(np.linalg.norm(edge))
        if length < 10:
            lines.append(None)
            continue
        direction = edge / length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        dist = np.abs((pts - p1) @ normal)
        proj = (pts - p1) @ direction
        band = max(2.0, length * 0.015)
        mask = (dist < band) & (proj > length * 0.1) & (proj < length * 0.9)
        if int(mask.sum()) < max(10, int(length * 0.15)):
            lines.append(None)
            continue

        vx, vy, x0, y0 = cv2.fitLine(pts[mask], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float32), np.array([vx, vy], dtype=np.float32)))

    refined = quad.copy()
    for i in range(4):
        line_a = lines[i - 1]
        line_b = lines[i]
        if line_a is None or line_b is None:
            continue
        corner = _line_intersection(line_a[0], line_a[1], line_b[0], line_b[1])
        if corner is None:
            continue
        # 与原始角点偏差过大时视为拟合失败
        if np.linalg.norm(corner - quad[i]) > max(8.0, 0.04 * np.linalg.norm(quad[(i + 1) % 4] - quad[i])):
            continue
        refined[i] = corner
    return refined


def warp_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """把任意四边形区域透视变换为端正矩形（类似 PS 透视裁切）。"""
    src = _order_points(quad.reshape(4, 2))
    width = max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3]),
    )
    height = max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1]),
    )
    out_w, out_h = max(1, int(round(width))), max(1, int(round(height)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# --------------------------------------------------------------------------- #
# 前景检测
# --------------------------------------------------------------------------- #
def estimate_background_color(bgr: np.ndarray) -> np.ndarray:
    """
    用四角区域的主色估计纸张/扫描背景色。
    取主色而非均值，可避免扫描仪黑边等少量异色像素的干扰。
    """
    h, w = bgr.shape[:2]
    m = max(8, int(min(h, w) * 0.03))
    patches = [bgr[:m, :m], bgr[:m, -m:], bgr[-m:, :m], bgr[-m:, -m:]]
    pix = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.int32)
    quant = pix // 16
    keys = (quant[:, 0] << 10) | (quant[:, 1] << 5) | quant[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    dominant = vals[int(np.argmax(counts))]
    selected = pix[keys == dominant]
    return np.median(selected, axis=0).astype(np.float32)


def build_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (原始前景掩膜, 闭运算后的掩膜)。
    原始掩膜保留照片之间的空白间隙，用于拆分粘连区域。
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bg = estimate_background_color(bgr)
    diff = np.abs(bgr.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)).max(axis=2)
    diff = cv2.medianBlur(diff.astype(np.uint8), 3)
    _, content = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blur, 40, 120)
    ek = _odd(int(min(h, w) * 0.0025), 3)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)))

    raw = cv2.bitwise_or(content, edges)

    # 只做很小的闭运算连接边界断点，避免把相邻照片粘成一块
    ck = _odd(int(min(h, w) * 0.004), 5)
    closed = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    )

    # 去掉扫描仪/纸张边缘的黑边干扰
    b = max(4, int(min(h, w) * 0.005))

    # 贴边且细长的近黑色连通域 = 扫描仪黑边/相框，整条剔除；
    # 否则它会沿页面四周连成一圈，把整页当成一张照片。
    dark = (gray < 90).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    frame = np.zeros((h, w), np.uint8)
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        thin = min(bw, bh) <= max(4, int(0.01 * min(h, w)))
        ring_like = area <= 0.15 * bw * bh
        if touches and (thin or ring_like):
            frame[labels == i] = 255

    for mask in (raw, closed):
        mask[frame > 0] = 0
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
    return raw, closed


def _zero_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """返回布尔数组中连续 False 的区间 [start, end)。"""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(flags):
        if not value and start is None:
            start = i
        elif value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _tighten_box(mask: np.ndarray, ox: int, oy: int, min_area: float) -> Optional[Box]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()) + ox, int(xs.max()) + ox
    y1, y2 = int(ys.min()) + oy, int(ys.max()) + oy
    if (x2 - x1 + 1) * (y2 - y1 + 1) < min_area:
        return None
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def split_by_white_gaps(
    box: Box, raw: np.ndarray, min_area: float, depth: int = 0
) -> List[Box]:
    """
    若多张照片被误连成一个大区域，则沿掩膜中“整行/整列都没有前景”的空白带切开。
    空白带必须贯穿整个区域，因此不会把照片内部的白色内容误切开。
    """
    if depth > 3:
        return [box]

    x, y, w, h = box
    sub = raw[y : y + h, x : x + w] > 0
    if sub.size == 0:
        return [box]

    for axis in (0, 1):  # 0: 按列切（左右相邻），1: 按行切（上下相邻）
        profile = sub.any(axis=axis)
        if profile.all() or not profile.any():
            continue

        segments: List[Tuple[int, int]] = []
        cursor = 0
        for start, end in _zero_runs(profile):
            if start > cursor:
                segments.append((cursor, start))
            cursor = end
        if cursor < len(profile):
            segments.append((cursor, len(profile)))
        if len(segments) < 2:
            continue

        min_gap = max(3, int(len(profile) * 0.006))
        max_gap = max(12, int(len(profile) * 0.05))
        gaps = [segments[k + 1][0] - segments[k][1] for k in range(len(segments) - 1)]
        parts: List[Box] = []
        valid = True
        for seg_start, seg_end in segments:
            if seg_end - seg_start < 4:
                valid = False
                break
            if axis == 0:
                part_mask = sub[:, seg_start:seg_end]
                part = _tighten_box(part_mask, x + seg_start, y, min_area)
            else:
                part_mask = sub[seg_start:seg_end, :]
                part = _tighten_box(part_mask, x, y + seg_start, min_area)
            if part is None:
                valid = False
                break
            parts.append(part)

        # 间隙太窄不是真正的照片间距；间隙太宽则多半是照片内部发白的天空/白墙，
        # 也不能据此拆分。
        if valid and len(parts) >= 2 and all(min_gap <= g <= max_gap for g in gaps):
            result: List[Box] = []
            for part in parts:
                result.extend(split_by_white_gaps(part, raw, min_area, depth + 1))
            return result

    return [box]


def _rectangularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    return area / (rw * rh) if rw * rh else 0.0


def _contour_metrics(
    contour: np.ndarray, min_area: float, strict: bool = True
) -> Optional[Tuple[Box, float]]:
    """
    用凸包评估轮廓：即使照片边界很淡、掩膜只留下断续的轮廓线，
    其凸包仍能覆盖照片的大部分区域，从而避免漏检。

    strict=True  仅接受完整的矩形轮廓（用于边缘图）；
    strict=False 允许不完整的轮廓/淡边界（用于前景连通域兜底）。
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    if hull_area < min_area * 0.3:
        return None

    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]
    if rw < 10 or rh < 10:
        return None
    rect_fill = hull_area / (rw * rh) if rw * rh else 0.0

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(bw * bh)
    if bbox_area <= 0:
        return None
    hull_ratio = hull_area / bbox_area

    if strict:
        if hull_area < min_area or rect_fill < 0.80:
            return None
    else:
        if bbox_area < min_area:
            return None
        aspect = max(bw, bh) / float(min(bw, bh))
        if aspect > 8 or hull_ratio < 0.45:
            return None
        if rect_fill < 0.80:
            # 只接受“细的边界残片”（淡边界照片），拒绝填充的圆形/色块。
            # 大面积实心区域仍可能是照片（例如照片边界与纸色接近、只剩主体轮廓），
            # 因此仅在区域较小时才拒绝。
            contour_fill = abs(cv2.contourArea(contour)) / bbox_area
            if contour_fill >= 0.35 and bbox_area < 5 * min_area:
                return None

    return (x, y, bw, bh), hull_area


def _nms_regions(
    candidates: List[Tuple[Box, np.ndarray, float]]
) -> List[Tuple[Box, np.ndarray]]:
    """面积优先的非极大值抑制，同时去掉嵌套的内部轮廓。"""
    candidates = sorted(candidates, key=lambda t: t[2], reverse=True)
    kept: List[Tuple[Box, np.ndarray]] = []
    for box, contour, _ in candidates:
        duplicate = False
        for kbox, _ in kept:
            ix = max(0, min(box[0] + box[2], kbox[0] + kbox[2]) - max(box[0], kbox[0]))
            iy = max(0, min(box[1] + box[3], kbox[1] + kbox[3]) - max(box[1], kbox[1]))
            inter = ix * iy
            if inter == 0:
                continue
            union = box[2] * box[3] + kbox[2] * kbox[3] - inter
            small = min(box[2] * box[3], kbox[2] * kbox[3])
            if (inter / union if union else 0) > 0.2 or (inter / small if small else 0) > 0.75:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, contour))
    return kept


def _box_contains(outer: Box, inner: Box) -> float:
    """inner 被 outer 覆盖的比例。"""
    ix = max(0, min(outer[0] + outer[2], inner[0] + inner[2]) - max(outer[0], inner[0]))
    iy = max(0, min(outer[1] + outer[3], inner[1] + inner[3]) - max(outer[1], inner[1]))
    inter = ix * iy
    area = inner[2] * inner[3]
    return inter / area if area else 0.0


# 纸张间隙与纸色的允许色差（仅作为粗筛上限，真正的判别见下面的色度/亮度）
PAPER_GAP_DIFF = 80
# 纸张间隙不应明显比页面背景更亮：老照片里过曝的天空常常比纸更白，
# 借此把“白色天空”与“纸色间隙”区分开。
PAPER_BRIGHT_TOL = 3.0
# 色度容差：纸缝在阴影里只是变暗，色相/饱和度仍与纸张一致；
# 而老照片的天空通常带蓝/黄等色偏，色度会明显偏离纸张。
PAPER_CHROMA_TOL = 0.012
# 判定“纸色像素”的色差阈值，用于判断间隙是否与页面背景连通
PAPER_PIXEL_DIFF = 30


def _band_stats(
    work: np.ndarray, bg: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> Optional[Tuple[float, float, float]]:
    """
    窄条区域（去掉两侧过渡像素）的
    (与纸色的中位色差, 比纸色的中位亮度差, 与纸张的最大色度差)。
    亮度差为正表示该区域比纸张更亮（例如老照片里过曝的白色天空）；
    色度差可识别带蓝/黄色偏的天空——阴影中的纸缝只是变暗，色度基本不变。
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(work.shape[1], x2), min(work.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sx = max(1, int((x2 - x1) * 0.2))
    sy = max(1, int((y2 - y1) * 0.2))
    patch = work[y1 + sy : y2 - sy, x1 + sx : x2 - sx]
    if patch.size == 0:
        return None
    pixels = patch.reshape(-1, 3).astype(np.float32)
    distance = np.median(np.abs(pixels - bg.reshape(1, 3)).max(axis=1))
    med = np.median(pixels, axis=0)
    bright_delta = float(med.mean() - float(bg.mean()))

    def chroma(color: np.ndarray) -> np.ndarray:
        total = float(color.sum())
        return color / total if total > 1e-6 else np.zeros(3, np.float32)

    chroma_delta = float(np.abs(chroma(med) - chroma(bg)).max())
    return float(distance), bright_delta, chroma_delta


def _band_is_paper(
    work: np.ndarray, bg: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> bool:
    """
    窄条是否像纸张间隙：
      1. 不比纸张更亮（排除过曝的白色天空）；
      2. 色度与纸张一致（排除带蓝/黄色偏的天空，同时允许阴影里的纸缝）。
    """
    stats = _band_stats(work, bg, x1, y1, x2, y2)
    if stats is None:
        return False
    distance, bright_delta, chroma_delta = stats
    return (
        distance < PAPER_GAP_DIFF
        and bright_delta <= PAPER_BRIGHT_TOL
        and chroma_delta <= PAPER_CHROMA_TOL
    )


def _band_reaches_page(
    paper_labels: np.ndarray,
    page_labels: set,
    x1: int, y1: int, x2: int, y2: int,
) -> bool:
    """
    窄条里的纸色像素是否与页面外部背景连通。
    照片内部被围住的天空/白墙等区域虽然颜色接近纸，但不与页面背景连通，
    因此不会被当成照片之间的间隙。
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(paper_labels.shape[1], x2), min(paper_labels.shape[0], y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return False
    labels = np.unique(paper_labels[y1:y2, x1:x2])
    return any(int(v) in page_labels for v in labels if v > 0)


def _build_paper_map(work: np.ndarray) -> Tuple[np.ndarray, np.ndarray, set]:
    """返回 (纸色, 纸色连通标记图, 与页面外部连通的标记集合)。"""
    bg = estimate_background_color(work)
    h, w = work.shape[:2]
    # 扫描仪黑边/相框会把页面纸色与图像边界隔断，先把最外圈还原成纸色再连通，
    # 否则会误判“页面纸色不通向外部”，把整页当成一张照片。
    m = max(2, int(0.005 * min(h, w)))
    padded = work.copy()
    padded[:m, :] = bg
    padded[-m:, :] = bg
    padded[:, :m] = bg
    padded[:, -m:] = bg
    paper_mask = (
        np.abs(padded.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)).max(axis=2)
        <= PAPER_PIXEL_DIFF
    ).astype(np.uint8)
    _, labels = cv2.connectedComponents(paper_mask, connectivity=4)
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    page_labels = {int(v) for v in np.unique(border) if v > 0}
    return bg, labels, page_labels


def _union_box(a: Box, b: Box) -> Box:
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _should_merge_without_paper(
    work: np.ndarray,
    bg: np.ndarray,
    paper_labels: np.ndarray,
    page_labels: set,
    a: Box,
    b: Box,
) -> bool:
    """
    两个框紧邻、且交界处没有纸张背景（不是纸色，或纸色区域被照片包围）时，
    判定它们属于同一张照片被误拆的两个部分。
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # 两个框直接相交：真实的两张照片不会互相重叠，说明是同一张照片的碎片
    # （常见于照片上部过曝白墙未被检出，只有局部暗部独立成块）。
    # 用重叠区域的颜色判断，避免把斜放照片包围盒的角落重叠误合并。
    ox1, ox2 = max(ax, bx), min(ax2, bx2)
    oy1, oy2 = max(ay, by), min(ay2, by2)
    if ox2 - ox1 > 0 and oy2 - oy1 > 0:
        if not _band_is_paper(work, bg, ox1, oy1, ox2, oy2):
            return True

    def gap_has_no_paper(band: Box, gap: int, limit: int) -> bool:
        if gap > limit:
            return False
        if gap < 2:
            return True
        if not _band_is_paper(work, bg, *band):
            return True
        return not _band_reaches_page(paper_labels, page_labels, *band)

    # 间隙上限按页面尺度取，避免把真正的纸张间距放进合并判断
    max_gap = max(16, int(0.03 * min(work.shape[:2])))

    # 上下紧邻（水平方向有足够重叠）
    ox1, ox2 = max(ax, bx), min(ax2, bx2)
    if ox2 - ox1 >= 0.5 * min(aw, bw):
        limit = max_gap
        if ay2 <= by:
            gap = by - ay2
            if gap_has_no_paper((ox1, ay2, ox2, by), gap, limit):
                return True
        if by2 <= ay:
            gap = ay - by2
            if gap_has_no_paper((ox1, by2, ox2, ay), gap, limit):
                return True

    # 左右紧邻（垂直方向有足够重叠）
    oy1, oy2 = max(ay, by), min(ay2, by2)
    if oy2 - oy1 >= 0.5 * min(ah, bh):
        limit = max_gap
        if ax2 <= bx:
            gap = bx - ax2
            if gap_has_no_paper((ax2, oy1, bx, oy2), gap, limit):
                return True
        if bx2 <= ax:
            gap = ax - bx2
            if gap_has_no_paper((bx2, oy1, ax, oy2), gap, limit):
                return True

    return False


def _merge_without_paper(
    work: np.ndarray,
    bg: np.ndarray,
    paper_labels: np.ndarray,
    page_labels: set,
    regions: List[Tuple[Box, Optional[np.ndarray]]],
) -> List[Tuple[Box, Optional[np.ndarray]]]:
    """把被误拆且中间没有纸张间隙的区域重新合并。"""
    items: List[List] = [[box, contour] for box, contour in regions]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items):
                if _should_merge_without_paper(
                    work, bg, paper_labels, page_labels, items[i][0], items[j][0]
                ):
                    items[i][0] = _union_box(items[i][0], items[j][0])
                    items[i][1] = None
                    items.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1
    return [(box, contour) for box, contour in items]


def _gap_is_paperlike(
    work: np.ndarray,
    bg: np.ndarray,
    parent: Box,
    children: List[Box],
    paper_labels: Optional[np.ndarray] = None,
    page_labels: Optional[set] = None,
) -> bool:
    """
    判断父框内的多个子框是否真的能各自独立成一张照片。
    要求：所有子框都能通过“窄的纸张间隙”连成一片。
    这样处理的理由：
      * 真正的多张照片之间是一条窄纸缝，彼此都能连起来；
      * 一张照片内部被大片发白天空/白墙分隔出的碎片，碎片之间的“间隙”很宽，
        无法通过窄缝连通，因此父框应保留为整张照片，而不是拆成多块。
    """
    if len(children) < 2:
        return False

    max_thickness = max(12, int(0.04 * min(work.shape[:2])))

    def linked(a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        bands: List[Box] = []
        if ax2 <= bx:
            bands.append((ax2, max(ay, by), bx, min(ay2, by2)))
        elif bx2 <= ax:
            bands.append((bx2, max(ay, by), ax, min(ay2, by2)))
        if ay2 <= by:
            bands.append((max(ax, bx), ay2, min(ax2, bx2), by))
        elif by2 <= ay:
            bands.append((max(ax, bx), by2, min(ax2, bx2), ay))

        for x1, y1, x2, y2 in bands:
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            # 真正的照片间距是一条窄缝
            if min(x2 - x1, y2 - y1) > max_thickness:
                continue
            if not _band_is_paper(work, bg, x1, y1, x2, y2):
                continue
            if paper_labels is not None and page_labels is not None:
                if not _band_reaches_page(paper_labels, page_labels, x1, y1, x2, y2):
                    continue
            return True
        return False

    reached = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(len(children)):
            if j not in reached and linked(children[i], children[j]):
                reached.add(j)
                stack.append(j)
    return len(reached) == len(children)


def _grow_to_content(work: np.ndarray, bg: np.ndarray, box: Box) -> Box:
    """
    过曝白墙/天空与纸张同色时，检测框会向内缩一截。
    逐列/逐行向外扩展，直到该列（行）几乎全是纸张色为止。
    真实照片之间的纸张间隙整列都是纸色，因此不会被越过。
    """
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    H, W = work.shape[:2]

    def column_is_paper(xx: int) -> bool:
        col = work[y1:y2, xx].astype(np.int16) - bg.reshape(1, 3).astype(np.int16)
        return float((np.abs(col).max(axis=1) <= PAPER_PIXEL_DIFF).mean()) >= 0.98

    def row_is_paper(yy: int) -> bool:
        row = work[yy, x1:x2].astype(np.int16) - bg.reshape(1, 3).astype(np.int16)
        return float((np.abs(row).max(axis=1) <= PAPER_PIXEL_DIFF).mean()) >= 0.98

    max_dx, max_dy = int(w * 0.5), int(h * 0.5)
    for _ in range(max_dx):
        if x1 <= 0 or column_is_paper(x1 - 1):
            break
        x1 -= 1
    for _ in range(max_dx):
        if x2 >= W - 1 or column_is_paper(x2):
            break
        x2 += 1
    for _ in range(max_dy):
        if y1 <= 0 or row_is_paper(y1 - 1):
            break
        y1 -= 1
    for _ in range(max_dy):
        if y2 >= H - 1 or row_is_paper(y2):
            break
        y2 += 1
    return (x1, y1, x2 - x1, y2 - y1)


def _detect_edge_regions(work: np.ndarray, min_area: float) -> List[Tuple[Box, np.ndarray]]:
    """
    直接在边缘图上找闭合轮廓。
    相比连通域，它能在一组相邻/轻微粘连的照片中分别识别每张照片的边界。
    """
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    h, w = work.shape[:2]
    # 页面最外圈的扫描仪黑边会形成一条包围整页的闭合轮廓，先清掉
    b = max(4, int(min(h, w) * 0.005))
    edges[:b, :] = 0
    edges[-b:, :] = 0
    edges[:, :b] = 0
    edges[:, -b:] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[Box, np.ndarray, float]] = []
    for contour in contours:
        metrics = _contour_metrics(contour, min_area)
        if metrics is None:
            continue
        box, hull_area = metrics
        if box[2] < w * 0.03 or box[3] < h * 0.03:
            continue
        candidates.append((box, contour, hull_area))

    # 若某个外轮廓内部包含两个及以上“照片级”的矩形轮廓，说明它可能是多张照片的外包络。
    # 只有这些子框之间确实是纸张背景时才认定是合并区域，避免把照片内部的纹理误判为多张照片。
    bg, paper_labels, page_labels = _build_paper_map(work)

    filtered: List[Tuple[Box, np.ndarray, float]] = []
    for i, (box, contour, area) in enumerate(candidates):
        children = [
            (other_box, other_area)
            for j, (other_box, _, other_area) in enumerate(candidates)
            if j != i
            and 0.2 * area <= other_area <= 0.95 * area
            and _box_contains(box, other_box) > 0.8
        ]
        if len(children) >= 2 and sum(a for _, a in children) >= 0.55 * area:
            if _gap_is_paperlike(
                work, bg, box, [b for b, _ in children], paper_labels, page_labels
            ):
                continue
        filtered.append((box, contour, area))

    return _nms_regions(filtered)


def sort_reading_order(boxes: List[Box]) -> List[Box]:
    """按阅读顺序（先上后下、同一行先左后右）排序。"""
    rows: List[List[Box]] = []
    for box in sorted(boxes, key=lambda b: b[1]):
        cy = box[1] + box[3] / 2.0
        for row in rows:
            top = min(b[1] for b in row)
            bottom = max(b[1] + b[3] for b in row)
            if top - 0.35 * box[3] <= cy <= bottom + 0.35 * box[3]:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: List[Box] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def _merge_by_border_continuity(
    work: np.ndarray, regions: List[Tuple[Box, Optional[np.ndarray]]]
) -> List[Tuple[Box, Optional[np.ndarray]]]:
    """
    若两块区域之间只有一条极窄的纸色缝，但照片的上下（或左右）边框线
    在缝里仍然连续，说明它们其实是同一张照片被白天空/白墙切开的两半，合并之。
    真正相邻的两张照片在纸缝处边框线是断开的，不会被合并。
    """
    if len(regions) < 2:
        return regions

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thin = max(10, int(0.012 * min(work.shape[:2])))

    def horizontal_coverage(y: int, x1: int, x2: int) -> float:
        y0, y1 = max(0, y - 2), min(mag.shape[0], y + 3)
        x0, x1c = max(0, x1), min(mag.shape[1], x2)
        sub = mag[y0:y1, x0:x1c]
        if sub.size == 0:
            return 0.0
        return float((sub.max(axis=0) > 60).mean())

    def vertical_coverage(x: int, y1: int, y2: int) -> float:
        x0, x1 = max(0, x - 2), min(mag.shape[1], x + 3)
        y0, y1c = max(0, y1), min(mag.shape[0], y2)
        sub = mag[y0:y1c, x0:x1]
        if sub.size == 0:
            return 0.0
        return float((sub.max(axis=1) > 60).mean())

    def same_photo(a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        # 左右相邻
        if ax2 <= bx or bx2 <= ax:
            gap_x1, gap_x2 = (ax2, bx) if ax2 <= bx else (bx2, ax)
            if gap_x2 - gap_x1 <= thin and min(ay2, by2) - max(ay, by) >= 0.5 * min(ah, bh):
                top = min(ay, by)
                bottom = max(ay2, by2)
                return (
                    horizontal_coverage(top, gap_x1, gap_x2) > 0.6
                    or horizontal_coverage(bottom, gap_x1, gap_x2) > 0.6
                )

        # 上下相邻
        if ay2 <= by or by2 <= ay:
            gap_y1, gap_y2 = (ay2, by) if ay2 <= by else (by2, ay)
            if gap_y2 - gap_y1 <= thin and min(ax2, bx2) - max(ax, bx) >= 0.5 * min(aw, bw):
                left = min(ax, bx)
                right = max(ax2, bx2)
                return (
                    vertical_coverage(left, gap_y1, gap_y2) > 0.6
                    or vertical_coverage(right, gap_y1, gap_y2) > 0.6
                )
        return False

    items: List[List] = [[box, contour] for box, contour in regions]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items):
                if same_photo(items[i][0], items[j][0]):
                    items[i][0] = _union_box(items[i][0], items[j][0])
                    items[i][1] = None
                    items.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return [(box, contour) for box, contour in items]


def _merge_aligned_thin_gap(
    work: np.ndarray, regions: List[Tuple[Box, Optional[np.ndarray]]]
) -> List[Tuple[Box, Optional[np.ndarray]]]:
    """
    两块区域之间只有 1–2px 的纯白细缝、且上下（或左右）边完全齐平时，
    判定它们是被过曝天空/白墙切开的两半，合并回一张照片。
    真正的照片间距通常更宽，或者缝里有阴影/杂色，不会满足“纯白 + 完全齐平”。
    """
    if len(regions) < 2:
        return regions

    bg = estimate_background_color(work)
    bg_mean = float(bg.mean())
    thin = max(8, int(0.01 * min(work.shape[:2])))

    def whitish(x1: int, y1: int, x2: int, y2: int) -> bool:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(work.shape[1], x2), min(work.shape[0], y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return False
        pixels = work[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
        med = np.median(pixels, axis=0)
        return float(med.mean()) >= bg_mean - 3 and float(np.abs(med - bg).max()) <= 10

    def align_tol(span: int) -> int:
        return max(3, int(0.012 * span))

    def same_photo(a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        # 左右相邻且上下边齐平
        if ax2 <= bx:
            gap = bx - ax2
            if gap <= thin and abs(ay - by) <= align_tol(min(ah, bh)) and abs(ay2 - by2) <= align_tol(min(ah, bh)):
                return whitish(ax2, max(ay, by), bx, min(ay2, by2))
        if bx2 <= ax:
            gap = ax - bx2
            if gap <= thin and abs(ay - by) <= align_tol(min(ah, bh)) and abs(ay2 - by2) <= align_tol(min(ah, bh)):
                return whitish(bx2, max(ay, by), ax, min(ay2, by2))

        # 上下相邻且左右边齐平
        if ay2 <= by:
            gap = by - ay2
            if gap <= thin and abs(ax - bx) <= align_tol(min(aw, bw)) and abs(ax2 - bx2) <= align_tol(min(aw, bw)):
                return whitish(max(ax, bx), ay2, min(ax2, bx2), by)
        if by2 <= ay:
            gap = ay - by2
            if gap <= thin and abs(ax - bx) <= align_tol(min(aw, bw)) and abs(ax2 - bx2) <= align_tol(min(aw, bw)):
                return whitish(max(ax, bx), by2, min(ax2, bx2), ay)
        return False

    items: List[List] = [[box, contour] for box, contour in regions]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items):
                if same_photo(items[i][0], items[j][0]):
                    items[i][0] = _union_box(items[i][0], items[j][0])
                    items[i][1] = None
                    items.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return [(box, contour) for box, contour in items]


def _absorb_stray_content(
    work: np.ndarray, bg: np.ndarray, regions: List[Tuple[Box, Optional[np.ndarray]]]
) -> List[Tuple[Box, Optional[np.ndarray]]]:
    """
    扫描页上除纸张背景外的内容都属于某张照片。
    把落在所有检测框之外的散落前景（例如与主体断开的树冠、阴影块）并入最近的框，
    避免照片被裁掉一截。
    """
    if not regions:
        return regions
    h, w = work.shape[:2]
    diff = np.abs(work.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)).max(axis=2)
    content = (diff > 30).astype(np.uint8)
    edge = max(1, int(min(h, w) * 0.003))
    content[:edge] = 0
    content[-edge:] = 0
    content[:, :edge] = 0
    content[:, -edge:] = 0

    covered = np.zeros((h, w), np.uint8)
    for (x, y, bw, bh), _ in regions:
        covered[max(0, y) : y + bh, max(0, x) : x + bw] = 1
    rest = cv2.bitwise_and(content, cv2.bitwise_not(covered))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not rest.any():
        return regions

    count, labels, stats, _ = cv2.connectedComponentsWithStats(rest, 8)
    min_piece = 0.0004 * h * w
    reach = 0.10 * min(h, w)
    boxes = [[float(v) for v in box] for box, _ in regions]
    for idx in range(1, count):
        x, y, bw, bh, area = stats[idx]
        if area < min_piece:
            continue
        best, best_gap = -1, float("inf")
        for k, (bx, by, bw_, bh_) in enumerate(boxes):
            gx = max(0.0, max(bx - (x + bw), x - (bx + bw_)))
            gy = max(0.0, max(by - (y + bh), y - (by + bh_)))
            gap = float(np.hypot(gx, gy))
            if gap < best_gap:
                best, best_gap = k, gap
        if best < 0 or best_gap > reach:
            continue
        bx, by, bw_, bh_ = boxes[best]
        nx1, ny1 = min(bx, x), min(by, y)
        nx2, ny2 = max(bx + bw_, x + bw), max(by + bh_, y + bh)
        boxes[best] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
    return [
        ((int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))), c)
        for b, (_, c) in zip(boxes, regions)
    ]


def detect_regions(
    img: np.ndarray,
    min_area_ratio: float = 0.004,
    max_side: int = 1800,
    debug: bool = False,
) -> Tuple[List[Tuple[Box, Optional[np.ndarray]]], float, Optional[np.ndarray]]:
    """
    检测照片候选区域。
    返回 ([(原图坐标框, 工作分辨率轮廓[可为 None]), ...], scale, 调试图)。
    """
    h0, w0 = img.shape[:2]
    scale = min(1.0, max_side / float(max(h0, w0)))
    if scale < 1.0:
        work = cv2.resize(
            img,
            (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = img.copy()

    raw, closed = build_masks(work)
    h, w = work.shape[:2]
    page_area = float(h * w)
    min_area = min_area_ratio * page_area

    # 来源一：边缘图上的独立闭合轮廓（可分辨相邻/轻微粘连的照片）
    edge_regions = _detect_edge_regions(work, min_area)

    # 来源二：前景连通域（兜底，适合边界很淡但色差明显的照片）
    mask_candidates: List[Tuple[Box, np.ndarray, float]] = []
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        metrics = _contour_metrics(contour, min_area, strict=False)
        if metrics is None:
            continue
        box, hull_area = metrics
        if box[2] < w * 0.015 or box[3] < h * 0.015:
            continue
        mask_candidates.append((box, contour, hull_area))
    mask_regions = _nms_regions(mask_candidates)

    # 如果某个连通域内部已经识别出多张照片，说明它把相邻照片粘在了一起，丢弃
    merged_candidates: List[Tuple[Box, Optional[np.ndarray]]] = []
    for box, contour in mask_regions:
        inside = [eb for eb, _ in edge_regions if _box_contains(box, eb) > 0.6]
        if len(inside) >= 2:
            continue
        if len(inside) == 1:
            eb = inside[0]
            if eb[2] * eb[3] >= 0.7 * box[2] * box[3]:
                continue  # 与边缘框基本重复
        merged_candidates.append((box, contour))

    combined: List[Tuple[Box, Optional[np.ndarray]]] = list(edge_regions)
    combined.extend(merged_candidates)

    # 对未被拆开的粘连区域再按空白间隙切分
    split_regions: List[Tuple[Box, Optional[np.ndarray]]] = []
    for box, contour in combined:
        parts = split_by_white_gaps(box, raw, min_area)
        if len(parts) == 1:
            split_regions.append((box, contour))
        else:
            split_regions.extend((part, None) for part in parts)

    # 合并后再做一次去重（保留轮廓可能为 None 的拆分结果）
    final_regions: List[Tuple[Box, Optional[np.ndarray]]] = []
    for box, contour in sorted(
        split_regions, key=lambda t: t[0][2] * t[0][3], reverse=True
    ):
        duplicate = False
        for kbox, _ in final_regions:
            ix = max(0, min(box[0] + box[2], kbox[0] + kbox[2]) - max(box[0], kbox[0]))
            iy = max(0, min(box[1] + box[3], kbox[1] + kbox[3]) - max(box[1], kbox[1]))
            inter = ix * iy
            if inter == 0:
                continue
            small = min(box[2] * box[3], kbox[2] * kbox[3])
            union = box[2] * box[3] + kbox[2] * kbox[3] - inter
            if (inter / union if union else 0) > 0.2 or (inter / small if small else 0) > 0.75:
                duplicate = True
                break
        if not duplicate:
            final_regions.append((box, contour))

    # 照片内部的大面积天空/白墙可能被拆成两块，交界处并没有纸张背景，需要重新合并
    bg, paper_labels, page_labels = _build_paper_map(work)
    final_regions = _merge_without_paper(
        work, bg, paper_labels, page_labels, final_regions
    )
    # 过曝白墙/天空与纸张同色，检测框会少一截；向外扩展到整列/整行都变成纸色为止
    final_regions = [
        (_grow_to_content(work, bg, box), contour) for box, contour in final_regions
    ]
    # 所有非背景内容都应属于某张照片：把遗落在框外的碎片并入最近的框
    final_regions = _absorb_stray_content(work, bg, final_regions)

    # 映射回原图坐标（保留极小的安全余量，供四角精修使用）
    pad = max(2, int(round(min(h0, w0) * 0.002)))
    results: List[Tuple[Box, Optional[np.ndarray]]] = []
    for (x, y, bw, bh), contour in final_regions:
        x1 = max(0, int(round(x / scale)) - pad)
        y1 = max(0, int(round(y / scale)) - pad)
        x2 = min(w0, int(round((x + bw) / scale)) + pad)
        y2 = min(h0, int(round((y + bh) / scale)) + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        results.append(((x1, y1, x2 - x1, y2 - y1), contour))

    by_box = {b: c for b, c in results}
    ordered_boxes = sort_reading_order([b for b, _ in results])
    results = [(b, by_box[b]) for b in ordered_boxes]

    debug_img = None
    if debug:
        debug_img = work.copy()
        for idx, ((x, y, bw, bh), _) in enumerate(results, 1):
            wx, wy = int(round(x * scale)), int(round(y * scale))
            ww, wh = int(round(bw * scale)), int(round(bh * scale))
            cv2.rectangle(debug_img, (wx, wy), (wx + ww, wy + wh), (0, 0, 255), 3)
            cv2.putText(
                debug_img,
                str(idx),
                (wx + 6, wy + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return results, scale, debug_img


# --------------------------------------------------------------------------- #
# 四角精修 + 透视裁切
# --------------------------------------------------------------------------- #
def _quad_bbox(quad: np.ndarray) -> Box:
    xs, ys = quad[:, 0], quad[:, 1]
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max() - xs.min())) + 1,
        int(np.ceil(ys.max() - ys.min())) + 1,
    )


def _box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _box_coverage(box: Box, ref: Box) -> float:
    """box 覆盖 ref 的比例（交面积 / ref 面积）。"""
    ix = max(0, min(box[0] + box[2], ref[0] + ref[2]) - max(box[0], ref[0]))
    iy = max(0, min(box[1] + box[3], ref[1] + ref[3]) - max(box[1], ref[1]))
    ref_area = ref[2] * ref[3]
    return (ix * iy) / ref_area if ref_area else 0.0


def _find_quad_in_roi(roi: np.ndarray, expected: Box) -> Optional[np.ndarray]:
    """
    在局部区域中找到最贴近 expected 框的照片四角（ROI 局部坐标）。
    使用位置/面积相似度评分，避免把相邻的另一张照片误当成目标。
    若最佳轮廓只覆盖照片的一部分（例如天空与地面被拆成两个闭合轮廓），
    则把这些部分轮廓合并成完整照片的四角。
    """
    h, w = roi.shape[:2]
    if h < 20 or w < 20:
        return None

    ex, ey, ew, eh = expected
    exp_area = float(max(1, ew * eh))
    exp_cx, exp_cy = ex + ew / 2.0, ey + eh / 2.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 25, 80)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found: List[Tuple[float, np.ndarray, Box]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.15 * exp_area:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        rect_area = rw * rh
        if rw < 15 or rh < 15 or rect_area < 0.2 * exp_area:
            continue
        rect_fill = area / rect_area if rect_area else 0.0
        if rect_fill < 0.78:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        is_quad = len(approx) == 4 and cv2.isContourConvex(approx)
        if is_quad:
            quad = approx.reshape(4, 2).astype(np.float32)
            quad = _fit_side_lines(_order_points(quad), contour)
        else:
            quad = cv2.boxPoints(rect).astype(np.float32)

        box = _quad_bbox(quad)
        overlap = _box_iou(box, (ex, ey, ew, eh))
        if overlap < 0.2:
            continue

        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        center_penalty = min(1.0, abs(cx - exp_cx) / max(ew, 1) + abs(cy - exp_cy) / max(eh, 1))
        area_similarity = min(rect_area, exp_area) / max(rect_area, exp_area)

        score = (0.55 * overlap + 0.3 * area_similarity + 0.15 * (1 - center_penalty)) * rect_fill
        if is_quad:
            score *= 1.1
        if score >= 0.2:
            found.append((score, quad, box))

    if not found:
        return None
    found.sort(key=lambda t: t[0], reverse=True)
    best_score, best_quad, best_box = found[0]
    if best_score < 0.25:
        return None
    if _box_coverage(best_box, (ex, ey, ew, eh)) >= 0.85:
        return best_quad

    # 最佳轮廓只覆盖了照片的一部分：合并落在 expected 内的部分轮廓
    parts = [quad for _, quad, box in found if _box_contains((ex, ey, ew, eh), box) >= 0.8]
    if len(parts) >= 2:
        points = np.concatenate(parts).astype(np.float32)
        combined = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(points))).astype(np.float32)
        combo_box = _quad_bbox(combined)
        if (
            _box_coverage(combo_box, (ex, ey, ew, eh)) >= 0.85
            and _box_contains((ex, ey, ew, eh), combo_box) >= 0.8
        ):
            return _order_points(combined)

    # 只找到照片的一部分（例如大片天空与纸色接近、边缘不可见），
    # 交给上层用检测轮廓/检测框兜底，避免把照片裁掉一块。
    return None


def _fit_quad_by_content(roi: np.ndarray, expected: Box) -> Optional[np.ndarray]:
    """
    边框很淡（照片上部与纸张同为白色）时，闭合轮廓找不到；
    改为沿四条边找“纸张→照片内容”的过渡点，再拟合四条直线求交点。
    只在能看见内容的边段取点，直线自然延伸到看不见的白色区域。
    """
    h, w = roi.shape[:2]
    ex, ey, ew, eh = expected
    bg = estimate_background_color(roi)
    diff = np.abs(roi.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)).max(axis=2)
    mask = (diff > 50).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tall = cv2.erode(mask, np.ones((9, 1), np.uint8))   # 连续 9 行有内容
    wide = cv2.erode(mask, np.ones((1, 9), np.uint8))   # 连续 9 列有内容

    tol_x, tol_y = 0.15 * ew, 0.15 * eh

    def collect(points, xlim, ylim):
        out = [(x, y) for x, y in points if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]]
        return np.array(out, dtype=np.float32).reshape(-1, 2)

    def walk(seed_x: int, seed_y: int, dx: int, dy: int):
        """从照片内部一点沿某方向走到内容尽头，返回边界点（避免把相邻照片的边当成这条边）。"""
        mat = wide if dy == 0 else tall
        hh, ww = mat.shape[:2]
        x, y = seed_x, seed_y
        if not (0 <= x < ww and 0 <= y < hh) or not mat[y, x]:
            return None
        while 0 <= x + dx < ww and 0 <= y + dy < hh and mat[y + dy, x + dx]:
            x += dx
            y += dy
        return (x, y)

    def scan(dx: int, dy: int, along_x: bool):
        pts = []
        if along_x:  # 上下边：逐列，从框内 8% 处向边方向走
            seed = int(round(ey + (0.08 if dy < 0 else 0.92) * eh))
            for x in range(int(ex + 0.02 * ew), int(ex + 0.98 * ew)):
                p = walk(x, seed, dx, dy)
                if p is not None:
                    pts.append(p)
        else:  # 左右边：逐行
            seed = int(round(ex + (0.08 if dx < 0 else 0.92) * ew))
            for y in range(int(ey + 0.02 * eh), int(ey + 0.98 * eh)):
                p = walk(seed, y, dx, dy)
                if p is not None:
                    pts.append(p)
        return pts

    top = collect(scan(0, -1, True), (ex + 0.02 * ew, ex + 0.98 * ew),
                  (ey - 0.10 * eh, ey + 0.10 * eh))
    bottom = collect(scan(0, 1, True), (ex + 0.02 * ew, ex + 0.98 * ew),
                     (ey + 0.90 * eh, ey + eh + 0.03 * eh))
    left = collect(scan(-1, 0, False), (ex - 0.03 * ew, ex + 0.15 * ew),
                   (ey + 0.02 * eh, ey + 0.98 * eh))
    right = collect(scan(1, 0, False), (ex + 0.90 * ew, ex + ew + 0.03 * ew),
                    (ey + 0.02 * eh, ey + 0.98 * eh))

    def fit_line(pts: np.ndarray, min_span: float):
        """RANSAC 拟合：边线只占取点的一部分（其余是照片内部纹理），需要抗差。"""
        n = len(pts)
        if n < 20:
            return None
        rng = np.random.default_rng(0)
        best = None
        for _ in range(300):
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            p, q = pts[i], pts[j]
            d = q - p
            length = float(np.linalg.norm(d))
            if length < 20:
                continue
            normal = np.array([-d[1], d[0]], dtype=np.float32) / length
            inliers = np.abs((pts - p) @ normal) < 2.5
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers)
        if best is None or best[0] < max(15, int(0.2 * n)):
            return None
        inl = best[1]
        if max(np.ptp(pts[inl][:, 0]), np.ptp(pts[inl][:, 1])) < min_span:
            return None
        line = cv2.fitLine(pts[inl], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        return np.array(line[2:4], np.float32), np.array(line[:2], np.float32)

    lines = [
        fit_line(top, 0.25 * ew),
        fit_line(right, 0.25 * eh),
        fit_line(bottom, 0.25 * ew),
        fit_line(left, 0.25 * eh),
    ]
    if any(line is None for line in lines):
        return None

    corners = []
    for i in range(4):
        corner = _line_intersection(lines[i][0], lines[i][1], lines[i - 1][0], lines[i - 1][1])
        if corner is None:
            return None
        corners.append(corner)
    quad = _order_points(np.array(corners, dtype=np.float32))

    # 合理性校验：面积接近检测框、四角不越界
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if not (0.70 * ew * eh <= area <= 1.15 * ew * eh):
        return None
    for x, y in quad:
        if not (ex - tol_x <= x <= ex + ew + tol_x and ey - tol_y <= y <= ey + eh + tol_y):
            return None
    return quad


def _is_near_rect(quad: np.ndarray, max_angle_dev: float = 2.0,
                  min_opposite_ratio: float = 0.75) -> bool:
    """
    照片扫描件在纸面上可能轻微倾斜，但本身一定接近矩形：
    四个内角与 90° 的偏差不超过 2°，对边长度也基本相等。
    用于剔除把边缘/阴影拟合成细长条或任意四边形的错误结果。
    """
    q = _order_points(np.asarray(quad, dtype=np.float32).reshape(4, 2))
    sides = [q[(i + 1) % 4] - q[i] for i in range(4)]
    lengths = [float(np.linalg.norm(s)) for s in sides]
    if min(lengths) < 8:
        return False
    for a, b in ((0, 2), (1, 3)):
        lo, hi = min(lengths[a], lengths[b]), max(lengths[a], lengths[b])
        if hi <= 0 or lo / hi < min_opposite_ratio:
            return False
    # 长宽比约束：扫描照片不会细长如纸条
    side_a = 0.5 * (lengths[0] + lengths[2])
    side_b = 0.5 * (lengths[1] + lengths[3])
    if min(side_a, side_b) <= 0 or max(side_a, side_b) / min(side_a, side_b) > 4.5:
        return False
    for i in range(4):
        u, v = q[i] - q[i - 1], q[(i + 1) % 4] - q[i]
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom <= 0:
            return False
        cos = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
        if abs(np.degrees(np.arccos(cos)) - 90.0) > max_angle_dev:
            return False
    return True


def _cover_rect(quad: np.ndarray, box: Box) -> np.ndarray:
    """
    取四边形体现出的倾斜角度，生成一个同角度、但能完全覆盖检测框的矩形。
    这样既保留方向修正，又不会因为拟合线略微内缩而裁掉照片内容。
    """
    q = _order_points(np.asarray(quad, np.float32).reshape(4, 2))
    edge = q[1] - q[0]
    if np.linalg.norm(edge) < np.linalg.norm(q[3] - q[0]):
        edge = q[3] - q[0]
    ang = float(np.degrees(np.arctan2(edge[1], edge[0])))
    while ang > 45.0:
        ang -= 90.0
    while ang < -45.0:
        ang += 90.0

    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    t = np.radians(ang)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    corners = np.array(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]], np.float32
    )
    local = (corners - [cx, cy]) @ rot  # 转到倾斜坐标系
    x1, y1 = local.min(axis=0)
    x2, y2 = local.max(axis=0)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return (rect @ rot.T + [cx, cy]).astype(np.float32)


def _extract_photo(
    img: np.ndarray, box: Box, contour: Optional[np.ndarray], scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """按检测框在局部重新找四角，透视校正后裁出照片。返回 (裁剪图, 原图坐标四角)。"""
    img_h, img_w = img.shape[:2]
    x, y, w, h = box

    margin = max(8, int(round(max(w, h) * 0.04)))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_w, x + w + margin), min(img_h, y + h + margin)
    roi = img[y1:y2, x1:x2]

    expected = (x - x1, y - y1, w, h)
    ex, ey, ew, eh = expected
    quad = _find_quad_in_roi(roi, expected)
    if quad is None:
        quad = _fit_quad_by_content(roi, expected)
    if quad is not None:
        # 约束：照片纸边必须是接近矩形的四边形，面积也要与检测框接近，
        # 且不能比检测框明显内缩（内缩说明拟合到了照片内部纹理，会裁掉内容）。
        quad = np.asarray(quad, np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(quad.reshape(-1, 1, 2))))
        box_area = float(max(1, w * h))
        qx1, qy1 = float(quad[:, 0].min()), float(quad[:, 1].min())
        qx2, qy2 = float(quad[:, 0].max()), float(quad[:, 1].max())
        tol_x, tol_y = 0.015 * ew, 0.015 * eh
        shrunk = (qx1 > ex + tol_x or qy1 > ey + tol_y
                  or qx2 < ex + ew - tol_x or qy2 < ey + eh - tol_y)
        if not _is_near_rect(quad) or not (0.45 <= area / box_area <= 1.15):
            quad = None
        elif shrunk:
            # 角度可信，但边线略向内：保留角度，扩成能覆盖检测框的同角度矩形
            quad = _cover_rect(quad, (ex, ey, ew, eh))
    if quad is not None:
        quad = quad + np.array([x1, y1], dtype=np.float32)
    else:
        # 回退：轮廓足够完整时用最小外接矩形；否则（淡边界只剩一段阴影线）用检测框
        pts = None
        if contour is not None and len(contour) >= 4:
            pts = contour.reshape(-1, 2).astype(np.float32) / scale
        if pts is not None:
            bbox_area = float(max(1, w * h))
            contour_area = abs(cv2.contourArea(pts))
            hull_area = cv2.contourArea(cv2.convexHull(pts))
            if contour_area >= 0.25 * bbox_area and hull_area >= 0.5 * bbox_area:
                quad = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
                if not _is_near_rect(quad):  # 轮廓本身若是一条细线，外接矩形也不可信
                    quad = None
            if quad is None:
                quad = np.array(
                    [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
                    dtype=np.float32,
                )
        else:
            quad = np.array(
                [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
                dtype=np.float32,
            )

    return warp_quad(img, quad), quad


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _is_output_file(path: Path, root: Path) -> bool:
    """避免把上次运行生成的 *_split/ 输出再次当作输入。"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part.endswith("_split") for part in relative.parts[:-1])


def collect_inputs(inputs: Sequence[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(
                f
                for f in sorted(p.glob(pattern))
                if f.suffix.lower() in IMAGE_EXTS and not _is_output_file(f, p)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"[警告] 找不到输入: {item}", file=sys.stderr)

    seen = set()
    unique = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def process_one(
    path: Path,
    output_dir: Path,
    fmt: str,
    quality: int,
    min_area_ratio: float,
    max_side: int,
    debug: bool,
    prefix: Optional[str],
) -> int:
    img = imread_unicode(path)
    if img is None:
        print(f"[跳过] 无法读取图片: {path}", file=sys.stderr)
        return 0

    regions, scale, debug_img = detect_regions(
        img, min_area_ratio=min_area_ratio, max_side=max_side, debug=debug
    )
    if not regions:
        print(f"[未检测到照片] {path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix if prefix else path.stem
    ext = ".png" if fmt == "png" else ".jpg"
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, quality]
        if ext == ".jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    written = 0
    quads: List[np.ndarray] = []
    if debug:
        for i, (box, _) in enumerate(regions, 1):
            print(f"  区域{i}: x={box[0]} y={box[1]} w={box[2]} h={box[3]}")
    for box, contour in regions:
        crop, quad = _extract_photo(img, box, contour, scale)
        if crop.size == 0:
            continue
        quads.append(quad)
        out_path = output_dir / f"{stem}_{written + 1:02d}{ext}"
        if not imwrite_unicode(out_path, crop, params):
            print(f"[写入失败] {out_path}", file=sys.stderr)
            continue
        written += 1

    if debug and debug_img is not None:
        for quad in quads:
            pts = np.round(quad * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
        dbg_path = output_dir / f"{stem}_debug.jpg"
        imwrite_unicode(dbg_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  调试图: {dbg_path}")

    print(f"[完成] {path.name}: 检测到 {len(regions)} 张，已保存 {written} 个文件 → {output_dir}")
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动拆分 A4 扫描件中的多张照片（jpg/png → 单张图片文件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="输入图片或目录（可多个）")
    parser.add_argument("-o", "--output", default=None, help="输出目录（默认：输入文件旁 <文件名>_split/）")
    parser.add_argument("--format", choices=["png", "jpg"], default="png", help="输出格式")
    parser.add_argument("--quality", type=int, default=95, help="JPG 输出质量 (1-100)")
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.004,
        help="照片最小面积占整页比例，用于过滤小噪点",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1800,
        help="检测阶段的工作分辨率（长边像素），越大越精细但越慢",
    )
    parser.add_argument("--debug", action="store_true", help="输出带检测框的调试图")
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("没有可处理的图片。", file=sys.stderr)
        return 1

    if args.output:
        common_output = Path(args.output)
        use_common = True
    else:
        common_output = None
        use_common = False

    total = 0
    for path in files:
        out_dir = common_output if use_common else path.parent / f"{path.stem}_split"
        assert out_dir is not None
        total += process_one(
            path=path,
            output_dir=out_dir,
            fmt=args.format,
            quality=args.quality,
            min_area_ratio=args.min_area,
            max_side=args.max_side,
            debug=args.debug,
            prefix=path.stem if use_common else None,
        )

    print(f"\n共处理 {len(files)} 个文件，输出 {total} 张照片。")
    return 0 if total > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
