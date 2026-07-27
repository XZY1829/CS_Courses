"""
矩形检测：基于颜色分割 + 轮廓逼近的方法，
比 Harris+Hough 组合对"找矩形"这个任务更加精准。
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    suffix = Path(path).suffix.lower()
    ext = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp"} else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def order_points(pts):
    """将 4 个点按 [左上, 右上, 右下, 左下] 排序。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def detect_rectangle(bgr):
    """
    返回检测到的最大矩形的 4 个角点坐标（按顺序），
    以及中间过程的 debug 图像。
    """
    h, w = bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── 方法 1：灰度自适应阈值 ──
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=5,
    )

    # ── 方法 2：HSV 饱和度分割（灰色物体饱和度低）──
    sat = hsv[:, :, 1]
    _, thresh_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 方法 3：Otsu 全局阈值 ──
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_rect = None
    best_score = 0
    best_method = ""
    debug_masks = {}

    margin = 5

    for name, mask in [("adaptive", thresh_adapt), ("saturation", thresh_sat), ("otsu", thresh_otsu)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        debug_masks[name] = cleaned

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.05 * img_area or area > 0.95 * img_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if 4 <= len(approx) <= 6:
                rect_candidate = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect_candidate)
                pts = box.astype(np.float32)

                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bx <= margin and by <= margin and bx + bw >= w - margin and by + bh >= h - margin:
                    continue

                ordered = order_points(pts)
                w1 = np.linalg.norm(ordered[1] - ordered[0])
                h1 = np.linalg.norm(ordered[3] - ordered[0])
                aspect = max(w1, h1) / (min(w1, h1) + 1e-6)
                if aspect < 3.0:
                    score = area
                    if score > best_score:
                        best_score = score
                        best_rect = ordered
                        best_method = name

    return best_rect, best_method, debug_masks


def main():
    input_path = Path(r"D:\autotest\datas\opencv_debug.png")
    output_dir = Path(r"d:\WorkCode\D5Engine\.agentic\shape2d-debug\outputs\case-002")
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = imread_unicode(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    h, w = bgr.shape[:2]
    print(f"[INFO] Image: {input_path.name}  size={w}x{h}")

    rect, method, debug_masks = detect_rectangle(bgr)

    if rect is None:
        print("[WARN] No rectangle detected!")
        return

    print(f"[INFO] Rectangle found via '{method}' method")
    print(f"[INFO] Corners (TL, TR, BR, BL):")
    for i, label in enumerate(["TL", "TR", "BR", "BL"]):
        print(f"       {label}: ({rect[i][0]:.1f}, {rect[i][1]:.1f})")

    # ── 可视化 1：检测到的矩形 ──
    vis_rect = bgr.copy()
    pts_int = rect.astype(np.int32)
    cv2.polylines(vis_rect, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    for i, (x, y) in enumerate(pts_int):
        cv2.circle(vis_rect, (x, y), 8, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis_rect, ["TL","TR","BR","BL"][i], (x+12, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    imwrite_unicode(output_dir / "10_rect_detected.png", vis_rect)

    # ── 可视化 2：各方法的分割掩码 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original", fontsize=13)

    for ax, (name, mask) in zip(axes.flat[1:4], debug_masks.items()):
        ax.imshow(mask, cmap="gray")
        marker = " *" if name == method else ""
        ax.set_title(f"Mask: {name}{marker}", fontsize=13)

    axes[1, 1].imshow(cv2.cvtColor(vis_rect, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Detected Rectangle", fontsize=13)

    # 透视校正
    side = int(max(
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[1]),
        np.linalg.norm(rect[3] - rect[2]),
        np.linalg.norm(rect[0] - rect[3]),
    ))
    dst_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(bgr, M, (side, side))
    axes[1, 2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Perspective-Corrected", fontsize=13)
    imwrite_unicode(output_dir / "12_rect_warped.png", warped)

    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle(f"Rectangle Detection ({method} method)", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_dir / "11_rect_montage.png"), dpi=150)
    plt.close()

    print(f"\n[DONE] Saved to: {output_dir}")
    for f in sorted(output_dir.glob("1*.png")):
        print(f"  -> {f.name}")


if __name__ == "__main__":
    main()
