"""
统一跑三种方法：传统CV (Harris+Hough+Canny)、OpenCV轮廓法、SAM prompt。
用法: python run_all_methods.py <input_image> <output_dir>
"""
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harris_cornerScript import detect_multiscale_harris, imread_unicode, imwrite_unicode
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines


def draw_hough_line(image, rho, theta, color, thickness=2):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
    pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
    cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


# ═══════════════════════════════════════════════════════════════
# 方法 1: 传统 CV (Harris + Hough + Canny)
# ═══════════════════════════════════════════════════════════════
def run_traditional_cv(bgr, gray, output_dir):
    print("\n" + "=" * 50)
    print("[Method 1] Traditional CV: Harris + Hough + Canny")
    print("=" * 50)
    h, w = gray.shape

    smooth = cv2.GaussianBlur(gray, (7, 7), 2.0)
    edges = cv2.Canny(smooth, 30, 90)
    n_edges = np.count_nonzero(edges)
    if n_edges < 50:
        edges = cv2.Canny(smooth, 15, 45)
        n_edges = np.count_nonzero(edges)
    print(f"  Canny edges: {n_edges}")

    edge_vis = bgr.copy()
    edge_vis[edges > 0] = (0, 255, 255)
    imwrite_unicode(output_dir / "01_canny_edges.png", edges)
    imwrite_unicode(output_dir / "02_edge_overlay.png", edge_vis)

    scales = [2.0, 3.0, 4.0, 5.0]
    corners = detect_multiscale_harris(smooth, scales=scales, k=0.04, threshold_ratio=0.01)
    print(f"  Harris corners: {len(corners)}")

    corner_vis = bgr.copy()
    for _, x, y, sigma in corners:
        r = max(3, int(round(2.5 * sigma)))
        cv2.circle(corner_vis, (x, y), r, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(corner_vis, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
    imwrite_unicode(output_dir / "03_harris_corners.png", corner_vis)

    rho_res, theta_res = 1.0, np.deg2rad(1.0)
    hough_acc, rho_scale, theta_scale = myHoughTransform(edges, rho_res, theta_res)
    rho_idx, theta_idx = myHoughLines(hough_acc, nLines=10, nms_window_size=21)
    print(f"  Hough lines: {len(rho_idx)}")

    colors = [(255,0,0),(0,180,0),(0,128,255),(255,0,255)]
    line_vis = bgr.copy()
    for i, (ri, ti) in enumerate(zip(rho_idx, theta_idx)):
        draw_hough_line(line_vis, rho_scale[ri], theta_scale[ti], colors[i % len(colors)])
    imwrite_unicode(output_dir / "04_hough_lines.png", line_vis)

    combined = bgr.copy()
    combined[edges > 0] = (0, 255, 255)
    for i, (ri, ti) in enumerate(zip(rho_idx, theta_idx)):
        draw_hough_line(combined, rho_scale[ri], theta_scale[ti], colors[i % len(colors)])
    for _, x, y, sigma in corners:
        r = max(3, int(round(2.5 * sigma)))
        cv2.circle(combined, (x, y), r, (0, 0, 255), 2, cv2.LINE_AA)
    imwrite_unicode(output_dir / "05_combined_cv.png", combined)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [
        (bgr, "Original"), (edges, "Canny Edges"), (edge_vis, "Edge Overlay"),
        (corner_vis, "Harris Corners"), (line_vis, "Hough Lines"), (combined, "Combined"),
    ]
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=13); ax.axis("off")
    plt.suptitle("Method 1: Traditional CV", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_dir / "06_cv_montage.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 方法 2: OpenCV 轮廓法（多策略分割）
# ═══════════════════════════════════════════════════════════════
def run_contour_method(bgr, gray, output_dir):
    print("\n" + "=" * 50)
    print("[Method 2] OpenCV Contour (HSV + Adaptive + Otsu)")
    print("=" * 50)
    h, w = bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    sat = hsv[:, :, 1]
    _, thresh_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    all_candidates = []
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
            if area < 0.02 * img_area or area > 0.95 * img_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bx <= margin and by <= margin and bx+bw >= w-margin and by+bh >= h-margin:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n_verts = len(approx)

            rect_c = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect_c).astype(np.float32)
            box_area = cv2.contourArea(box.astype(np.int32))
            rectangularity = area / (box_area + 1e-6)

            all_candidates.append({
                "method": name, "contour": cnt, "approx": approx, "box": box,
                "area": area, "n_verts": n_verts, "rectangularity": rectangularity,
            })
            print(f"  [{name}] verts={n_verts} area={area:.0f} rect={rectangularity:.3f}")

    vis_contour = bgr.copy()
    if all_candidates:
        all_candidates.sort(key=lambda c: c["area"], reverse=True)
        best = all_candidates[0]
        cv2.drawContours(vis_contour, [best["contour"]], -1, (0, 255, 0), 3, cv2.LINE_AA)
        ordered = order_points(best["box"])
        pts_int = ordered.astype(np.int32)
        cv2.polylines(vis_contour, [pts_int], True, (0, 255, 255), 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts_int):
            cv2.circle(vis_contour, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis_contour, ["TL","TR","BR","BL"][j], (x+12, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"  Best: {best['method']} verts={best['n_verts']} rect={best['rectangularity']:.3f}")
    else:
        print("  [WARN] No shape detected")

    imwrite_unicode(output_dir / "10_contour_detected.png", vis_contour)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [(bgr, "Original")]
    for name, mask in debug_masks.items():
        panels.append((mask, f"Mask: {name}"))
    panels.append((vis_contour, "Detected Shape"))
    panels.append((bgr, ""))
    for ax, (img, title) in zip(axes.flat, panels[:6]):
        if title:
            ax.imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                      cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=13)
        ax.axis("off")
    plt.suptitle("Method 2: Contour Detection", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_dir / "11_contour_montage.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 方法 3: SAM prompt 模式（GPU）
# ═══════════════════════════════════════════════════════════════
def run_sam_prompt(bgr, input_path, output_dir):
    print("\n" + "=" * 50)
    print("[Method 3] SAM Prompt Mode (MobileSAM + GPU)")
    print("=" * 50)
    import torch
    from ultralytics import SAM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    model = SAM("mobile_sam.pt")

    _ = model(str(input_path), points=[[cx, cy]], labels=[1], device=device, verbose=False)

    t0 = time.perf_counter()
    results = model(str(input_path), points=[[cx, cy]], labels=[1], device=device, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  Prompt inference: {dt:.3f}s")

    masks = results[0].masks
    if masks is None:
        print("  [WARN] No mask returned")
        return

    mask_data = masks.data.cpu().numpy()
    best_mask = mask_data[0].astype(bool)
    n_pixels = best_mask.sum()
    print(f"  Mask pixels: {n_pixels} ({100*n_pixels/(h*w):.1f}%)")

    m_uint8 = (mask_data[0] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
    print(f"  Contour vertices: {len(approx)}")

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    cnt_area = cv2.contourArea(cnt)
    box_area = cv2.contourArea(box.astype(np.int32))
    print(f"  Rectangularity: {cnt_area / (box_area + 1e-6):.3f}")

    vis = bgr.copy()
    vis[~best_mask] = (vis[~best_mask] * 0.3).astype(np.uint8)
    vis[best_mask] = (0.7 * vis[best_mask] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 3, cv2.LINE_AA)
    if len(approx) <= 8:
        for pt in approx:
            x, y = pt[0]
            cv2.circle(vis, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), 10, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "prompt", (cx+14, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    imwrite_unicode(output_dir / "20_sam_prompt.png", vis)

    imwrite_unicode(output_dir / "21_sam_mask.png", m_uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(m_uint8, cmap="gray"); axes[1].set_title("SAM Mask")
    axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); axes[2].set_title(f"SAM Result ({dt:.2f}s)")
    for ax in axes: ax.axis("off")
    plt.suptitle("Method 3: SAM Prompt", fontsize=15, y=1.0)
    plt.tight_layout()
    plt.savefig(str(output_dir / "22_sam_montage.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) >= 3:
        input_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    else:
        input_path = Path(r"D:\autotest\datas\opencv_debug1.png")
        output_dir = Path(r"d:\WorkCode\D5Engine\.agentic\shape2d-debug\outputs\case-003")

    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = imread_unicode(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"[INFO] Image: {input_path.name}  size={w}x{h}")

    run_traditional_cv(bgr, gray, output_dir)
    run_contour_method(bgr, gray, output_dir)
    run_sam_prompt(bgr, input_path, output_dir)

    print(f"\n{'=' * 50}")
    print(f"[DONE] All outputs -> {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  -> {f.name}")


if __name__ == "__main__":
    main()
