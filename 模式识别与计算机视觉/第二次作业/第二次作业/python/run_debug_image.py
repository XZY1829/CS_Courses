"""
对指定图片同时执行 Harris 角点检测、Hough 直线检测和 Canny 边缘候选点可视化，
结果输出到指定目录。
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harris_cornerScript import (
    compute_harris_response,
    detect_multiscale_harris,
    imread_unicode,
    imwrite_unicode,
)
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines


def draw_hough_line(image, rho, theta, color=(0, 0, 255), thickness=2):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
    pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
    cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def save_accumulator(accumulator, rho_scale, theta_scale, save_path):
    plt.figure(figsize=(10, 5))
    plt.imshow(
        accumulator, cmap="inferno", aspect="auto", origin="lower",
        extent=[theta_scale[0], theta_scale[-1], rho_scale[0], rho_scale[-1]],
    )
    plt.colorbar(label="Votes")
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\rho$ (pixel)")
    plt.title("Hough Accumulator")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=180)
    plt.close()


def main():
    input_path = Path(r"D:\autotest\datas\opencv_debug.png")
    output_dir = Path(r"d:\WorkCode\D5Engine\.agentic\shape2d-debug\outputs\case-002")
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = imread_unicode(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"[INFO] Image: {input_path.name}  size={w}x{h}")

    smooth = cv2.GaussianBlur(gray, (9, 9), 3.0)

    # ── 1. Canny 边缘（候选边缘点）──────────────────────────
    edges = cv2.Canny(smooth, 15, 30)
    edge_pts = np.nonzero(edges)
    print(f"[INFO] Canny edge points: {len(edge_pts[0])}")

    edge_vis_black = np.zeros_like(bgr)
    edge_vis_black[edges > 0] = (0, 255, 255)
    imwrite_unicode(output_dir / "01_canny_edges.png", edges)

    edge_vis = bgr.copy()
    edge_mask = edges > 0
    edge_vis[edge_mask] = (0, 255, 255)
    imwrite_unicode(output_dir / "02_edge_overlay.png", edge_vis)
    imwrite_unicode(output_dir / "02b_edge_yellow_on_black.png", edge_vis_black)

    # ── 2. Harris 角点检测 ──────────────────────────────────
    scales = [2.0, 3.0, 4.0, 5.0]
    corners = detect_multiscale_harris(
        smooth, scales=scales, k=0.04, threshold_ratio=0.02,
    )
    print(f"[INFO] Harris corners: {len(corners)}")

    corner_vis = bgr.copy()
    for _, x, y, sigma in corners:
        radius = max(4, int(round(2.5 * sigma)))
        cv2.circle(corner_vis, (x, y), radius, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(corner_vis, (x, y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    imwrite_unicode(output_dir / "03_harris_corners.png", corner_vis)

    # ── 3. Hough 直线检测 ───────────────────────────────────
    rho_res = 1.0
    theta_res = np.deg2rad(1.0)
    hough_acc, rho_scale, theta_scale = myHoughTransform(edges, rho_res, theta_res)
    rho_idx, theta_idx = myHoughLines(hough_acc, nLines=12, nms_window_size=21)
    print(f"[INFO] Hough lines: {len(rho_idx)}")

    line_vis = bgr.copy()
    colors = [(255, 0, 0), (0, 180, 0), (0, 128, 255), (255, 0, 255)]
    for i, (ri, ti) in enumerate(zip(rho_idx, theta_idx)):
        c = colors[i % len(colors)]
        draw_hough_line(line_vis, rho_scale[ri], theta_scale[ti], color=c, thickness=2)
    imwrite_unicode(output_dir / "04_hough_lines.png", line_vis)

    save_accumulator(hough_acc, rho_scale, theta_scale, output_dir / "05_hough_accumulator.png")

    # ── 4. 综合可视化 ──────────────────────────────────────
    combined = bgr.copy()

    combined[edges > 0] = (0, 255, 255)

    for i, (ri, ti) in enumerate(zip(rho_idx, theta_idx)):
        c = colors[i % len(colors)]
        draw_hough_line(combined, rho_scale[ri], theta_scale[ti], color=c, thickness=2)

    for _, x, y, sigma in corners:
        radius = max(4, int(round(2.5 * sigma)))
        cv2.circle(combined, (x, y), radius, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(combined, (x, y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    imwrite_unicode(output_dir / "06_combined.png", combined)

    # ── 5. Matplotlib 分区对比图 ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    titles = [
        "Original", "Canny Edges", "Edge Overlay (yellow)",
        "Harris Corners (red)", "Hough Lines (blue)", "Combined"
    ]
    images = [bgr, edges, edge_vis, corner_vis, line_vis, combined]
    for ax, img, title in zip(axes.flat, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    plt.suptitle(f"Corner + Line + Edge Detection: {input_path.name}", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_dir / "07_montage.png"), dpi=150)
    plt.close()

    print(f"\n[DONE] All outputs saved to: {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  -> {f.name}")


if __name__ == "__main__":
    main()
