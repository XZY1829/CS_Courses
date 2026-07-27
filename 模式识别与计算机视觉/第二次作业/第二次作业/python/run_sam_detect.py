"""
使用 SAM (Segment Anything) + GPU 对图像做自动分割，
找出最大的矩形区域并标注角点。
"""
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import SAM


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_unicode(path, image):
    ext = Path(path).suffix.lower()
    ext = ext if ext in {".jpg", ".jpeg", ".png", ".bmp"} else ".png"
    ok, enc = cv2.imencode(ext, image)
    if ok:
        enc.tofile(str(path))


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def main():
    input_path = Path(r"D:\autotest\datas\opencv_debug.png")
    output_dir = Path(r"d:\WorkCode\D5Engine\.agentic\shape2d-debug\outputs\case-002")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    bgr = imread_unicode(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    h, w = bgr.shape[:2]
    img_area = h * w
    print(f"[INFO] Image: {input_path.name}  size={w}x{h}")

    # ── SAM 自动分割 ──
    print("[INFO] Loading SAM model (mobile_sam, first run downloads ~30MB)...")
    model = SAM("mobile_sam.pt")

    print("[INFO] Running SAM 'segment everything' ...")
    t0 = time.perf_counter()
    results = model(str(input_path), device=device, verbose=False)
    dt = time.perf_counter() - t0
    print(f"[INFO] SAM inference: {dt:.2f}s")

    masks = results[0].masks
    if masks is None or masks.data is None:
        print("[WARN] SAM returned no masks!")
        return

    mask_data = masks.data.cpu().numpy()
    n_masks = mask_data.shape[0]
    print(f"[INFO] SAM found {n_masks} segments")

    # ── 找最像矩形的区域 ──
    best_idx = -1
    best_rect = None
    best_score = 0
    margin = 10

    for i in range(n_masks):
        m = (mask_data[i] * 255).astype(np.uint8)
        area = m.sum() / 255
        if area < 0.05 * img_area or area > 0.90 * img_area:
            continue

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx <= margin and by <= margin and bx + bw >= w - margin and by + bh >= h - margin:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box.astype(np.int32))
        cnt_area = cv2.contourArea(cnt)

        rectangularity = cnt_area / (box_area + 1e-6)
        print(f"  Mask {i}: area={cnt_area:.0f}  rect={rectangularity:.3f}  bbox=({bx},{by},{bw},{bh})")
        if rectangularity > 0.85 and cnt_area > best_score:
            best_score = cnt_area
            best_idx = i
            best_rect = order_points(box.astype(np.float32))

    if best_rect is None:
        print("[WARN] No rectangular segment found, showing largest segment")
        areas = [mask_data[i].sum() for i in range(n_masks)]
        best_idx = int(np.argmax(areas))

    # ── 可视化 ──
    # 1. 所有 SAM 分割
    overlay_all = bgr.copy()
    for i in range(n_masks):
        color = np.random.randint(50, 255, 3).tolist()
        m = mask_data[i].astype(bool)
        overlay_all[m] = (0.5 * overlay_all[m] + 0.5 * np.array(color)).astype(np.uint8)
    imwrite_unicode(output_dir / "20_sam_all_segments.png", overlay_all)

    # 2. 最佳矩形区域
    vis_best = bgr.copy()
    best_mask = mask_data[best_idx].astype(bool)
    vis_best[~best_mask] = (vis_best[~best_mask] * 0.3).astype(np.uint8)
    vis_best[best_mask] = (0.7 * vis_best[best_mask] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)

    if best_rect is not None:
        pts_int = best_rect.astype(np.int32)
        cv2.polylines(vis_best, [pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts_int):
            cv2.circle(vis_best, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis_best, ["TL","TR","BR","BL"][j], (x+12, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"[INFO] Rectangle corners:")
        for j, label in enumerate(["TL","TR","BR","BL"]):
            print(f"       {label}: ({best_rect[j][0]:.1f}, {best_rect[j][1]:.1f})")
    imwrite_unicode(output_dir / "21_sam_best_rect.png", vis_best)

    # 3. 透视校正
    if best_rect is not None:
        side = int(max(
            np.linalg.norm(best_rect[1] - best_rect[0]),
            np.linalg.norm(best_rect[2] - best_rect[1]),
        ))
        dst = np.array([[0,0],[side,0],[side,side],[0,side]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(best_rect, dst)
        warped = cv2.warpPerspective(bgr, M, (side, side))
        imwrite_unicode(output_dir / "22_sam_warped.png", warped)

    # 4. Montage
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, [overlay_all, vis_best, warped if best_rect is not None else bgr],
                               ["SAM All Segments", "Best Rectangle", "Perspective-Corrected"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.suptitle(f"SAM Rectangle Detection  ({dt:.2f}s on {device.upper()})", fontsize=15, y=1.0)
    plt.tight_layout()
    plt.savefig(str(output_dir / "23_sam_montage.png"), dpi=150)
    plt.close()

    print(f"\n[DONE] Saved to: {output_dir}")
    for f in sorted(output_dir.glob("2*.png")):
        print(f"  -> {f.name}")


if __name__ == "__main__":
    main()
