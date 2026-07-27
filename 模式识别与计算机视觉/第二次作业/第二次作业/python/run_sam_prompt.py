"""
SAM prompt 模式：给一个点提示，只跑 1 次推理。
对比 segment-everything 模式的速度差异。
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
    print(f"[INFO] Device: {device}")

    bgr = imread_unicode(input_path)
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"[INFO] Image: {w}x{h},  point prompt: ({cx}, {cy})")

    model = SAM("mobile_sam.pt")

    # ── warmup ──
    print("[INFO] Warmup...")
    _ = model(str(input_path), points=[[cx, cy]], labels=[1], device=device, verbose=False)

    # ── 计时：prompt 模式（1 个点）──
    print("[INFO] Running prompt mode (1 point)...")
    t0 = time.perf_counter()
    results = model(str(input_path), points=[[cx, cy]], labels=[1], device=device, verbose=False)
    dt_prompt = time.perf_counter() - t0
    print(f"[INFO] Prompt mode: {dt_prompt:.3f}s")

    # ── 计时：segment everything ──
    print("[INFO] Running segment-everything mode...")
    t0 = time.perf_counter()
    results_all = model(str(input_path), device=device, verbose=False)
    dt_all = time.perf_counter() - t0
    print(f"[INFO] Everything mode: {dt_all:.2f}s")

    speedup = dt_all / dt_prompt if dt_prompt > 0 else float("inf")
    print(f"[INFO] Speedup: {speedup:.0f}x faster with prompt")

    # ── prompt 结果处理 ──
    masks = results[0].masks
    if masks is None:
        print("[WARN] No mask returned")
        return

    mask_data = masks.data.cpu().numpy()
    best_mask = mask_data[0]
    m = (best_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered = order_points(box)
    cnt_area = cv2.contourArea(cnt)
    box_area = cv2.contourArea(box.astype(np.int32))
    rectangularity = cnt_area / (box_area + 1e-6)

    print(f"[INFO] Rectangularity: {rectangularity:.3f}")
    print(f"[INFO] Corners:")
    for j, label in enumerate(["TL","TR","BR","BL"]):
        print(f"       {label}: ({ordered[j][0]:.1f}, {ordered[j][1]:.1f})")

    # ── 可视化 ──
    vis = bgr.copy()
    vis[~best_mask.astype(bool)] = (vis[~best_mask.astype(bool)] * 0.3).astype(np.uint8)
    vis[best_mask.astype(bool)] = (0.7 * vis[best_mask.astype(bool)] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    pts_int = ordered.astype(np.int32)
    cv2.polylines(vis, [pts_int], True, (0, 255, 0), 3, cv2.LINE_AA)
    for j, (x, y) in enumerate(pts_int):
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), 10, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "prompt", (cx + 14, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    imwrite_unicode(output_dir / "30_sam_prompt_result.png", vis)

    # ── 速度对比图 ──
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = ["Prompt\n(1 point)", "Everything\n(auto grid)"]
    times = [dt_prompt, dt_all]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.barh(methods, times, color=colors, height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_title(f"SAM Inference Speed Comparison  ({speedup:.0f}x faster)", fontsize=14)
    ax.set_xlim(0, max(times) * 1.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "31_speed_comparison.png"), dpi=150)
    plt.close()

    print(f"\n[DONE] -> 30_sam_prompt_result.png, 31_speed_comparison.png")


if __name__ == "__main__":
    main()
