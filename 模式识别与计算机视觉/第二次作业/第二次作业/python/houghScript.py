from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from myHoughLines import myHoughLines
from myHoughTransform import myHoughTransform


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


def draw_hough_line(image, rho, theta, color=(0, 0, 255), thickness=2):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
    pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
    cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def save_hough_accumulator(accumulator, rho_scale, theta_scale, save_path):
    plt.figure(figsize=(9, 5))
    plt.imshow(
        accumulator,
        cmap="inferno",
        aspect="auto",
        origin="lower",
        extent=[theta_scale[0], theta_scale[-1], rho_scale[0], rho_scale[-1]],
    )
    plt.colorbar(label="Votes")
    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\rho$ (pixel)")
    plt.title("Hough Accumulator")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    result_dir = script_dir.parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    # Global parameters (used for all images).
    rho_res = 1.0
    theta_res = np.deg2rad(1.0)
    canny_low = 70
    canny_high = 180
    hough_line_count = 12

    for idx, image_path in enumerate(image_paths):
        bgr = imread_unicode(image_path)
        if bgr is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)

        img_hough, rho_scale, theta_scale = myHoughTransform(edges, rho_res, theta_res)
        rho_idx, theta_idx = myHoughLines(img_hough, hough_line_count, nms_window_size=11)

        out = bgr.copy()
        for r_i, t_i in zip(rho_idx, theta_idx):
            draw_hough_line(out, rho_scale[r_i], theta_scale[t_i], color=(0, 0, 255), thickness=2)

        # OpenCV result for comparison only.
        line_segments = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=80,
            minLineLength=40,
            maxLineGap=8,
        )
        if line_segments is not None:
            for seg in line_segments[:, 0]:
                x1, y1, x2, y2 = seg
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        stem = image_path.stem
        edge_path = result_dir / f"{stem}_edge.png"
        line_path = result_dir / f"{stem}_hough_lines.png"

        imwrite_unicode(edge_path, edges)
        imwrite_unicode(line_path, out)
        print(f"[INFO] Saved: {line_path}")

        # Save one accumulator map for report Q3.3
        if idx == 0:
            acc_path = result_dir / "img01_hough_accumulator.png"
            save_hough_accumulator(img_hough, rho_scale, theta_scale, acc_path)
            print(f"[INFO] Saved: {acc_path}")


if __name__ == "__main__":
    main()
