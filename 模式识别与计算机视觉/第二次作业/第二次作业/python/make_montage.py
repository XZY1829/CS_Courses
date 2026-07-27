from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = Path(path).suffix.lower()
    ok, encoded = cv2.imencode(ext if ext else ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def create_grid(image_paths, output_path, title):
    loaded = []
    labels = []
    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            continue
        loaded.append(img)
        labels.append(path.stem.replace("_hough_lines", "").replace("_harris_multiscale", ""))

    if not loaded:
        return

    cell_w, cell_h = 420, 280
    cols = 2
    rows = int(np.ceil(len(loaded) / cols))
    canvas = np.full((rows * cell_h + 60, cols * cell_w, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)

    for idx, (img, label) in enumerate(zip(loaded, labels)):
        r = idx // cols
        c = idx % cols
        y0 = 60 + r * cell_h
        x0 = c * cell_w

        resized = cv2.resize(img, (cell_w, cell_h - 35), interpolation=cv2.INTER_AREA)
        h_resized, w_resized = resized.shape[:2]
        canvas[y0 + 30 : y0 + 30 + h_resized, x0 : x0 + w_resized] = resized
        cv2.rectangle(canvas, (x0, y0 + 30), (x0 + w_resized - 1, y0 + 30 + h_resized - 1), (160, 160, 160), 1)
        cv2.putText(canvas, label, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    imwrite_unicode(output_path, canvas)
    print(f"[INFO] Saved: {output_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    result_dir = script_dir.parent / "results"

    hough_paths = sorted(result_dir.glob("img*_hough_lines.png"))
    harris_paths = sorted(result_dir.glob("img*_harris_multiscale.png"))

    create_grid(hough_paths, result_dir / "hough_all_grid.png", "All images - My Hough lines (red) + OpenCV segments (green)")
    create_grid(harris_paths, result_dir / "harris_all_grid.png", "All images - Multi-scale Harris corners")


if __name__ == "__main__":
    main()
