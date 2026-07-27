from pathlib import Path

import cv2
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


def compute_harris_response(gray_float, sigma, k=0.04):
    blurred = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ix = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    sxx = cv2.GaussianBlur(ix * ix, (0, 0), sigmaX=1.5 * sigma, sigmaY=1.5 * sigma)
    syy = cv2.GaussianBlur(iy * iy, (0, 0), sigmaX=1.5 * sigma, sigmaY=1.5 * sigma)
    sxy = cv2.GaussianBlur(ix * iy, (0, 0), sigmaX=1.5 * sigma, sigmaY=1.5 * sigma)

    det_m = sxx * syy - sxy * sxy
    trace_m = sxx + syy
    response = det_m - k * (trace_m ** 2)
    return response


def detect_multiscale_harris(gray, scales, k=0.04, threshold_ratio=0.01):
    gray_float = gray.astype(np.float64) / 255.0
    h, w = gray.shape

    candidates = []
    for sigma in scales:
        response = compute_harris_response(gray_float, sigma=sigma, k=k)
        response32 = response.astype(np.float32)
        max_r = float(response32.max()) if response32.size > 0 else 0.0
        if max_r <= 0:
            continue

        thresh = threshold_ratio * max_r
        window = max(3, int(2 * round(2.5 * sigma) + 1))
        if window % 2 == 0:
            window += 1
        local_max = cv2.dilate(response32, np.ones((window, window), np.uint8))
        mask = (response32 == local_max) & (response32 > thresh)

        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            candidates.append((float(response32[y, x]), int(x), int(y), float(sigma)))

    if not candidates:
        return []

    # Cross-scale suppression: keep stronger points and suppress nearby weaker ones.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, x, y, sigma in candidates:
        keep = True
        for _, sx, sy, ss in selected:
            distance = np.hypot(x - sx, y - sy)
            radius = 1.6 * min(sigma, ss)
            if distance < radius:
                keep = False
                break
        if keep:
            selected.append((score, x, y, sigma))
        if len(selected) >= 250:
            break
    return selected


def draw_corners(image_bgr, corners):
    out = image_bgr.copy()
    for _, x, y, sigma in corners:
        radius = max(2, int(round(2.5 * sigma)))
        cv2.circle(out, (x, y), radius, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    return out


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    result_dir = script_dir.parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    scales = [1.0, 1.6, 2.2, 3.0, 4.0]

    for image_path in image_paths:
        bgr = imread_unicode(image_path)
        if bgr is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = detect_multiscale_harris(
            gray,
            scales=scales,
            k=0.04,
            threshold_ratio=0.01,
        )
        vis = draw_corners(bgr, corners)

        save_path = result_dir / f"{image_path.stem}_harris_multiscale.png"
        imwrite_unicode(save_path, vis)
        print(f"[INFO] Saved: {save_path} (corners={len(corners)})")


if __name__ == "__main__":
    main()
