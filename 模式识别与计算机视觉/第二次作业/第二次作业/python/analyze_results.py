from pathlib import Path

import cv2
import numpy as np

from harris_cornerScript import detect_multiscale_harris
from myHoughTransform import myHoughTransform


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    image_paths = sorted(data_dir.glob("*.jpg"))

    rho_res = 1.0
    theta_res = np.deg2rad(1.0)
    scales = [1.0, 1.6, 2.2, 3.0, 4.0]

    print("image,edge_pixels,max_hough_vote,harris_corners,opencv_segments")
    for image_path in image_paths:
        bgr = imread_unicode(image_path)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 180)
        img_hough, _, _ = myHoughTransform(edges, rho_res, theta_res)
        max_vote = int(np.max(img_hough)) if img_hough.size > 0 else 0
        corners = detect_multiscale_harris(gray, scales=scales, k=0.04, threshold_ratio=0.01)
        segs = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi / 180, threshold=80, minLineLength=40, maxLineGap=8)
        seg_count = 0 if segs is None else int(len(segs))

        print(
            f"{image_path.name},{int(np.count_nonzero(edges))},"
            f"{max_vote},{len(corners)},{seg_count}"
        )


if __name__ == "__main__":
    main()
