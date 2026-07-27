import cv2
import numpy as np


def myHoughLines(img_hough, nLines, nms_window_size=9):
    """
    Extract top line peaks from a Hough accumulator using non-max suppression.

    Parameters
    ----------
    img_hough : np.ndarray
        Hough accumulator array.
    nLines : int
        Number of peaks to return.
    nms_window_size : int
        Local neighborhood size for non-max suppression.

    Returns
    -------
    rhos : np.ndarray
        Peak row indices in accumulator.
    thetas : np.ndarray
        Peak column indices in accumulator.
    """
    if img_hough.ndim != 2:
        raise ValueError("img_hough must be a 2D array.")
    if nLines <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    accumulator = img_hough.astype(np.float32, copy=False)
    window = max(3, int(nms_window_size))
    if window % 2 == 0:
        window += 1

    kernel = np.ones((window, window), dtype=np.uint8)
    local_max = cv2.dilate(accumulator, kernel)
    peak_mask = (accumulator == local_max) & (accumulator > 0)

    peak_coords = np.argwhere(peak_mask)
    if peak_coords.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    peak_values = accumulator[peak_mask]
    order = np.argsort(-peak_values)
    top_coords = peak_coords[order[:nLines]]

    rhos = top_coords[:, 0].astype(np.int64)
    thetas = top_coords[:, 1].astype(np.int64)
    return rhos, thetas
