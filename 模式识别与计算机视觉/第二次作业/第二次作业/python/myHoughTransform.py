import numpy as np


def myHoughTransform(img_threshold, rhoRes, thetaRes):
    """
    Compute the Hough accumulator for a thresholded edge image.

    Parameters
    ----------
    img_threshold : np.ndarray
        Binary/thresholded edge magnitude image; non-zero pixels vote.
    rhoRes : float
        Rho resolution in pixels.
    thetaRes : float
        Theta resolution in radians.

    Returns
    -------
    img_hough : np.ndarray
        Accumulator with shape (len(rhoScale), len(thetaScale)).
    rhoScale : np.ndarray
        Discretized rho values in [0, max_rho].
    thetaScale : np.ndarray
        Discretized theta values in [0, 2*pi).
    """
    if img_threshold.ndim != 2:
        raise ValueError("img_threshold must be a single-channel image.")
    if rhoRes <= 0 or thetaRes <= 0:
        raise ValueError("rhoRes and thetaRes must be positive.")

    height, width = img_threshold.shape
    max_rho = float(np.hypot(width, height))
    rhoScale = np.arange(0.0, max_rho + rhoRes, rhoRes, dtype=np.float64)
    thetaScale = np.arange(0.0, 2.0 * np.pi, thetaRes, dtype=np.float64)

    img_hough = np.zeros((rhoScale.size, thetaScale.size), dtype=np.float64)

    ys, xs = np.nonzero(img_threshold > 0)
    if xs.size == 0:
        return img_hough, rhoScale, thetaScale

    # Convert to 1-based coordinates to align with assignment notation.
    xs = xs.astype(np.float64) + 1.0
    ys = ys.astype(np.float64) + 1.0

    cos_t = np.cos(thetaScale)
    sin_t = np.sin(thetaScale)
    inv_rho_res = 1.0 / rhoRes

    for x, y in zip(xs, ys):
        rhos = x * cos_t + y * sin_t
        valid_mask = (rhos >= 0.0) & (rhos <= max_rho)
        valid_t_idx = np.nonzero(valid_mask)[0]
        if valid_t_idx.size == 0:
            continue
        rho_idx = np.round(rhos[valid_mask] * inv_rho_res).astype(np.int64)
        rho_idx = np.clip(rho_idx, 0, rhoScale.size - 1)
        np.add.at(img_hough, (rho_idx, valid_t_idx), 1.0)

    return img_hough, rhoScale, thetaScale
