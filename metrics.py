"""
metrics.py — Image quality metrics: RMSE, SSIM, PSNR, HFEN, Pearson correlation.
All functions replicate MATLAB-equivalent behaviour.
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Gaussian kernel (used by SSIM)
# ---------------------------------------------------------------------------

def gkernel(sigma: float, sk: tuple) -> np.ndarray:
    """3-D Gaussian kernel.  sk is a tuple of half-sizes, e.g. (2,2,2) → 5×5×5."""
    x = np.arange(-sk[0], sk[0] + 1)
    y = np.arange(-sk[1], sk[1] + 1)
    z = np.arange(-sk[2], sk[2] + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    radius_sq = X ** 2 + Y ** 2 + Z ** 2
    gaussKernel = np.exp(-radius_sq / (2 * sigma ** 2))
    return gaussKernel


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def compute_rmse(chi_recon: np.ndarray, chi_true: np.ndarray) -> float:
    """RMSE = 100 × ‖chi_recon − chi_true‖ / ‖chi_true‖  (MATLAB convention)."""
    chi_recon = np.asarray(chi_recon, dtype=np.float64)
    chi_true = np.asarray(chi_true, dtype=np.float64)
    num = np.linalg.norm(chi_recon.ravel() - chi_true.ravel())
    den = np.linalg.norm(chi_true.ravel())
    if den < 1e-12:
        return 0.0
    return float(100.0 * num / den)


# Alias used in training pipeline
compute_rmse_matlab = compute_rmse


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    sw: tuple = (2, 2, 2),
    ind=None,
):
    """SSIM for 2-D or 3-D arrays (numpy), matching MATLAB behaviour.

    Parameters
    ----------
    img1, img2 : ndarray  — same shape
    sw         : half-size for Gaussian kernel, e.g. (2,2,2)
    ind        : optional indices tuple (e.g. np.where(mask)) for ROI

    Returns
    -------
    mssim    : float
    ssim_map : ndarray same shape as img1
    """
    if img1.shape != img2.shape:
        return -np.inf, -np.inf

    img1 = img1.astype(np.float64).copy()
    img2 = img2.astype(np.float64).copy()

    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] -= min_img
    img2[img2 != 0] -= min_img

    max_img = max(img1.max(), img2.max())
    if max_img == 0:
        return -np.inf, -np.inf

    img1 = 255.0 * img1 / max_img
    img2 = 255.0 * img2 / max_img

    if ind is None:
        ind = np.where(img1 != 0)

    window = gkernel(1.5, sw)
    window = window / np.sum(window)

    K1, K2 = 0.01, 0.03
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = convolve(img1, window, mode="reflect")
    mu2 = convolve(img2, window, mode="reflect")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window, mode="reflect") - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, mode="reflect") - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode="reflect") - mu1_mu2

    if (C1 > 0) and (C2 > 0):
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (
            denominator1[index] * denominator2[index]
        )
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    temp = np.zeros_like(img1)
    temp[ind] = 1
    iind = np.where(temp == 0)
    ssim_map[iind] = 1.0

    mssim = 0.0 if np.size(ind[0]) == 0 else float(np.mean(ssim_map[ind]))
    return mssim, ssim_map


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(chi_recon: np.ndarray, chi_true: np.ndarray) -> float:
    """PSNR matching MATLAB normalisation (0–255 rescale before MSE)."""
    img1 = chi_recon.astype(np.float64).copy()
    img2 = chi_true.astype(np.float64).copy()

    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] -= min_img
    img2[img2 != 0] -= min_img

    max_img = max(img1.max(), img2.max())
    if max_img == 0:
        return float("-inf")

    img1 = 255.0 * img1 / max_img
    img2 = 255.0 * img2 / max_img

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / mse))


# ---------------------------------------------------------------------------
# HFEN
# ---------------------------------------------------------------------------

def compute_hfen(
    img1: np.ndarray,
    img2: np.ndarray,
    filt_size: int = 15,
    sigma: float = 1.5,
) -> float:
    """HFEN via 3-D Laplacian-of-Gaussian filter (faithful MATLAB translation)."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    # Normalise to [0, 1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-12)
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-12)

    filt_size_arr = np.array([filt_size] * 3, dtype=int)
    sigma_arr = np.array([sigma] * 3, dtype=np.float64)

    siz = (filt_size_arr - 1) // 2
    x, y, z = np.mgrid[
        -siz[0]:siz[0] + 1,
        -siz[1]:siz[1] + 1,
        -siz[2]:siz[2] + 1,
    ]

    h = np.exp(
        -(
            x ** 2 / (2 * sigma_arr[0] ** 2)
            + y ** 2 / (2 * sigma_arr[1] ** 2)
            + z ** 2 / (2 * sigma_arr[2] ** 2)
        )
    )
    h /= np.sum(h)

    arg = (
        x ** 2 / sigma_arr[0] ** 4
        + y ** 2 / sigma_arr[1] ** 4
        + z ** 2 / sigma_arr[2] ** 4
        - (1 / sigma_arr[0] ** 2 + 1 / sigma_arr[1] ** 2 + 1 / sigma_arr[2] ** 2)
    )
    H = arg * h
    H -= np.sum(H) / np.prod(2 * siz + 1)

    img1_log = convolve(img1, H, mode="nearest")
    img2_log = convolve(img2, H, mode="nearest")
    return compute_rmse(img1_log, img2_log)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
):
    """Compute all metrics (PSNR, SSIM, Pearson r, HFEN, RMSE) inside the mask.

    Parameters
    ----------
    gt, pred, mask : numpy arrays with shape (Z, Y, X)

    Returns
    -------
    psnr (dB), ssim, corr (Pearson r), hfen, rmse (%)
    """
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask_bool = np.asarray(mask) > 0

    if gt.shape != pred.shape or gt.shape != mask_bool.shape:
        raise ValueError(
            f"Shape mismatch: gt {gt.shape}, pred {pred.shape}, mask {mask_bool.shape}"
        )

    if mask_bool.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    gt_vol = gt.copy()
    pred_vol = pred.copy()
    gt_vol[~mask_bool] = 0.0
    pred_vol[~mask_bool] = 0.0

    psnr_val = compute_psnr(pred_vol, gt_vol)

    ind = np.where(mask_bool)
    try:
        ssim_val, _ = compute_ssim(gt_vol, pred_vol, sw=(2, 2, 2), ind=ind)
    except Exception:
        ssim_val = 0.0

    gt_masked = gt[mask_bool]
    pred_masked = pred[mask_bool]
    if np.std(gt_masked) < 1e-8 or np.std(pred_masked) < 1e-8:
        corr = 0.0
    else:
        corr = float(pearsonr(gt_masked.ravel(), pred_masked.ravel())[0])

    try:
        hfen_val = compute_hfen(gt_vol, pred_vol)
    except Exception:
        hfen_val = 0.0

    rmse_val = compute_rmse_matlab(pred_vol, gt_vol)

    return float(psnr_val), float(ssim_val), float(corr), float(hfen_val), float(rmse_val)
