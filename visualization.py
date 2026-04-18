"""
visualization.py — Visualisation helpers for QSM volumes.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """Zero-pad a 2-D image to a square."""
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def save_vis(
    cosmos: np.ndarray,
    chi: np.ndarray,
    mask: np.ndarray,
    out_path: str = "vis.png",
    show: bool = False,
    clim: tuple | None = None,
) -> None:
    """Save a 2-row × 3-col figure showing COSMOS and reconstructed χ.

    Row 1 — COSMOS (axial, coronal, sagittal centre slices)
    Row 2 — Reconstructed χ

    Parameters
    ----------
    cosmos   : (Z, Y, X) numpy array
    chi      : (Z, Y, X) numpy array
    mask     : (Z, Y, X) numpy array (bool or float)
    out_path : output file path (PNG)
    show     : call plt.show() after saving
    clim     : optional (vmin, vmax) for both rows; auto-computed if None
    """
    cosmos = np.squeeze(cosmos)
    chi = np.squeeze(chi)
    mask = np.squeeze(mask).astype(bool)

    cosmos = cosmos * mask
    chi = chi * mask

    cz = cosmos.shape[0] // 2
    cy = cosmos.shape[1] // 2
    cx = cosmos.shape[2] // 2

    fig, axs = plt.subplots(2, 3, figsize=(12, 9))

    vmin = clim[0] if clim else None
    vmax = clim[1] if clim else None

    # Row 1 — COSMOS
    axs[0, 0].imshow(_pad_to_square(cosmos[cz]), cmap="gray", vmin=vmin, vmax=vmax)
    axs[0, 1].imshow(_pad_to_square(cosmos[:, cy, :]), cmap="gray", vmin=vmin, vmax=vmax)
    axs[0, 2].imshow(_pad_to_square(cosmos[:, :, cx]), cmap="gray", vmin=vmin, vmax=vmax)

    # Row 2 — Reconstructed χ
    axs[1, 0].imshow(_pad_to_square(chi[cz]), cmap="gray", vmin=vmin, vmax=vmax)
    axs[1, 1].imshow(_pad_to_square(chi[:, cy, :]), cmap="gray", vmin=vmin, vmax=vmax)
    axs[1, 2].imshow(_pad_to_square(chi[:, :, cx]), cmap="gray", vmin=vmin, vmax=vmax)

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    offset_inches = 0.38
    y_row1 = 1 - (offset_inches / fig.get_size_inches()[1])
    y_row2 = 0.56 - (offset_inches / fig.get_size_inches()[1])

    fig.text(0.5, y_row1, "COSMOS", ha="center", va="bottom", fontsize=16)
    fig.text(
        0.5, y_row2, "Reconstructed Susceptibility",
        ha="center", va="bottom", fontsize=16,
    )

    plt.subplots_adjust(top=0.97, hspace=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    out_path: str = "loss_curves.png",
    show: bool = False,
) -> None:
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : list of per-epoch training losses
    val_losses   : list of per-epoch validation losses
    out_path     : output file path
    show         : call plt.show() after saving
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.semilogy(epochs, val_losses, label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_summary(
    results: dict,
    out_path: str = "metrics_summary.png",
    show: bool = False,
) -> None:
    """Bar chart of mean ± std for PSNR, SSIM, RMSE, HFEN, Corr.

    Parameters
    ----------
    results  : dict returned by testing.test()
    out_path : output file path
    show     : call plt.show() after saving
    """
    metrics = ["psnr", "ssim", "corr", "hfen", "rmse"]
    labels = ["PSNR (dB)", "SSIM", "Pearson r", "HFEN", "RMSE (%)"]
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Test-Set Metrics (mean ± std)", fontsize=14)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
