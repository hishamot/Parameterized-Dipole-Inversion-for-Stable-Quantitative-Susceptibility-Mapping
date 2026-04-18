"""
testing.py — Testing / evaluation loop with QSM quality metrics.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import QSM_Dataset
from model import UNet
from loss import predict_chi
from metrics import compute_metrics
from visualization import save_vis


# ---------------------------------------------------------------------------
# FLOPs estimator
# ---------------------------------------------------------------------------

def compute_reconstruction_flops(volume_shape: tuple) -> int:
    """Estimate FLOPs: forward FFT + element-wise multiply + inverse FFT."""
    N = volume_shape[0] * volume_shape[1] * volume_shape[2]
    fft_flops = 5 * N * np.log2(N)
    mult_flops = N
    return int(2 * fft_flops + mult_flops)


# ---------------------------------------------------------------------------
# Test loop
# ---------------------------------------------------------------------------

def test(
    test_csv: str,
    root: str,
    model_path: str,
    dinv_path: str,
    device: str = "cuda",
    n_vis: int = 30,
    vis_dir: str = ".",
) -> dict:
    """Evaluate the trained model on the test set.

    Parameters
    ----------
    test_csv   : path to test CSV
    root       : data root directory
    model_path : path to saved model weights (.pt)
    dinv_path  : path to saved D_inv tensor (.pt)
    device     : 'cuda' or 'cpu'
    n_vis      : save visualisations for the first n_vis samples
    vis_dir    : directory for visualisation images

    Returns
    -------
    results : dict with lists and means/stds for each metric
    """
    test_ds = QSM_Dataset(test_csv, root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    net = UNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    best_D_inv = torch.load(dinv_path, map_location=device)

    psnr_list, ssim_list, corr_list, hfen_list, rmse_list = [], [], [], [], []
    recon_times = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            phase = batch["phase"].to(device)
            mask = batch["mask"].to(device)
            cosmos = batch["cosmos"].to(device)

            phi = phase.unsqueeze(1)

            # Timed reconstruction
            start = time.time()
            chi_torch = predict_chi(phi, best_D_inv, mask)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            recon_times.append(elapsed)

            # FLOPs (first sample only)
            if i == 0:
                H, W, D = phi.shape[2], phi.shape[3], phi.shape[4]
                flops = compute_reconstruction_flops((H, W, D))
                print(f"\n⚡ Reconstruction FLOPs = {flops / 1e9:.3f} GFLOPs\n")

            # Convert to numpy
            chi = chi_torch.cpu().numpy()[0, 0]
            cosmos_np = cosmos.cpu().numpy()[0]
            mask_np = mask.cpu().numpy()[0] > 0

            # Clip to COSMOS range inside mask
            cosmos_masked = cosmos_np[mask_np]
            if cosmos_masked.size > 0:
                chi = np.clip(chi, float(cosmos_masked.min()), float(cosmos_masked.max()))

            # Compute metrics
            psnr_v, ssim_v, corr_v, hfen_v, rmse_v = compute_metrics(
                cosmos_np, chi, mask_np
            )
            psnr_list.append(psnr_v)
            ssim_list.append(ssim_v)
            corr_list.append(corr_v)
            hfen_list.append(hfen_v)
            rmse_list.append(rmse_v)

            print(
                f"[Test {i:03d}] PSNR={psnr_v:.2f}  SSIM={ssim_v:.3f}  "
                f"Corr={corr_v:.3f}  HFEN={hfen_v:.4f}  RMSE={rmse_v:.4f}%"
            )

            if i < n_vis:
                save_vis(
                    cosmos_np,
                    chi,
                    mask_np,
                    out_path=f"{vis_dir}/vis_test{i:03d}.png",
                )

    # Summary
    print("\n" + "=" * 55)
    print("Summary over entire test set")
    print("=" * 55)
    print(f"  PSNR  : {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"  SSIM  : {np.mean(ssim_list):.3f} ± {np.std(ssim_list):.3f}")
    print(f"  Corr  : {np.mean(corr_list):.3f} ± {np.std(corr_list):.3f}")
    print(f"  HFEN  : {np.mean(hfen_list):.4f} ± {np.std(hfen_list):.4f}")
    print(f"  RMSE  : {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f} %")
    print(f"  Time  : {np.mean(recon_times):.4f} ± {np.std(recon_times):.4f} s")

    return {
        "psnr": psnr_list,
        "ssim": ssim_list,
        "corr": corr_list,
        "hfen": hfen_list,
        "rmse": rmse_list,
        "recon_times": recon_times,
    }
