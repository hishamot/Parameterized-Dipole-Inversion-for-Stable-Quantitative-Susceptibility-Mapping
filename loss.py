"""
loss.py — Physics-informed loss and reconstruction helpers for QSM.
"""

import torch
import torch.nn.functional as F
import torch.fft as fft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def twochan_to_complex(t: torch.Tensor) -> torch.Tensor:
    """Convert a 2-channel real tensor (B,2,Z,Y,X) → complex tensor (B,Z,Y,X)."""
    return torch.complex(t[:, 0], t[:, 1])


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Isotropic 3-D total-variation of x (B, C, Z, Y, X)."""
    dz = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    dy = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
    dx = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
    return dz + dy + dx


def predict_chi(
    phi: torch.Tensor,
    D_inv: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct susceptibility χ from phase φ and learned dipole inverse D_inv.

    Parameters
    ----------
    phi   : (B, 1, Z, Y, X)
    D_inv : (B, Z, Y, X)  complex
    mask  : (B, Z, Y, X)  float, optional

    Returns
    -------
    chi : (B, 1, Z, Y, X)
    """
    Phi_f = fft.fftn(phi.squeeze(1), dim=(-3, -2, -1), norm="ortho")
    chi_f = D_inv * Phi_f
    chi = fft.ifftn(chi_f, dim=(-3, -2, -1), norm="ortho").real
    chi = chi.unsqueeze(1)
    if mask is not None:
        chi = chi * mask.unsqueeze(1)
    return chi


def forward_dipole(chi: torch.Tensor, dipole: torch.Tensor) -> torch.Tensor:
    """Forward dipole model: φ = IFFT( D · FFT(χ) ).

    Parameters
    ----------
    chi    : (B, 1, Z, Y, X)
    dipole : (B, Z, Y, X)  complex

    Returns
    -------
    phi : (B, 1, Z, Y, X)
    """
    Chi_f = fft.fftn(chi.squeeze(1), dim=(-3, -2, -1), norm="ortho")
    Phi_pred = dipole * Chi_f
    phi = fft.ifftn(Phi_pred, dim=(-3, -2, -1), norm="ortho").real
    return phi.unsqueeze(1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    phi: torch.Tensor,
    dipole: torch.Tensor,
    cosmos: torch.Tensor,
    mask: torch.Tensor,
    net: torch.nn.Module,
    lam_tv: float = 1e-6,
    lam_reg: float = 1e-6,
    lam_id: float = 5e-5,
    lam_dip: float = 1.0,
):
    """Physics-informed loss for QSM inversion.

    Parameters
    ----------
    phi    : (B, 1, Z, Y, X) — local field map (phase)
    dipole : (B, Z, Y, X)   — complex dipole kernel
    cosmos : (B, Z, Y, X)   — COSMOS ground-truth χ
    mask   : (B, Z, Y, X)   — brain mask
    net    : UNet instance

    Returns
    -------
    loss   : scalar tensor
    comps  : dict of loss components (for logging) and D_inv
    """
    # Build 2-channel real input from complex dipole
    dipole_r = dipole.real.unsqueeze(1)
    dipole_i = dipole.imag.unsqueeze(1)
    net_in = torch.cat([dipole_r, dipole_i], dim=1)

    out = net(net_in)
    D_inv = twochan_to_complex(out)

    chi = predict_chi(phi, D_inv, mask)

    # Data fidelity (L1 inside mask)
    loss_data = F.l1_loss(chi.squeeze() * mask, cosmos * mask)
    # Spatial regularity
    loss_tv = total_variation(chi)
    # D_inv magnitude regularisation
    loss_reg = (D_inv.abs() ** 2).mean()
    # Identity: D · D_inv ≈ 1
    DDinv = dipole * D_inv
    loss_id = F.l1_loss(DDinv.real, torch.ones_like(DDinv.real))
    # Dipole consistency: forward(χ) ≈ φ
    phi_pred = forward_dipole(chi, dipole)
    loss_dip = F.l1_loss(phi_pred * mask, phi * mask)

    loss = (
        loss_data
        + lam_tv * loss_tv
        + lam_reg * loss_reg
        + lam_id * loss_id
        + lam_dip * loss_dip
    )

    comps = {
        "data": loss_data.item(),
        "tv": loss_tv.item(),
        "reg": loss_reg.item(),
        "id": loss_id.item(),
        "dip": loss_dip.item(),
        "D_inv": D_inv.detach().cpu(),
    }
    return loss, comps
