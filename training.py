"""
training.py — Training and validation loops for the QSM UNet.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import QSM_Dataset
from model import UNet
from loss import compute_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    net: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch.  Returns mean loss."""
    net.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")

    for batch in pbar:
        phase = batch["phase"].to(device)
        dipole = batch["dipole"].to(device)
        mask = batch["mask"].to(device)
        cosmos = batch["cosmos"].to(device)

        phi = phase.unsqueeze(1)
        loss, _ = compute_loss(phi, dipole, cosmos, mask, net)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3e}"})

    return running_loss / max(1, len(loader))


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    net: UNet,
    loader: DataLoader,
    device: str,
):
    """Evaluate on validation set.  Returns (mean_loss, last_D_inv)."""
    net.eval()
    val_loss = 0.0
    last_D_inv = None

    with torch.no_grad():
        for batch in loader:
            phase = batch["phase"].to(device)
            dipole = batch["dipole"].to(device)
            mask = batch["mask"].to(device)
            cosmos = batch["cosmos"].to(device)

            phi = phase.unsqueeze(1)
            loss, comps = compute_loss(phi, dipole, cosmos, mask, net)
            val_loss += loss.item()
            last_D_inv = comps["D_inv"]

    return val_loss / max(1, len(loader)), last_D_inv


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(
    train_csv: str,
    val_csv: str,
    root: str,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: str = "best_model.pt",
    dinv_save_path: str = "best_dipole_inverse.pt",
):
    """Train the QSM UNet and save the best checkpoint.

    Parameters
    ----------
    train_csv      : path to training CSV
    val_csv        : path to validation CSV
    root           : data root directory
    device         : 'cuda' or 'cpu'
    epochs         : number of training epochs
    lr             : initial learning rate
    save_path      : where to save best model weights
    dinv_save_path : where to save best D_inv tensor
    """
    train_ds = QSM_Dataset(train_csv, root)
    val_ds = QSM_Dataset(val_csv, root)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    net = UNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(
            net, train_loader, optimizer, device, ep, epochs
        )
        val_loss, best_D_inv = validate(net, val_loader, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"[Epoch {ep:03d}] Train Loss = {train_loss:.4e} | "
            f"Val Loss = {val_loss:.4e}"
        )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), save_path)
            torch.save(best_D_inv, dinv_save_path)
            print(
                f"  ✅ Saved best model → {save_path}  "
                f"(val_loss={val_loss:.4e})"
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4e}")
    return train_loss_history, val_loss_history
