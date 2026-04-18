"""
dataset.py — QSM Dataset and dipole-kernel utilities.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Dipole kernel
# ---------------------------------------------------------------------------

def dipole_kernel(
    matrix_size,
    voxel_size=(1, 1, 1),
    B0_dir=(0, 0, 1),
) -> torch.Tensor:
    """Compute the 3-D dipole kernel in k-space (complex64 tensor)."""
    Y, X, Z = np.meshgrid(
        np.linspace(-matrix_size[1] // 2, matrix_size[1] // 2 - 1, matrix_size[1]),
        np.linspace(-matrix_size[0] // 2, matrix_size[0] // 2 - 1, matrix_size[0]),
        np.linspace(-matrix_size[2] // 2, matrix_size[2] // 2 - 1, matrix_size[2]),
        indexing="ij",
    )
    kx = X / (matrix_size[0] * voxel_size[0])
    ky = Y / (matrix_size[1] * voxel_size[1])
    kz = Z / (matrix_size[2] * voxel_size[2])
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[k2 == 0] = np.inf
    D = (
        1 / 3
        - (kx * B0_dir[0] + ky * B0_dir[1] + kz * B0_dir[2]) ** 2 / k2
    )
    D = np.fft.ifftshift(D).astype(np.float32)
    return torch.from_numpy(D).to(torch.complex64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QSM_Dataset(Dataset):
    """Load QSM volumes from a CSV file.

    CSV format (with header):
        patient_id, cosmos_path, mask_path, phase_path

    All paths are relative to *root*.
    """

    def __init__(
        self,
        csv_file: str,
        root: str,
        voxel_size=(1, 1, 1),
        B0_dir=(0, 0, 1),
    ):
        self.samples = []
        with open(csv_file, "r") as f:
            for line in f.readlines()[1:]:  # skip header
                line = line.strip()
                if not line:
                    continue
                pid, cosmos, mask, phase = line.split(",")
                self.samples.append((pid, cosmos.strip(), mask.strip(), phase.strip()))
        self.root = root
        self.voxel_size = voxel_size
        self.B0_dir = B0_dir

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_mat(self, path: str) -> np.ndarray:
        data = loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No valid variable found in {path}")
        return data[keys[0]].astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        pid, cosmos_path, mask_path, phase_path = self.samples[idx]
        cosmos = self._load_mat(os.path.join(self.root, cosmos_path))
        mask = self._load_mat(os.path.join(self.root, mask_path))
        phase = self._load_mat(os.path.join(self.root, phase_path))
        matrix_size = phase.shape
        dipole = dipole_kernel(matrix_size, self.voxel_size, self.B0_dir)
        return {
            "patient_id": pid,
            "phase": torch.from_numpy(phase).float(),
            "mask": torch.from_numpy(mask).float(),
            "cosmos": torch.from_numpy(cosmos).float(),
            "dipole": dipole,
        }
