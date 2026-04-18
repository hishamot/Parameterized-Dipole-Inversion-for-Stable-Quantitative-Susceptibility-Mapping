"""
model.py — 3-D UNet that predicts the complex dipole inverse (2-channel real output).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv3d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """3-D UNet.

    Input  : (B, in_ch,  Z, Y, X)  — default in_ch=2 (real & imag of dipole)
    Output : (B, out_ch, Z, Y, X)  — default out_ch=2 (real & imag of D_inv)
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 2):
        super().__init__()
        c1, c2, c3, c4 = 16, 32, 64, 128

        # Encoder
        self.down1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottom = DoubleConv(c3, c4)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.conv3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.conv2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.conv1 = DoubleConv(c1 + c1, c1)

        # Output head
        self.out_conv = nn.Conv3d(c1, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        xb = self.bottom(self.pool3(x3))

        x = self.up3(xb)
        x = self.conv3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, x1], dim=1))

        return self.out_conv(x)
