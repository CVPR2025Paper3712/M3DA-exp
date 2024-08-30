import numpy as np
import torch
from dpipe.layers import PostActivation3d
from torch import nn

from ..functional import softmax


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, n_features: int = 256, si_eps: float = 1e-30):
        super().__init__()
        self.in_channels = in_channels
        self.n_features = n_features
        n = n_features
        self.discriminator = nn.Sequential(
            SelfInformation(eps=si_eps),
            nn.BatchNorm3d(in_channels),
            PostActivation3d(in_channels, n, kernel_size=3, padding=1, activation_module=nn.LeakyReLU),
            PostActivation3d(n, n, kernel_size=3, padding=1, activation_module=nn.LeakyReLU),
            nn.AvgPool3d(2, 2),
            PostActivation3d(n, n // 2, kernel_size=3, padding=1, activation_module=nn.LeakyReLU),
            nn.AvgPool3d(2, 2),
            PostActivation3d(n // 2, n // 2, kernel_size=3, padding=1, activation_module=nn.LeakyReLU),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(n // 2, 1)
        )

    def forward(self, x):
        return self.discriminator(x)


class SelfInformation(nn.Module):
    def __init__(self, eps: float = 1e-30):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        proba = softmax(x)
        return -torch.mul(proba, torch.log2(proba + self.eps)) / np.log2(proba.shape[1])


class EntropyLoss(nn.Module):
    def __init__(self, eps: float = 1e-30):
        super().__init__()
        self.si = SelfInformation(eps=eps)

    def forward(self, x):
        return torch.mean(self.si(x))
