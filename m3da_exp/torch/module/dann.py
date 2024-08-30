from typing import Tuple, Callable

import torch
from dpipe.layers import PostActivation3d
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 240, n_features: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_features = n_features
        n = n_features
        self.discriminator = nn.Sequential(
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


class DANN(nn.Module):
    def __init__(self, segmentator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        self.segmentator = segmentator
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, features = self.segmentator(x, return_features=True)
        domain_score = self.discriminator(features)
        return logits, domain_score


class DANNLoss(nn.Module):
    def __init__(self, segmentation_criterion: Callable, discriminator_criterion: Callable, alpha: float = 1.0):
        super().__init__()
        self.segmentation_criterion = segmentation_criterion
        self.discriminator_criterion = discriminator_criterion
        if alpha < 0:
            raise ValueError(f"Expected alpha to be non negative, got {alpha}")
        self.alpha = alpha

    def compute_segm(self, xs, y) -> dict:
        logits, d_score_src, d_score_tgt = xs
        segm_loss = self.segmentation_criterion(logits, y)
        disc_loss_source = self.discriminator_criterion(d_score_src, torch.zeros_like(d_score_src))
        disc_loss_target = self.discriminator_criterion(d_score_tgt, torch.ones_like(d_score_tgt))
        disc_loss_total = disc_loss_source + disc_loss_target
        return {
            "segm_loss": segm_loss,
            "disc_loss_source": disc_loss_source,
            "disc_loss_target": disc_loss_target,
            "disc_loss_total": disc_loss_total,
            "adversarial_loss": segm_loss - self.alpha * disc_loss_total
        }

    def compute_disc(self, xs) -> dict:
        d_score_src, d_score_tgt = xs
        disc_loss_source = self.discriminator_criterion(d_score_src, torch.zeros_like(d_score_src))
        disc_loss_target = self.discriminator_criterion(d_score_tgt, torch.ones_like(d_score_tgt))
        return {"discriminator_loss": disc_loss_source + disc_loss_target}


class TrainWrapper(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (*self.architecture(source), *self.architecture(target, "target"))


class InferenceWrapper(nn.Module):
    def __init__(self, architecture, mode: str = "segm"):
        super().__init__()
        if mode not in ("segm", "domain"):
            raise ValueError(f"Inference mode must be 'segm' or 'domain', got {mode}")
        self.architecture = architecture
        self.mode = mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.architecture(x)[0] if self.mode == "segm" else self.architecture(x)[1]


def get_resizing_features_modules(ndim: int, resize_features_to: str):
    if ndim not in (2, 3, ):
        raise ValueError(f'`ndim` should be in (2, 3). However, {ndim} is given.')

    ds16 = nn.AvgPool2d(16, 16, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(16, 16, ceil_mode=True)
    ds8 = nn.AvgPool2d(8, 8, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(8, 8, ceil_mode=True)
    ds4 = nn.AvgPool2d(4, 4, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(4, 4, ceil_mode=True)
    ds2 = nn.AvgPool2d(2, 2, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(2, 2, ceil_mode=True)
    identity = nn.Identity()
    us2 = nn.Upsample(scale_factor=2, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us4 = nn.Upsample(scale_factor=4, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us8 = nn.Upsample(scale_factor=8, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us16 = nn.Upsample(scale_factor=16, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)

    if resize_features_to == 'x16':
        return ds16, ds8, ds4, ds2, identity
    elif resize_features_to == 'x8':
        return ds8, ds4, ds2, identity, us2
    elif resize_features_to == 'x4':
        return ds4, ds2, identity, us2, us4
    elif resize_features_to == 'x2':
        return ds2, identity, us2, us4, us8
    elif resize_features_to == 'x1':
        return identity, us2, us4, us8, us16
    else:
        resize_features_to__options = ('x1', 'x2', 'x4', 'x8', 'x16')
        raise ValueError(f'`resize_features_to` should be in {resize_features_to__options}. '
                         f'However, {resize_features_to} is given.')
