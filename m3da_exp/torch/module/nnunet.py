from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 init_bias: float = None,
                 return_features_from: int = -1,
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        self.init_bias = init_bias
        self.return_features_from = return_features_from

    def forward(self, x, return_features: bool = False):
        skips = self.encoder(x)
        return (self.decoder(skips), skips[self.return_features_from]) if return_features else self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), \
            "just give the image size without color/feature channels or batch channel. " \
            "Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) \
            + self.decoder.compute_conv_feature_map_size(input_size)

    def set_deep_supervision(self, deep_supervision: bool):
        self.decoder.deep_supervision = deep_supervision

    def get_deep_supervision(self):
        return self.decoder.deep_supervision

    def update_bias(self, background_logit_idx: int = 0):
        bias = self.decoder.seg_layers[-1].bias.detach()
        if self.init_bias is not None:
            bias[background_logit_idx] = self.init_bias
        self.decoder.seg_layers[-1].bias = nn.Parameter(bias)


class DeepSupervisionTargetWrapper(nn.Module):
    def __init__(self, ds_loss, pool_op_kernel_sizes, labels_list, device):
        """One more wrapper to adapt the target to Deep Supervision."""
        super().__init__()
        self.ds_loss = ds_loss
        self.pool_kernel_sizes = pool_op_kernel_sizes[1:-1]
        self.labels_list = torch.tensor(labels_list, device=device)
        self.device = device

        self.pools = nn.ModuleList([nn.Upsample(scale_factor=tuple([1. / s for s in kernel_size]), mode='trilinear')
                                    for kernel_size in self.pool_kernel_sizes])

    def forward(self, outputs, target):
        targets = [target]
        for pool in self.pools:
            target = self._pool(target, pool)
            targets.append(target)
        return self.ds_loss(outputs, targets)

    def _pool(self, x, pool):
        onehot = (self.labels_list == x[..., None]).transpose(1, -1)[..., 0]
        return pool(onehot.double()).argmax(axis=1).type(dtype=x.dtype)[:, None, ...]
