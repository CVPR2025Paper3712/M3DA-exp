from itertools import chain
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from dpipe.itertools import zip_equal
from dpipe.torch import load_model_state
from torch.nn.modules.batchnorm import _BatchNorm

from ..utils import idx2train


def lr_onecycle_cosine_schedule_fn(epoch, lr_base=1e-2, lr_min=1e-6, last_linear_epoch=10, n_epochs=100):
    if epoch <= last_linear_epoch:
        lr = lr_base * epoch / last_linear_epoch
    else:
        lr = lr_base * 0.5 * (1 + np.cos((epoch - last_linear_epoch) / (n_epochs - last_linear_epoch) * np.pi))
    return lr if lr > 0 else lr_min


def nnunet_lr_schedule_fn(epoch: int, lr_base: float = 1e-2, pw: float = 0.9, n_epochs: int = 100):
    return lr_base * (1 - epoch / n_epochs) ** pw


def lr_finder_schedule_fn(epoch, n_epochs, lr_min: float = 1e-6, lr_max: float = 10.):
    return lr_min * np.power(lr_max / lr_min, epoch / n_epochs)


def load_gan_state(models: List[nn.Module], paths: List[str]) -> None:
    for model, path in zip_equal(models, paths):
        load_model_state(model, path)


def load_model_state_from_one2all(architecture, dataset_name: str, n_folds: int = 6):
    from adabn.paths import EXP_BASE_PATH
    from adabn.split import SELECTED_PAIRS

    exp_idx = int(Path('.').absolute().name.split('_')[-1])
    base_exp_id = exp_idx if (dataset_name == 'vsseg') else\
        idx2train(SELECTED_PAIRS[dataset_name][exp_idx], n_folds=n_folds)
    load_path = EXP_BASE_PATH / f'{dataset_name}/one2all/experiment_{base_exp_id}/model.pth'
    load_model_state(architecture, load_path)


def switch_grad(*models: nn.Module, mode: bool = False) -> None:
    if not isinstance(mode, bool):
        raise ValueError(f"Expected `mode` to be True or False, got {mode}")
    
    for p in chain.from_iterable(m.parameters() for m in models):
        p.requires_grad = mode
        

def detach(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.detach()
    return [t.detach() for t in x]


def batch_norm_track_mode(*models: nn.Module, mode: bool = False) -> None:
    if not isinstance(mode, bool):
        raise ValueError(f"Expected `mode` to be True or False, got {mode}")
    
    for model in models:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.track_running_stats = mode
                

def init_normal_weights(*models: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    print(f"Initialize models with mean {mean} and std {std}")
    for model in models:
        for module in model.modules():
            if isinstance(module, _BatchNorm) or not hasattr(module, "weight"):
                continue
            module.apply(nn.init.normal_(module.weight.data, mean, std))
