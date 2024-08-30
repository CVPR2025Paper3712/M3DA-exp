from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
import torch
from dpipe.im.utils import dmap
from dpipe.torch import optimizer_step
from dpipe.torch.utils import sequence_to_var, to_np, save_model_state
from torch.nn import Module
from torch.optim import Optimizer

from .module.dann import DANN
from ..batch_iter.pipeline import SPATIAL_DIMS
from .utils import detach


def _save_at_epoch(epoch, architecture, save_condition_fn: Callable):
    e = int(epoch)
    if save_condition_fn(e):
        architecture.eval()
        save_model_state(architecture, f'model_{e}.pth')
        architecture.train()


# ### DANN: ###


def train_step_dann(*inputs: np.ndarray, architecture: DANN, criterion: Callable, optimizer_segm: Optimizer,
                    optimizer_disc: Optimizer, loss_key_segm: str, loss_key_disc: str, gradient_accumulation_steps=None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None, clip_grad: Optional[float] = None,
                    **optimizer_params) -> Dict[str, np.ndarray]:
        
    architecture.train()
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:len(inputs) - 1], inputs[len(inputs) - 1:]
    
    xs, xt = inputs
    
    src_logits, src_features = architecture.segmentator(xs, return_features=True)
    tgt_logits, tgt_features = architecture.segmentator(xt, return_features=True)

    src_domain_score = architecture.discriminator(src_features)
    tgt_domain_score = architecture.discriminator(tgt_features)
    
    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        loss_segm = criterion.compute_segm((src_logits, src_domain_score, tgt_domain_score), *targets)
    
    optimizer_step(optimizer_segm, loss_segm[loss_key_segm], scaler=scaler, clip_grad=clip_grad, **optimizer_params)
    optimizer_disc.zero_grad()
    
    src_domain_score = architecture.discriminator(detach(src_features))
    tgt_domain_score = architecture.discriminator(detach(tgt_features))
    
    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        loss_disc = criterion.compute_disc((src_domain_score, tgt_domain_score))
    
    optimizer_step(optimizer_disc, loss_disc[loss_key_disc], scaler=scaler, clip_grad=clip_grad, **optimizer_params)
    
    return dmap(to_np, {**loss_segm, **loss_disc})


# ### Original train step (with checkpoints saving): ###


def train_step(*inputs: np.ndarray, architecture: Module, criterion: Callable, optimizer: Optimizer,
               loss_key: Optional[str] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None,
               clip_grad: Optional[float] = None, accumulate: bool = False, gradient_accumulation_steps: int = 1,
               epoch=None, **optimizer_params):

    # called from adabn.train.base::train_and_save() at the end of the epoch
    if epoch is not None:
        _save_at_epoch(epoch, architecture, save_condition_fn=lambda e: e > 0 and e % 100 == 0)
        return

    architecture.train()
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:len(inputs) - 1], inputs[len(inputs) - 1:]

    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        loss = criterion(architecture(*inputs), *targets)

    if loss_key is not None:
        optimizer_step(optimizer, loss[loss_key] / gradient_accumulation_steps,
                       scaler=scaler, clip_grad=clip_grad, accumulate=accumulate, **optimizer_params)

        return dmap(to_np, loss)

    optimizer_step(optimizer, loss / gradient_accumulation_steps,
                   scaler=scaler, clip_grad=clip_grad, accumulate=accumulate, **optimizer_params)

    return to_np(loss)


# ### Self-ensembling: ###


def train_step_se(*inputs, student, teacher, ema_alpha: float, gamma: float, criterion_task, criterion_cons,
                  optimizer, scaler, random_state: np.random.RandomState, activation: Callable,
                  clip_grad: Optional[float] = None, gradient_accumulation_steps=None,
                  p: float = 3 / 4, max_gamma: float = 2.0, **optimizer_params):
    loss_dict = {}
    student.train(), teacher.train()
    deep_supervision = student.get_deep_supervision()
    images_s, images_t, targets = sequence_to_var(*inputs, device=student)

    g = random_state.choice((random_state.uniform(1 / max_gamma, 1), random_state.uniform(1, max_gamma))).item()
    logits_s = student(images_s ** g)  # additional gamma augmentation

    augm_list: list[Callable] = [lambda x: x]
    if random_state.rand() < p:
        flip_dim = random_state.choice(SPATIAL_DIMS)
        augm_list.append(partial(torch.flip, dims=[flip_dim]))

    pre_augm = images_t
    for augm in augm_list:
        pre_augm = augm(pre_augm)
    probas_student_t = activation(student(pre_augm)[0]) if deep_supervision else activation(student(pre_augm))

    with torch.no_grad():
        post_augm = teacher(images_t)[0] if deep_supervision else teacher(images_t)
    for augm in augm_list:
        post_augm = augm(post_augm)
    probas_teacher_t = activation(post_augm)
        
    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        loss_task = criterion_task(logits_s, targets)
        loss_cons = criterion_cons(probas_student_t, probas_teacher_t)
        loss_dict['task'] = loss_task
        loss_dict['consistency'] = loss_cons

    optimizer_step(optimizer, loss_task + gamma * loss_cons, scaler=scaler, clip_grad=clip_grad, **optimizer_params)

    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data = (1.0 - ema_alpha) * t.data + ema_alpha * s.detach().clone()
    
    return dmap(to_np, loss_dict)


# ### Adaptive BN: ###


def train_step_adabn_dummy(*inputs: np.ndarray, architecture: Module, scaler: torch.cuda.amp.GradScaler = None,
                           **kwargs):
    architecture.train()
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:len(inputs) - 1], inputs[len(inputs) - 1:]
    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        _ = architecture(*inputs)
    return 0


# ### Entropy Minimization ###


def train_step_minent(*inputs, architecture, criterion_segm, criterion_ent, optimizer, lambda_ent: float = 0.001,
                      loss_key: Optional[str] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                      clip_grad: Optional[float] = None, accumulate: bool = False,
                      gradient_accumulation_steps: int = 1, epoch=None, **optimizer_params):

    # called from adabn.train.base::train_and_save() at the end of the epoch
    if epoch is not None:
        _save_at_epoch(epoch, architecture, save_condition_fn=lambda e: e > 0 and e % 100 == 0)
        return

    architecture.train()
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:len(inputs) - 1], inputs[len(inputs) - 1:]

    xs, xt = inputs

    with torch.cuda.amp.autocast(scaler is not None or torch.is_autocast_enabled()):
        loss_segm = criterion_segm(architecture(xs), *targets)
        loss_ent = criterion_ent(architecture(xt)[0] if architecture.get_deep_supervision() else architecture(xt))

    optimizer_step(optimizer, loss_segm + lambda_ent * loss_ent,
                   scaler=scaler, clip_grad=clip_grad, accumulate=accumulate, **optimizer_params)

    return dmap(to_np, {"criterion_segm": loss_segm, "criterion_entropy": loss_ent})
