import torch
from torch.nn.functional import cross_entropy
from dpipe.torch import weighted_cross_entropy_with_logits, dice_loss_with_logits


def softmax(x, dim: int = 1):
    return torch.softmax(x, dim=dim)


def focal_tversky_loss_with_logits(logit, target, beta, gamma):
    spatial_dims = list(range(2, target.dim()))

    proba = torch.sigmoid(logit)

    intersection = torch.sum(proba * target, dim=spatial_dims)
    tp = torch.sum(proba ** 2 * target, dim=spatial_dims)
    fp = torch.sum(proba ** 2 * (1 - target), dim=spatial_dims)
    fn = torch.sum((1 - proba ** 2) * target, dim=spatial_dims)
    tversky_index = intersection / (tp + beta * fn + (1 - beta) * fp + 1)
    loss = (1 - tversky_index) ** gamma

    return loss.mean()


def small_target_segm_loss(logit, target):
    return (weighted_cross_entropy_with_logits(logit, target)
            + focal_tversky_loss_with_logits(logit, target, beta=0.7, gamma=1))


def combined_loss_with_logits(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                              alpha=0.5, beta=0.7, adaptive_bce=False):
    return (1 - alpha) * dice_loss_with_logits(logit, target) \
           + alpha * weighted_cross_entropy_with_logits(logit, target, weight=weight, alpha=beta, adaptive=adaptive_bce)


def multiclass_dice_loss_by_averaging(logit: torch.Tensor, target: torch.Tensor, background_idx: int = 0):
    positive_class_idxs = set(range(logit.shape[1])) - {background_idx}

    loss = 0
    for cls_idx in positive_class_idxs:
        loss += dice_loss_with_logits(logit[:, cls_idx, ...], (target[:, 0, ...] == cls_idx).type(dtype=logit.dtype))

    return loss / len(positive_class_idxs)


def combined_loss_with_logits_mc(logit: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor = None,
                                 alpha: float = 0.5, background_idx: int = 0):
    # TODO: probably, we need to add `beta=0.7` (0.7 is a default value) to reduce positive class weights in CE.
    ce = cross_entropy(logit, target[:, 0, ...].long(), weight=class_weights)
    dl = multiclass_dice_loss_by_averaging(logit, target, background_idx=background_idx)
    return alpha * ce + (1 - alpha) * dl


def entropy_loss(proba: torch.Tensor, eps: float = 1e-4):
    if proba.shape[1] == 1:
        return -torch.mean(proba * torch.log(proba + eps) + (1 - proba) * torch.log(1 - proba + eps))
    else:
        return -torch.mean(proba * torch.log(proba + eps))
