from typing import Callable, Optional
from warnings import warn

import numpy as np
from dpipe.torch import has_batchnorm
from dpipe.train import Logger, Checkpoints, ValuePolicy, Policy, EarlyStopping
from dpipe.train.base import _DummyCheckpoints, _build_context_manager, _DummyLogger


def train_and_save(
    train_step: Callable,
    batch_iter: Callable,
    n_epochs: int = np.inf,
    logger: Optional[Logger] = None,
    checkpoints: Optional[Checkpoints] = None,
    validate: Optional[Callable] = None,
    gradient_accumulation_steps: int = 1,
    **kwargs,
) -> None:
    """
    Performs a series of train and validation steps.

    Parameters
    ----------
    train_step: Callable
        a function to perform train step.
    batch_iter: Callable
        batch iterator.
    n_epochs: int
        maximal number of training epochs
    logger: Logger, None, optional
    checkpoints: Checkpoints, None, optional
    validate: Callable, None, optional
        a function to calculate metrics on the validation set.
    gradient_accumulation_steps: int
    kwargs
        additional keyword arguments passed to ``train_step``.
        For instances of `ValuePolicy` their `value` attribute is passed.
        Other policies are used for early stopping.

    References
    ----------
    See the :doc:`tutorials/training` tutorial for more details.
    """

    def get_policy_values():
        return {k: v.value for k, v in policies.items() if isinstance(v, ValuePolicy)}

    def broadcast_event(method, *args, **kw):
        for name, policy in policies.items():
            getattr(policy, method.__name__)(*args, **kw)

    if checkpoints is None:
        checkpoints = _DummyCheckpoints()
    if logger is None:
        logger = _DummyLogger()
    if not hasattr(batch_iter, '__enter__'):
        batch_iter = _build_context_manager(batch_iter)

    epoch = checkpoints.restore()
    scalars = {name: value for name, value in kwargs.items() if not isinstance(value, Policy)}
    policies = {name: value for name, value in kwargs.items() if isinstance(value, Policy)}

    assert isinstance(gradient_accumulation_steps, int)

    if gradient_accumulation_steps > 1 and has_batchnorm(kwargs['architecture']):
        warn(
            "Be careful! Implemented gradient accumulation is naive and doesn't take into account specifity of "
            "BatchNorm you are using."
        )

    with batch_iter as iterator:
        try:
            while epoch < n_epochs:
                broadcast_event(Policy.epoch_started, epoch)

                train_losses = []
                for idx, inputs in enumerate(iterator()):
                    broadcast_event(Policy.train_step_started, epoch, idx)
                    train_losses.append(
                        train_step(
                            *inputs,
                            accumulate=(idx + 1) % gradient_accumulation_steps != 0,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            **scalars,
                            **get_policy_values(),
                        )
                    )
                    broadcast_event(Policy.train_step_finished, epoch, idx, train_losses[-1])

                train_step(*inputs, epoch=epoch, **scalars, **get_policy_values())  # saves

                logger.train(train_losses, epoch)
                logger.policies(get_policy_values(), epoch)
                broadcast_event(Policy.validation_started, epoch, train_losses)

                metrics = None
                if validate is not None:
                    metrics = validate()
                    logger.metrics(metrics, epoch)

                broadcast_event(Policy.epoch_finished, epoch, train_losses,
                                metrics=metrics, policies=get_policy_values())

                checkpoints.save(epoch, train_losses, metrics)
                epoch += 1

        except EarlyStopping:
            pass
