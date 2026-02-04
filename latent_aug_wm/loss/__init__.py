import os
import time
import torch

from torch import nn

import torch.nn.functional as F
import numpy as np

# from munch import Munch
from loguru import logger as glogger

mse_loss = nn.MSELoss()


def construct_loss_fn(list_of_fn, log_fn_time=False):
    def _loss_fn(nets, weight, batch, step, eval=False, **kwargs):
        new_batch = {k: v for k, v in batch.items()}
        used_modules = []

        for fn in list_of_fn:
            start = time.time()
            result, used_module = fn(nets=nets, step=step, eval=eval, **new_batch)

            new_batch.update(result)
            used_modules += used_module

            end = time.time()
            if log_fn_time:
                glogger.debug(f"function {fn} tooks in total {end - start} sec")

        total_loss = 0
        loss_items = {}

        for k, v in new_batch.items():
            if "loss" in k:
                total_loss = total_loss + weight.get(k, 1.0) * v
                loss_items.update({k: v.detach().item()})
        # loss, logging for loss values, module for update

        if not eval:
            return total_loss, loss_items, used_modules

        return total_loss, loss_items, used_modules, new_batch

    return _loss_fn


def create_eval_only_loss_fn(loss_fn):
    def eval_only_loss_fn(nets=None, step=None, eval=False, **kwargs):
        if not eval:
            return kwargs, []
        return loss_fn(nets=nets, step=step, eval=eval, **kwargs)

    return eval_only_loss_fn


def loss_fn_add_per_step_scaling(
    loss_fn=None, initial_scale=0.1, start_step=0, final_step=1000
):
    def modified_loss(nets=None, step=None, eval=False, **kwargs):

        if step is None or step < start_step:
            return kwargs, []
        if step >= final_step:
            current_scale = 1.0
        else:
            current_scale = initial_scale + (1.0 - initial_scale) * (
                (step - start_step) / (final_step - start_step)
            )
        return loss_fn(nets=nets, step=step, scale=current_scale, eval=eval, **kwargs)

    return modified_loss


def loss_fn_add_start_step(loss_fn=None, start_step=None):
    """
    add step control to style loss
    """

    def modified_loss(nets=None, step=None, **kwargs):

        if step is None or step < start_step:
            return kwargs
        return loss_fn(nets=nets, step=step, **kwargs)

    return modified_loss


def loss_fn_add_end_step(loss_fn=None, end_step=None):
    def modified_loss(nets=None, step=None, **kwargs):

        if step is None or step > end_step:
            return kwargs
        return loss_fn(nets=nets, step=step, **kwargs)

    return modified_loss


class LossInverseScaler:
    def __init__(self, end_beta, step_value, loss_names=[], **kwargs):
        self.end_beta = end_beta
        self.step_value = step_value
        self.loss_names = loss_names

    def __call__(self, nets=None, step=None, **kwargs):
        if step is None:
            return kwargs

        current_beta = max(1.0 - step * self.step_value, self.end_beta)
        if not self.loss_names:
            for k in kwargs.keys():
                if not "kl" in k:
                    continue
                kwargs[k] = current_beta * kwargs[k]
            return kwargs
        for loss_name in self.loss_names:
            if loss_name in self.loss_names:
                if loss_name not in kwargs:
                    continue
                kwargs[loss_name] = current_beta * kwargs[loss_name]
        return kwargs
