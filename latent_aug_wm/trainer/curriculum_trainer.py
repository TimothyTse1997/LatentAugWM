import os
import os.path as osp
import gc
from pathlib import Path
import re
import sys
import yaml
import shutil
import click
import warnings


import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from latent_aug_wm.trainer.base import Trainer


class CurriculumTrainer(Trainer):
    def __init__(
        self,
        *args,
        curriculum_config={
            # "model_list": [["noise_encoder"], ["detector", "classifier"]],
            "curriculum_steps": [500, 500],
            "loss_fns": [],
        },
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.curriculum_config = curriculum_config
        self.curriculum_config["cumsum_steps"] = np.cumsum(
            self.curriculum_config["curriculum_steps"]
        )

    def set_module_require_grad(self, model, target=True):
        for k, v in model.named_parameters():
            v.requires_grad = target

    # def set_curriculum(self):
    #     curriculum_step = self.steps % sum(self.curriculum_config["curriculum_steps"])
    #     curriculum_id = None
    #     for train_model_id, cumsum_step in enumerate(self.curriculum_config["cumsum_steps"]):
    #         curriculum_id = train_model_id
    #         if cumsum_step > curriculum_step: break

    #     target_module = self.curriculum_config["model_list"]

    #     for k, v in self.model.items():
    #         try:
    #             if k in target_module:
    #                 self.set_module_require_grad(v, True)
    #             else:
    #                 self.set_module_require_grad(v, False)
    #         except Exception as e:
    #             print(e)
    #             continue

    def set_curriculum_fn(self):
        curriculum_step = self.steps % sum(self.curriculum_config["curriculum_steps"])
        curriculum_id = None
        for train_model_id, cumsum_step in enumerate(
            self.curriculum_config["cumsum_steps"]
        ):
            curriculum_id = train_model_id
            if cumsum_step > curriculum_step:
                break

        current_loss_fn = self.curriculum_config["loss_fns"][curriculum_id]
        return current_loss_fn

    def _train_step(self, batch):
        loss_fn = self.set_curriculum_fn()

        ### load data
        if isinstance(batch, list):
            batch = [(b.to(self.device) if torch.is_tensor(b) else b) for b in batch]
        else:
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        # train the discriminator (by target reference)

        self.optimizer.zero_grad()

        if self.fp16_run:  # scaler is not None:
            # with torch.cuda.amp.autocast():
            # with torch.autocast(device_type=self.device, dtype=torch.float16):
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                loss, losses_ref, used_modules = loss_fn(
                    self.model,
                    self.args.loss_weights,
                    batch,
                    step=self.steps,
                    eval=False,
                )
            # scaler.scale(loss).backward()
        else:
            loss, losses_ref, used_modules = loss_fn(
                self.model, self.args.loss_weights, batch, step=self.steps, eval=False
            )
        if torch.isnan(loss):
            return losses_ref
        loss.backward()
        # for n, p in self.model.noise_encoder.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.grad.abs().mean())

        for m in used_modules:
            # self.optimizer.step(m, scaler=scaler)
            max_norm = self.args.get("max_norm", 12.0)
            if self.args.get("max_norm_dict", None) and m in self.args["max_norm_dict"]:
                max_norm = self.args["max_norm_dict"][m]
            clip_grad_norm_(self.model[m].parameters(), max_norm)
            self.optimizer.step(m)
            model_norm = self.get_grad_norm(m)
            self.writer.add_scalar("train/%s_grad_norm" % m, model_norm, self.steps)

        batch = self.batch_detach(batch)
        if (
            self.args.get("ema_modules", None) is not None
            and self.model_ema is not None
        ):
            for module_name in self.args.ema_modules:
                self.moving_average(module_name, beta=0.999)

        self.optimizer.scheduler()

        return losses_ref
