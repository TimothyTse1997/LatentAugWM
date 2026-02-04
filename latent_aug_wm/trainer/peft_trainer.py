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


class PeftTrainer(Trainer):
    # TODO: ema on PEFT model

    def __init__(self, *args, peft_module_names=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.peft_module_names = peft_module_names

    # def _get_state_dict(self, module_name, module):
    #     if module_name in self.peft_module_names:
    #         return get_peft_model_state_dict(module)
    #     return module.state_dict()

    def save_checkpoint(self, checkpoint_path, peft_checkpoint_path={}):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """

        model_state_dicts = {
            key: self.model[key].state_dict()
            for key in self.args.save_modules
            if (key not in self.peft_module_names)
        }

        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": model_state_dicts,
        }
        if self.model_ema is not None:
            ema_model_state_dicts = {
                key: self.model_ema[key].state_dict
                for key in self.args.save_modules
                if (key not in self.args.peft_module_names)
            }
            if ema_model_state_dicts:
                state_dict["model_ema"] = ema_model_state_dicts

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

        if not self.peft_module_names:
            return

        if not peft_checkpoint_path:
            peft_checkpoint_path = self.create_peft_checkpoint_path(checkpoint_path)

        for peft_key in self.peft_module_names:
            if peft_key not in self.args.save_modules:
                continue
            peft_module = self.model[peft_key]
            peft_module.save_pretrained(peft_checkpoint_path[peft_key])

    def load_checkpoint(
        self, checkpoint_path, load_only_params=False, peft_checkpoint_path={}
    ):
        # TODO
        # the loading of peft model isn't as easy as we think
        # might need a separate function
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            if not key in state_dict["model"]:
                continue
            if key in self.peft_module_names:
                continue
            else:
                self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                if key in self.peft_module_names:
                    continue
                if not key in state_dict["model_ema"]:
                    continue
                self._load(state_dict["model_ema"][key], self.model_ema[key])

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def create_peft_checkpoint_path(self, checkpoint_path):
        peft_checkpoint_path = {}
        for peft_key in self.peft_module_names:
            save_dir_name = (
                Path(checkpoint_path).name.split(".")[0] + f"_peft_{peft_key}"
            )
            peft_checkpoint_path[peft_key] = (
                Path(checkpoint_path).parent / save_dir_name
            )
        return peft_checkpoint_path

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    if not force_load:
                        continue

                    min_shape = np.minimum(
                        np.array(val.shape), np.array(model_states[key].shape)
                    )
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    def moving_average(self, module_name, beta=0.999):
        if module_name in self.peft_module_names:
            self._moving_average_peft(
                self.model[module_name], self.model_ema[module_name]
            )
        else:
            self._moving_average(self.model[module_name], self.model_ema[module_name])

    @staticmethod
    def _moving_average_peft(model, model_test, beta=0.999):
        # TODO:
        # Lora REALLY don't work with ema, lets try this later ...
        return
        skipped_layers = 0
        for param_name, param in model.named_parameters():
            if not "lora" in param_name:
                skipped_layers += 1
                continue
            param_test = model_test.parameters[param_name]
            param_test.data = torch.lerp(param.data, param_test.data, beta)
        # self.logger.info(f"ema skipped layers: {skipped_layers}")
