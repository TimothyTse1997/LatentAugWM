# -*- coding: utf-8 -*-
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
import traceback


import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(
        self,
        args=None,
        model=None,
        model_ema=None,
        optimizer=None,
        device="cuda",
        train_dataloader=None,
        val_dataloader=None,
        initial_steps=0,
        initial_epochs=0,
        fp16_run=False,
        log_audio_batchs=10,
        loss_fn=None,
        log_fn=None,
        metrics=None,
        log_dir="./",
        best_measure_by_metric="f1",
        metric_measurement="highest",  # or lowest
        step_per_eval=100,
        **kwargs,
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # self.device = torch.device(device)
        self.device = device

        self.finish_train = False
        self.fp16_run = fp16_run
        self.log_audio_batchs = log_audio_batchs
        self.loss_fn = loss_fn
        self.log_fn = log_fn
        self.metrics = metrics
        self.best_measure_by_metric = best_measure_by_metric
        self.best_eval_result = None
        self.metric_measurement = metric_measurement
        self.step_per_eval = step_per_eval

        self.log_dir = log_dir

        Trainer._create_directory(self.log_dir)

        self.logger = Trainer._config_logging(self.log_dir)
        self.writer = Trainer._config_tensorboard(self.log_dir)

        if self.args.get("pretrained_model", "") != "":
            self.load_checkpoint(
                self.args["pretrained_model"],
                load_only_params=self.args.get("load_only_params", True),
            )

    @staticmethod
    def _create_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_attr_model_to_device(model_attr, device):
        for k, m in model_attr.items():
            try:
                _ = m.to(device)
            except:
                continue
        return device

    @staticmethod
    def _config_tensorboard(log_dir):
        if not Path(log_dir).exists():
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir + "/tensorboard")
        return writer

    @staticmethod
    def _config_logging(log_dir):
        # write logs
        open(osp.join(log_dir, "train.log"), "w").close()
        file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
        )
        logger.addHandler(file_handler)
        return logger

    def zero_grad(self):
        for k, v in self.model.items():
            _ = v.zero_grad()

    def save_checkpoint(self, checkpoint_path):
        return
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """

        model_state_dicts = {
            key: self.model[key].state_dict() for key in self.args.save_modules
        }
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": model_state_dicts,
        }
        if self.model_ema is not None:
            ema_model_state_dicts = {
                key: self.model_ema[key].state_dict() for key in self.args.save_modules
            }
            state_dict["model_ema"] = ema_model_state_dicts

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            if not key in state_dict["model"]:
                continue
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:

                if not key in state_dict["model_ema"]:
                    continue
                self._load(state_dict["model_ema"][key], self.model_ema[key])

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

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

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param_name, param in model.parameters.items():
            param_test = model_test.parameters[param_name]
            param_test.data = torch.lerp(param.data, param_test.data, beta)

        # for param, param_test in zip(model.parameters(), model_test.parameters()):
        #     param_test.data = torch.lerp(param.data, param_test.data, beta)

    def get_grad_norm(self, model_name):
        grads = [
            param.grad.detach().flatten()
            for param in self.model[model_name].parameters()
            if param.grad is not None
        ]
        try:
            norm = torch.cat(grads).norm()
        except:
            norm = 0
        return norm

    def batch_detach(self, batch):
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = [v.detach() if torch.is_tensor(v) else v for v in batch]
        else:
            batch = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in batch.items()
            }
        return batch

    def _train_step(self, batch):  # , scaler):

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
                loss, losses_ref, used_modules = self.loss_fn(
                    self.model,
                    self.args.loss_weights,
                    batch,
                    step=self.steps,
                    eval=False,
                )
            # scaler.scale(loss).backward()
        else:
            loss, losses_ref, used_modules = self.loss_fn(
                self.model, self.args.loss_weights, batch, step=self.steps, eval=False
            )
        loss.backward()
        # for n, p in self.model.noise_encoder.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.grad.abs().mean())

        for m in used_modules:
            # self.optimizer.step(m, scaler=scaler)
            max_norm = self.args.get("max_norm", 12.0)
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
                self.moving_average(
                    self.model["module_name"], self.model_ema["module_name"], beta=0.999
                )

        self.optimizer.scheduler()

        return losses_ref

    def _train_epoch(self):

        train_losses = defaultdict(list)
        _ = [self.model[k].train() for k in self.model]
        # scaler = (
        #     torch.amp.GradScaler()  # self.device)
        #     if (("cuda" in str(self.device)) and self.fp16_run)
        #     else None
        # )

        for train_steps_per_epoch, batch in enumerate(
            tqdm(self.train_dataloader, desc="[train]", leave=False), 1
        ):

            gc.collect()
            try:
                loss_items = self._train_step(batch)  # , scaler)
            except Exception as e:
                traceback.print_exc()
                batch = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for key in loss_items:
                train_losses["train/%s" % key].append(loss_items[key])
                self.writer.add_scalar("train/%s" % key, loss_items[key], self.steps)

            self.steps += 1

            if self.args.get(
                "steps_per_epoch"
            ) is not None and train_steps_per_epoch > self.args.get("steps_per_epoch"):
                break
            if self.steps % self.step_per_eval == 0:
                eval_results = self._eval_epoch(end_of_epoch=False)
                del eval_results

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        self.epochs += 1
        return train_losses

    @torch.no_grad()
    def _eval_step(self, batch):
        if self.fp16_run:
            # with torch.autocast(device_type=self.device, dtype=torch.float16):
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                loss, losses_ref, _, batch_output = self.loss_fn(
                    self.model,
                    self.args.loss_weights,
                    batch,
                    step=self.steps,
                    eval=True,
                )
        else:
            loss, losses_ref, _, batch_output = self.loss_fn(
                self.model, self.args.loss_weights, batch, step=self.steps, eval=True
            )

        # batch = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch_output = {
            k: (v.detach().float().cpu() if torch.is_tensor(v) else v)
            for k, v in batch_output.items()
        }

        del batch
        return loss, losses_ref, batch_output

    @torch.no_grad()
    def _eval_epoch(self, end_of_epoch=False):

        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model if hasattr(self.model[k], "eval")]
        audios, mels, figure = {}, {}, {}

        style_encoding, labels = [], []

        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.val_dataloader, desc="[eval]"), 1
        ):

            gc.collect()
            ### load data
            if isinstance(batch, list):
                batch = [
                    (b.to(self.device) if torch.is_tensor(b) else b) for b in batch
                ]
            else:
                batch = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }

            loss, loss_items, batch_output = self._eval_step(batch)

            eval_losses["total_loss"].append(loss.detach().cpu())

            for key in loss_items:
                eval_losses["eval/%s" % key].append(loss_items[key])

            # metric and logging
            if self.metrics is not None:
                self.metrics.step_process(batch_output)

            if eval_steps_per_epoch < self.log_audio_batchs and self.log_fn is not None:
                self.log_fn(
                    # self.writer, batch_output, self.epochs, eval_steps_per_epoch
                    self.writer,
                    batch_output,
                    self.steps,
                    eval_steps_per_epoch,
                )

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}

        # compute metrics
        if self.metrics is not None:
            # metrics_outcome = self.metrics.write(self.writer, self.epochs)
            metrics_outcome = self.metrics.write(self.writer, self.steps)
            eval_losses.update(metrics_outcome)

        _ = [self.model[k].train() for k in self.model]

        if not end_of_epoch:
            self.check_eval_and_save_checkpoint(eval_losses)

        return eval_losses

    def check_eval_and_save_checkpoint(self, eval_results):

        save_checkpoint = True

        if self.best_measure_by_metric is not None:
            if self.best_eval_result is None:
                self.best_eval_result = eval_results[self.best_measure_by_metric]
            elif (
                self.metric_measurement == "highest"
                and self.best_eval_result > eval_results[self.best_measure_by_metric]
            ):
                self.logger.info(
                    f"not saving checkingpoint in {self.epochs}, as the {self.best_measure_by_metric} {eval_results[self.best_measure_by_metric]} is lower then {self.best_eval_result}"
                )
                save_checkpoint = False
            elif self.best_eval_result < eval_results[self.best_measure_by_metric]:
                self.logger.info(
                    f"not saving checkingpoint in {self.epochs}, as the {self.best_measure_by_metric} {eval_results[self.best_measure_by_metric]} is higher then {self.best_eval_result}"
                )
                save_checkpoint = False

        if save_checkpoint:
            self.save_checkpoint(
                osp.join(
                    self.log_dir,
                    f"epoch_{self.epochs}_step_{self.steps}_best_{self.best_measure_by_metric}.pth",
                )
            )

    def fit(self):
        save_checkpoint = True
        for _ in trange(1, self.args.get("epochs", 100) + 1):
            epoch = self.epochs
            train_results = self._train_epoch()
            eval_results = self._eval_epoch(end_of_epoch=True)
            self.logger.info("--- epoch %d ---" % epoch)

            if self.best_measure_by_metric is not None:
                if self.best_eval_result is None:
                    self.best_eval_result = eval_results[self.best_measure_by_metric]
                elif (
                    self.metric_measurement == "highest"
                    and self.best_eval_result
                    > eval_results[self.best_measure_by_metric]
                ):
                    self.logger.info(
                        f"not saving checkingpoint in {self.epochs}, as the {self.best_measure_by_metric} {eval_results[self.best_measure_by_metric]} is lower then {self.best_eval_result}"
                    )
                    save_checkpoint = False
                elif self.best_eval_result < eval_results[self.best_measure_by_metric]:
                    self.logger.info(
                        f"not saving checkingpoint in {self.epochs}, as the {self.best_measure_by_metric} {eval_results[self.best_measure_by_metric]} is higher then {self.best_eval_result}"
                    )
                    save_checkpoint = False

            if (epoch % self.args.get("save_freq", 1)) == 0:
                self.save_checkpoint(osp.join(self.log_dir, "epoch_%05d.pth" % epoch))
