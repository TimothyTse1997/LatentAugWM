import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_aug_wm.f5_infer.forward_backward import (
    sample_from_model,
    forward_backward_sample_from_model,
)
from latent_aug_wm.loss.w2v import W2VPeftLatentLoss


def curriculum_noise_encoder_generate_mel_from_f5tts_with_aug_noise(
    nets=None, step=None, eval=False, **kwargs
):
    # ONLY works with model.encoder.UniqueNoiseEncoder

    # orig_inv_noise = kwargs["orig_inv_noise"]
    wm_noise = kwargs["aug_noise"]

    wm_out = nets.f5tts(
        fix_noise=wm_noise, use_grad_checkpoint=(not eval), eval=eval, **kwargs
    )

    wm_out = {("wm_" + k): v for k, v in wm_out.items()}
    return wm_out, []


class W2VLatentLossFakeOnly(W2VPeftLatentLoss):
    def __call__(self, nets=None, step=None, scale=1.0, **kwargs):
        fake_input = kwargs["wm_gr_wave"]
        batch_size, seq_len = fake_input.shape

        self.transform = self.transforms.to(fake_input.device)

        fake_input_resample = self.transforms(fake_input)

        fake_logits = torch.zeros(
            batch_size, dtype=torch.long, device=fake_input.device
        )

        fake_out = nets.detector(fake_input_resample)

        fake_detector_logits, fake_loss = nets.classifier.compute_loss(
            fake_out.last_hidden_state, fake_logits
        )

        return {
            "detector_loss": fake_loss * scale,
            "fake_detector_logits": fake_detector_logits.detach().cpu(),
        }, []


class W2VLatentLossRealOnly(W2VPeftLatentLoss):
    def __call__(self, nets=None, step=None, scale=1.0, **kwargs):
        real_input = kwargs["wm_gr_wave"]
        batch_size, seq_len = real_input.shape

        self.transform = self.transforms.to(real_input.device)

        real_input_resample = self.transforms(real_input)

        real_logits = torch.ones(batch_size, dtype=torch.long, device=real_input.device)

        real_out = nets.detector(real_input_resample)

        real_detector_logits, real_loss = nets.classifier.compute_loss(
            real_out.last_hidden_state, real_logits
        )

        return {
            "detector_loss": real_loss * scale,
            "real_detector_logits": real_detector_logits.detach().cpu(),
        }, []


class W2VDetectorStageLoss(W2VLatentLossRealOnly):
    def __call__(self, nets=None, step=None, scale=1.0, **kwargs):
        real_input = kwargs["wm_gr_wave"]
        fake_input = kwargs["rand_gr_wave"]
        batch_size, seq_len = real_input.shape

        self.transform = self.transforms.to(real_input.device)

        real_input_resample = self.transforms(real_input)
        fake_input_resample = self.transforms(fake_input)

        real_logits = torch.ones(batch_size, dtype=torch.long, device=real_input.device)
        fake_logits = torch.zeros(
            batch_size, dtype=torch.long, device=fake_input.device
        )

        real_out = nets.detector(real_input_resample)
        fake_out = nets.detector(fake_input_resample)

        real_detector_logits, real_loss = nets.classifier.compute_loss(
            real_out.last_hidden_state, real_logits
        )
        fake_detector_logits, fake_loss = nets.classifier.compute_loss(
            fake_out.last_hidden_state, fake_logits
        )

        return {
            "detector_loss": (real_loss + fake_loss) * scale,
            "real_detector_logits": real_detector_logits.detach().cpu(),
            "fake_detector_logits": fake_detector_logits.detach().cpu(),
        }, ["detector", "classifier"]
