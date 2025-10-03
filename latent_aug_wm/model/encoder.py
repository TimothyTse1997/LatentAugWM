import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# from torch.nn.utils.parametrizations import weight_norm
from torch import linalg as LA


class BasicEncoder(nn.Module):
    # Simple latent editting
    def __init__(
        self, *args, num_channel=100, max_length=2000, use_addition=False, **kwargs
    ):
        super().__init__()
        self.num_channel = num_channel
        self.max_length = max_length
        self.special_latent = torch.nn.Parameter(
            torch.randn(self.max_length, self.num_channel), requires_grad=True
        )
        self.use_addition = use_addition

    def forward(self, x, **kwargs):
        batch_size, seq_length, num_channel = x.shape
        assert num_channel == self.num_channel
        assert seq_length <= self.max_length
        # current_noise = torch.stack([torch.clone(self.special_latent) for _ in range(batch_size)])
        current_noise = self.special_latent.unsqueeze(0).repeat(batch_size, 1, 1)
        current_noise = current_noise[:, :seq_length, :]
        if self.use_addition:
            current_noise += x
        return current_noise


class UniqueNoiseEncoder(nn.Module):
    # set an initial noise such that there is something to compare with
    # (prevent the model to generate gibberish)
    def __init__(
        self,
        common_latent,
        num_channel=100,
        max_length=2000,
        initialize_magnitude=0.001,
        **kwargs
    ):
        super().__init__()
        self.num_channel = num_channel
        self.max_length = max_length
        self.common_latent = common_latent[:, : self.max_length].permute(1, 0)

        self.initialize_magnitude = initialize_magnitude
        self.special_latent = torch.nn.Parameter(
            torch.randn(self.max_length, self.num_channel) * initialize_magnitude,
            requires_grad=True,
        )
        if self.common_latent.shape != self.special_latent.shape:
            print(self.common_latent.shape, self.special_latent.shape)
        assert self.common_latent.shape == self.special_latent.shape

    def forward(self, x):
        batch_size, seq_length, num_channel = x.shape
        common_latent = self.common_latent.to(x.device).type(x.dtype)

        assert num_channel == self.num_channel
        assert seq_length <= self.max_length
        # current_noise = torch.stack([torch.clone(self.special_latent) for _ in range(batch_size)])
        current_noise = (
            (self.special_latent + common_latent).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        current_noise = current_noise[:, :seq_length, :]
        return current_noise

    def get_non_wm_latent(self, x):
        # created for contraining generation range
        batch_size, seq_length, num_channel = x.shape
        common_latent = self.common_latent.to(x.device).type(x.dtype)

        assert num_channel == self.num_channel
        assert seq_length <= self.max_length
        # current_noise = torch.stack([torch.clone(self.special_latent) for _ in range(batch_size)])
        current_noise = common_latent.unsqueeze(0).repeat(batch_size, 1, 1)

        current_noise = current_noise[:, :seq_length, :]
        return current_noise


class UniqueNoiseEncoderRemoveLen(nn.Module):
    # remove noise thats at the reference audio
    # also use weight norm as regularization
    def __init__(
        self,
        common_latent,
        num_channel=100,
        max_length=2000,
        max_weight_norm=0.01,
        **kwargs
    ):
        super().__init__()
        self.num_channel = num_channel
        self.max_length = max_length
        # self.common_latent = common_latent
        self.common_latent = common_latent[:, : self.max_length].permute(1, 0)
        self.max_weight_norm = max_weight_norm
        self.special_latent = torch.nn.Parameter(
            torch.zeros(self.max_length, self.num_channel),
            requires_grad=True,
        )
        assert self.common_latent.shape == self.special_latent.shape

    def forward(self, x, lens):
        batch_size, seq_length, num_channel = x.shape
        common_latent = self.common_latent.to(x.device).type(x.dtype)

        assert num_channel == self.num_channel
        assert seq_length <= self.max_length

        # normalize special latent
        special_latent = self.special_latent
        weight_norm = LA.norm(self.special_latent)

        if weight_norm > self.max_weight_norm:
            special_latent = self.max_weight_norm * special_latent / weight_norm

        current_noise = (
            special_latent + common_latent  # .unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # current_noise = current_noise[:, :seq_length, :]
        for batch_id, length in enumerate(lens):
            current_seq_length = seq_length - length
            x[batch_id, length:] = current_noise[:current_seq_length]
        return current_noise

    def get_non_wm_latent(self, x, lens):
        # created for contraining generation range
        batch_size, seq_length, num_channel = x.shape
        common_latent = self.common_latent.to(x.device).type(x.dtype)

        assert num_channel == self.num_channel
        assert seq_length <= self.max_length
        # current_noise = torch.stack([torch.clone(self.special_latent) for _ in range(batch_size)])
        current_noise = common_latent.unsqueeze(0)  # .repeat(batch_size, 1, 1)

        current_noise = current_noise[:, :seq_length, :]
        for batch_id, length in enumerate(lens):
            current_seq_length = seq_length - length
            x[batch_id, length:] = current_noise[:current_seq_length]
        return current_noise


if __name__ == "__main__":
    common_latent = torch.randn((2000, 100))
    noise_update_fn = UniqueNoiseEncoder(common_latent).half().cuda()

    from latent_aug_wm.dataset.mel_dataset import get_combine_dataloader

    ref_wav_file = "/home/tst000/projects/datasets/selected_ref_files.txt"
    gen_txt_fname = "/home/tst000/projects/datasets/selected_gen_text.txt"

    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    tmp_dir = "/home/tst000/projects/tmp/libriTTS"

    data_iter = get_combine_dataloader(
        ref_wav_file=ref_wav_file,
        gen_txt_fname=gen_txt_fname,
        mel_spec_kwargs=mel_spec_kwargs,
        tmp_dir="/home/tst000/projects/tmp/libriTTS",
        shuffle=False,
        unsorted_batch_size=256,
        batch_size=32,
        allowed_padding=50,
    )

    batch = next(data_iter)

    from latent_aug_wm.f5_infer.infer import F5TTSBatchInferencer

    sampler = F5TTSBatchInferencer(
        device="cuda", noise_update_fn=noise_update_fn, train=True
    )

    generated_dict = sampler(
        cond=batch["cond"].cuda(),
        texts=batch["texts"],
        durations=batch["durations"].cuda(),
        lens=batch["lens"].cuda(),
    )
    for k, v in generated_dict.items():
        print(k, v.shape)
