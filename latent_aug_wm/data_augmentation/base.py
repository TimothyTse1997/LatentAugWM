import random
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

import torchaudio
import torch_audiomentations

from torch_audiomentations.augmentations.padding import Padding
from torch_audiomentations.augmentations.shuffle_channels import ShuffleChannels
from torch_audiomentations.augmentations.splice_out import SpliceOut
from torch_audiomentations.core.transforms_interface import ModeNotSupportedException
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.augmentations.random_crop import RandomCrop

SPECIAL_AUG = ["HighPassFilter", "LowPassFilter"]


class CustomRandomCrop:
    def __init__(self, max_length, min_length, sampling_rate=24000):
        self.max_frame = int(max_length * sampling_rate)
        self.min_length = int(min_length * sampling_rate)
        self.sampling_rate = sampling_rate

    def __call__(self, audio):
        # audio: 1, 1, audio_length
        _, _, length = audio.shape
        crop_length = int(random.uniform(self.min_length, self.max_frame))
        if length <= crop_length:
            return audio

        max_start_spot = length - crop_length
        start_index = int(random.uniform(0, max_start_spot - 1))
        end_index = start_index + crop_length
        return audio[:, :, start_index:end_index]


# class CustomSpeedAugmentation:
#     def __init__(self, max_speed=1.10, min_speed=0.9, sampling_rate=24000):
#         self.max_speed = max_speed
#         self.min_speed = min_speed
#         self.sampling_rate = sampling_rate

#     def __call__(self, audio):
#         # audio: 1, 1, audio_length
#         speed = random.uniform(self.max_speed, self.min_speed)
#         speed_fn = torchaudio.transforms.Speed(self.sampling_rate, speed)

#         aug_audio, _ = speed_fn(audio.squeeze(0)).unsqueeze(0)
#         return aug_audio


class BaseBatchAugmentation:
    """
    Current doesn't support multi-augmentation
    *** cannot implement highpass-lowpass filter in fp16
    """

    def __init__(
        self,
        sampling_rate=24000,
        transform_configs: dict = {},
        add_no_aug=True,
        crop_config={},
        speed_config={
            "orig_freq": 24000,
            "factors": [0.9, 0.95, 1.0, 1.0, 1.0, 1.05, 1.1],
        },
    ):

        # transforms = [
        #     name: getattr(torch_audiomentations, name)(**tc) \
        #         for name, tc in transform_configs.items()
        # ]
        # self.transform = torch_audiomentations.Compose(
        #     transforms=transforms
        # )
        self.transforms = []
        self.sampling_rate = sampling_rate
        self.transforms = [
            (name, getattr(torch_audiomentations, name)(**tc))
            for name, tc in transform_configs.items()  # if hasattr(torch_audiomentations, name)
        ]

        # add no augmentation as option
        self.add_no_aug = add_no_aug
        if self.add_no_aug:
            self.transforms.append(("no_aug", (lambda x, s: x)))
        self.crop_config = crop_config
        if self.crop_config:
            # self.crop_fn = RandomCrop(**self.crop_config)
            self.crop_fn = CustomRandomCrop(**self.crop_config)
        self.speed_config = speed_config

        self.speed_fn = None
        if self.speed_config:
            self.speed_fn = torchaudio.transforms.SpeedPerturbation(**speed_config)

    def speed_adj(self, input):
        if self.speed_fn is None:
            return input
        return self.speed_fn(input)

    def __call__(self, input):
        name, transform = random.choice(self.transforms)
        # if name in SPECIAL_AUG and input.dtype == torch.float16:
        #    return transform(
        #        input.float(), self.sampling_rate).half()
        if self.crop_config:
            input = self.crop_fn(input)
        input, _ = self.speed_adj(input)
        return transform(input, self.sampling_rate)


if __name__ == "__main__":
    transform_configs = {
        # "HighPassFilter": {
        #     "mode": "per_example",
        #     "p": 1.0, "min_cutoff_freq": 400,
        #     "max_cutoff_freq": 600
        # },
        # "LowPassFilter": {
        #     "mode": "per_example",
        #     "p": 1.0, "min_cutoff_freq": 8000,
        #     "max_cutoff_freq": 8000
        # },
        # "AddColoredNoise": {
        #     "mode": "per_example",
        #     "p": 1.0,
        #     "min_snr_in_db": 10.0,
        #     "max_snr_in_db": 10.0,
        # }
        "BandPassFilter": {
            "mode": "per_example",
            "p": 1.0,
        }
    }
    # scaler = torch.cuda.amp.GradScaler()
    aug_obj = BaseBatchAugmentation(
        sampling_rate=24000, transform_configs=transform_configs
    )
    dummy_input = torch.randn(16, 1, 1000).cuda()
    # test fp16
    for i in range(10):
        with torch.cuda.amp.autocast():
            print(aug_obj(dummy_input).shape)
