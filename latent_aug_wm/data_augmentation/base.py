import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

import torch_audiomentations

from torch_audiomentations.augmentations.padding import Padding
from torch_audiomentations.augmentations.shuffle_channels import ShuffleChannels
from torch_audiomentations.augmentations.splice_out import SpliceOut
from torch_audiomentations.core.transforms_interface import ModeNotSupportedException
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.utils.io import Audio

SPECIAL_AUG = ["HighPassFilter", "LowPassFilter"]


class BaseBatchAugmentation:
    """
    Current doesn't support multi-augmentation
    *** cannot implement highpass-lowpass filter in fp16
    """

    def __init__(
        self, sampling_rate=24000, transform_configs: dict = {}, add_no_aug=True
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
            for name, tc in transform_configs.items()
        ]
        # add no augmentation as option
        self.add_no_aug = add_no_aug
        if self.add_no_aug:
            self.transforms.append(("no_aug", (lambda x, s: x)))

    def __call__(self, input):
        name, transform = random.choice(self.transforms)
        # if name in SPECIAL_AUG and input.dtype == torch.float16:
        #    return transform(
        #        input.float(), self.sampling_rate).half()
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
        "AddColoredNoise": {
            "mode": "per_example",
            "p": 1.0,
            "min_snr_in_db": 10.0,
            "max_snr_in_db": 10.0,
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
