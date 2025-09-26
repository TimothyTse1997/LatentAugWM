import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from torch_audiomentations import (
    PolarityInversion,
    Gain,
    PeakNormalization,
    Compose,
    Shift,
    AddBackgroundNoise,
    ApplyImpulseResponse,
    AddColoredNoise,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    PitchShift,
    BandStopFilter,
    TimeInversion,
    Identity,
)
from torch_audiomentations.augmentations.padding import Padding
from torch_audiomentations.augmentations.shuffle_channels import ShuffleChannels
from torch_audiomentations.augmentations.splice_out import SpliceOut
from torch_audiomentations.core.transforms_interface import ModeNotSupportedException
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.utils.io import Audio

mode = "per_example"
SAMPLE_RATE = 24000

output_dir = "./demo_output/"

os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(43)
np.random.seed(43)
random.seed(43)

filename = "/home/tst000/projects/LatentAugWM/latent_aug_wm/f5_infer/0_gr.wav"
filenames = [filename]
audio = Audio(SAMPLE_RATE, mono=True)
samples = audio(filename).unsqueeze(0)


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s

    Warning: The time resolution used here may be limited to 1 ms
    """

    def __init__(self, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        if self.verbose:
            print("{}: {:.3f} s".format(self.description, self.execution_time))


transforms = [
    {"get_instance": lambda: BandPassFilter(mode=mode, p=1.0), "num_runs": 5},
    {"get_instance": lambda: BandStopFilter(mode=mode, p=1.0), "num_runs": 5},
    {
        "get_instance": lambda: Compose(
            transforms=[
                Gain(min_gain_in_db=-18.0, max_gain_in_db=-16.0, mode=mode, p=1.0),
                PeakNormalization(mode=mode, p=1.0),
            ],
            shuffle=True,
        ),
        "name": "Shuffled Compose with Gain and PeakNormalization",
        "num_runs": 5,
    },
    {
        "get_instance": lambda: Compose(
            transforms=[
                Gain(min_gain_in_db=-18.0, max_gain_in_db=-16.0, mode=mode, p=0.5),
                PolarityInversion(mode=mode, p=0.5),
            ],
            shuffle=True,
        ),
        "name": "Compose with Gain and PolarityInversion",
        "num_runs": 5,
    },
    {"get_instance": lambda: Gain(mode=mode, p=1.0), "num_runs": 5},
    {
        "get_instance": lambda: HighPassFilter(
            mode=mode, p=1.0, min_cutoff_freq=600, max_cutoff_freq=600
        ),
        "num_runs": 5,
    },
    {"get_instance": lambda: Identity(mode=mode, p=1.0), "num_runs": 1},
    {
        "get_instance": lambda: LowPassFilter(
            mode=mode, p=1.0, min_cutoff_freq=8000, max_cutoff_freq=8000
        ),
        "num_runs": 5,
    },
    {"get_instance": lambda: Padding(mode=mode, p=1.0), "num_runs": 5},
    {"get_instance": lambda: PeakNormalization(mode=mode, p=1.0), "num_runs": 1},
    {
        "get_instance": lambda: PitchShift(sample_rate=SAMPLE_RATE, mode=mode, p=1.0),
        "num_runs": 5,
    },
    {"get_instance": lambda: PolarityInversion(mode=mode, p=1.0), "num_runs": 1},
    {"get_instance": lambda: Shift(mode=mode, p=1.0), "num_runs": 5},
    {"get_instance": lambda: ShuffleChannels(mode=mode, p=1.0), "num_runs": 5},
    {
        "get_instance": lambda: SpliceOut(mode=mode, p=1.0),
        "num_runs": 5,
    },
    {
        "get_instance": lambda: AddColoredNoise(
            mode=mode, p=1.0, min_snr_in_db=10, max_snr_in_db=10.0
        ),
        "num_runs": 5,
    },
    {"get_instance": lambda: TimeInversion(mode=mode, p=1.0), "num_runs": 1},
]

execution_times = {}
for transform in transforms:
    try:
        augmenter = transform["get_instance"]()
    except ModeNotSupportedException:
        continue
    transform_name = (
        transform.get("name") if transform.get("name") else augmenter.__class__.__name__
    )
    execution_times[transform_name] = []
    for i in range(transform["num_runs"]):
        with timer() as t:
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
            print(
                augmenter.__class__.__name__,
                "is output ObjectDict:",
                type(augmented_samples) is ObjectDict,
            )
            augmented_samples = (
                augmented_samples.samples.numpy()
                if type(augmented_samples) is ObjectDict
                else augmented_samples.numpy()
            )
        execution_times[transform_name].append(t.execution_time)
        for example_idx, original_filename in enumerate(filenames):
            output_file_path = os.path.join(
                output_dir,
                "{}_{}_{:03d}_{}.wav".format(
                    transform_name, mode, i, Path(original_filename).stem
                ),
            )
            wavfile.write(
                output_file_path,
                rate=SAMPLE_RATE,
                data=augmented_samples[example_idx].transpose(),
            )

for transform_name in execution_times:
    if len(execution_times[transform_name]) > 1:
        print(
            "{:<52} {:.3f} s (std: {:.3f} s)".format(
                transform_name,
                np.mean(execution_times[transform_name]),
                np.std(execution_times[transform_name]),
            )
        )
    else:
        print(
            "{:<52} {:.3f} s".format(
                transform_name, np.mean(execution_times[transform_name])
            )
        )
