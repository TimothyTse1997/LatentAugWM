import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


def apply_effect(waveform, sample_rate, effect):
    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)


def echo(audio_path, fix_length=True):
    waveform1, sample_rate = torchaudio.load(audio_path, channels_first=False)
    effect = (
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
    )
    waveform2 = apply_effect(waveform1, sample_rate, effect)
    if fix_length:
        initial_length = waveform1.shape[0]
        waveform2 = waveform2[:initial_length, :]
        assert waveform2.shape == waveform1.shape
    return waveform2.transpose(1, 0)


def echo_file_conversion(source_file, target_file, sampling_rate=16000):
    wav = echo(source_file)
    torchaudio.save(target_file, wav, sample_rate=sampling_rate)


if __name__ == "__main__":
    audio_path = "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/diffwave/mp3/501.mp3"
    print(echo(audio_path).shape)
