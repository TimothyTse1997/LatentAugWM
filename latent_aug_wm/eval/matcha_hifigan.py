import json
from mpl_toolkits.mplot3d import Axes3D

import datetime as dt
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
import torch.nn.functional as F
from tqdm.auto import tqdm

import torch.nn as nn

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.model import denormalize

MEL_PARAMETERS = {
    "n_fft": 1024,
    "n_mels": 80,
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "f_min": 0.0,
    "f_max": 8000,
}

VCTK_MEL_STAT = {"mel_mean": -6.630575, "mel_std": 2.482914}


def get_mel(filepath, sample_rate=22050):
    audio, sr = ta.load(filepath)
    if not sr == MEL_PARAMETERS["sample_rate"]:
        print("resampling!!")
        audio = ta.functional.resample(
            audio, orig_freq=sr, new_freq=MEL_PARAMETERS["sample_rate"]
        )
    # assert sr == MEL_PARAMETERS["sample_rate"]
    mel = mel_spectrogram(
        audio,
        MEL_PARAMETERS["n_fft"],
        MEL_PARAMETERS["n_mels"],
        MEL_PARAMETERS["sample_rate"],
        MEL_PARAMETERS["hop_length"],
        MEL_PARAMETERS["win_length"],
        MEL_PARAMETERS["f_min"],
        MEL_PARAMETERS["f_max"],
        center=False,
    ).squeeze()
    # mel = mel_spectrogram(
    #    audio,
    #    center=False,
    #    **MEL_PARAMETERS,
    # ).squeeze()
    if mel.shape[-1] % 2 != 0:
        print("before pad:", mel.shape)
        mel = F.pad(mel, (0, 1), "constant", 0)
        print("after pad:", mel.shape)
    mel = normalize(mel, VCTK_MEL_STAT["mel_mean"], VCTK_MEL_STAT["mel_std"])
    return mel


@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()


def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["generator"]
    )
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


cache_dir = Path("/home/tst000/projects/tst000/.local/share")


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MATCHA_CHECKPOINT = cache_dir / "matcha_vctk.ckpt"
HIFIGAN_CHECKPOINT = cache_dir / "g_02500000"  # "hifigan_T2_v1" / "generator_v1"

vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
denoiser = Denoiser(vocoder, mode="zeros")

test_file = "/home/tst000/projects/tst000/LatentAugWM/0.wav"
mel = get_mel(test_file)

audio = to_waveform(mel.cuda(), vocoder)

sf.write(
    "/home/tst000/projects/tst000/LatentAugWM/0_matcha_hifigan.wav",
    audio,
    22050,
    "PCM_24",
)
