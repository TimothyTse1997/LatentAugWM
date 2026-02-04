import os
from tqdm import tqdm
import random
from pathlib import Path
import time
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import torchaudio

from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import save_spectrogram, load_vocoder

vocos = load_vocoder()

# gaussian_noise = torch.randn(1, 240000)
mel_spec_kwargs = {
    "target_sample_rate": 24000,
    # "target_sample_rate": 22050,
    "n_mel_channels": 100,
    "hop_length": 256,
    # "hop_length": 235,
    "win_length": 1024,
    # "win_length": 941,
    "n_fft": 1024,
    # "n_fft": 941,
    "mel_spec_type": "vocos",
}

mel_spec = MelSpec(**mel_spec_kwargs)
# audio_mel = mel_spec(gaussian_noise)
# save_spectrogram(
#    audio_mel.float().cpu().detach()[0],
#    "mel_just_noise.png",
# )


# rand_mel = torch.randn(100, 1000)
rand_mel_bin = [
    torch.randn(20, 64),
    torch.randn(20, 128),
    torch.randn(20, 256),
    torch.randn(20, 512),
    torch.randn(20, 1024),
]
# rand_mel_bin = [
#     torch.randn(20, 256),
#     torch.randn(20, 256),
#     torch.randn(20, 256),
#     torch.randn(20, 256),
#     torch.randn(20, 256),
# ]

rand_mel = torch.zeros((100, 1000))
start_index = 0

for bin_ in rand_mel_bin:
    end_index = start_index + 20
    rand_mel[start_index:end_index]

    latent_size = bin_.shape[1]
    max_length = 1000
    num_repeat = 1 + (max_length // latent_size)
    current_latent = bin_

    print(f"before repeat {num_repeat}: ", current_latent.shape, max_length)

    current_latent = current_latent.repeat(1, num_repeat)
    print(f"after repeat {num_repeat}: ", current_latent.shape)

    rand_mel[start_index:end_index, :] = current_latent[:, :max_length]
    start_index = end_index

save_spectrogram(
    rand_mel.float().cpu().detach(),
    "mel_just_noise.png",
)

device = "cuda"


wav_gen = vocos.decode(rand_mel.unsqueeze(0).cuda())
wav_gen_float = wav_gen.squeeze(0)


from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

source = "/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-22050Hz/snapshots/4188503131602dc234f48d7f22eebea93d788736/"


hifi_gan = HIFIGAN.from_hparams(
    source=source  # savedir="pretrained_models/tts-hifigan-libritts-16kHz"
)

wav = torchaudio.functional.resample(wav_gen_float, 24000, 22050)
spectrogram, _ = mel_spectogram(
    audio=wav.squeeze(),
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True,
)
with torch.inference_mode():
    waveforms = hifi_gan.decode_batch(spectrogram)

waveforms = waveforms.squeeze(1)

wav = torchaudio.functional.resample(waveforms, 22050, 24000)
recon_mel = mel_spec(wav)
save_spectrogram(
    recon_mel.float().cpu().detach()[0],
    "mel_recon.png",
)
