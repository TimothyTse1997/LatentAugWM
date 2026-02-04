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
from f5_tts.infer.utils_infer import save_spectrogram
from latent_aug_wm.model.starganv2 import StarGanDetector


def _get_mp3(wav_path):
    mp3_path = wav_path.parent / (wav_path.name.split(".")[0] + ".mp3")
    return mp3_path


def _get_all_wav_text(wav_file):
    with open(wav_file, "r") as f:
        all_wav_files = [Path(line.rstrip()) for line in f]
    return all_wav_files


def load_audio(audio_wav, sampling_rate=24000, target_rms=0.1):
    audio, sr = torchaudio.load(audio_wav)
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        audio = resampler(audio)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        audio = resampler(audio)
    return audio, sr


def resample_augmentation(audio, sr, aug_sr=16000):
    resampler = torchaudio.transforms.Resample(sr, aug_sr)
    audio = resampler(audio)
    # print(audio.shape)
    resampler = torchaudio.transforms.Resample(aug_sr, sr)
    audio = resampler(audio)
    # print(audio.shape)
    return audio


def _load(states, model, force_load=True):
    model_states = model.state_dict()
    for key, val in states.items():
        try:
            if key not in model_states:
                continue
            if isinstance(val, nn.Parameter):
                val = val.data

            if val.shape != model_states[key].shape:
                print("%s does not have same shape" % key)
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
            print("not exist ", key)


n_mel_channels = 100

detector = StarGanDetector(dim_in=n_mel_channels)
# checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector/version_debug_10_new_dataloader/epoch_14_step_2100_best_f1_0.6580027341842651.pth"
# checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector/version_debug_13_hifigan/epoch_15_step_9000_best_f1_0.3762376010417938.pth"
checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector/version_debug_13_hifigan/epoch_7_step_4000_best_f1_0.3762376010417938.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")

_load(state_dict["model"]["detector"], detector)

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

# data_file = "/home/tst000/projects/tst000/datasets/f5tts_dataset_periodic_test_dataset.txt"
# data_file = "/home/tst000/projects/tst000/datasets/f5tts_random_audio_test.txt"
# data_file = "/home/tst000/projects/tst000/datasets/selected_ref_files_train.txt"
# data_file = "/home/tst000/projects/tst000/datasets/hifigan_16k_random_audio_test.txt"
# all_paths = _get_all_wav_text(data_file)

# all_paths = list(Path("/home/tst000/projects/tst000/datasets/bigvgan_random_gen_audio/").glob("*.wav"))
# all_paths = list(Path("/home/tst000/projects/tst000/datasets/bigvgan_periodic_gen_audio/").glob("*.wav"))

# all_paths = list(Path("/home/tst000/projects/tst000/datasets/hifigan_16k_periodic_gen_audio/").glob("*.wav"))
# all_paths = list(Path("/home/tst000/projects/tst000/datasets/hifigan_16k_random_gen_audio/").glob("*.wav"))
all_paths = list(
    Path("/home/tst000/projects/tst000/datasets/bigvgan_22k_periodic_gen_audio/").glob(
        "*.wav"
    )
)

# all_paths = list(Path("/home/tst000/projects/tst000/datasets/f5tts_new_text_periodic_noise_test_set").glob("*.wav"))
# all_paths = list(Path("/home/tst000/projects/tst000/datasets/hifigan_16k_new_text_periodic_noise_test_set/").glob("*.wav"))

mel_spec = MelSpec(**mel_spec_kwargs)

detector.eval()
detector.cuda()
all_labels = []
print(all_paths[:2])
N = 0
for p in tqdm(all_paths):
    try:
        # if True:
        with torch.no_grad():
            mp3_fname = p  # _get_mp3(p)
            # audio, sr = load_audio(mp3_fname, sampling_rate=22050)
            audio, sr = load_audio(mp3_fname, sampling_rate=24000)

            # audio_aug = resample_augmentation(audio, sr, aug_sr=16000)

            audio_mel = mel_spec(audio)
            audio_mel = torchaudio.functional.resample(audio_mel, 22050, 24000)
            # audio_mel = mel_spec(audio_aug)
            if N < 2:
                save_spectrogram(
                    audio_mel.float().cpu().detach()[0],
                    f"mel_new_dataset_{N}.png",
                )
            audio_mel = audio_mel.permute(0, 2, 1).cuda()
            logit = detector.get_feature(audio_mel).squeeze(-1)
            label = torch.argmax(logit, dim=0)
            all_labels.append(label.item())
            N += 1
    except:
        continue
    # print(detector.get_feature(audio_mel))
# print(all_labels)
print()
print(Counter(all_labels))
print(all_paths[0])
print("here")
print("here")
print("here")
print("here")
print("here")
print("here")
print("here")
print("here")
