import os
from tqdm import tqdm
import random
from pathlib import Path
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio

from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import *


def get_ref_text_from_wav(wav_path):
    wav_path = Path(wav_path).absolute()
    wav_parent, wav_name = wav_path.parent, wav_path.name
    wav_name = wav_name.split(".")[0]
    text_fname = wav_parent / f"{wav_name}.normalized.txt"
    with open(text_fname, "r") as f:
        ref_text = f.readline().replace("\n", "")
    return ref_text


class MelDataset(Dataset):
    def __init__(
        self,
        ref_wav_file,
        gen_txt_fname,
        mel_spec_kwargs={},
        sampling_rate=24000,
        tmp_dir="./tmp",
        target_rms=0.1,
    ):
        self.ref_wav_file = ref_wav_file
        self.data = self._get_all_ref_wav_text(ref_wav_file)

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sampling_rate = sampling_rate
        self.tmp_dir = Path(tmp_dir)
        self.tmp_cache_files = {
            p.name: p.absolute() for p in self.tmp_dir.glob("*.wav")
        }
        self.target_rms = target_rms

    def _check_file_in_tmp(self, wav_fname):
        if (self.tmp_dir / Path(wav_fname).name).absolute() in self.tmp_cache_files:
            return True
        return False

    def _get_all_ref_wav_text(self, ref_wav_file):
        with open(ref_wav_file, "r") as f:
            all_wav_files = [line.rstrip() for line in f]
        return all_wav_files

    def load_audio(self, ref_audio_wav):
        audio, sr = torchaudio.load(ref_audio_wav)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
        return audio, sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_fname = self.data[index]
        ref_text = get_ref_text_from_wav(wav_fname)
        ref_audio_wav, ref_text = self.preprocess_ref_audio_text(
            ref_audio_orig, ref_text
        )

        audio, sr = self.load_audio(ref_audio_wav)

        cond = self.mel_spec(audio)
        cond = cond.permute(0, 2, 1)
        assert cond.shape[-1] == self.mel_spec.n_mel_channels

        return {
            "original_sr": sr,
            "ref_text": ref_text,
            "wav_fname": wav_fname,
            "ref_mel": cond,
        }

    def preprocess_ref_audio_text(
        self, index, ref_audio_orig, ref_text, show_info=print
    ):
        show_info("Converting audio...")

        # Compute a hash of the reference audio file
        with open(ref_audio_orig, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_hash = f"index_{index}_" + Path(ref_audio_orig).name

        if self._check_file_in_tmp(ref_audio_orig):
            show_info("Using cached preprocessed reference audio...")
            ref_audio = self.tmp_cache_files[audio_hash]

        else:  # first pass, do preprocess
            temp_path = self.tmp_dir / audio_hash

            aseg = AudioSegment.from_file(ref_audio_orig)

            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if (
                    len(non_silent_wave) > 6000
                    and len(non_silent_wave + non_silent_seg) > 12000
                ):
                    show_info("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if (
                        len(non_silent_wave) > 6000
                        and len(non_silent_wave + non_silent_seg) > 12000
                    ):
                        show_info("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 12s, clipping short. (3)")

            aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            aseg.export(temp_path, format="wav")
            ref_audio = temp_path

            # Cache the processed reference audio
            self.tmp_cache_files[audio_hash] = ref_audio.absolute()

        # Ensure ref_text ends with a proper sentence-ending punctuation
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        return ref_audio, ref_text
