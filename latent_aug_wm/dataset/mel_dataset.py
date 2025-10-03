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
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

from latent_aug_wm.dataset.data_utils import (
    F5TTSCollator,
    batch_filter,
    recursive_batch_filtering,
)


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
        mel_spec_kwargs={},
        sampling_rate=24000,
        tmp_dir="./tmp",
        target_rms=0.1,
        debug=False,
    ):
        self.ref_wav_file = ref_wav_file
        self.data = self._get_all_ref_wav_text(ref_wav_file)

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sampling_rate = sampling_rate
        self.tmp_dir = Path(tmp_dir)
        self.tmp_cache_files = {
            p.name: str(p.absolute()) for p in self.tmp_dir.glob("*.wav")
        }
        self.target_rms = target_rms
        self.dummy_print = lambda x: None
        self.debug = debug

    def _check_file_in_tmp(self, wav_fname):
        # cache_fname = str((self.tmp_dir / Path(wav_fname).name).absolute())

        # print(self.tmp_cache_files)
        # print("cache_fname", cache_fname)

        if wav_fname in self.tmp_cache_files:
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
            index,
            wav_fname,
            ref_text,
            show_info=(print if self.debug else self.dummy_print),
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

        if self._check_file_in_tmp(audio_hash):
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


class TextDataset(Dataset):
    def __init__(self, gen_txt_fname):
        self.gen_txt_fname = gen_txt_fname
        self.data = self._get_all_gen_text(self.gen_txt_fname)
        pass

    def _get_all_gen_text(self, gen_txt_fname):
        with open(gen_txt_fname, "r") as f:
            text = [line.rstrip() for line in f]
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_combine_dataloader(
    ref_wav_file="/home/tst000/projects/datasets/selected_ref_files.txt",
    gen_txt_fname="/home/tst000/projects/datasets/selected_gen_text.txt",
    mel_spec_kwargs={
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    },
    tmp_dir="/home/tst000/projects/tmp/libriTTS",
    shuffle=True,
    unsorted_batch_size=256,
    batch_size=32,
    allowed_padding=100,
    steps_per_epoch=3000,
):
    mel_dataset = MelDataset(
        ref_wav_file,
        mel_spec_kwargs=mel_spec_kwargs,
        sampling_rate=mel_spec_kwargs["target_sample_rate"],
        tmp_dir=tmp_dir,
        target_rms=0.1,
    )
    text_dataset = TextDataset(gen_txt_fname)
    collator = F5TTSCollator()
    mel_dataloader = DataLoader(
        mel_dataset,
        batch_size=unsorted_batch_size,
        collate_fn=collator,
        shuffle=shuffle,
    )
    text_dataloader = DataLoader(
        text_dataset, batch_size=unsorted_batch_size, shuffle=shuffle
    )

    class DataIter:
        def __init__(self):
            self.index = 0
            self.storage = []
            self.max_batch = min(len(mel_dataloader), len(text_dataloader))
            pass

        def __len__(self):
            return max(self.max_batch, steps_per_epoch)

        def __iter__(self):
            self.index = 0
            self.storage = []
            return self

        def __next__(self):
            if self.index > steps_per_epoch:
                raise StopIteration
            self.index += 1
            mel_data, text_data = next(iter(mel_dataloader)), next(
                iter(text_dataloader)
            )
            mel_data["gen_texts"] = text_data

            if self.storage:
                # print(len(self.storage))
                return self.storage.pop(0)

            for result in recursive_batch_filtering(
                final_batch_size=batch_size, allowed_padding=allowed_padding, **mel_data
            ):
                result["lens"] = torch.tensor(result["lens"], dtype=torch.long)
                result["duration"] = torch.tensor(result["durations"], dtype=torch.long)
                result["cond"] = pad_sequence(result["cond"], batch_first=True)

                result["combine_texts"] = [
                    rt + gt for rt, gt in zip(result["ref_texts"], result["gen_texts"])
                ]
                result["text"] = [
                    sum(convert_char_to_pinyin(ct), [])
                    for ct in result["combine_texts"]
                ]
                self.storage.append(result)
                # yield result
            # print(len(self.storage))
            return self.storage.pop(0)

    # def data_iter():
    #     for mel_data, text_data in zip(mel_dataloader, text_dataloader):
    #         mel_data["gen_texts"] = text_data

    #         for result in recursive_batch_filtering(
    #             final_batch_size=batch_size, allowed_padding=allowed_padding, **mel_data
    #         ):
    #             result["lens"] = torch.tensor(result["lens"], dtype=torch.long)
    #             result["duration"] = torch.tensor(result["durations"], dtype=torch.long)
    #             result["cond"] = pad_sequence(result["cond"], batch_first=True)

    #             result["combine_texts"] = [
    #                 rt + gt for rt, gt in zip(result["ref_texts"], result["gen_texts"])
    #             ]
    #             result["text"] = [
    #                 sum(convert_char_to_pinyin(ct), [])
    #                 for ct in result["combine_texts"]
    #             ]

    #             yield result
    start = time.time()
    dataiter = DataIter()
    print("iter create time: ", time.time() - start)
    return dataiter


if __name__ == "__main__":
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
        unsorted_batch_size=2048,
        batch_size=32,
    )

    for i, result in enumerate(tqdm.tqdm(data_iter)):
        print("lens afterward", result["lens"])
        # print(pad_sequence(result["cond"], batch_first=True).shape)
        print("dur: ", result["durations"])
        if i > 50:
            break
