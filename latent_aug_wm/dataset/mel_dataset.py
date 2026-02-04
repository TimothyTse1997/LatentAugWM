import json
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

        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

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


class MelDatasetWithAug(Dataset):
    def __init__(
        self,
        wav_fname_file,
        mel_spec_kwargs={},
        use_mp3=True,
        mp3_prob=0.1,
        sampling_rate=24000,
        target_label=0,
        target_rms=0.1,
        fname_aug_fn=None,
        aug_obj=None,
    ):
        self.data = self._get_all_wav_text(wav_fname_file)
        self.aug_obj = aug_obj
        self.use_mp3 = use_mp3
        self.mp3_prob = mp3_prob
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sampling_rate = sampling_rate
        self.target_rms = target_rms
        self.target_label = target_label
        self.fname_aug_fn = fname_aug_fn

    def _get_mp3(self, wav_path):
        mp3_path = wav_path.parent / (wav_path.name.split(".")[0] + ".mp3")
        return mp3_path

    def _get_all_wav_text(self, wav_file):
        with open(wav_file, "r") as f:
            all_wav_files = [Path(line.rstrip()) for line in f]
        return all_wav_files

    def load_audio(self, audio_wav):
        audio, sr = torchaudio.load(audio_wav)
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

    @staticmethod
    def collation(batch):
        audio_mel = [b["audio_mel"][0] for b in batch]
        labels = [b["label"] for b in batch]
        labels = torch.tensor(labels, dtype=torch.long)
        audio_mel = pad_sequence(audio_mel, batch_first=True)

        audios = [b["audio_wave"][0] for b in batch]
        audios = pad_sequence(audios, batch_first=True)
        return {"audio_mel": audio_mel, "audio_wave": audios, "label": labels}

    def __getitem__(self, index):
        wav_fname = self.data[index]
        if self.fname_aug_fn is not None:
            wav_fname = self.fname_aug_fn(wav_fname)
        if not self.use_mp3 or random.uniform(0, 1) > self.mp3_prob:
            audio, sr = self.load_audio(wav_fname)
        else:
            mp3_fname = self._get_mp3(wav_fname)
            audio, sr = self.load_audio(mp3_fname)
        if self.aug_obj is not None:
            audio = self.aug_obj(audio.unsqueeze(1)).squeeze(1)
        audio_mel = self.mel_spec(audio)
        audio_mel = audio_mel.permute(0, 2, 1)
        assert audio_mel.shape[-1] == self.mel_spec.n_mel_channels
        return {"audio_mel": audio_mel, "audio_wave": audio, "label": self.target_label}


class MelDatasetWithAugV2(MelDatasetWithAug):
    """
    generalize with updated metadata
    """

    def __init__(
        self,
        metadata=None,
        metadata_path=None,
        mel_spec_kwargs={},
        sampling_rate=24000,
        target_label=0,
        target_rms=0.1,
        fname_aug_fn=None,
        aug_obj=None,
    ):
        if metadata is None:
            self.data = json.load(open(metadata_path, "r"))
        else:
            self.data = metadata
        self.aug_obj = aug_obj

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sampling_rate = sampling_rate
        self.target_rms = target_rms
        self.target_label = target_label
        self.fname_aug_fn = fname_aug_fn

    def __getitem__(self, index):
        metadata = self.data[index]
        # choose from mp3 mp4 or wav
        wav_fname = random.choice(metadata["audio_paths"])

        if self.fname_aug_fn is not None:
            wav_fname = self.fname_aug_fn(wav_fname)

        audio, sr = self.load_audio(wav_fname)
        if self.aug_obj is not None:
            audio = self.aug_obj(audio.unsqueeze(1)).squeeze(1)

        audio_mel = self.mel_spec(audio)
        audio_mel = audio_mel.permute(0, 2, 1)

        assert audio_mel.shape[-1] == self.mel_spec.n_mel_channels
        return {"audio_mel": audio_mel, "audio_wave": audio, "label": self.target_label}


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


def load_basic_dataloader(
    real_wav_fname_file, fake_wav_fname_file, mel_spec_kwargs, batch_size=10, **kwargs
):
    real_basic_dataset = MelDatasetWithAug(
        real_wav_fname_file, mel_spec_kwargs=mel_spec_kwargs, target_label=1, **kwargs
    )
    fake_basic_dataset = MelDatasetWithAug(
        fake_wav_fname_file, mel_spec_kwargs=mel_spec_kwargs, target_label=0, **kwargs
    )

    combine_dataset = torch.utils.data.ConcatDataset(
        [real_basic_dataset, fake_basic_dataset]
    )
    dataloader = DataLoader(
        combine_dataset,
        batch_size=batch_size,
        collate_fn=MelDatasetWithAug.collation,
        shuffle=True,
    )
    return dataloader


def load_multi_simple_dataloader(
    dataset_configs, batch_size=10, dataset_cls=MelDatasetWithAug, **kwargs
):
    datasets = [
        dataset_cls(**dataset_config, **kwargs) for dataset_config in dataset_configs
    ]

    combine_dataset = torch.utils.data.ConcatDataset(datasets)

    dataloader = DataLoader(
        combine_dataset,
        batch_size=batch_size,
        collate_fn=MelDatasetWithAug.collation,
        shuffle=True,
    )
    return dataloader


if __name__ == "__main__":
    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    wav_fname_file = (
        "/home/tst000/projects/tst000/datasets/f5tts_dataset_periodic_train_dataset.txt"
    )
    basic_dataset = MelDatasetWithAug(wav_fname_file, mel_spec_kwargs=mel_spec_kwargs)
    fake_basic_dataset = MelDatasetWithAug(
        wav_fname_file, mel_spec_kwargs=mel_spec_kwargs, target_label=1
    )
    combine_dataset = torch.utils.data.ConcatDataset(
        [basic_dataset, fake_basic_dataset]
    )
    dataloader = DataLoader(
        combine_dataset,
        batch_size=10,
        collate_fn=MelDatasetWithAug.collation,
        shuffle=True,
    )
    print(next(iter(dataloader)))

    # ref_wav_file = "/home/tst000/projects/datasets/selected_ref_files.txt"
    # gen_txt_fname = "/home/tst000/projects/datasets/selected_gen_text.txt"

    # mel_spec_kwargs = {
    #     "target_sample_rate": 24000,
    #     "n_mel_channels": 100,
    #     "hop_length": 256,
    #     "win_length": 1024,
    #     "n_fft": 1024,
    #     "mel_spec_type": "vocos",
    # }
    # tmp_dir = "/home/tst000/projects/tmp/libriTTS"

    # data_iter = get_combine_dataloader(
    #     ref_wav_file=ref_wav_file,
    #     gen_txt_fname=gen_txt_fname,
    #     mel_spec_kwargs=mel_spec_kwargs,
    #     tmp_dir="/home/tst000/projects/tmp/libriTTS",
    #     shuffle=False,
    #     unsorted_batch_size=2048,
    #     batch_size=32,
    # )

    # for i, result in enumerate(tqdm.tqdm(data_iter)):
    #     print("lens afterward", result["lens"])
    #     # print(pad_sequence(result["cond"], batch_first=True).shape)
    #     print("dur: ", result["durations"])
    #     if i > 50:
    #         break
