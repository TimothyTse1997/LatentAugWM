import random
import json
from pathlib import Path
from collections import defaultdict

from importlib.resources import files
from tqdm import tqdm
from functools import partial
from cached_path import cached_path

import numpy as np
import torch
import torchaudio

from datasets import load_dataset

from omegaconf import OmegaConf
from hydra.utils import get_class

import torch_audiomentations

from f5_tts.api import F5TTS
from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import save_spectrogram, load_model

from latent_aug_wm.dataset.mel_dataset import (
    get_combine_dataloader,
    get_ref_text_from_wav,
)
from latent_aug_wm.f5_infer.infer import (
    F5TTSFixNoiseInferencer,
    F5TTSPeriodicFixNoiseInferencer,
)
from latent_aug_wm.data_augmentation.torchaudio_io_augmentation import echo


def get_ref_text_fname_from_wav(wav_path):
    wav_path = Path(wav_path).absolute()
    wav_parent, wav_name = wav_path.parent, wav_path.name
    wav_name = wav_name.split(".")[0]
    text_fname = wav_parent / f"{wav_name}.normalized.txt"
    return text_fname


def main_random_latent(target_format="mp3"):
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts"
    )
    sampler = F5TTS(device="cuda")
    text_dataset = load_dataset("jakeazcona/short-text-labeled-emotion-classification")[
        "test"
    ]

    fix_latent_path = "/gpfs/fs3c/nrc/dt/tst000/datasets/periodic_special_latent.pt"

    gen_wav_save_dir = save_dir / "wav"
    # augmentations which cannot be done in the mel-space
    gen_echo_save_dir = save_dir / "echo_wav"
    gen_mp3_save_dir = save_dir / "mp3"

    if not source_noise_save_dir.exists():
        source_noise_save_dir.mkdir()

    if not gen_wav_save_dir.exists():
        gen_wav_save_dir.mkdir()

    if not gen_mp3_save_dir.exists():
        gen_mp3_save_dir.mkdir()

    if not gen_wav_save_dir.exists():
        gen_wav_save_dir.mkdir()

    final_json = []

    ref_wav_file = "/home/tst000/projects/tst000/datasets/selected_ref_files.txt"

    all_wav_files = defaultdict(list)
    with open(ref_wav_file, "r") as f:
        for line in f:
            file_path = line.rstrip()
            ref_fname = Path(file_path).name
            speaker = ref_fname.split("_")[0]
            all_wav_files[speaker].append(file_path)

    all_speaker = list(all_wav_files.keys())
    all_speaker.shuffle()
    # use 30 speaker as testset
    train_set_speakers = all_speaker[:-30]
    test_set_speakers = all_speaker[-30:]

    testset_size = 300

    # dummy_fix_latent = torch.randn(100, 1000)
    data_id = 0
    for speaker_loop in range(50):
        # generate 50 speech per speaker
        for speaker in train_set_speakers:
            fname = f"{i}.pt"

            ref_text = get_ref_text_from_wav(wav_fname)
            gen_text = text_dataset[i]["sample"]

    for i, wav_fname in enumerate(all_wav_files):
        if i >= max_data:
            break

        fname = f"{i}.pt"

        ref_text = get_ref_text_from_wav(wav_fname)
        gen_text = text_dataset[i]["sample"]

        audio_save_path = str(audio_save_dir / f"{i}.wav")
        with torch.no_grad():
            _, _, _, trajectories = sampler.infer(
                ref_file=wav_fname,
                ref_text=ref_text,
                gen_text=gen_text,
                seed=None,
                remove_ref_from_trajectory=True
                # file_wave=audio_save_path
            )
        del _
        initial_noise = trajectories[0][0].detach().cpu().float()
        final_mel = trajectories[0][-1].detach().cpu().float()

        with torch.no_grad():
            audio = sampler.vocoder.decode(trajectories[0][-1])
            audio = audio.detach().cpu().float()

        orig_noise_save_path = str((source_noise_save_dir / fname).absolute())
        gen_tensor_save_path = str((gen_mel_save_dir / fname).absolute())

        torch.save(initial_noise, orig_noise_save_path)
        torch.save(final_mel, gen_tensor_save_path)
        torchaudio.save(audio_save_path, audio, sample_rate=24000)

        metadata_dict = {
            "orig_noise": orig_noise_save_path,
            "generated_tensor": gen_tensor_save_path,
            "gen_text": gen_text,
            "ref_text": ref_text,
            "ref_audio": wav_fname,
            "gen_audio": audio_save_path,
        }
        final_json.append(metadata_dict)

    json_fname = save_dir / "metadata.json"
    json.dump(final_json, open(json_fname.absolute(), "w"), indent=4)


class F5TTSMel:
    def __init__(
        self,
        mel_spec_kwargs={
            "target_sample_rate": 24000,
            "n_mel_channels": 100,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "mel_spec_type": "vocos",
        },
        sampling_rate=24000,
        target_rms=0.1,
    ):
        self.mel_spec_kwargs = mel_spec_kwargs
        self.mel_spec = MelSpec(**self.mel_spec_kwargs)
        self.sampling_rate = sampling_rate
        self.target_rms = target_rms

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

    def __call__(self, audio_fname):
        audio, sr = self.load_audio(audio_fname)
        mel = self.mel_spec(audio)
        return mel


@torch.no_grad()
def main_convert_audio():
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts"
    )
    sampler = F5TTS(device="cuda")

    audio_save_dir = save_dir / "wav"
    if not audio_save_dir.exists():
        audio_save_dir.mkdir()

    json_fname = save_dir / "metadata.json"
    metadata = json.load(open(json_fname, "r"))

    for md in tqdm(metadata):
        mel_path = md["generated_tensor"]
        generated_tensor = torch.load(mel_path).cuda()
        audio = sampler.vocoder.decode(generated_tensor)
        audio_id = Path(mel_path).name.split(".")[0]
        fname = f"{audio_id}.wav"
        audio_path = audio_save_dir / fname
        audio = audio.detach().float().cpu()
        torchaudio.save(audio_path, audio, sample_rate=24000)


def main_convert_audio_to_mel(target_format="mp3", aug_fn=None, aug_name=""):
    # create mel-spec function for f5tts
    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    sampling_rate = 24000
    mel_spec = F5TTSMel(mel_spec_kwargs=mel_spec_kwargs, sampling_rate=sampling_rate)

    # convert dir of audio back to mel
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/"
    )
    audio_dir = save_dir / f"{target_format}/"

    new_mel_dir = save_dir / f"{target_format+aug_name}_melspec/"
    if not new_mel_dir.exists():
        new_mel_dir.mkdir()

    # all_audio_fname = audio_dir.glob("*.mp3")

    json_fname = save_dir / f"metadata.json"
    metadata = json.load(open(json_fname, "r"))

    updated_metadata = []
    updated_metadata_fname = save_dir / f"{target_format+aug_name}_metadata.json"

    # some format can obtain more info, e.g mp3_16k (bit rate)
    file_type = target_format.split("_")[0]

    for md in tqdm(metadata):
        target_fname = Path(md["orig_noise"]).name
        target_id = target_fname.split(".")[0]
        mp3_fname = target_id + f".{file_type}"

        audio_file = audio_dir / mp3_fname
        audio, _ = mel_spec.load_audio(audio_file)
        if aug_fn is not None:
            audio = aug_fn(audio.unsqueeze(0)).squeeze(0)
        mel = mel_spec.mel_spec(audio)
        save_fname = new_mel_dir / target_fname

        torch.save(mel, save_fname)
        new_md = md
        new_md[f"{target_format}_mel"] = str(save_fname.absolute())
        updated_metadata.append(new_md)

    json.dump(updated_metadata, open(updated_metadata_fname.absolute(), "w"), indent=4)


def main_aug_echo():
    # create mel-spec function for f5tts
    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    sampling_rate = 24000
    mel_spec = F5TTSMel(mel_spec_kwargs=mel_spec_kwargs, sampling_rate=sampling_rate)

    # convert dir of audio back to mel
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/"
    )

    new_mel_dir = save_dir / "echo_melspec/"
    if not new_mel_dir.exists():
        new_mel_dir.mkdir()

    # all_audio_fname = audio_dir.glob("*.mp3")

    json_fname = save_dir / "metadata.json"
    metadata = json.load(open(json_fname, "r"))

    updated_metadata = []
    updated_metadata_fname = save_dir / "aug_echo_metadata.json"

    for i, md in enumerate(tqdm(metadata)):
        target_fname = Path(md["orig_noise"]).name
        target_id = target_fname.split(".")[0]

        audio_file = md["gen_audio"]
        aug_audio = echo(audio_file)

        if i == 0:
            audio_save_path = save_dir / "echo_example.wav"
            torchaudio.save(audio_save_path, aug_audio, sample_rate=24000)

        mel = mel_spec.mel_spec(aug_audio)
        save_fname = new_mel_dir / target_fname

        torch.save(mel, save_fname)
        new_md = md
        new_md["aug_mel"] = str(save_fname.absolute())
        updated_metadata.append(new_md)

    json.dump(updated_metadata, open(updated_metadata_fname.absolute(), "w"), indent=4)


def main_aug_hifigan():
    # create mel-spec function for f5tts
    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    sampling_rate = 24000
    mel_spec = F5TTSMel(mel_spec_kwargs=mel_spec_kwargs, sampling_rate=sampling_rate)

    # convert dir of audio back to mel
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/"
    )
    hifigan_dir = save_dir / "hifigan_22k_wav/"

    new_mel_dir = save_dir / "hifigan_22k_mel/"
    if not new_mel_dir.exists():
        new_mel_dir.mkdir()

    # all_audio_fname = audio_dir.glob("*.mp3")

    json_fname = save_dir / "metadata.json"
    metadata = json.load(open(json_fname, "r"))

    updated_metadata = []
    updated_metadata_fname = save_dir / "hifigan_22k_metadata.json"

    for i, md in enumerate(tqdm(metadata)):
        target_fname = Path(md["orig_noise"]).name
        target_id = target_fname.split(".")[0]

        audio_file = hifigan_dir / f"{target_id}.wav"

        mel = mel_spec(audio_file)
        save_fname = new_mel_dir / target_fname

        torch.save(mel, save_fname)
        new_md = md
        new_md["aug_mel"] = str(save_fname.absolute())
        updated_metadata.append(new_md)

    json.dump(updated_metadata, open(updated_metadata_fname.absolute(), "w"), indent=4)


class DataGenerator:
    """data generation For detector training"""

    def __init__(
        self,
        use_fix_noise=True,
        save_dir="/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/",
    ):
        self.device = "cuda"
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir()
        self.text_dataset = load_dataset(
            "jakeazcona/short-text-labeled-emotion-classification"
        )["train"]
        self.gen_audio_save_dir = self.save_dir / "gen_audio"

        if not self.gen_audio_save_dir.exists():
            self.gen_audio_save_dir.mkdir()
        for audio_format in ["wav", "mp3", "mp4"]:
            format_dir = self.gen_audio_save_dir / f"{audio_format}/"
            if not format_dir.exists():
                format_dir.mkdir()

        self.fix_latent_path = (
            "/gpfs/fs3c/nrc/dt/tst000/datasets/periodic_special_latent.pt"
        )
        if use_fix_noise:
            self.sampler = F5TTSPeriodicFixNoiseInferencer(
                fix_latent_path=self.fix_latent_path, device=self.device
            )
        else:
            self.sampler = F5TTS(device=self.device)

        self.ref_wav_file = (
            "/home/tst000/projects/tst000/datasets/selected_ref_files.txt"
        )
        self.all_wav_files = self.get_wav_files(self.ref_wav_file)
        (
            self.train_set_speakers,
            self.test_set_speakers,
            self.eval_set_speakers,
        ) = self.get_train_test_speaker_split(self.all_wav_files)

        self.audio_per_speaker = 100
        self.max_train_data = 10000
        self.max_test_data = 1000

    def get_wav_files(self, ref_wav_file):
        all_wav_files = defaultdict(list)
        with open(ref_wav_file, "r") as f:
            for line in f:
                file_path = line.rstrip()
                ref_fname = Path(file_path).name
                speaker = ref_fname.split("_")[0]
                all_wav_files[speaker].append(file_path)
        return all_wav_files

    def get_train_test_speaker_split(self, all_wav_files):

        all_speaker = list(all_wav_files.keys())
        random.shuffle(all_speaker)
        # use 30 speaker as testset
        train_set_speakers = all_speaker[:-60]
        test_set_speakers = all_speaker[-60:-30]
        eval_set_speakers = all_speaker[-30:]
        return train_set_speakers, test_set_speakers, eval_set_speakers

    def generate_mel(self, wav_fname, gen_text, remove_ref_from_trajectory=True):
        ref_text = get_ref_text_from_wav(wav_fname)
        print(wav_fname, ref_text, gen_text)
        with torch.no_grad():
            _, _, _, trajectories = self.sampler.infer(
                ref_file=wav_fname,
                ref_text=ref_text,
                gen_text=gen_text,
                seed=None,
                remove_ref_from_trajectory=remove_ref_from_trajectory
                # file_wave=audio_save_path
            )
        final_mel = trajectories[0][-1].detach().float()
        return final_mel

    def extract_audio(self, final_mel):
        with torch.no_grad():
            audio = self.sampler.vocoder.decode(final_mel)
            audio = audio.detach().cpu().float()
        return audio

    def step(self, wav_fname, gen_text):
        final_mel = self.generate_mel(wav_fname, gen_text)
        audio = self.extract_audio(final_mel)
        return audio

    def save_audios(self, file_id, audio, save_dir=None):
        if save_dir is None:
            save_dir = self.gen_audio_save_dir
        else:
            save_dir = Path(save_dir)

        mp3_save_dir = save_dir / "mp3/"
        mp4_save_dir = save_dir / "mp4/"
        wav_save_dir = save_dir / "wav/"

        torchaudio.save(
            (mp4_save_dir / f"{file_id}.mp4"),
            audio,
            format="mp4",
            sample_rate=24000,
            # bits_per_sample=32
        )
        torchaudio.save(
            (mp3_save_dir / f"{file_id}.mp3"),
            audio,
            format="mp3",
            sample_rate=24000,
            bits_per_sample=32,
        )
        torchaudio.save(
            (wav_save_dir / f"{file_id}.wav"),
            audio,
            format="wav",
            sample_rate=24000,
            bits_per_sample=32,
        )

        return [
            str(mp4_save_dir / f"{file_id}.mp4"),
            str(mp3_save_dir / f"{file_id}.mp3"),
            str(wav_save_dir / f"{file_id}.wav"),
        ]

    def generate_data_split(
        self, speakers, max_data, split_name="train", starting_text_id=0
    ):
        text_index = starting_text_id
        data_size = 0
        full_meta_data = []

        for i in range(self.audio_per_speaker):
            for speaker in speakers:
                total_ref_files = len(self.all_wav_files[speaker])
                wav_fname = self.all_wav_files[speaker][i % total_ref_files]
                gen_text = self.text_dataset[text_index]["sample"]
                audio = self.step(wav_fname, gen_text)
                save_paths = self.save_audios(data_size, audio)
                data_size += 1
                text_index += 1
                metadata = {
                    "wav_fname": str(wav_fname),
                    "gen_text": str(gen_text),
                    "audio_paths": save_paths,
                }
                full_meta_data.append(metadata)
                if data_size % 500 == 0:
                    print(f"{split_name} dataset generate: {data_size}")
                if data_size >= max_data:
                    break

        json_fname = self.save_dir / f"{split_name}_metadata.json"
        json.dump(full_meta_data, open(json_fname.absolute(), "w"), indent=4)
        return text_index, json_fname

    def generate_from_metadata(
        self, metadata_list: list, save_metadata_path, save_dir=None
    ):
        new_meta_data_list = []
        for metadata in tqdm(metadata_list):
            audio_key = "wav_fname" if "wav_fname" in metadata else "gen_audio"
            wav_fname = Path(metadata[audio_key])

            gen_text = metadata["gen_text"]
            gen_path = Path(metadata["audio_paths"][0])
            gen_id = gen_path.name.split(".")[0]

            audio = self.step(wav_fname, gen_text)
            save_paths = self.save_audios(gen_id, audio, save_dir=save_dir)
            new_metadata = metadata
            new_metadata.update(
                {
                    audio_key: str(wav_fname),
                    "gen_text": str(gen_text),
                    "audio_paths": save_paths,
                }
            )
            new_meta_data_list.append(new_metadata)

        json.dump(
            new_meta_data_list, open(Path(save_metadata_path).absolute(), "w"), indent=4
        )

    def run(self):
        text_index = 0
        # generate training data
        # self.audio_per_speaker = 50
        # self.max_train_data = 10000
        # self.max_test_data = 1000

        text_index, train_json = self.generate_data_split(
            speakers=self.train_set_speakers,
            max_data=self.max_train_data,
            split_name="train",
            starting_text_id=text_index,
        )

        text_index, test_json = self.generate_data_split(
            speakers=self.test_set_speakers,
            max_data=self.max_test_data,
            split_name="test",
            starting_text_id=text_index,
        )

        text_index, eval_json = self.generate_data_split(
            speakers=self.eval_set_speakers,
            max_data=self.max_test_data,
            split_name="eval",
            starting_text_id=text_index,
        )

        return {
            "train": train_json,
            "test": test_json,
            "eval": eval_json,
        }


def main_fix_latent():
    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/"
    )
    sampler = F5TTS(device="cuda")
    text_dataset = load_dataset("jakeazcona/short-text-labeled-emotion-classification")[
        "train"
    ]

    source_noise_save_dir = save_dir / "source_noise"
    gen_mel_save_dir = save_dir / "gen_mel"

    if not source_noise_save_dir.exists():
        source_noise_save_dir.mkdir()
    if not gen_mel_save_dir.exists():
        gen_mel_save_dir.mkdir()

    audio_save_dir = save_dir / "wav"
    if not audio_save_dir.exists():
        audio_save_dir.mkdir()

    final_json = []

    ref_wav_file = "/home/tst000/projects/tst000/datasets/selected_ref_files.txt"
    with open(ref_wav_file, "r") as f:
        all_wav_files = [line.rstrip() for line in f]

    max_data = 500

    dummy_fix_latent = torch.randn(100, 1000)

    for i, wav_fname in enumerate(all_wav_files):
        if i >= max_data:
            break

        fname = f"{i}.pt"

        ref_text = get_ref_text_from_wav(wav_fname)
        gen_text = text_dataset[i]["sample"]

        audio_save_path = str(audio_save_dir / f"{i}.wav")
        with torch.no_grad():
            _, _, _, trajectories = sampler.infer(
                ref_file=wav_fname,
                ref_text=ref_text,
                gen_text=gen_text,
                seed=None,
                remove_ref_from_trajectory=True
                # file_wave=audio_save_path
            )
        del _
        initial_noise = trajectories[0][0].detach().cpu().float()
        final_mel = trajectories[0][-1].detach().cpu().float()

        with torch.no_grad():
            audio = sampler.vocoder.decode(trajectories[0][-1])
            audio = audio.detach().cpu().float()

        orig_noise_save_path = str((source_noise_save_dir / fname).absolute())
        gen_tensor_save_path = str((gen_mel_save_dir / fname).absolute())

        torch.save(initial_noise, orig_noise_save_path)
        torch.save(final_mel, gen_tensor_save_path)
        torchaudio.save(audio_save_path, audio, sample_rate=24000)

        metadata_dict = {
            "orig_noise": orig_noise_save_path,
            "generated_tensor": gen_tensor_save_path,
            "gen_text": gen_text,
            "ref_text": ref_text,
            "ref_audio": wav_fname,
            "gen_audio": audio_save_path,
        }
        final_json.append(metadata_dict)

    json_fname = save_dir / "metadata.json"
    json.dump(final_json, open(json_fname.absolute(), "w"), indent=4)


def main_detector_train_data():
    # data_generator = DataGenerator(
    #     use_fix_noise=True,
    #     save_dir = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/"
    # )
    # metadata_paths = data_generator.run()
    # del data_generator
    meta_data_path = Path(
        "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/"
    )
    metadata_paths = {
        "train": meta_data_path / "train_metadata.json",
        "test": meta_data_path / "test_metadata.json",
        "eval": meta_data_path / "eval_metadata.json",
    }

    rand_latent_generator = DataGenerator(
        use_fix_noise=False,
        save_dir="/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/",
    )

    audio_save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/rand_latent_mel"
    )
    if not audio_save_dir.exists():
        audio_save_dir.mkdir()

    for audio_format in ["wav", "mp3", "mp4"]:
        format_dir = audio_save_dir / f"{audio_format}/"
        if not format_dir.exists():
            format_dir.mkdir()

    for split in ["train", "test", "eval"]:
        save_metadata_path = f"/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/{split}_rand_latent_metadata.json"
        metadata_list = json.load(open(metadata_paths[split], "r"))
        rand_latent_generator.generate_from_metadata(
            metadata_list, save_metadata_path, save_dir=audio_save_dir
        )


if __name__ == "__main__":
    # main_random_latent()
    # main_convert_audio()
    # from latent_aug_wm.eval.convert_mp3 import MP3_BIT_RATE

    # for bit_rate in MP3_BIT_RATE:
    #     target_format = f"mp3_{bit_rate}"
    #     main_convert_audio_to_mel(target_format=target_format)

    # for cutoff_freq in [500, 600, 700, 800, 1000]:
    #     aug_kwargs = {
    #         "mode": "per_example",
    #         "p": 1.0, "min_cutoff_freq": cutoff_freq,
    #         "max_cutoff_freq": cutoff_freq
    #     }
    #     lp_aug = getattr(torch_audiomentations, "LowPassFilter")(**aug_kwargs)
    #     aug_fn = lambda x: lp_aug(x, 24000)
    #     main_convert_audio_to_mel(target_format="wav", aug_fn=aug_fn, aug_name=f"LowPass_{cutoff_freq}")

    # main_aug_echo()
    # main_aug_hifigan()
    main_detector_train_data()
