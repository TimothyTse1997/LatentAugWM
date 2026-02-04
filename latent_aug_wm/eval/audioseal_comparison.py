import json
import random
import os
import random
import time
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
import matplotlib.pyplot as plt

import torchaudio
import torch_audiomentations

# import audioseal
from audioseal import AudioSeal

from latent_aug_wm.eval.contrastive_test import (
    SimilarityChecker,
    default_distance_function,
)
from latent_aug_wm.eval.generate_f5tts_pretrain_dataset import F5TTSMel
from latent_aug_wm.model.starganv2 import StarGanDetector
from latent_aug_wm.eval.convert_mp3 import mp3_file_conversion, MP3_BIT_RATE
from latent_aug_wm.data_augmentation.torchaudio_io_augmentation import (
    echo_file_conversion,
)
from torch_audiomentations.augmentations.random_crop import RandomCrop

F5TTS_MEL_SPEC_KWARGS = {
    "target_sample_rate": 24000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
    "mel_spec_type": "vocos",
}


def default_distance_function(A, B):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(A, B)


def custom_l1(A, B):
    batch_size_A = A.shape[0]
    batch_size_B = B.shape[0]
    assert batch_size_B <= batch_size_A
    if batch_size_A == batch_size_B:
        return F.l1_loss(A, B)
    repeated_B = B.repeat(batch_size_A, 1)
    assert repeated_B.shape == A.shape
    return F.l1_loss(A, repeated_B)


def custom_l2(A, B):
    batch_size_A = A.shape[0]
    batch_size_B = B.shape[0]
    assert batch_size_B <= batch_size_A
    if batch_size_A == batch_size_B:
        return F.l1_loss(A, B)
    repeated_B = B.repeat(batch_size_A, 1)
    assert repeated_B.shape == A.shape
    return F.mse_loss(A, repeated_B)


def custom_dot_product(A, B):
    return torch.mm(B, A.T).mean()


class BasicCosineSimilarityDetector(SimilarityChecker):
    def __init__(
        self,
        mel_spec_kwargs=F5TTS_MEL_SPEC_KWARGS,
        sampling_rate=24000,
        distance_function=default_distance_function,
    ):
        self.mel_spec_kwargs = mel_spec_kwargs
        self.sampling_rate = sampling_rate
        self.distance_function = distance_function
        self.mel_spec = F5TTSMel(
            mel_spec_kwargs=self.mel_spec_kwargs, sampling_rate=self.sampling_rate
        )
        self.use_fake_source_noise = False

    def detect(self, source_noise, audio, N=500):  # , confidence=0.95):
        target_tensor = self.mel_spec.mel_spec(audio)
        target_tensor = self.pad_to_same_length(source_noise, target_tensor)
        if self.use_fake_source_noise:
            current_source_noise = torch.randn_like(source_noise)
        else:
            current_source_noise = source_noise

        distribution = self.get_distribution_2d(
            current_source_noise, target_tensor, N=N
        )
        source_distance = self.get_source_similarity(
            current_source_noise, target_tensor
        )

        success = (source_distance.numpy() > distribution.numpy()).sum() / N
        return success  # bool(success > confidence)


class CustomDetector:
    # def __init__(self, checkpoint_path, mel_spec_kwargs=F5TTS_MEL_SPEC_KWARGS, sampling_rate=24000, device="cuda"):
    def __init__(
        self,
        checkpoint_path,
        mel_spec_kwargs=F5TTS_MEL_SPEC_KWARGS,
        sampling_rate=24000,
        device="cpu",
    ):
        self.mel_spec_kwargs = mel_spec_kwargs
        self.sampling_rate = sampling_rate
        self.mel_spec = F5TTSMel(
            mel_spec_kwargs=self.mel_spec_kwargs, sampling_rate=self.sampling_rate
        )

        self.device = device
        self.model = {
            "detector": StarGanDetector(dim_in=self.mel_spec_kwargs["n_mel_channels"])
        }
        self._load_checkpoint(
            checkpoint_path, load_only_params=False, specify_model_keys=None
        )
        for k, v in self.model.items():
            _ = v.to(self.device)

    def _load(self, states, model, force_load=True):
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

    def _load_checkpoint(
        self, checkpoint_path, load_only_params=False, specify_model_keys=None
    ):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            if not key in state_dict["model"]:
                continue
            if (specify_model_keys is not None) and not (key in specify_model_keys):
                continue

            print(f"loading module {key} from {checkpoint_path}")
            self._load(state_dict["model"][key], self.model[key])

    def detect(self, audio):

        target_tensor = self.mel_spec.mel_spec(audio).permute(0, 2, 1)
        target_tensor = target_tensor.to(self.device)
        with torch.no_grad():
            logit = self.model["detector"].get_feature(target_tensor).detach().cpu()
            # print(logit)
        label = F.softmax(logit.squeeze(-1), dim=-1).numpy()
        return float(label[1])


class AudioSealWMDatasetGenerator:
    def __init__(self, metadata_json: str, max_data=500, aug_fn=None, device="cuda"):
        self.metadata = json.load(open(metadata_json, "r"))
        self.device = device
        self.aug_fn = aug_fn
        self.sampling_rate = 16000
        self.audioseal_wm = AudioSeal.load_generator(
            "audioseal_wm_16bits", device=self.device
        )
        # self.audioseal_detector = AudioSeal.load_detector("audioseal_detector_16bits", device="cuda")
        self.audioseal_wm.eval()

    def load_audio(self, audio_wav):
        audio, sr = torchaudio.load(audio_wav)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
        return audio, sr

    def watermarking(self, wav_path):
        audio, sr = self.load_audio(wav_path)
        audio = audio.unsqueeze(0).cuda()
        watermark = self.audioseal_wm.get_watermark(audio)
        watermarked_audio = audio + watermark
        return watermarked_audio.squeeze(0).cpu()

    def wm_file_convert(self, source_fname, target_fname):
        wm_audio = self.watermarking(source_fname)
        torchaudio.save(target_fname, wm_audio, sample_rate=self.sampling_rate)
        # file_wm_score = self.detect_from_file(target_fname)
        # print(file_wm_score)

    # def detect_from_file(self, fp):
    #     # put here to make sure the watermark is present from generated dataset (without any augentation)
    #     audio, _ = self.load_audio(fp)
    #     result, message = self.audioseal_detector.detect_watermark(
    #         audio.unsqueeze(0).cuda(), self.sampling_rate)

    #     print(result, message)
    #     return result #result[:, 1, :].mean()

    def run(self, out_dir: str, updated_meta_data_fname: str):
        updated_metadata = []
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()

        for md in tqdm(self.metadata):
            wav_fname = md["gen_audio"]
            target_fname = out_dir / Path(wav_fname).name
            self.wm_file_convert(wav_fname, target_fname)
            new_md = md
            new_md["audioseal_gen_audio"] = str(target_fname.absolute())
            updated_metadata.append(new_md)

        json.dump(updated_metadata, open(updated_meta_data_fname, "w"), indent=4)


class AudioSealAugmentationTest(AudioSealWMDatasetGenerator):
    def __init__(self, tmp_dir: str, wm_dir: str, non_wm_dir: str, device="cuda"):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_wm_dir = self.tmp_dir / "watermarked/"
        self.tmp_non_wm_dir = self.tmp_dir / "non_watermarked/"

        if not self.tmp_wm_dir.exists():
            self.tmp_wm_dir.mkdir()
            self.tmp_non_wm_dir.mkdir()

        self.clean_tmp_dirs()

        self.wm_dir = Path(wm_dir)
        self.non_wm_dir = Path(non_wm_dir)
        self.sampling_rate = 16000
        self.device = device
        self.audioseal_detector = AudioSeal.load_detector(
            "audioseal_detector_16bits", device=device
        )

    def clean_tmp_dirs(self):
        total_removed = 0
        for p in self.tmp_wm_dir.glob("*"):
            os.remove(p)
            total_removed += 1
        for p in self.tmp_non_wm_dir.glob("*"):
            os.remove(p)
            total_removed += 1
        print(f"tmp cleaning completed, file removed: {total_removed}")

    def create_tmp_file(self, source_fname, aug_fn, watermarked=None, file_type=None):
        # aug_fn should be loading and saving the audio only
        # we only provide the tmp file name and source fname
        target_dir = self.tmp_wm_dir if watermarked else self.tmp_non_wm_dir
        if file_type is not None:
            fname = source_fname.name.split(".")[0] + f".{file_type}"
        else:
            fname = source_fname.name

        tmp_fname = target_dir / fname
        aug_fn(source_fname, tmp_fname, self.sampling_rate)
        return tmp_fname

    def create_tmp_dataset(self, aug_fn, file_type=None):
        print("start creating tmp files...")
        for wm_file in tqdm(self.wm_dir.glob("*")):
            _ = self.create_tmp_file(
                wm_file, aug_fn, watermarked=True, file_type=file_type
            )
        for non_wm_file in tqdm(self.non_wm_dir.glob("*")):
            _ = self.create_tmp_file(
                non_wm_file, aug_fn, watermarked=False, file_type=file_type
            )

    def detect_from_file(self, fp):
        audio, _ = self.load_audio(fp)
        # result, message = self.audioseal_detector.detect_watermark(audio.unsqueeze(0).cuda(), self.sampling_rate)
        result, message = self.audioseal_detector.detect_watermark(
            audio.unsqueeze(0).to(self.device), self.sampling_rate
        )

        # print(result, message)
        return result  # result[:, 1, :].mean()

    def get_logit_labels(self):
        print("start getting logits...")
        all_wm_tmp_files = list(self.tmp_wm_dir.glob("*"))
        wm_logit = []
        for fp in tqdm(all_wm_tmp_files):
            logit = self.detect_from_file(fp)
            if logit is None:
                continue
            wm_logit.append(logit)

        all_non_wm_tmp_files = list(self.tmp_non_wm_dir.glob("*"))
        non_wm_logit = []
        for fp in tqdm(all_non_wm_tmp_files):
            logit = self.detect_from_file(fp)
            if logit is None:
                continue
            non_wm_logit.append(logit)

        return {
            "real_logit": wm_logit,
            "fake_logit": non_wm_logit,
        }

    def run(self, aug_fn, file_type=None):
        self.clean_tmp_dirs()
        self.create_tmp_dataset(aug_fn, file_type=file_type)
        out = self.get_logit_labels()
        self.clean_tmp_dirs()
        return out


class AudioSealAugmentationTestMetadata(AudioSealAugmentationTest):
    def __init__(self, metadata_json, tmp_dir: str, device="cuda"):
        self.metadata = json.load(open(metadata_json, "r"))
        self.tmp_dir = Path(tmp_dir)
        self.tmp_raw_wm_dir = self.tmp_dir / "raw_watermarked/"
        self.tmp_wm_dir = self.tmp_dir / "watermarked/"
        self.tmp_non_wm_dir = self.tmp_dir / "non_watermarked/"

        if not self.tmp_raw_wm_dir.exists():
            self.tmp_raw_wm_dir.mkdir()
        if not self.tmp_non_wm_dir.exists():
            self.tmp_non_wm_dir.mkdir()
        if not self.tmp_raw_wm_dir.exists():
            self.tmp_raw_wm_dir.mkdir()

        self.clean_tmp_dirs()
        self.sampling_rate = 16000
        self.device = device
        self.audioseal_detector = AudioSeal.load_detector(
            "audioseal_detector_16bits", device=device
        )
        self.audioseal_wm = AudioSeal.load_generator(
            "audioseal_wm_16bits", device=self.device
        )
        self.audioseal_wm.eval()
        self.wm_files, self.non_wm_files = None, None

    def clean_tmp_dirs(self):
        super().clean_tmp_dirs()

        total_removed = 0
        for p in self.tmp_raw_wm_dir.glob("*"):
            os.remove(p)
            total_removed += 1
        print(f"tmp raw wm audio files removed: {total_removed}")

    def get_audio(self, audio_paths):
        for ap in audio_paths:
            if Path(ap).name.endswith(".wav"):
                return ap

    def watermarking(self, wav_path):
        audio, sr = self.load_audio(wav_path)
        audio = audio.unsqueeze(0).cuda()
        watermark = self.audioseal_wm.get_watermark(audio)
        watermarked_audio = audio + watermark

        save_path = self.tmp_raw_wm_dir / wav_path.name
        torchaudio.save(
            save_path,
            watermarked_audio.squeeze(0).cpu(),
            sample_rate=self.sampling_rate,
        )

        return save_path

    def create_tmp_dataset(self, aug_fn, file_type=None):
        print("start creating tmp files...")
        wm_files = []
        non_wm_files = []

        for wm_md in tqdm(self.metadata):
            # for wm_file in tqdm(self.wm_dir.glob("*")):
            non_wm_file = Path(self.get_audio(wm_md["audio_paths"]))

            wm_raw_audio_path = self.watermarking(non_wm_file)

            wm_tmp_fname = self.create_tmp_file(
                wm_raw_audio_path, aug_fn, watermarked=True, file_type=file_type
            )
            non_wm_tmp_fname = self.create_tmp_file(
                non_wm_file, aug_fn, watermarked=False, file_type=file_type
            )

            wm_files.append(wm_tmp_fname)
            non_wm_files.append(non_wm_tmp_fname)

        self.wm_files = wm_files
        self.non_wm_files = non_wm_files

    def get_logit_labels(self):
        print("start getting logits...")
        all_wm_tmp_files = self.wm_files

        wm_logit = []
        for fp in tqdm(all_wm_tmp_files):
            logit = self.detect_from_file(fp)
            if logit is None:
                continue
            wm_logit.append(logit)

        all_non_wm_tmp_files = self.non_wm_files

        non_wm_logit = []
        for fp in tqdm(all_non_wm_tmp_files):
            logit = self.detect_from_file(fp)
            if logit is None:
                continue
            non_wm_logit.append(logit)

        self.wm_files, self.non_wm_files = None, None
        return {
            "real_logit": wm_logit,
            "fake_logit": non_wm_logit,
        }


class SimpleDataAugGenerator(AudioSealAugmentationTest):
    def __init__(
        self,
        metadata_json,
        tmp_dir: str,
        cosim_detector=BasicCosineSimilarityDetector(),
    ):
        self.metadata = json.load(open(metadata_json, "r"))
        self.tmp_dir = Path(tmp_dir)
        self.tmp_wm_dir = self.tmp_dir / "watermarked/"
        self.tmp_non_wm_dir = self.tmp_dir / "non_watermarked/"

        if not self.tmp_wm_dir.exists():
            self.tmp_wm_dir.mkdir()
            self.tmp_non_wm_dir.mkdir()

        self.clean_tmp_dirs()
        self.sampling_rate = 24000
        self.cosim_detector = cosim_detector

        self.wm_file_pair, self.non_wm_file_pair = [], []

    def clean_tmp_dirs(self):
        total_removed = 0
        for p in self.tmp_wm_dir.glob("*"):
            os.remove(p)
            total_removed += 1
        for p in self.tmp_non_wm_dir.glob("*"):
            os.remove(p)
            total_removed += 1
        self.wm_file_pair, self.non_wm_file_pair = [], []
        print(f"tmp cleaning completed, file removed: {total_removed}")

    def create_tmp_dataset(self, aug_fn, file_type=None):
        print("start creating tmp files...")
        wm_file_pair = []
        non_wm_file_pair = []

        for wm_md in tqdm(self.metadata):
            # for wm_file in tqdm(self.wm_dir.glob("*")):
            wm_file = Path(wm_md["gen_audio"])
            wm_tmp_fname = self.create_tmp_file(
                wm_file, aug_fn, watermarked=True, file_type=file_type
            )
            wm_file_pair.append((wm_md["orig_noise"], wm_tmp_fname))

        all_source, all_audio = zip(*wm_file_pair)
        all_source, all_audio = list(all_source), list(all_audio)

        non_wm_audio = [all_audio[-1]] + all_audio[:-1]
        non_wm_file_pair = list(zip(all_source, non_wm_audio))
        self.wm_file_pair, self.non_wm_file_pair = wm_file_pair, non_wm_file_pair

    def detect_from_file(self, noise_fp, audio_fp):
        audio, _ = self.load_audio(audio_fp)
        source_noise = torch.load(noise_fp)
        result = self.cosim_detector.detect(source_noise, audio)

        # print(result, message)
        return result  # result[:, 1, :].mean()

    def get_logit_labels(self):
        print("start getting logits...")
        # all_wm_tmp_files = list(self.tmp_wm_dir.glob("*"))
        self.cosim_detector.use_fake_source_noise = False
        wm_logit = [
            self.detect_from_file(s_fp, fp) for s_fp, fp in tqdm(self.wm_file_pair)
        ]

        self.cosim_detector.use_fake_source_noise = True
        non_wm_logit = [
            self.detect_from_file(s_fp, fp) for s_fp, fp in tqdm(self.non_wm_file_pair)
        ]
        return {
            "real_logit": wm_logit,
            "fake_logit": non_wm_logit,
        }


class SimpleDetectorAugmentationTest(AudioSealAugmentationTest):
    def __init__(
        self,
        detector: CustomDetector,
        tmp_dir: str,
        wm_metadata_json,
        non_wm_metadata_json,
    ):
        self.detector = detector

        self.tmp_dir = Path(tmp_dir)
        self.tmp_wm_dir = self.tmp_dir / "watermarked/"
        self.tmp_non_wm_dir = self.tmp_dir / "non_watermarked/"

        if not self.tmp_wm_dir.exists():
            self.tmp_wm_dir.mkdir()
            self.tmp_non_wm_dir.mkdir()

        self.clean_tmp_dirs()

        self.wm_metadata = json.load(open(wm_metadata_json, "r"))
        self.non_wm_metadata = json.load(open(non_wm_metadata_json, "r"))

        self.sampling_rate = 24000

    def get_audio(self, audio_paths):
        for ap in audio_paths:
            if Path(ap).name.endswith(".wav"):
                return ap

    def create_tmp_dataset(self, aug_fn, file_type=None):
        print("start creating tmp files...")

        for wm_md in tqdm(self.wm_metadata):
            wm_files = wm_md["audio_paths"]
            wm_file = self.get_audio(wm_files)
            _ = self.create_tmp_file(
                Path(wm_file), aug_fn, watermarked=True, file_type=file_type
            )

        for non_wm_md in tqdm(self.non_wm_metadata):
            non_wm_files = non_wm_md["audio_paths"]
            non_wm_file = self.get_audio(non_wm_files)
            _ = self.create_tmp_file(
                Path(non_wm_file), aug_fn, watermarked=False, file_type=file_type
            )

    def detect_from_file(self, fp):
        audio, _ = self.load_audio(fp)
        if audio.shape[-1] < 20000:
            return None
        try:
            result = self.detector.detect(audio)
        except Exception as e:
            print(e)
            result = 0

        return result  # result[:, 1, :].mean()


def simple_comparison_plot(
    results,
    model_score_cutoff,
    x_axis_values,
    x_axis_label,
    save_path="accuracy_comparison.png",
    comparing_title="no augmentation",
):
    """
    Plot and save a comparison graph of model accuracy.

    Parameters
    ----------
    results : list of dict
        Each element corresponds to one x-axis value.
        [
            {
                model_name_A: {"real_logit": [...], "fake_logit": [...]},
                model_name_B: {"real_logit": [...], "fake_logit": [...]}
            },
            ...
        ]

    model_score_cutoff : dict
        {model_name: cutoff_value}

    x_axis_values : list
        Values for the x-axis (same length as results)

    x_axis_label : str
        Label for the x-axis

    save_path : str
        File path to save the figure
    """

    assert len(results) == len(
        x_axis_values
    ), "results and x_axis_values must match in length"

    # Collect model names
    model_names = list(model_score_cutoff.keys())

    # Store accuracies per model
    model_accuracies = {model: [] for model in model_names}

    for step_result in results:
        for model in model_names:
            logits = step_result[model]
            cutoff = model_score_cutoff[model]

            real_logits = np.array(logits["real_logit"])
            fake_logits = np.array(logits["fake_logit"])

            # Accuracy definitions
            real_acc = np.mean(real_logits >= cutoff) if len(real_logits) > 0 else 0.0
            fake_acc = np.mean(fake_logits < cutoff) if len(fake_logits) > 0 else 0.0

            overall_acc = 0.5 * (real_acc + fake_acc)
            model_accuracies[model].append(overall_acc)

    # Plot
    plt.figure(figsize=(8, 5))

    for model, acc_values in model_accuracies.items():
        plt.plot(
            x_axis_values,
            acc_values,
            marker="o",
            label=model,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy Comparison ({comparing_title})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(
        save_path
    ) else None
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {save_path}")


def default_aug(source_file, target_file, sampling_rate=16000):
    audio, sr = torchaudio.load(source_file)
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        audio = resampler(audio)
    torchaudio.save(target_file, audio, sample_rate=sampling_rate)


def creat_compression_aug_fn(target_bit_rate=None, new_format="mp3"):
    def _compress_aug(source_file, target_file, sampling_rate=16000):
        mp3_file_conversion(
            source_file,
            target_file,
            new_format=new_format,
            target_bit_rate=target_bit_rate,
        )

    return _compress_aug


def create_torchaug_fn(aug_name, aug_config):
    import torch_audiomentations

    aug_transform = getattr(torch_audiomentations, aug_name)(**aug_config)

    def _audiomentations_aug(source_file, target_file, sampling_rate=16000):
        audio, sr = torchaudio.load(source_file)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)
        aug_audio = aug_transform(
            audio.unsqueeze(0), sample_rate=sampling_rate
        ).squeeze(0)
        torchaudio.save(target_file, aug_audio, sample_rate=sampling_rate)

    return _audiomentations_aug


def create_randcrop_fn(max_length, sampling_rate=16000):
    crop_fn = RandomCrop(max_length=max_length, sampling_rate=sampling_rate)

    def _randcrop_fn(source_file, target_file, sampling_rate=sampling_rate):
        audio, sr = torchaudio.load(source_file)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)
        aug_audio = crop_fn(audio.unsqueeze(0)).squeeze(0)
        torchaudio.save(target_file, aug_audio, sample_rate=sampling_rate)

    return _randcrop_fn


def create_scale_fn(scale, sampling_rate=16000):
    def _randscale_fn(source_file, target_file, sampling_rate=sampling_rate):
        audio, sr = torchaudio.load(source_file)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)
        aug_audio = audio * scale
        torchaudio.save(target_file, aug_audio, sample_rate=sampling_rate)

    return _randscale_fn


def create_speed_aug_fn(speed, sampling_rate=16000):
    def _speed_scale_fn(source_file, target_file, sampling_rate=sampling_rate):
        speed_fn = torchaudio.transforms.Speed(sampling_rate, speed)
        audio, sr = torchaudio.load(source_file)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)
        aug_audio, _ = speed_fn(audio)
        torchaudio.save(target_file, aug_audio, sample_rate=sampling_rate)

    return _speed_scale_fn


def parse_number(s: str) -> int:
    s = s.strip().lower()

    if s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    else:
        return int(s)


def get_all_crop_and_scale_augs(sampling_rate=24000):
    all_augs = {}
    crop_augs = {
        f"max_length_{max_length}": create_randcrop_fn(
            max_length, sampling_rate=sampling_rate
        )
        for max_length in [0.5, 1, 2, 3]
    }

    scale_augs = {
        f"scale_{scale}": create_scale_fn(scale, sampling_rate=sampling_rate)
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]
    }
    all_augs.update(crop_augs)
    all_augs.update(scale_augs)
    return all_augs


def get_speed_augs(sampling_rate=24000):
    return {
        f"speed_{speed}": create_speed_aug_fn(speed, sampling_rate=sampling_rate)
        for speed in [0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1]
    }


def get_all_audioseal_augs():
    all_aug = {"no augmentation": default_aug}

    mp3_augs = {
        f"mp3_{bitrate}": creat_compression_aug_fn(
            target_bit_rate=bitrate, new_format="mp3"
        )
        for bitrate in MP3_BIT_RATE
    }
    all_aug.update(mp3_augs)
    mp4_aug = {"mp4": creat_compression_aug_fn(new_format="mp4")}
    all_aug.update(mp4_aug)

    lp_augs = {
        f"lp_{cutoff_freq}": create_torchaug_fn(
            aug_name="LowPassFilter",
            aug_config={
                "mode": "per_example",
                "p": 1.0,
                "min_cutoff_freq": cutoff_freq,
                "max_cutoff_freq": cutoff_freq,
            },
        )
        for cutoff_freq in [500, 600, 700, 800, 1000]
    }
    all_aug.update(lp_augs)

    hp_augs = {
        f"hp_{cutoff_freq}": create_torchaug_fn(
            aug_name="HighPassFilter",
            aug_config={
                "mode": "per_example",
                "p": 1.0,
                "min_cutoff_freq": cutoff_freq,
                "max_cutoff_freq": cutoff_freq,
            },
        )
        for cutoff_freq in [500, 1000, 2000, 3000, 4000]
    }
    all_aug.update(hp_augs)

    white_noise_aug = {
        f"white_noise_db_{db}": create_torchaug_fn(
            aug_name="AddColoredNoise",
            aug_config={
                "mode": "per_example",
                "p": 1.0,
                "min_f_decay": 0,
                "min_f_decay": 0,
                "min_snr_in_db": db,
                "max_snr_in_db": db,
            },
        )
        for db in [10, 20, 30, 40]
    }
    pink_noise_aug = {
        f"pink_noise_db_{db}": create_torchaug_fn(
            aug_name="AddColoredNoise",
            aug_config={
                "mode": "per_example",
                "p": 1.0,
                "min_f_decay": 1,
                "min_f_decay": 1,
                "min_snr_in_db": db,
                "max_snr_in_db": db,
            },
        )
        for db in [10, 20, 30, 40]
    }
    all_aug.update(white_noise_aug)
    all_aug.update(pink_noise_aug)

    echo_aug = {"echo": echo_file_conversion}
    all_aug.update(echo_aug)

    return all_aug


def audioseal_result_from_all_aug():

    metadata_json = "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/metadata.json"
    out_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioseal_wm_wav/"
    updated_meta_data_fname = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioseal_metadata.json"

    # audioseal_generator = AudioSealWMDatasetGenerator(metadata_json=metadata_json)
    # audioseal_generator.run(out_dir, updated_meta_data_fname)

    tmp_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/tmp/"
    eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results.json"
    # eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_echo.json"
    all_audio_aug = {}

    wm_dir = out_dir
    non_wm_dir = "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/wav/"
    # audioseal_detector = AudioSealAugmentationTest(tmp_dir, wm_dir, non_wm_dir, device="cuda")
    audioseal_detector = AudioSealAugmentationTest(
        tmp_dir, wm_dir, non_wm_dir, device="cuda"
    )

    all_augs = get_all_audioseal_augs()
    # all_augs = {
    # "echo": echo_file_conversion
    # }

    for aug_name, aug_fn in all_augs.items():
        print(f"Current Augmentation: {aug_name}")
        file_type = None
        if "mp3" in aug_name:
            file_type = "mp3"
        if "mp4" in aug_name:
            file_type = "mp4"

        result = audioseal_detector.run(aug_fn=aug_fn, file_type=file_type)
        all_audio_aug[aug_name] = result

    with open(eval_result_path, "w") as f:
        json.dump(all_audio_aug, f, indent=4)


def audioseal_result_from_all_aug_from_metadata():

    non_wm_metadata_json = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/test_rand_latent_metadata.json"

    tmp_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/tmp2/"
    # eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_from_metadata_crop_scale.json"
    eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_from_metadata_speed2.json"
    all_audio_aug = {}

    # audioseal_detector = AudioSealAugmentationTestMetadata(non_wm_metadata_json, tmp_dir=tmp_dir, device="cuda")
    audioseal_detector = AudioSealAugmentationTestMetadata(
        non_wm_metadata_json, tmp_dir=tmp_dir, device="cuda"
    )

    # all_augs = get_all_audioseal_augs()
    # all_augs = get_all_crop_and_scale_augs()
    # all_augs.update(get_speed_augs())

    all_augs = get_speed_augs()

    all_augs = update_aug_override_sampling_rate(all_augs, override_sampling_rate=16000)

    for aug_name, aug_fn in all_augs.items():
        print(f"Current Augmentation: {aug_name}")
        file_type = None
        if "mp3" in aug_name:
            file_type = "mp3"
        if "mp4" in aug_name:
            file_type = "mp4"

        result = audioseal_detector.run(aug_fn=aug_fn, file_type=file_type)
        all_audio_aug[aug_name] = result

    with open(eval_result_path, "w") as f:
        json.dump(all_audio_aug, f, indent=4)


def update_aug_override_sampling_rate(all_augs, override_sampling_rate=16000):
    out = {}

    def _update_aug_sampling_rate(aug_fn):
        def _new_aug(source_file, target_file, sampling_rate=16000):
            # print(f"overided sampling rate: {override_sampling_rate}")
            return aug_fn(source_file, target_file, override_sampling_rate)

        return _new_aug

    for aug_name, aug_fn in all_augs.items():
        out[aug_name] = _update_aug_sampling_rate(aug_fn)
        # print(out[aug_name])
    return out


def cosim_result_from_all_aug(
    cosim_detector=BasicCosineSimilarityDetector(),
    eval_result_path="/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results_16k.json",
):

    tmp_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/tmp/"

    # eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results.json"

    cosine_detector = SimpleDataAugGenerator(
        metadata_json="/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/metadata.json",
        tmp_dir=tmp_dir,
        cosim_detector=cosim_detector,
    )

    all_audio_aug = {}

    # all_augs = get_all_audioseal_augs()

    # all_augs = update_aug_override_sampling_rate(all_augs, override_sampling_rate=16000)

    all_augs = {"no augmentation": default_aug}

    for aug_name, aug_fn in all_augs.items():
        print(f"Current Augmentation: {aug_name}")
        file_type = None
        if "mp3" in aug_name:
            file_type = "mp3"
        if "mp4" in aug_name:
            file_type = "mp4"

        result = cosine_detector.run(aug_fn=aug_fn, file_type=file_type)
        all_audio_aug[aug_name] = result

    with open(eval_result_path, "w") as f:
        json.dump(all_audio_aug, f, indent=4)


def get_sim_scores_with_dis_fn(
    dis_fn=default_distance_function,
    eval_result_path="/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results_16k.json",
):
    cosim_detector = BasicCosineSimilarityDetector(distance_function=dis_fn)
    cosim_result_from_all_aug(
        cosim_detector=cosim_detector, eval_result_path=eval_result_path
    )


def get_acc_of_different_dis_fn_cutoff(
    dis_fns={
        "CoSim": default_distance_function,
        "L1": custom_l1,  # F.l1_loss,
        "L2": custom_l2,  # F.mse_loss,
        "DotProduct": custom_dot_product,
    },
    eval_result_source_dir="/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/",
    cutoffs=[0.8, 0.85, 0.9, 0.95, 1.0],
    acc_results_path="/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cutoff_dis_acc.json",
):

    eval_result_source_dir = Path(eval_result_source_dir)

    def get_acc_from_cutoff_logit(logits, cutoff):
        real_logits = np.array(logits["real_logit"])
        fake_logits = np.array(logits["fake_logit"])

        # Accuracy definitions
        real_acc = np.mean(real_logits >= cutoff) if len(real_logits) > 0 else 0.0
        fake_acc = np.mean(fake_logits < cutoff) if len(fake_logits) > 0 else 0.0

        overall_acc = 0.5 * (real_acc + fake_acc)
        return overall_acc

    def get_acc_from_eval_result_path(eval_result_path, cutoff):
        results = json.load(open(eval_result_path, "r"))["no augmentation"]
        acc = get_acc_from_cutoff_logit(results, cutoff)
        return acc

    acc_results = {}
    for dis_fn_name, dis_fn in dis_fns.items():
        eval_result_path = (
            eval_result_source_dir / f"cosim_eval_results_dis_fn_{dis_fn_name}.json"
        )
        get_sim_scores_with_dis_fn(dis_fn=dis_fn, eval_result_path=eval_result_path)
        acc_results[dis_fn_name] = {
            cutoff: get_acc_from_eval_result_path(eval_result_path, cutoff)
            for cutoff in cutoffs
        }

    with open(acc_results_path, "w") as f:
        json.dump(acc_results, f, indent=4)

    return acc_results


def detector_result_from_all_aug():

    tmp_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/tmp/"

    eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k.json"
    # eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k_speed2.json"
    # checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector_v2/version_debug_0/epoch_00009.pth"
    checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector_v2/version_debug_4_new_crop_and_speed/epoch_00038.pth"

    detector = CustomDetector(checkpoint_path, device="cuda")

    all_audio_aug = {}

    wm_metadata_json = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/test_rand_latent_metadata.json"
    non_wm_metadata_json = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/test_metadata.json"

    tester = SimpleDetectorAugmentationTest(
        detector, tmp_dir, wm_metadata_json, non_wm_metadata_json
    )

    all_augs = get_all_audioseal_augs()
    # all_augs = get_all_crop_and_scale_augs()
    # all_augs.update(get_speed_augs())
    # all_augs = get_speed_augs()

    all_augs = update_aug_override_sampling_rate(all_augs, override_sampling_rate=16000)

    for aug_name, aug_fn in all_augs.items():
        print(f"Current Augmentation: {aug_name}")
        file_type = None
        if "mp3" in aug_name:
            file_type = "mp3"
        if "mp4" in aug_name:
            file_type = "mp4"

        result = tester.run(aug_fn=aug_fn, file_type=file_type)
        all_audio_aug[aug_name] = result

    with open(eval_result_path, "w") as f:
        json.dump(all_audio_aug, f, indent=4)


def plot_mp3_results():

    cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results.json"
    audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results.json"

    cosim_eval_result = json.load(open(cosim_eval_result_path, "r"))
    audioseal_eval_result = json.load(open(audioseal_eval_result_path, "r"))
    # mp3_results = [
    # {k: v for k, v in cosim_eval_result.items() if k.startswith("mp3")},
    # {k: v for k, v in audioseal_eval_result.items() if k.startswith("mp3")},
    # ]
    combined_result = {
        k: {"Cosine similarity": v, "Audioseal": audioseal_eval_result[k]}
        for k, v in cosim_eval_result.items()
        if k.startswith("mp3")
    }

    x_axis_values = [parse_number(k.split("_")[-1]) for k in combined_result.keys()]
    x_axis_label = [k.replace("_", " ") for k in combined_result.keys()]
    simple_comparison_plot(
        list(combined_result.values()),
        model_score_cutoff={"Cosine similarity": 1.0, "Audioseal": 0.5},
        x_axis_values=x_axis_values,
        x_axis_label=x_axis_label,
        save_path="accuracy_comparison_mp3.png",
        comparing_title="MP3 compression",
    )


def plot_results(
    starting_key="lp",
    save_path="accuracy_comparison_lp.png",
    comparing_title="low pass filter",
    x_axis_label="low pass cut off",
):

    # cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results.json"
    cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results_16k.json"
    audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results.json"

    cosim_eval_result = json.load(open(cosim_eval_result_path, "r"))
    audioseal_eval_result = json.load(open(audioseal_eval_result_path, "r"))
    # mp3_results = [
    # {k: v for k, v in cosim_eval_result.items() if k.startswith("mp3")},
    # {k: v for k, v in audioseal_eval_result.items() if k.startswith("mp3")},
    # ]
    combined_result = {
        k: {"Cosine similarity": v, "Audioseal": audioseal_eval_result[k]}
        for k, v in cosim_eval_result.items()
        if k.startswith(starting_key)
    }

    x_axis_values = [parse_number(k.split("_")[-1]) for k in combined_result.keys()]
    # x_axis_label = list(combined_result.keys())

    simple_comparison_plot(
        list(combined_result.values()),
        model_score_cutoff={"Cosine similarity": 1.0, "Audioseal": 0.5},
        x_axis_values=x_axis_values,
        x_axis_label=x_axis_label,
        save_path=save_path,
        comparing_title=comparing_title,
    )


def plot_results_metadata(
    starting_key="lp",
    save_path="accuracy_comparison_lp.png",
    comparing_title="low pass filter",
    x_axis_label="low pass cut off",
    remove_key=["max_length_0.5"],
):

    detector_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k_speed.json"
    # detector_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k_speed2.json"
    # audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_from_metadata.json"
    # audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_from_metadata_crop_scale.json"
    audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_from_metadata_speed.json"

    detector_eval_result = json.load(open(detector_eval_result_path, "r"))
    audioseal_eval_result = json.load(open(audioseal_eval_result_path, "r"))
    # mp3_results = [
    # {k: v for k, v in cosim_eval_result.items() if k.startswith("mp3")},
    # {k: v for k, v in audioseal_eval_result.items() if k.startswith("mp3")},
    # ]
    combined_result = {
        k: {"Audio NoisePrinter": v, "Audioseal": audioseal_eval_result[k]}
        for k, v in detector_eval_result.items()
        if k.startswith(starting_key) and (k not in remove_key)
    }

    x_axis_values = [k.split("_")[-1] for k in combined_result.keys()]
    # x_axis_label = list(combined_result.keys())

    simple_comparison_plot(
        list(combined_result.values()),
        model_score_cutoff={"Audio NoisePrinter": 0.5, "Audioseal": 0.5},
        x_axis_values=x_axis_values,
        x_axis_label=x_axis_label,
        save_path=save_path,
        comparing_title=comparing_title,
    )


def plot_results_all(
    starting_key="lp",
    save_path="accuracy_comparison_lp.png",
    comparing_title="low pass filter",
    x_axis_label="low pass cut off",
):

    # cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results.json"
    cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results_16k.json"
    audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results.json"
    detector_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k.json"

    cosim_eval_result = json.load(open(cosim_eval_result_path, "r"))
    audioseal_eval_result = json.load(open(audioseal_eval_result_path, "r"))
    detector_eval_result = json.load(open(detector_eval_result_path, "r"))

    # mp3_results = [
    # {k: v for k, v in cosim_eval_result.items() if k.startswith("mp3")},
    # {k: v for k, v in audioseal_eval_result.items() if k.startswith("mp3")},
    # ]
    combined_result = {
        k: {
            "Cosine similarity": v,
            "Audioseal": audioseal_eval_result[k],
            "Audio NoisePrinter": detector_eval_result[k],
        }
        for k, v in cosim_eval_result.items()
        if k.startswith(starting_key)
    }

    x_axis_values = [parse_number(k.split("_")[-1]) for k in combined_result.keys()]
    # x_axis_label = list(combined_result.keys())

    simple_comparison_plot(
        list(combined_result.values()),
        model_score_cutoff={
            "Cosine similarity": 1.0,
            "Audioseal": 0.5,
            "Audio NoisePrinter": 0.5,
        },
        x_axis_values=x_axis_values,
        x_axis_label=x_axis_label,
        save_path=save_path,
        comparing_title=comparing_title,
    )


def print_results_all(
    starting_key="lp",
):

    # cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results.json"
    cosim_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/cosim_eval_results_16k.json"
    # audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results.json"
    audioseal_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioeval_eval_results_echo.json"
    detector_eval_result_path = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/detector_eval_results_16k.json"

    cosim_eval_result = json.load(open(cosim_eval_result_path, "r"))
    audioseal_eval_result = json.load(open(audioseal_eval_result_path, "r"))
    detector_eval_result = json.load(open(detector_eval_result_path, "r"))

    # mp3_results = [
    # {k: v for k, v in cosim_eval_result.items() if k.startswith("mp3")},
    # {k: v for k, v in audioseal_eval_result.items() if k.startswith("mp3")},
    # ]
    model_score_cutoff = {
        "Cosine similarity": 1.0,
        "Audioseal": 0.5,
        "Audio NoisePrinter": 0.5,
    }

    def get_acc(logits, cutoff):
        real_logits = np.array(logits["real_logit"])
        fake_logits = np.array(logits["fake_logit"])

        real_acc = np.mean(real_logits >= cutoff) if len(real_logits) > 0 else 0.0
        fake_acc = np.mean(fake_logits < cutoff) if len(fake_logits) > 0 else 0.0

        overall_acc = 0.5 * (real_acc + fake_acc)
        return overall_acc

    combined_result = {
        k: {
            "Cosine similarity": get_acc(v, model_score_cutoff["Cosine similarity"]),
            "Audioseal": get_acc(
                audioseal_eval_result[k], model_score_cutoff["Audioseal"]
            ),
            "Audio NoisePrinter": get_acc(
                detector_eval_result[k], model_score_cutoff["Audio NoisePrinter"]
            ),
        }
        for k, v in cosim_eval_result.items()
        if k.startswith(starting_key)
    }
    # x_axis_values = [parse_number(k.split("_")[-1]) for k in combined_result.keys()]
    # x_axis_label = list(combined_result.keys())

    print(combined_result)


if __name__ == "__main__":
    # test_audio = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/gen_audio/wav/0.wav"
    # wm_test_audio = "/home/tst000/projects/tst000/official_dataset/fix_latent_detector/f5tts/fix_latent/rand_latent_mel/wav/3.wav"
    # checkpoint_path = "/home/tst000/projects/tst000/checkpoint/latent_aug_wm/simple_detector_v2/version_debug_0/epoch_00009.pth"

    # detector = CustomDetector(checkpoint_path, device="cpu")

    # audio, _ = torchaudio.load(test_audio)
    # print(audio.shape)
    # result = detector.detect(audio)
    # print(result)
    # audio, _ = torchaudio.load(wm_test_audio)
    # result = detector.detect(audio)
    # print(result)

    # metadata_json = "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/metadata.json"
    # out_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioseal_wm_wav/"
    # updated_meta_data_fname = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/f5tts/audioseal_metadata.json"

    # # audioseal_generator = AudioSealWMDatasetGenerator(metadata_json=metadata_json)
    # # audioseal_generator.run(out_dir, updated_meta_data_fname)

    # tmp_dir = "/home/tst000/projects/tst000/official_dataset/audioseal_comparison/tmp/"
    # #wm_dir = out_dir
    # #non_wm_dir = "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/wav/"
    # #audioseal_detector = AudioSealAugmentationTest(tmp_dir, wm_dir, non_wm_dir)
    # cosine_detector = SimpleDataAugGenerator(
    #     metadata_json="/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts/metadata.json",
    #     tmp_dir=tmp_dir
    # )

    # #result = audioseal_detector.run(aug_fn=default_aug, file_type=None)
    # result = cosine_detector.run(aug_fn=default_aug, file_type=None)
    # print(result)

    # debug_result = [
    #     {
    #         "modelA": result,
    #         "modelB": result,
    #     },
    #     {
    #         "modelA": result,
    #         "modelB": result,
    #     }
    # ]
    # debug_model_score_cutoff = {
    #         "modelA": 0.5,
    #         "modelB": 0.5,
    #     }
    # simple_comparison_plot(
    #     debug_result,
    #     model_score_cutoff=debug_model_score_cutoff,
    #     x_axis_values=[1, 2],
    #     x_axis_label=,
    #     save_path="accuracy_comparison.png",
    #     comparing_title="no augmentation")

    # audioseal_result_from_all_aug()
    # cosim_result_from_all_aug()
    # audioseal_result_from_all_aug_from_metadata()
    # detector_result_from_all_aug()
    # plot_results()
    # plot_results(
    #     starting_key="mp3",
    #     save_path="accuracy_comparison_mp3.pdf",
    #     comparing_title="MP3 compression",
    #     x_axis_label="bit-rate",
    # )
    # plot_results(
    #     starting_key="hp",
    #     save_path="accuracy_comparison_hp.pdf",
    #     comparing_title="high pass filter",
    #     x_axis_label="high pass cut off",
    # )
    # plot_results(
    #     starting_key="lp",
    #     save_path="accuracy_comparison_lp.pdf",
    #     comparing_title="low pass filter",
    #     x_axis_label="low pass cut off",
    # )
    # plot_results(
    #     starting_key="white_noise_db_",
    #     save_path="accuracy_comparison_white_noise.pdf",
    #     comparing_title="White Noise",
    #     x_axis_label="Signal-to-Noise Ratio in db",
    # )
    # plot_results(
    #     starting_key="pink_noise_db_",
    #     save_path="accuracy_comparison_pink_noise.pdf",
    #     comparing_title="Pink Noise",
    #     x_axis_label="Signal-to-Noise Ratio in db",
    # )
    # plot_results_metadata(
    #     starting_key="max_length",
    #     save_path="accuracy_comparison_corp.pdf",
    #     comparing_title="Audio Cropping",
    #     x_axis_label="remaining audio length (seconds)",
    # )
    # plot_results_metadata(
    #    starting_key="scale",
    #    save_path="accuracy_comparison_scale.pdf",
    #    comparing_title="Audio volumn scaling",
    #    x_axis_label="Scaling factor",
    # )
    # plot_results_metadata(
    #    starting_key="speed",
    #    save_path="accuracy_comparison_speed.pdf",
    #    comparing_title="Speed",
    #    x_axis_label="Speed scaling factor",
    # )

    # plot_results_all(
    #     starting_key="mp3",
    #     save_path="accuracy_comparison_mp3.pdf",
    #     comparing_title="MP3 compression",
    #     x_axis_label="bit-rate",
    # )
    # plot_results_all(
    #     starting_key="hp",
    #     save_path="accuracy_comparison_hp.pdf",
    #     comparing_title="high pass filter",
    #     x_axis_label="high pass cut off",
    # )
    # plot_results_all(
    #     starting_key="lp",
    #     save_path="accuracy_comparison_lp.pdf",
    #     comparing_title="low pass filter",
    #     x_axis_label="low pass cut off",
    # )
    # plot_results_all(
    #     starting_key="white_noise_db_",
    #     save_path="accuracy_comparison_white_noise.pdf",
    #     comparing_title="White Noise",
    #     x_axis_label="Signal-to-Noise Ratio in db",
    # )
    # plot_results_all(
    #     starting_key="pink_noise_db_",
    #     save_path="accuracy_comparison_pink_noise.pdf",
    #     comparing_title="Pink Noise",
    #     x_axis_label="Signal-to-Noise Ratio in db",
    # )
    # print_results_all(starting_key="echo")
    get_acc_of_different_dis_fn_cutoff()
