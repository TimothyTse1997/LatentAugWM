import json
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from argparse import ArgumentParser, BooleanOptionalAction

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


class DiffWaveInferencer:
    def __init__(self, model_dir, device="cpu", params=None):
        models = {}
        self.device = device
        if not model_dir in models:
            if os.path.exists(f"{model_dir}/weights.pt"):
                checkpoint = torch.load(f"{model_dir}/weights.pt")
            else:
                checkpoint = torch.load(model_dir)
            model = DiffWave(AttrDict(base_params)).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            models[model_dir] = model

        self.model = models[model_dir]
        self.model.params.override(params)
        self.params = params

    @torch.no_grad()
    def infer(self, spectrogram, fast_sampling=False):
        training_noise_schedule = np.array(self.model.params.noise_schedule)
        inference_noise_schedule = (
            np.array(self.model.params.inference_noise_schedule)
            if fast_sampling
            else training_noise_schedule
        )

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        if not self.model.params.unconditional:
            if (
                len(spectrogram.shape) == 2
            ):  # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(self.device)
            audio = torch.randn(
                spectrogram.shape[0],
                self.model.params.hop_samples * spectrogram.shape[-1],
                device=self.device,
            )
        else:
            audio = torch.randn(1, self.params.audio_len, device=self.device)

        initial_noise = audio.detach().cpu()
        noise_scale = (
            torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(self.device)
        )

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            audio = c1 * (
                audio
                - c2
                * self.model(
                    audio, torch.tensor([T[n]], device=audio.device), spectrogram
                ).squeeze(1)
            )
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)

        return audio, self.model.params.sample_rate, initial_noise


def main():
    model_path = "/home/tst000/projects/tst000/checkpoint/vocoders/diffwave-ljspeech-22kHz-1000578.pt"
    inferencer = DiffWaveInferencer(model_dir=model_path, device="cuda", params=None)
    spectrogram_dir = Path("/home/tst000/projects/tst000/datasets/LJSpeech-1.1/wavs/")

    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/diffwave/"
    )
    source_noise_save_dir = save_dir / "source_noise"
    gen_mel_save_dir = save_dir / "gen_mel"

    if not source_noise_save_dir.exists():
        source_noise_save_dir.mkdir()
    if not gen_mel_save_dir.exists():
        gen_mel_save_dir.mkdir()

    final_json = []

    max_data = 500

    for i, spectrogram_path in enumerate(tqdm(spectrogram_dir.glob("*.spec.npy"))):

        spectrogram = torch.from_numpy(np.load(spectrogram_path))
        audio, sr, initial_noise = inferencer.infer(spectrogram, fast_sampling=True)

        audio, initial_noise = (
            audio.detach().cpu().float(),
            initial_noise.detach().cpu().float(),
        )
        assert audio.shape == initial_noise.shape

        fname = f"{i}.pt"
        orig_noise_save_path = str((source_noise_save_dir / fname).absolute())
        gen_tensor_save_path = str((gen_mel_save_dir / fname).absolute())

        torch.save(initial_noise, orig_noise_save_path)
        torch.save(audio, gen_tensor_save_path)

        metadata_dict = {
            "orig_noise": orig_noise_save_path,
            "generated_tensor": gen_tensor_save_path,
            "spectrogram_path": str(spectrogram_path),
        }
        final_json.append(metadata_dict)
        if i > max_data:
            break

    json_fname = save_dir / "metadata.json"
    json.dump(final_json, open(json_fname.absolute(), "w"), indent=4)


def convert_tensor_dir_to_wav(
    target_dir="/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/diffwave/gen_mel/",
    output_dir="/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/diffwave/wav/",
):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    get_name = lambda x: x.name.split(".")[0] + ".wav"
    get_new_path = lambda x: (output_dir / get_name(x))

    for tensor_file in tqdm(Path(target_dir).glob("*.pt")):
        audio_save_path = get_new_path(tensor_file)
        target_tensor = torch.load(tensor_file)

        torchaudio.save(audio_save_path, target_tensor, sample_rate=24000)


if __name__ == "__main__":
    # main()
    convert_tensor_dir_to_wav()
