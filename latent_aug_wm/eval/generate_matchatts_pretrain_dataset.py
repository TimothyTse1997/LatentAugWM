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

import sys

sys.path.insert(0, "/home/tst000/projects/tst000/tts_watermark_eval")
from src.perturbations import AudioPerturbations

# import gdown

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS, InverseMatchaTTS
from matcha.text import sequence_to_text, _symbol_to_id  # text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.audio import mel_spectrogram

from matcha.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.pylab as plt

matplotlib.use("Agg")

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


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def sym_to_sequence(symbles):
    sequence = []
    for symbol in symbles:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence, symbles


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    output_mel = output["mel"].cpu().numpy()
    np.save(folder / f"{filename}", output_mel)
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")
    save_spectrogram(output_mel[0], folder / f"{filename}_mel.png")


def save_trajectory(trajectory, folder, speaker=0):
    if not folder.exists():
        folder.mkdir()
    for step, step_mel in enumerate(trajectory):
        step_mel = step_mel.detach().cpu()
        torch.save(step_mel, folder / f"speaker_{speaker}_step_{step}_mel.pt")
        step_mel = step_mel.cpu().numpy()
        step_fname = folder / f"speaker_{speaker}_step_{step}_mel.png"
        save_spectrogram(step_mel[0], step_fname)


def plot_kde(A, B, fname):
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot A (fixed color and label)
    sns.kdeplot(A, label="original distribution", color="red", linewidth=2)

    # Normalize the keys for colormap
    keys = sorted(B.keys())
    norm = Normalize(vmin=min(keys), vmax=max(keys))
    cmap = plt.get_cmap("viridis")

    # Plot each distribution in B with a color from the continuous colormap
    for key in keys:
        color = cmap(norm(key))
        sns.kdeplot(B[key], label=str(key), color=color, linewidth=2, alpha=0.8)

    # Create colorbar as legend for B's keys
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = plt.colorbar(sm, pad=0.01, ax=plt.gca())
    cbar.set_label("EC loops")

    plt.title("original distribution vs reverse distribution")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend(title="Legend")
    plt.tight_layout()
    plt.savefig(fname)


def plot_original_distribution(A, fname):

    # Sample list of values
    plt.figure()
    values = A

    # Create the KDE plot
    sns.kdeplot(values, fill=True)  # fill=True gives a shaded area under the curve

    # Add labels and title
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Kernel Density Estimate (KDE)")
    plt.savefig(fname)


def plot_with_markers(
    values, marker_indices, marker="o", line_color="blue", marker_color="red", fname=""
):
    """
    Plots a line graph from a list of values and places markers only at specified indices.

    Parameters:
        values (list or array): The y-values of the line plot.
        marker_indices (list of int): Indices at which markers should appear.
        marker (str): Marker style (default: 'o').
        line_color (str): Color of the line (default: 'blue').
        marker_color (str): Color of the markers (default: 'red').
    """
    # Plot the line without markers
    print(values.shape)
    plt.figure()
    plt.plot(values, color=line_color)

    # Plot only the markers at specified indices
    marker_x = marker_indices

    print(values.shape)
    print(values.shape)
    print(values.shape)

    marker_y = [values[i] for i in marker_indices]
    plt.plot(marker_x, marker_y, marker=marker, linestyle="None", color=marker_color)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Line Plot with Selective Markers")
    plt.grid(True)
    plt.savefig(fname)


class MatchTTSGenerator:
    n_timesteps = 10
    ## Changes to the speaking rate
    length_scale = 1.0
    ## Sampling temperature
    # temperature = 0.667
    temperature = 1.0

    def __init__(
        self,
        cache_dir="/home/tst000/projects/tst000/.local/share",
        device=None,
    ):
        self.cache_dir = Path(cache_dir)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        MATCHA_CHECKPOINT = self.cache_dir / "matcha_vctk.ckpt"
        HIFIGAN_CHECKPOINT = (
            self.cache_dir / "g_02500000"
        )  # "hifigan_T2_v1" / "generator_v1"
        self.count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

        self.model = self.load_model(MATCHA_CHECKPOINT, self.device)
        print(f"Model loaded! Parameter count: {self.count_params(self.model)}")

        self.vocoder = self.load_vocoder(HIFIGAN_CHECKPOINT, device)
        self.denoiser = Denoiser(self.vocoder, mode="zeros")
        print("Vocoder loaded!")

    @staticmethod
    def load_model(checkpoint_path, device):
        model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        return model

    @staticmethod
    def load_vocoder(checkpoint_path, device):
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        hifigan.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["generator"]
        )
        _ = hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan

    @staticmethod
    @torch.inference_mode()
    def process_text(text: str, phone: str, device=None):
        # x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2'])[0], 0),dtype=torch.long, device=device)[None]
        x = torch.tensor(
            intersperse(sym_to_sequence(phone)[0], 0), dtype=torch.long, device=device
        )[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())
        return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

    @torch.inference_mode()
    def synthesise(self, text, phone, spks=None, preset_noise=None, temperature=1.0):
        text_processed = self.process_text(text, phone, device=self.device)
        start_t = dt.datetime.now()
        output = self.model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=self.n_timesteps,
            temperature=self.temperature,
            spks=spks,
            length_scale=self.length_scale,
            # preset_noise=preset_noise
        )
        # merge everything to one dict
        output.update({"start_t": start_t, **text_processed})
        return output

    @torch.inference_mode()
    def to_waveform(self, mel):
        audio = self.vocoder(mel).clamp(-1, 1)
        audio = self.denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
        return audio.cpu().squeeze()

    def run(self, text, phone, spk: int):

        output = self.synthesise(text, phone, spks=torch.tensor([spk]).to(self.device))
        output["waveform"] = self.to_waveform(output["mel"])

        t = (dt.datetime.now() - output["start_t"]).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])

        ## Pretty print
        print(f"{'*' * 53}")
        print(f"Input text")
        print(f"{'-' * 53}")
        print(output["x_orig"])
        print(f"{'*' * 53}")
        print(f"Phonetised text")
        print(f"{'-' * 53}")
        print(output["x_phones"])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        return output


def main():
    matchtts_generator = MatchTTSGenerator()
    text_json = "/home/tst000/projects/tst000/official_dataset/short-text-labeled-emotion-classification_phoneme.json"
    texts = json.load(open(text_json, "r"))

    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/matchatts"
    )
    source_noise_save_dir = save_dir / "source_noise"
    gen_mel_save_dir = save_dir / "gen_mel"

    if not source_noise_save_dir.exists():
        source_noise_save_dir.mkdir()
    if not gen_mel_save_dir.exists():
        gen_mel_save_dir.mkdir()

    final_json = []

    max_data = 500
    current_gen_num = 0
    for text, phone in texts:
        for spk in range(10):
            if current_gen_num % 10 == 0:
                print(f"current step: {current_gen_num}")
            output = matchtts_generator.run(text, phone, spk)
            final_mel = output["decoder_outputs"][-1].detach().cpu().float()
            initial_noise = output["decoder_outputs"][0].detach().cpu().float()
            print("final_mel", final_mel.shape)
            print("initial_noise", initial_noise.shape)

            fname = f"{current_gen_num}.pt"

            orig_noise_save_path = str((source_noise_save_dir / fname).absolute())
            gen_tensor_save_path = str((gen_mel_save_dir / fname).absolute())

            torch.save(initial_noise, orig_noise_save_path)
            torch.save(final_mel, gen_tensor_save_path)
            metadata_dict = {
                "orig_noise": orig_noise_save_path,
                "generated_tensor": gen_tensor_save_path,
                "gen_text": text,
                "spk": spk,
            }
            final_json.append(metadata_dict)

            current_gen_num += 1
        if current_gen_num > max_data:
            break

    json_fname = save_dir / "metadata.json"
    json.dump(final_json, open(json_fname.absolute(), "w"), indent=4)


if __name__ == "__main__":
    main()
