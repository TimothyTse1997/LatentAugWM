device = "cuda"

from pathlib import Path

import torch
import torchaudio
import bigvgan
import librosa
from meldataset import get_mel_spectrogram

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_22khz_80band", use_cuda_kernel=False
)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)


def _get_all_wav_text(wav_file):
    with open(wav_file, "r") as f:
        all_wav_files = [Path(line.rstrip()) for line in f]
    return all_wav_files


# audio_dir = "/gpfs/fs3c/nrc/dt/tst000/LatentAugWM/experiments"
data_file = (
    "/home/tst000/projects/tst000/datasets/f5tts_dataset_periodic_test_dataset.txt"
)
# data_file = "/home/tst000/projects/tst000/datasets/f5tts_random_audio_test.txt"

all_paths = _get_all_wav_text(data_file)

# out_dir = Path("/home/tst000/projects/tst000/datasets/bigvgan_22k_random_gen_audio")
out_dir = Path("/home/tst000/projects/tst000/datasets/bigvgan_22k_periodic_gen_audio")
if not out_dir.exists():
    out_dir.mkdir()

for p in all_paths:
    outpath = out_dir / str(p.name.split(".")[0] + "_bigvgan_22k.wav")

    # load wav file and compute mel spectrogram
    wav_path = str(p)
    wav, sr = librosa.load(
        wav_path, sr=model.h.sampling_rate, mono=True
    )  # wav is np.ndarray with shape [T_time] and values in [-1, 1]
    wav = torch.FloatTensor(wav).unsqueeze(
        0
    )  # wav is FloatTensor with shape [B(1), T_time]

    # compute mel spectrogram from the ground truth audio
    mel = get_mel_spectrogram(wav, model.h).to(
        device
    )  # mel is FloatTensor with shape [B(1), C_mel, T_frame]

    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = model(
            mel
        )  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
    wav_gen_float = wav_gen.squeeze(
        0
    ).cpu()  # wav_gen is FloatTensor with shape [1, T_time]
    print(sr)
    print(str(p))
    print(outpath)
    torchaudio.save(outpath, wav_gen_float, sr)
