import torch
import torchaudio
from torchaudio import transforms

import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

# fastpitch, generator_train_setup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch')

hifigan, vocoder_train_setup, denoiser = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_hifigan"
)

CHECKPOINT_SPECIFIC_ARGS = [
    "sampling_rate",
    "hop_length",
    "win_length",
    "p_arpabet",
    "text_cleaners",
    "symbol_set",
    "max_wav_value",
    "prepend_space_to_text",
    "append_space_to_text",
]


# for k in CHECKPOINT_SPECIFIC_ARGS:

#     #v1 = generator_train_setup.get(k, None)
#     v2 = vocoder_train_setup.get(k, None)
#     print(k, v2)
print(vocoder_train_setup)
target_sr = vocoder_train_setup.get("sampling_rate", None)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft=1024,
    num_mels=80,
    sampling_rate=target_sr,
    hop_size=256,
    win_size=1024,
    fmin=0,
    fmax=8000,
    center=False,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    # spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    print(spec.shape)
    spec = spectral_normalize_torch(torch.view_as_real(spec))

    return spec


# mel_spec = transforms.MelSpectrogram(
#     sample_rate=target_sr,
#     n_fft=vocoder_train_setup.get('filter_length', None),
#     win_length=vocoder_train_setup.get('win_length', None),
#     hop_length=vocoder_train_setup.get('hop_length', None),
#     f_min=vocoder_train_setup.get('mel_fmin', None),
#     f_max=vocoder_train_setup.get('mel_fmax', None),
#     n_mels=vocoder_train_setup.get('num_mels', None),
#     power=1,
# )

# def get_mel(wave):
#     return mel_spec(wave)


test_wave = "./0.wav"

wav, sr = torchaudio.load(test_wave)

if sr != target_sr:
    with torch.no_grad():
        transform = transforms.Resample(sr, target_sr)
        wav = transform(wav)
        # mel = get_mel(wav)
        mel = mel_spectrogram(wav)

        audio = hifigan(mel).float()
        audio = audio.squeeze(1)

plt.figure(figsize=(10, 12))
res_mel = mel[0].detach().cpu().numpy()
plt.imshow(res_mel, origin="lower")
plt.xlabel("time")
plt.ylabel("frequency")
_ = plt.title("Spectrogram")
plt.savefig("hifi_Spectrogram.png")

audio_numpy = audio[0].cpu().numpy()
# Audio(audio_numpy, rate=22050)

from scipy.io.wavfile import write

write("0_hifigan.wav", vocoder_train_setup["sampling_rate"], audio_numpy)
