from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram


def _get_all_wav_text(wav_file):
    with open(wav_file, "r") as f:
        all_wav_files = [Path(line.rstrip()) for line in f]
    return all_wav_files


class SpeechBrainHiFiGANConverter:
    hifigan_mel_kwargs = {
        22050: {
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "n_fft": 1024,
            "f_min": 0.0,
            "f_max": 8000.0,
            "power": 1,
        },
        16000: {
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "n_fft": 1024,
            "f_min": 0.0,
            "f_max": 8000.0,
            "power": 1,
        },
    }

    def __init__(
        self,
        source_path="/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-22050Hz/snapshots/4188503131602dc234f48d7f22eebea93d788736/",
        # source = "/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-16kHz/snapshots/4d32c6f19e7ff9316c5c56d7903d3d345ddcace4"
        sampling_rate=22050,  # 16000
    ):
        self.hifi_gan = HIFIGAN.from_hparams(source=source_path)
        self.sampling_rate = sampling_rate

    def load_audio(self, p):
        # load wav file and compute mel spectrogram
        wav_path = str(p)
        wav, sr = torchaudio.load(wav_path)
        original_wav_size = wav.shape[1]
        wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        wav = wav[0].squeeze()
        return wav, original_wav_size

    def get_mel(self, wav):
        spectrogram, _ = mel_spectogram(
            audio=wav.squeeze(),
            sample_rate=self.sampling_rate,
            normalized=False,
            min_max_energy_norm=True,
            norm="slaney",
            mel_scale="slaney",
            compression=True,
            **self.hifigan_mel_kwargs[self.sampling_rate]
        )
        return spectrogram

    def spec_to_wav(self, spectrogram):
        with torch.inference_mode():
            waveforms = self.hifi_gan.decode_batch(spectrogram)
        return waveforms

    def prun_and_pad_to_shape(self, current_tensor, target_length):
        current_length = current_tensor.shape[-1]
        if current_length == target_length:
            return current_tensor

        if current_length > target_length:
            return current_tensor[:, :target_length]

        num_pad = target_length - current_length

        return F.pad(current_tensor, (0, num_pad), "constant", 0)

    def __call__(self, p, outpath, new_sampling_rate=None):
        wav, original_wav_size = self.load_audio(p)
        mel = self.get_mel(wav)
        new_wav = self.spec_to_wav(mel)

        if new_sampling_rate:
            new_wav = torchaudio.functional.resample(
                new_wav, self.sampling_rate, new_sampling_rate
            )
            new_wav = self.prun_and_pad_to_shape(new_wav, original_wav_size)
            torchaudio.save(outpath, new_wav.squeeze(1), new_sampling_rate)

        else:
            torchaudio.save(outpath, new_wav.squeeze(1), self.sampling_rate)


def main():
    source = "/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-22050Hz/snapshots/4188503131602dc234f48d7f22eebea93d788736/"
    converter = SpeechBrainHiFiGANConverter(source_path=source, sampling_rate=22050)

    save_dir = Path(
        "/home/tst000/projects/tst000/official_dataset/spatial_correlation_exp/f5tts"
    )
    target_dir = save_dir / "wav/"
    output_dir = save_dir / "hifigan_22k_wav/"
    assert target_dir.exists()
    if not output_dir.exists():
        output_dir.mkdir()

    for audio_path in tqdm(target_dir.glob("*.wav")):
        outpath = output_dir / audio_path.name
        converter(audio_path, outpath, new_sampling_rate=24000)


if __name__ == "__main__":
    # Load a pretrained HIFIGAN Vocoder
    # source = "/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-16kHz/snapshots/4d32c6f19e7ff9316c5c56d7903d3d345ddcace4"
    # source = "/home/tst000/projects/tst000/.cache/huggingface/hub/models--speechbrain--tts-hifigan-libritts-22050Hz/snapshots/4188503131602dc234f48d7f22eebea93d788736/"

    # converter = SpeechBrainHiFiGANConverter(source_path=source, sampling_rate=22050)

    # test_audio = "echo_example.wav"
    # outpath = "echo_example_hifigan_22k.wav"

    # converter(test_audio, outpath, new_sampling_rate=24000)
    main()
