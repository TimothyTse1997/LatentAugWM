import torch
import torch.nn as nn
from torchaudio.transforms import GriffinLim, InverseMelScale

CONFIG_FOR_GRIFFINLIM = {
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
    "n_iter": 32,
    "n_mels": 100,
    "n_stft": 1024 // 2 + 1,
    "power": 1,
}


class CustomGriffinLim(nn.Module):
    def __init__(self, griffin_lim_config=CONFIG_FOR_GRIFFINLIM):
        super().__init__()
        self.griffin_lim_config = griffin_lim_config
        self.inv_melscale = InverseMelScale(
            n_stft=griffin_lim_config["n_stft"], n_mels=griffin_lim_config["n_mels"]
        )
        self.griffinlim = GriffinLim(
            n_fft=griffin_lim_config["n_fft"],
            n_iter=griffin_lim_config["n_iter"],
            win_length=griffin_lim_config["win_length"],
            hop_length=griffin_lim_config["hop_length"],
            power=griffin_lim_config["hop_length"],
        )

    def forward(self, x):
        inv_mel_x = self.inv_melscale(x)
        return self.griffinlim(inv_mel_x)


if __name__ == "__main__":
    cgl = CustomGriffinLim()
    rand_mel = torch.randn((2, 100, 1273))
    print(rand_mel.shape)
    print(cgl(rand_mel).shape)
