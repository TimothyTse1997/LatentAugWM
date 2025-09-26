import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from audioseal.models import AudioSealDetector
from audioseal.loader import load_model_checkpoint, _update_state_dict


class BaseAudioSealClassifier(AudioSealDetector):
    def __init__(
        self,
        input_dim=1024 * 2,
        model_kwargs={
            "activation": "ELU",
            "activation_params": {"alpha": 1.0},
            "causal": False,
            "channels": 1,
            "compress": 2,
            "dilation_base": 2,
            "dimension": 128,
            "disable_norm_outer_blocks": 0,
            "kernel_size": 7,
            "last_kernel_size": 7,
            "lstm": 2,
            "n_filters": 32,
            "n_residual_layers": 1,
            "norm": "weight_norm",
            "norm_params": {},
            "pad_mode": "constant",
            "ratios": [8, 5, 4, 2],
            "residual_kernel_size": 3,
            "true_skip": True,
            "output_dim": 32,
            "nbits": 10,
        },
    ):
        super().__init__(**model_kwargs)
        self.loss_fn = nn.NLLLoss()
        self.input_dim = input_dim
        # self.proj = nn.Linear(input_dim, model_kwargs["dimension"])
        pass

    def calculate_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,  # binary labels, (B,)
        sample_rate=None,
    ):
        # x = self.proj(x)#.permute(0, 2, 1)  # B, output_dim, seq_len
        seq_logits, _ = self.forward(x, sample_rate=sample_rate)
        logits = seq_logits[:, :2, :].mean(-1)
        logits = F.log_softmax(logits)  # (B, 2)
        loss = self.loss_fn(logits, labels)
        return logits, loss

    def forward(
        self,
        x: torch.Tensor,
        sample_rate=None,
    ):
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
            sample_rate: The sample rate of the input audio
        """
        result = self.detector(x)  # b x 2+nbits
        # REMOVED hardcode softmax on 2 first units used for detection

        # result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message


if __name__ == "__main__":
    batch_size = 16
    seq_len = 100
    detector = BaseAudioSealClassifier(input_dim=1).cuda()
    test_input = torch.randn(batch_size, seq_len)

    with torch.no_grad():
        out, _ = detector.calculate_loss(
            test_input.unsqueeze(1).cuda(),
            torch.ones(batch_size, dtype=torch.long, device="cuda"),
        )
    print(out.shape)
