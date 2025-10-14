import torch
import torch.nn as nn
import torchaudio


class W2VDetectorLoss:
    # loss which need to account for audio resampling
    def __init__(initially_sampling_rate=24000, w2v_sampling_rate=16000):
        self.initially_sampling_rate = initially_sampling_rate
        self.w2v_sampling_rate = w2v_sampling_rate
        self.transforms = torchaudio.transforms.Resample(
            self.initially_sampling_rate, self.w2v_sampling_rate
        )

    def __call__(self, nets=None, step=None, scale=None, eval=False, **kwargs):
        real_input, fake_input = kwargs["rand_gr_wave"], kwargs["wm_gr_wave"]
        batch_size, seq_len = real_input.shape

        self.transform = self.transforms.to(real_input.device)

        real_input_resample = self.transforms(real_input)
        fake_input_resample = self.transforms(fake_input)

        real_logits = torch.ones(batch_size, dtype=torch.long, device=real_input.device)
        fake_logits = torch.zeros(
            batch_size, dtype=torch.long, device=fake_input.device
        )

        real_detector_logits, real_loss = nets.detector(
            real_input_resample, labels=real_logits
        )
        fake_detector_logits, fake_loss = nets.detector(
            fake_input_resample, labels=fake_logits
        )

        return {
            "detector_loss": (real_loss + fake_loss) * scale,
            "real_detector_logits": real_detector_logits.detach().cpu(),
            "fake_detector_logits": fake_detector_logits.detach().cpu(),
        }, ["detector"]
