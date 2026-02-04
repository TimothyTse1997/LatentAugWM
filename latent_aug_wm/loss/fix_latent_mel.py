import torch
import torch.nn as nn
import torchaudio


class FixLatentForF5TTSInference:
    def __init__(self, fix_latent=None, fix_latent_path=None):
        if fix_latent is not None:
            self.fix_latent = fix_latent
        else:
            self.fix_latent = torch.load(fix_latent_path).permute(1, 0)
        self.fix_latent_length = self.fix_latent.shape[0]

    def update_random_noise_with_lens(self, random_noise, lens):
        self.fix_latent = self.fix_latent.to(random_noise.device).to(random_noise.dtype)
        max_length = random_noise.shape[1]
        for i, length in enumerate(lens):
            latent_length = max_length - length
            random_noise[i, length:, :] = self.fix_latent[:latent_length, :]
        return random_noise

    def __call__(self, nets=None, step=None, scale=None, eval=False, **kwargs):

        random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)
        aug_noise = random_noise.detach().clone()
        aug_noise = self.update_random_noise_with_lens(aug_noise, kwargs["lens"])

        return (
            {
                "orig_noise": random_noise,
                "aug_noise": aug_noise,
                # "orig_inv_noise": norm_orig_inv_noise,
            },
            [],
        )


def generate_multiple_mel_from_f5tts_with_aug_noise_no_grad(
    nets=None, step=None, eval=False, **kwargs
):
    orig_noise = kwargs["orig_noise"]
    wm_noise = kwargs["aug_noise"]

    with torch.no_grad():
        wm_out = nets.f5tts(
            fix_noise=wm_noise, use_grad_checkpoint=False, eval=eval, **kwargs
        )
        orig_out = nets.f5tts(
            fix_noise=orig_noise, use_grad_checkpoint=False, eval=eval, **kwargs
        )

    wm_out = {("wm_" + k): v for k, v in wm_out.items()}
    orig_out = {("rand_" + k): v for k, v in orig_out.items()}
    wm_out.update(orig_out)

    return wm_out, []


class DifferentialMelSpec:
    def __init__(
        self,
        target_keys=["wm_gr_wave", "rand_gr_wave"],
        mel_spec_kwargs={
            "target_sample_rate": 24000,
            "n_mel_channels": 100,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "mel_spec_type": "vocos",
        },
    ):
        # copied from F5TTS vocos mel
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_spec_kwargs["target_sample_rate"],
            n_fft=mel_spec_kwargs["n_fft"],
            win_length=mel_spec_kwargs["win_length"],
            hop_length=mel_spec_kwargs["hop_length"],
            n_mels=mel_spec_kwargs["n_mel_channels"],
            power=1,
            center=True,
        )
        self.target_keys = target_keys

    def __call__(self, nets=None, step=None, eval=False, **kwargs):
        out = {}
        for k in self.target_keys:

            self.transform = self.transform.to(kwargs[k].device)
            mel = self.transform(kwargs[k]).clamp(min=1e-5).log()
            out[k + "_mel"] = mel.permute(0, 2, 1)
        return out, []


def mel_detector_loss(nets=None, step=None, scale=1.0, **kwargs):
    real_input, fake_input = kwargs["rand_gr_wave_mel"], kwargs["wm_gr_wave_mel"]
    batch_size, seq_len, _ = real_input.shape

    real_logits = torch.ones(batch_size, dtype=torch.long, device=real_input.device)
    fake_logits = torch.zeros(batch_size, dtype=torch.long, device=fake_input.device)

    real_detector_logits, real_loss = nets.detector.calculate_loss(
        real_input.unsqueeze(1), real_logits
    )
    fake_detector_logits, fake_loss = nets.detector.calculate_loss(
        fake_input.unsqueeze(1), fake_logits
    )

    return {
        "detector_loss": (real_loss + fake_loss) * scale,
        "real_detector_logits": real_detector_logits.detach().cpu(),
        "fake_detector_logits": fake_detector_logits.detach().cpu(),
    }, ["detector"]


# def stargan_detector()

if __name__ == "__main__":
    from latent_aug_wm.model.starganv2 import Discriminator2d, StarGanDetector

    discriminator = StarGanDetector(dim_in=100)
    labels = torch.zeros(13, dtype=torch.long)
    out = discriminator.calculate_loss(torch.zeros(13, 1, 269, 100), labels)

    fix_latent_path = "/gpfs/fs3c/nrc/dt/tst000/datasets/special_latent.pt"
    fix_latent_fn = FixLatentForF5TTSInference(fix_latent_path=fix_latent_path)

    print(out)
