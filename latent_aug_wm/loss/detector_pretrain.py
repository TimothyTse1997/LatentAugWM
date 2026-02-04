import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_aug_wm.f5_infer.forward_backward import (
    sample_from_model,
    forward_backward_sample_from_model,
)
from latent_aug_wm.loss.w2v import W2VPeftLatentLoss


def mask_audio(lens, max_duration, gen_audio, hop_length=256, win_length=1024):
    audio_lens = [
        min(length * (hop_length + 1) + win_length, max_duration) for length in lens
    ]
    batch_size = len(lens)
    mask = torch.zeros_like(gen_audio, device=gen_audio.device, dtype=gen_audio.dtype)
    for i, al in enumerate(audio_lens):
        mask[i, :al] = 1.0
    return gen_audio * mask


class W2VLatentPretrainLoss(W2VPeftLatentLoss):
    def __init__(self, *args, aug_obj, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_obj = aug_obj

    def create_resample_mask(self, lens, audio):
        mask = torch.zeros_like(audio, device=audio.device, dtype=audio.dtype)
        max_dur = audio.shape[1]
        for i, length in enumerate(lens):
            resampled_length = (
                length // self.initially_sampling_rate
            ) * self.w2v_sampling_rate
            resampled_length = min(resampled_length, max_dur)
            mask[i, :resampled_length] = 1

        return mask

    def __call__(self, nets=None, step=None, scale=None, eval=False, **kwargs):
        with torch.no_grad():
            real_audio = nets.f5tts.vocoder.decode(kwargs["cond"].permute(0, 2, 1))
            real_masked_audio = nets.f5tts.mask_audio(
                kwargs["lens"], real_audio.shape[1], real_audio
            )

            fake_out = nets.f5tts(use_grad_checkpoint=False, eval=eval, **kwargs)

            # out:
            # {
            #     "generated_rebatched_mel": generated_rebatched,
            #     "gr_wave": gr_wave,
            #     "g_wave": g_wave,
            # }
            batch_size = kwargs["cond"].shape[0]

            generate_audio_lens = [
                dur - length for dur, length in zip(kwargs["duration"], kwargs["lens"])
            ]
            fake_masked_audio = nets.f5tts.mask_audio(
                generate_audio_lens, fake_out["gr_wave"].shape[1], fake_out["gr_wave"]
            )

            real_input = real_masked_audio
            fake_input = fake_masked_audio

            self.transforms = self.transforms.to(real_input.device)

            real_input = self.aug_obj(real_input.unsqueeze(1)).squeeze(1)
            fake_input = self.aug_obj(fake_input.unsqueeze(1)).squeeze(1)

            real_input_resample = self.transforms(real_input)
            fake_input_resample = self.transforms(fake_input)

            real_reampled_mask = self.create_resample_mask(
                kwargs["lens"], real_input_resample
            )
            fake_reampled_mask = self.create_resample_mask(
                generate_audio_lens, fake_input_resample
            )

            real_logits = torch.ones(
                batch_size, dtype=torch.long, device=real_input.device
            )
            fake_logits = torch.zeros(
                batch_size, dtype=torch.long, device=fake_input.device
            )

        real_out = nets.detector(real_input_resample, attention_mask=real_reampled_mask)
        fake_out = nets.detector(fake_input_resample, attention_mask=fake_reampled_mask)

        real_detector_logits, real_loss = nets.classifier.compute_loss(
            real_out.last_hidden_state, real_logits
        )
        fake_detector_logits, fake_loss = nets.classifier.compute_loss(
            fake_out.last_hidden_state, fake_logits
        )
        if scale is None:
            scale = 1.0
        return {
            "detector_pretrain_loss": (real_loss + fake_loss) * scale,
            "real_detector_logits": real_detector_logits.float().detach().cpu(),
            "fake_detector_logits": fake_detector_logits.float().detach().cpu(),
        }, ["detector", "classifier"]


if __name__ == "__main__":

    import addict
    import matplotlib.pyplot as plt

    from latent_aug_wm.f5_infer.infer import F5TTSBatchInferencer
    from latent_aug_wm.dataset.mel_dataset import get_combine_dataloader
    from latent_aug_wm.f5_infer.forward_backward import load_peft_f5tts_linear

    from f5_tts.infer.utils_infer import save_spectrogram
    import os

    data_dir = "/home/tst000/projects/tst000/datasets/"
    tmp_dir = "/home/tst000/projects/tst000/tmp/"

    inference_kwargs = {
        "target_rms": 0.1,
        "cross_fade_duration": 0.15,
        "sway_sampling_coef": -1,
        "cfg_strength": 2,
        "nfe_step": 32,
    }

    ref_wav_file = data_dir + "selected_ref_files.txt"
    gen_txt_fname = data_dir + "/selected_gen_text.txt"

    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    libriTTS_tmp_dir = tmp_dir + "libriTTS"

    data_iter = get_combine_dataloader(
        ref_wav_file=ref_wav_file,
        gen_txt_fname=gen_txt_fname,
        mel_spec_kwargs=mel_spec_kwargs,
        tmp_dir=libriTTS_tmp_dir,
        shuffle=False,
        unsorted_batch_size=256,
        batch_size=32,
        allowed_padding=50,
    )

    batch = next(data_iter)
    batch = {k: (b.to("cuda") if torch.is_tensor(b) else b) for k, b in batch.items()}

    print(batch["cond"].shape)
    length = batch["cond"].shape[1]
    f5tts = F5TTSBatchInferencer(device="cuda")
    from latent_aug_wm.model.w2vdetector import create_w2v_encoder
    from latent_aug_wm.model.w2vdetector import SimpleLinearClassifier

    nets = addict.Dict(
        {
            "f5tts": f5tts,
            "detector": create_w2v_encoder(
                model_name="facebook/wav2vec2-xls-r-300m"
            ).cuda(),
            "classifier": SimpleLinearClassifier(latent_dim=1024, num_label=2).cuda(),
        }
    )
    loss_fn = W2VLatentPretrainLoss()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        print(loss_fn(nets=nets, step=0, scale=1.0, **batch))
