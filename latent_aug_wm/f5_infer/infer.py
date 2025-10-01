import json
from pathlib import Path
from importlib.resources import files
from tqdm import tqdm
from functools import partial
from cached_path import cached_path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from omegaconf import OmegaConf
from hydra.utils import get_class

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import save_spectrogram, load_model


class F5TTSBatchInferencer(F5TTS):
    def __init__(
        self,
        *args,
        inference_kwargs={
            "target_rms": 0.1,
            "cross_fade_duration": 0.15,
            "sway_sampling_coef": -1,
            "cfg_strength": 2,
            "nfe_step": 32,
        },
        train=False,
        custum_vocoder=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inference_kwargs = inference_kwargs
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # for example, we can use griffinlim here
        if custum_vocoder is not None:
            self.vocoder = custum_vocoder

        for param in self.vocoder.parameters():
            param.requires_grad = False
        if train:
            self.vocoder = self.vocoder.half()

        pass

    def update_wm_function(self, use_wm=True):
        self.ema_model.use_wm = use_wm

    def rebatching_from_varying_length(self, generated, lens, duration):
        mels_removed_cond = []
        for mel, length, dur in zip(generated, lens, duration):
            mels_removed_cond.append(mel[length:dur, :])
        return pad_sequence(mels_removed_cond, batch_first=True)

    def sample(self, cond, text, duration, lens, fix_noise=None):
        generated, _ = self.ema_model._sample(
            cond=cond,
            text=text,
            duration=duration,
            lens=lens,
            steps=self.inference_kwargs["nfe_step"],
            cfg_strength=self.inference_kwargs["cfg_strength"],
            sway_sampling_coef=self.inference_kwargs["sway_sampling_coef"],
            fix_noise=fix_noise,
        )
        generated_rebatched = self.rebatching_from_varying_length(
            generated, lens, duration
        )
        return generated_rebatched.permute(0, 2, 1), generated.permute(0, 2, 1)

    def __call__(
        self, cond, text, duration, lens, fix_noise=None, eval=False, **kwargs
    ):
        generated, _ = self.ema_model._sample(
            cond=cond,
            text=text,
            duration=duration,
            lens=lens,
            steps=self.inference_kwargs["nfe_step"],
            cfg_strength=self.inference_kwargs["cfg_strength"],
            sway_sampling_coef=self.inference_kwargs["sway_sampling_coef"],
            fix_noise=fix_noise,
            use_grad_checkpoint=(not eval),
        )
        generated_rebatched = self.rebatching_from_varying_length(
            generated, lens, duration
        )
        gr_wave = self.vocoder.decode(generated_rebatched.permute(0, 2, 1))

        result = {
            "generated_rebatched_mel": generated_rebatched,
            "gr_wave": gr_wave,
        }
        if not eval:
            return result
        with torch.no_grad():
            result["g_wave"] = self.vocoder.decode(generated.permute(0, 2, 1))
        return result


def load_cfm(
    model="F5TTS_v1_Base",
    use_ema=True,
    device="cuda",
    ode_method="euler",
    vocab_file="",
    hf_cache_dir=None,
):
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    # override for previous models
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    if model == "F5TTS_Base":
        if mel_spec_type == "vocos":
            ckpt_step = 1200000
        elif mel_spec_type == "bigvgan":
            model = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif model == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    if not ckpt_file:
        ckpt_file = str(
            cached_path(
                f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}",
                cache_dir=hf_cache_dir,
            )
        )
    ema_model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type,
        vocab_file,
        ode_method,
        use_ema,
        device,
    )
    return ema_model


if __name__ == "__main__":

    from latent_aug_wm.datasets.mel_dataset import get_combine_dataloader

    ref_wav_file = "/home/tst000/projects/datasets/selected_ref_files.txt"
    gen_txt_fname = "/home/tst000/projects/datasets/selected_gen_text.txt"

    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    tmp_dir = "/home/tst000/projects/tmp/libriTTS"

    data_iter = get_combine_dataloader(
        ref_wav_file=ref_wav_file,
        gen_txt_fname=gen_txt_fname,
        mel_spec_kwargs=mel_spec_kwargs,
        tmp_dir="/home/tst000/projects/tmp/libriTTS",
        shuffle=False,
        unsorted_batch_size=256,
        batch_size=32,
        allowed_padding=50,
    )

    batch = next(data_iter)
    sampler = F5TTSBatchInferencer(device="cuda")

    def save_text(text, fname):
        with open(fname, "w") as f:
            f.write(text)

    generated_rebatched, generated = sampler.sample(
        cond=batch["cond"].cuda(),
        texts=batch["texts"],
        duration=batch["duration"].cuda(),
        lens=batch["lens"].cuda(),
    )
    for i, (gr, g) in enumerate(zip(generated_rebatched, generated)):
        if i > 2:
            break
        print(gr.shape)
        print(g.shape)
        gr_wave = (
            sampler.vocoder.decode(gr.unsqueeze(0).float()).squeeze().cpu().numpy()
        )
        g_wave = sampler.vocoder.decode(g.unsqueeze(0).float()).squeeze().cpu().numpy()

        save_spectrogram(gr.detach().cpu(), f"{i}_gr_mel.png")
        sampler.export_wav(gr_wave, f"{i}_gr.wav")
        save_text(batch["gen_texts"][i], f"{i}_gr_text.txt")

        save_spectrogram(g.detach().cpu(), f"{i}_g_mel.png")
        sampler.export_wav(g_wave, f"{i}_g.wav")
        save_text(batch["combine_texts"][i], f"{i}_g_text.txt")
