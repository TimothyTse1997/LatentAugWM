import random
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


def rebatching_from_varying_length(generated, lens, duration):
    mels_removed_cond = []
    for mel, length, dur in zip(generated, lens, duration):
        mels_removed_cond.append(mel[length:dur, :])
    return pad_sequence(mels_removed_cond, batch_first=True)


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

        # for example, we can use griffinlim here
        if custum_vocoder is not None:
            self.vocoder = custum_vocoder

        self.set_require_grad()

        # if train:
        #     self.vocoder = self.vocoder.half()

        pass

    def update_wm_function(self, use_wm=True):
        self.ema_model.use_wm = use_wm

    def set_require_grad(self):
        for param in self.vocoder.parameters():
            param.requires_grad = False

        for param in self.ema_model.parameters():
            param.requires_grad = False

    def mask_audio(
        self, lens, max_duration, gen_audio, return_mask=False
    ):  # , hop_length=256, win_length=1024):

        hop_length, win_length = (
            self.mel_spectrogram.hop_length,
            self.mel_spectrogram.win_length,
        )
        audio_lens = [
            min(length * (hop_length + 1) + win_length, max_duration) for length in lens
        ]
        batch_size = len(lens)
        mask = torch.zeros_like(
            gen_audio, device=gen_audio.device, dtype=gen_audio.dtype
        )
        for i, al in enumerate(audio_lens):
            mask[i, :al] = 1.0
        if not return_mask:
            return gen_audio * mask
        else:
            return gen_audio * mask, mask

    def train(self):
        # self.vocoder = self.vocoder.half()
        return

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
        self.ema_model.transformer.clear_cache()
        generated_rebatched = rebatching_from_varying_length(generated, lens, duration)
        return generated_rebatched.permute(0, 2, 1), generated.permute(0, 2, 1)

    def __call__(
        self,
        cond,
        text,
        duration,
        lens,
        fix_noise=None,
        use_grad_checkpoint=False,
        eval=False,
        cache=True,
        **kwargs,
    ):
        # print(cond.shape, fix_noise.shape, duration, lens)
        # print(text)
        self.ema_model.transformer.clear_cache()
        assert self.ema_model.transformer.text_cond is None
        generated, _ = self.ema_model._sample(
            cond=cond,
            text=text,
            duration=duration,
            lens=lens,
            steps=self.inference_kwargs["nfe_step"],
            cfg_strength=self.inference_kwargs["cfg_strength"],
            sway_sampling_coef=self.inference_kwargs["sway_sampling_coef"],
            fix_noise=fix_noise,
            use_grad_checkpoint=use_grad_checkpoint,
            cache=cache,
        )
        self.ema_model.transformer.clear_cache()
        assert self.ema_model.transformer.text_cond is None
        generated_rebatched = rebatching_from_varying_length(generated, lens, duration)
        gr_wave = self.vocoder.decode_with_grad(generated_rebatched.permute(0, 2, 1))

        result = {
            "generated_rebatched_mel": generated_rebatched,
            "gr_wave": gr_wave,
        }
        if not eval:
            return result
        with torch.no_grad():
            result["g_wave"] = self.vocoder.decode_with_grad(generated.permute(0, 2, 1))
            result["generated_mel"] = generated
        return result


class F5TTSFixNoiseInferencer(F5TTS):
    def __init__(self, *args, fix_latent=None, fix_latent_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if fix_latent is not None:
            self.fix_latent = fix_latent
        else:
            self.fix_latent = torch.load(fix_latent_path).permute(1, 0)
        self.ema_model.noise_update_fn = self.update_random_noise_with_lens

    def update_random_noise_with_lens(self, random_noise, lens, **kwargs):
        self.fix_latent = self.fix_latent.to(random_noise.device).to(random_noise.dtype)
        max_length = random_noise.shape[1]
        for i, length in enumerate(lens):
            latent_length = max_length - length
            random_noise[i, length:, :] = self.fix_latent[:latent_length, :]
        return random_noise


class F5TTSPeriodicFixNoiseInferencer(F5TTSFixNoiseInferencer):
    def update_random_noise_with_lens(self, random_noise, lens, **kwargs):
        latent_size, _ = self.fix_latent.shape
        current_latent = self.fix_latent.detach().clone()

        max_length = random_noise.shape[1]
        num_repeat = 1 + (max_length // latent_size)

        print(f"before repeat {num_repeat}: ", current_latent.shape, max_length)

        current_latent = current_latent.repeat(num_repeat, 1)
        print(f"after repeat {num_repeat}: ", current_latent.shape)

        current_latent = current_latent.to(random_noise.device).to(random_noise.dtype)

        for i, length in enumerate(lens):
            latent_length = max_length - length
            random_noise[i, length:, :] = current_latent[:latent_length, :]
        return random_noise


# class F5TTSPeriodicInterpolateNoiseInferencer(F5TTSFixNoiseInferencer):

#     noise_segment = [(20, 1/4), (20, 1/2), (20, 1), (20, 2), , (20, 4)]
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.repeating_latent = self.get_repeating_latent()

#     def get_repeating_latent(self):
#         repeating_latent = []
#         start_index = 0
#         channels, length = self.fix_latent.shape
#         for size, frac in self.noise_segment:
#             end_index = start_index + size
#             current_length = max(1, int(length * frac))
#             repeating_latent.append(
#                 self.fix_latent[start_index:end_index, :current_length]
#             )
#             start_index = end_index
#         return repeating_latent


#     def update_random_noise_with_lens(self, random_noise, lens, **kwargs):
#         j
#         for fix_latent in self.repeating_latent:
#             latent_size, _ = self.fix_latent.shape
#             current_latent = self.fix_latent.detach().clone()

#             max_length = random_noise.shape[1]
#             num_repeat = 1 + (max_length // latent_size)

#             print(f"before repeat {num_repeat}: ", current_latent.shape, max_length)

#             current_latent = current_latent.repeat(num_repeat, 1)
#             print(f"after repeat {num_repeat}: ", current_latent.shape)

#             current_latent = current_latent.to(random_noise.device).to(random_noise.dtype)

#             for i, length in enumerate(lens):
#                 latent_length = max_length - length
#                 random_noise[i, length:, :] = current_latent[:latent_length, :]
#         return random_noise


def load_cfm(
    model="F5TTS_v1_Base",
    use_ema=True,
    device="cuda",
    ode_method="euler",
    vocab_file="",
    hf_cache_dir=None,
    ckpt_file=None,
    bf16=False,
):
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    # override for previous models
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
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
    if not bf16:
        return ema_model
    return ema_model.to(torch.bfloat16)


if __name__ == "__main__":

    from latent_aug_wm.dataset.mel_dataset import (
        get_combine_dataloader,
        get_ref_text_from_wav,
    )

    def get_ref_text_fname_from_wav(wav_path):
        wav_path = Path(wav_path).absolute()
        wav_parent, wav_name = wav_path.parent, wav_path.name
        wav_name = wav_name.split(".")[0]
        text_fname = wav_parent / f"{wav_name}.normalized.txt"
        return text_fname

    # sampler = F5TTS(device="cuda")
    fix_latent_path = "/gpfs/fs3c/nrc/dt/tst000/datasets/periodic_special_latent.pt"
    sampler = F5TTSPeriodicFixNoiseInferencer(
        fix_latent_path=fix_latent_path, device="cuda"
    )

    ref_wav_file = "/home/tst000/projects/tst000/datasets/selected_ref_files.txt"
    gen_txt_fname = "/home/tst000/projects/tst000/datasets/selected_gen_text.txt"
    # out_dir = Path("/home/tst000/projects/tst000/datasets/f5tts_random_audio")
    out_dir = Path(
        "/home/tst000/projects/tst000/datasets/f5tts_new_text_periodic_noise_test_set"
    )
    if not out_dir.exists():
        out_dir.mkdir()
    # text_dataset = []
    with open(ref_wav_file, "r") as f:
        all_wav_files = [line.rstrip() for line in f]

    # with open(gen_txt_fname, "r") as f:
    #    text_dataset = [line.rstrip() for line in f]

    random.shuffle(all_wav_files)
    # random.shuffle(text_dataset)

    from datasets import load_dataset

    text_dataset = load_dataset("jakeazcona/short-text-labeled-emotion-classification")[
        "test"
    ]

    max_data = 500

    for i, wav_fname in enumerate(all_wav_files):
        text_fname = get_ref_text_fname_from_wav(wav_fname)
        wav, sr, spec, _ = sampler.infer(
            ref_file=wav_fname,
            ref_text=get_ref_text_from_wav(wav_fname),
            gen_text=text_dataset[i]["sample"],
            file_wave=str(out_dir / f"{i}.wav"),
            # file_spec="./api_out.png",
            seed=None,
            # cfg_strength=0,
        )
        if i > max_data:
            break

    exit()

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

    def save_text(text, fname):
        with open(fname, "w") as f:
            f.write(text)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_rebatched, generated = sampler.sample(
            cond=batch["cond"].cuda(),
            text=batch["text"],
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
            g_wave = (
                sampler.vocoder.decode(g.unsqueeze(0).float()).squeeze().cpu().numpy()
            )

            # save_spectrogram(gr.detach().cpu(), f"{i}_gr_mel.png")
            # sampler.export_wav(gr_wave, f"{i}_gr.wav")
            # save_text(batch["gen_texts"][i], f"{i}_gr_text.txt")

            # save_spectrogram(g.detach().cpu(), f"{i}_g_mel.png")
            # sampler.export_wav(g_wave, f"{i}_g.wav")
            # save_text(batch["combine_texts"][i], f"{i}_g_text.txt")
