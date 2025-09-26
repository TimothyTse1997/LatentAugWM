import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_aug_wm.model.encoder import UniqueNoiseEncoder
from latent_aug_wm.data_augmentation.base import BaseBatchAugmentation
from loguru import logger as glogger

l1_loss = nn.L1Loss()


def generate_multiple_mel_from_f5tts(nets=None, step=None, eval=False, **kwargs):
    # ONLY works with model.encoder.UniqueNoiseEncoder
    assert isinstance(nets.noise_encoder, UniqueNoiseEncoder)
    common_latent = nets.noise_encoder.common_latent

    random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)

    wm_noise = nets.noise_encoder(random_noise)
    orig_noise = nets.noise_encoder.get_non_wm_latent(random_noise)

    # with torch.no_grad():
    wm_out = nets.f5tts(fix_noise=wm_noise, use_grad_checkpoint=(not eval), **kwargs)
    with torch.no_grad():
        orig_out = nets.f5tts(fix_noise=orig_noise, **kwargs)
        out = nets.f5tts(fix_noise=random_noise, **kwargs)
    # out:
    # {
    #     "generated_rebatched": generated_rebatched,
    #     "gr_wave": gr_wave,
    #     "g_wave": g_wave,
    # }
    wm_out = {("wm_" + k): v for k, v in wm_out.items()}
    orig_out = {("orig_" + k): v for k, v in orig_out.items()}
    out = {("rand_" + k): v for k, v in out.items()}

    wm_out.update(out)
    wm_out.update(orig_out)

    return wm_out, ["noise_encoder"]


# TODO normalize the loss by duration!!
def mel_reg_L1_loss(nets=None, step=None, scale=None, **kwargs):
    original_mel = kwargs["orig_generated_rebatched"]
    watermark_mel = kwargs["wm_generated_rebatched"]

    mel_reg_loss = l1_loss(watermark_mel, original_mel)
    if scale is not None:
        mel_reg_loss = mel_reg_loss * scale
    return {"mel_reg_l1_loss": mel_reg_loss}, []


def detector_loss(nets=None, step=None, scale=None, **kwargs):
    real_input, fake_input = kwargs["rand_gr_wave"], kwargs["wm_gr_wave"]
    batch_size, seq_len = real_input.shape

    real_logits = torch.ones(batch_size, dtype=torch.long, device=real_input.device)
    fake_logits = torch.zeros(batch_size, dtype=torch.long, device=fake_input.device)

    real_detector_logits, real_loss = nets.detector.calculate_loss(
        real_input.unsqueeze(1), real_logits
    )
    fake_detector_logits, fake_loss = nets.detector.calculate_loss(
        fake_input.unsqueeze(1), fake_logits
    )

    return {
        "detector_loss": real_loss + fake_loss,
        "real_detector_logits": real_detector_logits.detach().cpu(),
        "fake_detector_logits": fake_detector_logits.detach().cpu(),
    }, ["detector"]


class AugApplier:
    def __init__(self, aug_obj: BaseBatchAugmentation):
        self.target_keys = ["rand_gr_wave", "wm_gr_wave"]
        self.aug_obj = aug_obj

    def __call__(self, nets=None, step=None, scale=None, **kwargs):
        return {
            k: self.aug_obj(kwargs[k].unsqueeze(1)).squeeze(1) for k in self.target_keys
        }, []


if __name__ == "__main__":
    from attrmap import AttrMap
    from latent_aug_wm.model.decoder import BaseAudioSealClassifier
    from latent_aug_wm.f5_infer.infer import F5TTSBatchInferencer

    common_latent = torch.randn((2000, 100))

    nets = AttrMap(
        {
            "noise_encoder": UniqueNoiseEncoder(common_latent).cuda().half(),
            # "f5tts": F5TTSBatchInferencer(model="F5TTS_Small", device="cuda", train=True),
            "f5tts": F5TTSBatchInferencer(device="cuda", train=True),
            "detector": BaseAudioSealClassifier(input_dim=1).cuda().half(),
        }
    )
    nets.f5tts.ema_model = nets.f5tts.ema_model.half()

    glogger.debug("created models")
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
        batch_size=16,
        allowed_padding=50,
    )
    glogger.debug("completed batching")
    batch = next(data_iter)
    if isinstance(batch, list):
        batch = [(b.to("cuda") if torch.is_tensor(b) else b) for b in batch]
    else:
        batch = {
            k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    from latent_aug_wm.loss import construct_loss_fn

    transform_configs = {
        "AddColoredNoise": {
            "mode": "per_example",
            "p": 1.0,
            "min_snr_in_db": 10.0,
            "max_snr_in_db": 10.0,
        }
    }
    # scaler = torch.cuda.amp.GradScaler()
    aug_obj = BaseBatchAugmentation(
        sampling_rate=24000, transform_configs=transform_configs
    )

    glogger.debug("completed aug obj")
    loss_fn = construct_loss_fn(
        [
            generate_multiple_mel_from_f5tts,
            mel_reg_L1_loss,
            AugApplier(aug_obj),
            detector_loss,
        ],
        log_fn_time=False,
    )
    glogger.debug("completed loss func")

    nets.f5tts.ema_model = nets.f5tts.ema_model.half()

    with torch.cuda.amp.autocast():
        _ = loss_fn(nets, {}, batch, 0, eval=False)
        del _
        total_loss, loss_items, _, items = loss_fn(nets, {}, batch, 0, eval=True)

    total_loss.backward()
    print(items.keys())
    print("loss", loss_items)
    for n, p in nets.noise_encoder.named_parameters():
        if p.requires_grad:
            print(n, p.grad.abs().mean())
