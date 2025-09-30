import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_aug_wm.model.encoder import UniqueNoiseEncoder, UniqueNoiseEncoderRemoveLen
from latent_aug_wm.data_augmentation.base import BaseBatchAugmentation
from loguru import logger as glogger


def generate_multiple_mel_from_f5tts_with_lens(
    nets=None, step=None, eval=False, **kwargs
):
    # ONLY works with model.encoder.UniqueNoiseEncoder
    assert isinstance(nets.noise_encoder, UniqueNoiseEncoderRemoveLen)
    common_latent = nets.noise_encoder.common_latent
    lens = kwargs["lens"]

    random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)

    wm_noise = nets.noise_encoder(random_noise, lens)
    orig_noise = nets.noise_encoder.get_non_wm_latent(random_noise, lens)

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
