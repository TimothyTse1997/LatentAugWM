import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_aug_wm.f5_infer.forward_backward import sample_from_model

l1_loss = nn.L1Loss()


class F5TTSForwardBackwardNoiseAug:
    def __init__(self, inference_kwargs, forward_backward_step=1):
        self.inference_kwargs = inference_kwargs
        self.forward_backward_step = forward_backward_step

    def __call__(self, nets=None, step=None, eval=False, **kwargs):
        random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)
        # only works if nets.noise_encoder was created via
        # latent_aug_wm.f5_infer.forward_backward.load_peft_f5tts_linear
        aug_noise = sample_from_model(
            fix_noise=random_noise,
            model=nets.noise_encoder,
            inference_kwargs=self.inference_kwargs,
            forward_backward_step=self.forward_backward_step,
            **kwargs,
        )
        return {"orig_noise": random_noise, "aug_noise": aug_noise}, ["noise_encoder"]


def len_to_mask(lens, max_dur):

    batch_size = int(lens.shape[0])
    mask = torch.ones((batch_size, max_dur), device=lens.device)  # , dtype=lens.dtype)
    for i, length in enumerate(lens):
        mask[i, :length] = 0
    return mask


def noise_reg_L1_loss(nets=None, step=None, scale=None, **kwargs):
    orig_noise = kwargs["orig_noise"]
    aug_noise = kwargs["aug_noise"]

    _, seq_len, num_channel = orig_noise.shape
    assert num_channel == 100

    lens = kwargs["lens"]
    mask = len_to_mask(lens, seq_len)
    noise_reg_loss = torch.abs(
        (orig_noise - aug_noise) * mask.unsqueeze(-1)
    ).sum() / torch.sum(mask)

    # noise_reg_loss = l1_loss(aug_noise, orig_noise)
    if scale is not None:
        noise_reg_loss = noise_reg_loss * scale
    return {"noise_reg_loss": noise_reg_loss}, []


def generate_multiple_mel_from_f5tts_with_aug_noise(
    nets=None, step=None, eval=False, **kwargs
):
    # ONLY works with model.encoder.UniqueNoiseEncoder

    orig_noise = kwargs["orig_noise"]
    wm_noise = kwargs["aug_noise"]

    # with torch.no_grad():
    wm_out = nets.f5tts(
        fix_noise=wm_noise, use_grad_checkpoint=(not eval), eval=eval, **kwargs
    )
    with torch.no_grad():
        orig_out = nets.f5tts(
            fix_noise=orig_noise, use_grad_checkpoint=False, eval=eval, **kwargs
        )
    # out:
    # {
    #     "generated_rebatched": generated_rebatched,
    #     "gr_wave": gr_wave,
    #     "g_wave": g_wave,
    # }
    wm_out = {("wm_" + k): v for k, v in wm_out.items()}
    orig_out = {("rand_" + k): v for k, v in orig_out.items()}

    wm_out.update(out)
    wm_out.update(orig_out)

    return wm_out, ["noise_encoder"]


if __name__ == "__main__":

    import addict
    from latent_aug_wm.f5_infer.infer import F5TTSBatchInferencer
    from latent_aug_wm.dataset.mel_dataset import get_combine_dataloader
    from latent_aug_wm.f5_infer.forward_backward import load_peft_f5tts_linear

    from f5_tts.infer.utils_infer import save_spectrogram

    inference_kwargs = {
        "target_rms": 0.1,
        "cross_fade_duration": 0.15,
        "sway_sampling_coef": -1,
        "cfg_strength": 2,
        "nfe_step": 32,
    }

    loss_fn = F5TTSForwardBackwardNoiseAug(inference_kwargs)

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
    lens = batch["lens"][0]

    test_seq_length = 2000
    test_mask = len_to_mask(batch["lens"], test_seq_length)

    f5tts = F5TTSBatchInferencer(device="cpu")
    noise_encoder = load_peft_f5tts_linear(f5tts)

    nets = addict.Dict({"f5tts": f5tts, "noise_encoder": noise_encoder})
    out = loss_fn(nets, step=0, **batch)
    _, seq_len, num_channel = out[0]["orig_noise"].shape
    mask = len_to_mask(batch["lens"], seq_len)
    assert num_channel == 100

    output = out[0]

    # save_spectrogram(out["orig_noise"].permute(0, 2, 1).detach()[0][:, lens:], f"orig_noise.png")
    # save_spectrogram(out["aug_noise"].permute(0, 2, 1).detach()[0][:, lens:], f"aug_noise.png")

    save_spectrogram(
        (output["orig_noise"] * mask.unsqueeze(-1)).permute(0, 2, 1).detach()[0],
        f"orig_noise.png",
    )
    save_spectrogram(
        (output["aug_noise"] * mask.unsqueeze(-1)).permute(0, 2, 1).detach()[0],
        f"aug_noise.png",
    )
    print(noise_reg_L1_loss(**output, **batch))

    # save_spectrogram((output["orig_noise"]*mask.unsqueeze(1)).permute(0, 2, 1).detach()[0], f"orig_noise.png")
    # save_spectrogram((output["aug_noise"]*mask.unsqueeze(1)).permute(0, 2, 1).detach()[0], f"aug_noise.png")
