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

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()


def match_stats_ignore_zeros_bnm_preserve_zeros(
    x1: torch.Tensor, x2: torch.Tensor, clamp_min=-7.0, clamp_max=7.0, eps=1e-6
):
    """
    Clamp x2, then normalize each (N, M) tensor in x2 to match the mean and variance of x1,
    ignoring zeros in stat computation and preserving zero positions in output.

    Args:
        x1 (torch.Tensor): Reference tensor of shape (B, N, M)
        x2 (torch.Tensor): Tensor to normalize, shape (B, N, M)
        clamp_min (float): Min value for clamping x2
        clamp_max (float): Max value for clamping x2
        eps (float): Epsilon to avoid divide-by-zero

    Returns:
        torch.Tensor: Normalized x2 with matching stats and zero positions preserved
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

    # Clamp x2
    x2_clamped = x2.clamp(min=clamp_min, max=clamp_max)

    # Create masks for non-zero elements
    mask1 = (x1 != 0).float()
    mask2 = (x2_clamped != 0).float()

    # Compute masked means and variances
    count1 = mask1.sum(dim=(1, 2), keepdim=True).clamp(min=1)
    count2 = mask2.sum(dim=(1, 2), keepdim=True).clamp(min=1)

    mean1 = (x1 * mask1).sum(dim=(1, 2), keepdim=True) / count1
    mean2 = (x2_clamped * mask2).sum(dim=(1, 2), keepdim=True) / count2

    var1 = (((x1 - mean1) * mask1) ** 2).sum(dim=(1, 2), keepdim=True) / count1
    var2 = (((x2_clamped - mean2) * mask2) ** 2).sum(dim=(1, 2), keepdim=True) / count2

    std1 = (var1 + eps).sqrt()
    std2 = (var2 + eps).sqrt()

    # Normalize and rescale
    x2_normalized = (x2_clamped - mean2) / std2
    x2_scaled = x2_normalized * std1 + mean1
    # x2_scaled = x2_clamped # - mean2) + mean1

    # Preserve original zero positions from x2
    x2_final = x2_scaled * mask2  # Zero out positions where x2 was originally 0

    return x2_final


class F5TTSForwardBackwardNoiseAug:
    def __init__(self, inference_kwargs, aug_weighted=0.1, forward_backward_step=1):
        self.inference_kwargs = inference_kwargs
        self.forward_backward_step = forward_backward_step
        self.aug_weighted = aug_weighted

    def __call__(self, nets=None, step=None, eval=False, **kwargs):
        # random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)
        random_noise = nets.noise_encoder.get_initial_noise(**kwargs).detach()
        # only works if nets.noise_encoder was created via
        # latent_aug_wm.f5_infer.forward_backward.load_peft_f5tts_linear

        # nets.f5tts.set_require_grad()
        aug_noise = sample_from_model(
            fix_noise=random_noise,
            model=nets.noise_encoder,
            inference_kwargs=self.inference_kwargs,
            forward_backward_step=self.forward_backward_step,
            **kwargs,
        )
        aug_noise = aug_noise.clamp(min=-7.0, max=7.0)

        # B, seq_length, n_mels
        # norm_aug_noise = random_noise.clone().detach() # * 0.6 + aug_noise * 0.4
        # norm_aug_noise[:, :, :60] = aug_noise[:, :, :60]

        norm_aug_noise = (
            random_noise.clone().detach() * (1 - self.aug_weighted)
            + aug_noise * self.aug_weighted
        )

        # match_stats_ignore_zeros_bnm_preserve_zeros(random_noise, aug_noise)
        # with torch.no_grad():
        #     orig_inv_noise = sample_from_model(
        #         fix_noise=random_noise,
        #         model=nets.f5tts.ema_model,
        #         inference_kwargs=self.inference_kwargs,
        #         forward_backward_step=self.forward_backward_step,
        #         **kwargs,
        #     )
        #     norm_orig_inv_noise = orig_inv_noise

        # match_stats_ignore_zeros_bnm_preserve_zeros(random_noise, orig_inv_noise)
        return (
            {
                "orig_noise": random_noise,
                "aug_noise": norm_aug_noise,
                # "orig_inv_noise": norm_orig_inv_noise,
            },
            ["noise_encoder"],
        )


class F5TTSForwardBackwardNoiseAugV2(F5TTSForwardBackwardNoiseAug):
    def __init__(self, *args, custom_t=[0.0, 0.0039, 0.0], **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_t = custom_t

    def __call__(self, nets=None, step=None, eval=False, **kwargs):
        # random_noise = nets.f5tts.ema_model.get_initial_noise(**kwargs)
        random_noise = nets.noise_encoder.get_initial_noise(**kwargs).detach()

        aug_noise, _ = forward_backward_sample_from_model(
            fix_noise=random_noise,
            model=nets.noise_encoder,
            inference_kwargs=self.inference_kwargs,
            # forward_backward_step=2,
            cache=False,
            custom_t=self.custom_t,
            # use_grad_checkpoint=True,
            **kwargs,
        )
        del _
        norm_aug_noise = aug_noise.clamp(min=-7.0, max=7.0)

        return (
            {
                "orig_noise": random_noise,
                "aug_noise": norm_aug_noise,
                # "orig_inv_noise": norm_orig_inv_noise,
            },
            ["noise_encoder"],
        )


def len_to_mask(lens, max_dur):

    batch_size = int(lens.shape[0])
    mask = torch.ones((batch_size, max_dur), device=lens.device)  # , dtype=lens.dtype)
    for i, length in enumerate(lens):
        mask[i, :length] = 0
    return mask


def noise_reg_L1_loss(nets=None, step=None, scale=None, eval=False, **kwargs):
    # orig_noise = kwargs["orig_inv_noise"]
    # nets.f5tts.set_require_grad()
    orig_noise = kwargs["orig_noise"]
    aug_noise = kwargs["aug_noise"]

    # _, seq_len, num_channel = orig_noise.shape
    # assert num_channel == 100

    # lens = kwargs["lens"]
    # mask = len_to_mask(lens, seq_len)
    # noise_reg_loss = torch.abs(
    #     (orig_noise - aug_noise) * mask.unsqueeze(-1)
    # ).sum() / torch.sum(mask)

    # noise_reg_loss = l1_loss(aug_noise, orig_noise)

    noise_reg_loss = l1_loss(aug_noise, orig_noise)
    if scale is not None:
        noise_reg_loss = noise_reg_loss * scale

    if not eval:
        return {"noise_reg_loss": noise_reg_loss}, []
    aug_diff_mel = (orig_noise - aug_noise).detach().cpu()
    return {
        "aug_diff_mel": aug_diff_mel,
        "aug_noise_mel": aug_noise,
        "noise_reg_loss": noise_reg_loss,
    }, []


def noise_reg_L2_loss(nets=None, step=None, scale=None, **kwargs):
    # orig_noise = kwargs["orig_inv_noise"]
    orig_noise = kwargs["orig_noise"]
    aug_noise = kwargs["aug_noise"]

    # _, seq_len, num_channel = orig_noise.shape
    # assert num_channel == 100

    # lens = kwargs["lens"]
    # mask = len_to_mask(lens, seq_len)
    # noise_reg_loss = torch.abs(
    #     (orig_noise - aug_noise) * mask.unsqueeze(-1)
    # ).sum() / torch.sum(mask)

    # noise_reg_loss = l1_loss(aug_noise, orig_noise)

    nets.f5tts.set_require_grad()
    noise_reg_loss = l2_loss(aug_noise, orig_noise)
    if scale is not None:
        noise_reg_loss = noise_reg_loss * scale
    return {"noise_reg_loss": noise_reg_loss}, []


def generate_multiple_mel_from_f5tts_with_aug_noise(
    nets=None, step=None, eval=False, **kwargs
):
    # ONLY works with model.encoder.UniqueNoiseEncoder

    # orig_inv_noise = kwargs["orig_inv_noise"]
    orig_noise = kwargs["orig_noise"]
    wm_noise = kwargs["aug_noise"]

    # with torch.no_grad():
    # nets.f5tts.set_require_grad()
    # with nets.f5tts.ema_model.disable_adapter():
    wm_out = nets.f5tts(
        fix_noise=wm_noise, use_grad_checkpoint=(not eval), eval=eval, **kwargs
    )
    with torch.no_grad():
        orig_out = nets.f5tts(
            fix_noise=orig_noise, use_grad_checkpoint=False, eval=eval, **kwargs
        )
        # orig_inv = nets.f5tts(
        #     fix_noise=orig_inv_noise, use_grad_checkpoint=False, eval=eval, **kwargs
        # )
    # out:
    # {
    #     "generated_rebatched_mel": generated_rebatched,
    #     "gr_wave": gr_wave,
    #     "g_wave": g_wave,
    # }
    wm_out = {("wm_" + k): v for k, v in wm_out.items()}
    orig_out = {("rand_" + k): v for k, v in orig_out.items()}
    # orig_inv_out = {("rand_inv_" + k): v for k, v in orig_inv.items()}

    # wm_out.update(out)
    wm_out.update(orig_out)
    # wm_out.update(orig_inv_out)

    return wm_out, []


if __name__ == "__main__":

    import addict
    import matplotlib.pyplot as plt

    from latent_aug_wm.f5_infer.infer import F5TTSBatchInferencer
    from latent_aug_wm.dataset.mel_dataset import get_combine_dataloader
    from latent_aug_wm.f5_infer.forward_backward import load_peft_f5tts_linear

    from f5_tts.infer.utils_infer import save_spectrogram
    import os

    def plot_two_tensor_distributions(
        tensor1,
        tensor2,
        bins=50,
        labels=("Tensor 1", "Tensor 2"),
        title="Tensor Value Distributions",
        save_path="tensor_histogram_comparison.png",
    ):
        """
        Plots and saves a histogram comparing two PyTorch tensors.

        Args:
            tensor1 (torch.Tensor): First tensor.
            tensor2 (torch.Tensor): Second tensor.
            bins (int): Number of histogram bins.
            labels (tuple): Labels for the legend corresponding to tensor1 and tensor2.
            title (str): Title of the plot.
            save_path (str): Path to save the PNG file.
        """
        if not isinstance(tensor1, torch.Tensor) or not isinstance(
            tensor2, torch.Tensor
        ):
            raise TypeError("Both inputs must be torch.Tensors.")

        # Flatten, move to CPU, filter out zeros
        data1 = tensor1.detach().cpu().numpy().flatten()
        data2 = tensor2.detach().cpu().numpy().flatten()

        data1 = data1[data1 != 0]
        data2 = data2[data2 != 0]

        # Plot
        plt.figure(figsize=(8, 6))
        plt.hist(
            data1,
            bins=bins,
            alpha=0.6,
            label=labels[0],
            color="blue",
            edgecolor="black",
        )
        plt.hist(
            data2,
            bins=bins,
            alpha=0.6,
            label=labels[1],
            color="orange",
            edgecolor="black",
        )

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save plot
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        print(f"Comparison histogram saved to {save_path}")

    data_dir = "/home/tst000/projects/tst000/datasets/"
    tmp_dir = "/home/tst000/projects/tst000/tmp/"

    inference_kwargs = {
        "target_rms": 0.1,
        "cross_fade_duration": 0.15,
        "sway_sampling_coef": -1,
        "cfg_strength": 2,
        "nfe_step": 32,
    }

    loss_fn = F5TTSForwardBackwardNoiseAug(inference_kwargs)
    gen_fn = generate_multiple_mel_from_f5tts_with_aug_noise

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

    from f5_tts.model.modules import MelSpec

    mel_spec = MelSpec(**mel_spec_kwargs)

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
    ref_audio = f5tts.vocoder.decode(batch["cond"].permute(0, 2, 1))

    print(ref_audio.shape)
    print(length * (256 + 1) + 1024)
    masked_audio = mask_audio(batch["lens"], ref_audio.shape[1], ref_audio)
    print(masked_audio.shape)
    import torchaudio

    for i, (text, ma) in enumerate(zip(batch["text"], masked_audio)):
        fname = f"masked_audio_{i}.wav"
        print("".join(text), fname)
        torchaudio.save(
            fname, ma.detach().float().cpu().unsqueeze(0), sample_rate=24000
        )

    exit()
    lens = batch["lens"][0]

    test_seq_length = 2000
    test_mask = len_to_mask(batch["lens"], test_seq_length)

    # f5tts = F5TTSBatchInferencer(device="cuda")
    noise_encoder = load_peft_f5tts_linear(f5tts)

    nets = addict.Dict({"f5tts": f5tts, "noise_encoder": noise_encoder})
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = loss_fn(nets, step=0, **batch)
            out[0].update(gen_fn(nets, step=0, **batch, **out[0])[0])

    _, seq_len, num_channel = out[0]["orig_noise"].shape
    mask = len_to_mask(batch["lens"], seq_len)
    assert num_channel == 100

    output = out[0]

    # save_spectrogram(out["orig_noise"].permute(0, 2, 1).detach()[0][:, lens:], f"orig_noise.png")
    # save_spectrogram(out["aug_noise"].permute(0, 2, 1).detach()[0][:, lens:], f"aug_noise.png")

    save_spectrogram(
        (output["orig_noise"] * mask.unsqueeze(-1)).cpu().permute(0, 2, 1).detach()[0],
        f"orig_noise.png",
    )
    save_spectrogram(
        (output["aug_noise"] * mask.unsqueeze(-1)).cpu().permute(0, 2, 1).detach()[0],
        f"aug_noise.png",
    )
    save_spectrogram(
        output["wm_generated_rebatched_mel"].cpu().permute(0, 2, 1).detach()[0],
        f"wm_generated_rebatched_mel.png",
    )
    save_spectrogram(
        output["rand_generated_rebatched_mel"].cpu().permute(0, 2, 1).detach()[0],
        f"rand_generated_rebatched_mel.png",
    )
    save_spectrogram(
        output["rand_inv_generated_rebatched_mel"].cpu().permute(0, 2, 1).detach()[0],
        f"rand_inv_generated_rebatched_mel.png",
    )

    save_spectrogram(
        ((output["orig_noise"] - output["orig_inv_noise"]).abs() * mask.unsqueeze(-1))
        .cpu()
        .permute(0, 2, 1)
        .detach()[0],
        f"orig_inv_diff_noise.png",
    )
    print(noise_reg_L1_loss(**output, **batch))

    # save_spectrogram((output["orig_noise"]*mask.unsqueeze(1)).permute(0, 2, 1).detach()[0], f"orig_noise.png")
    # save_spectrogram((output["aug_noise"]*mask.unsqueeze(1)).permute(0, 2, 1).detach()[0], f"aug_noise.png")
    plot_two_tensor_distributions(
        output["rand_inv_generated_rebatched_mel"].cpu().flatten(),
        output["rand_generated_rebatched_mel"].cpu().flatten(),
        labels=("rand_inv", "rand"),
        title="Mel Value Distributions",
        save_path="./tensor_histogram_comparison.png",
    )

    plot_two_tensor_distributions(
        output["orig_noise"].cpu().flatten(),
        output["orig_inv_noise"].cpu().flatten(),
        labels=("orig_noise", "orig_inv_noise"),
        title="Noise Value Distributions",
        save_path="./tensor_noise_histogram_comparison.png",
    )
