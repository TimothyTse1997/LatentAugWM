import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig
from latent_aug_wm.f5_infer.infer import (
    load_cfm,
    F5TTSBatchInferencer,
    rebatching_from_varying_length,
)


class F5TTSBatchFBInferencer(F5TTSBatchInferencer):
    # add forward backward functionarity to the inferencer

    def forward_backward(
        self, cond, text, duration, lens, fix_noise=None, forward_backward_step=1
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
            forward_backward_step=forward_backward_step,
        )

        return generated.permute(0, 2, 1)


def run_cfm_fb_noise(
    cfm,
    cond,
    text,
    duration,
    lens,
    fix_noise=None,
    forward_backward_step=None,
    inference_kwargs=None,
):
    generated, _ = cfm._sample(
        cond=cond,
        text=text,
        duration=duration,
        lens=lens,
        steps=inference_kwargs["nfe_step"],
        cfg_strength=inference_kwargs["cfg_strength"],
        sway_sampling_coef=inference_kwargs["sway_sampling_coef"],
        fix_noise=fix_noise,
        forward_backward_step=forward_backward_step,
    )

    return generated.permute(0, 2, 1)


def _get_peft_f5tts_names(fttts_inferencer: F5TTSBatchInferencer):
    module = fttts_inferencer.ema_model

    for n, m in module.named_modules():
        print(n, type(m))


def load_peft_f5tts_linear(fttts_inferencer: F5TTSBatchInferencer):
    module = fttts_inferencer.ema_model
    linear_layer_names = [
        n
        for n, m in module.named_modules()
        if isinstance(m, nn.modules.linear.Linear)
        and ("ff.ff" in n)
        and ("transformer_blocks" in n)
    ]
    config = LoraConfig(
        target_modules=linear_layer_names,
        modules_to_save=[],
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def load_peft_f5tts_q_k(fttts_inferencer: F5TTSBatchInferencer):
    module = fttts_inferencer.ema_model
    linear_layer_names = [
        n
        for n, m in module.named_modules()
        if isinstance(m, nn.modules.linear.Linear)
        and ("attn.to_q" in n or "attn.to_k" in n)
        and ("transformer_blocks" in n)
    ]
    config = LoraConfig(
        target_modules=linear_layer_names,
        modules_to_save=[],
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def load_peft_f5tts_full_transformer_block(fttts_inferencer: F5TTSBatchInferencer):
    module = fttts_inferencer.ema_model
    linear_layer_names = [
        n
        for n, m in module.named_modules()
        if isinstance(m, nn.modules.linear.Linear)
        and ("attn.to_" in n or "ff.ff" in n)
        and ("transformer_blocks" in n)
    ]
    config = LoraConfig(
        target_modules=linear_layer_names,
        modules_to_save=[],
    )
    peft_model = get_peft_model(module, config)
    print("created trainable parameters")
    peft_model.print_trainable_parameters()
    return peft_model


def forward_backward_sample_from_model(
    cond,
    text,
    duration,
    lens,
    fix_noise=None,
    use_grad_checkpoint=False,
    cache=True,
    model=None,
    inference_kwargs={},
    custom_t=[0.0, 0.0039, 0.0],
    **kwargs,
):
    custom_t = torch.tensor(custom_t).to(model.device)
    model.transformer.clear_cache()
    assert model.transformer.text_cond is None
    # generated, _ = model._sample(
    generated, _ = _sample(
        model,
        cond=cond,
        text=text,
        duration=duration,
        lens=lens,
        steps=inference_kwargs["nfe_step"],
        cfg_strength=inference_kwargs["cfg_strength"],
        sway_sampling_coef=inference_kwargs["sway_sampling_coef"],
        fix_noise=fix_noise,
        use_grad_checkpoint=use_grad_checkpoint,
        cache=cache,
        custom_t=custom_t,
    )
    model.transformer.clear_cache()
    assert model.transformer.text_cond is None
    for i, length in enumerate(lens):
        generated[i, :length, :] = fix_noise[i, :length, :]
    return generated, _


def sample_from_model(
    cond,
    text,
    duration,
    lens,
    fix_noise=None,
    use_grad_checkpoint=False,
    cache=True,
    model=None,
    inference_kwargs={},
    forward_backward_step=None,
    **kwargs,
):
    model.transformer.clear_cache()
    assert model.transformer.text_cond is None
    # generated, _ = model._sample(
    generated, _ = _sample(
        model,
        cond=cond,
        text=text,
        duration=duration,
        lens=lens,
        steps=inference_kwargs["nfe_step"],
        cfg_strength=inference_kwargs["cfg_strength"],
        sway_sampling_coef=inference_kwargs["sway_sampling_coef"],
        fix_noise=fix_noise,
        use_grad_checkpoint=use_grad_checkpoint,
        cache=cache,
        forward_backward_step=forward_backward_step,
    )
    model.transformer.clear_cache()
    assert model.transformer.text_cond is None

    return generated


def odeint(fn, y0, t, **kwargs):
    trajectory = []
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        dy = fn(t=t0, x=y0)
        y1 = y0 + dy * dt
        trajectory.append(y1)
        y0 = y1
    return trajectory


def odeint_grad_checkpoint(fn, y0, t, **kwarg):
    trajectory = []
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        # dy = fn(t=t0, x=y0, dt=dt, t1=t1)
        dy = checkpoint.checkpoint(fn, t0, y0)
        y1 = y0 + dy * dt
        trajectory.append(y1)
        y0 = y1
    return trajectory


def _sample(
    model,
    cond,
    text,
    duration,
    *,
    lens,
    steps=32,
    cfg_strength=1.0,
    sway_sampling_coef=None,
    seed=None,
    max_duration=4096,
    vocoder=None,
    use_epss=True,
    no_ref_audio=False,
    duplicate_test=False,
    t_inter=0.1,
    edit_mask=None,
    fix_noise=None,
    use_grad_checkpoint=False,
    cache=True,
    forward_backward_step=None,
    custom_t=None,
):
    # raw wave
    from f5_tts.model.utils import exists

    from f5_tts.model.utils import (
        lens_to_mask,
        list_str_to_tensor,
        list_str_to_idx,
        get_epss_timesteps,
    )

    if cond.ndim == 2:
        cond = model.mel_spec(cond)
        cond = cond.permute(0, 2, 1)
        assert cond.shape[-1] == model.num_channels

    cond = cond.to(next(model.parameters()).dtype)

    batch, cond_seq_len, device = *cond.shape[:2], cond.device
    if not exists(lens):
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

    # text

    if isinstance(text, list):
        if exists(model.vocab_char_map):
            text = list_str_to_idx(text, model.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    # duration

    cond_mask = lens_to_mask(lens)
    if edit_mask is not None:
        cond_mask = cond_mask & edit_mask

    if isinstance(duration, int):
        duration = torch.full((batch,), duration, device=device, dtype=torch.long)

    duration = torch.maximum(
        torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
    )  # duration at least text/audio prompt length plus one token, so something is generated
    duration = duration.clamp(max=max_duration)
    max_duration = duration.amax()

    # duplicate test corner for inner time step oberservation
    if duplicate_test:
        test_cond = F.pad(
            cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0
        )

    cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
    if no_ref_audio:
        cond = torch.zeros_like(cond)

    cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
    cond_mask = cond_mask.unsqueeze(-1)
    step_cond = torch.where(
        cond_mask, cond, torch.zeros_like(cond)
    )  # allow direct control (cut cond audio) with lens passed in

    if batch > 1:
        mask = lens_to_mask(duration)
    else:  # save memory and speed up, as single inference need no mask currently
        mask = None

    # neural ode

    def fn(t, x):
        if cfg_strength < 1e-5:
            # with NaNErrorMode(
            #     enabled=True, raise_error=False, print_stats=True, print_nan_index=False
            # ):
            pred = model.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
            )
            return pred

        pred_cfg = model.transformer(
            x=x,
            cond=step_cond,
            text=text,
            time=t,
            mask=mask,
            cfg_infer=True,
            cache=cache,
        )
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        return pred + (pred - null_pred) * cfg_strength

    y0 = fix_noise.type(step_cond.dtype)

    t_start = 0

    if custom_t is not None:
        t = custom_t.type(step_cond.dtype)

    else:
        if (
            t_start == 0 and use_epss
        ):  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=model.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(
                t_start, 1, steps + 1, device=model.device, dtype=step_cond.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        if forward_backward_step is not None:
            t = model.get_forward_backward_t(t, forward_backward_step)
    # print(t)
    trajectory = []
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        dy = fn(t=t0, x=y0)
        y1 = y0 + dy * dt
        trajectory.append(y1)
        y0 = y1

    # trajectory = odeint(fn, y0, t, **model.odeint_kwargs)

    model.transformer.clear_cache()

    sampled = trajectory[-1]
    out = sampled
    out = torch.where(cond_mask, cond, out)

    return out, trajectory


if __name__ == "__main__":
    from latent_aug_wm.model.decoder import BaseAudioSealClassifier
    from latent_aug_wm.loss.base import AugApplier
    from latent_aug_wm.data_augmentation.base import BaseBatchAugmentation
    from latent_aug_wm.trainer.optimizer import build_optimizer

    aug_obj = BaseBatchAugmentation(
        sampling_rate=24000,
        add_no_aug=False,
        transform_configs={
            "AddColoredNoise": {
                "mode": "per_example",
                "p": 1.0,
                "min_snr_in_db": 10.0,
                "max_snr_in_db": 10.0,
            }
        },
    )

    sampler = F5TTSBatchInferencer(device="cuda")
    detector = BaseAudioSealClassifier(input_dim=1).cuda()
    # _get_peft_f5tts_names(sampler)

    from latent_aug_wm.dataset.mel_dataset import get_combine_dataloader

    ref_wav_file = "/home/tst000/projects/tst000/datasets/selected_ref_files.txt"
    gen_txt_fname = "/home/tst000/projects/tst000/datasets/selected_gen_text.txt"

    mel_spec_kwargs = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    tmp_dir = "/home/tst000/projects/tst000/tmp/libriTTS"

    data_iter = get_combine_dataloader(
        ref_wav_file=ref_wav_file,
        gen_txt_fname=gen_txt_fname,
        mel_spec_kwargs=mel_spec_kwargs,
        tmp_dir="/home/tst000/projects/tst000/tmp/libriTTS",
        shuffle=False,
        unsorted_batch_size=1,
        batch_size=1,
        allowed_padding=50,
    )

    def save_text(text, fname):
        with open(fname, "w") as f:
            f.write(text)

    model = load_cfm().cuda().to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = build_optimizer(
        model_dict={"noise_encoder": model},
        scheduler_params_dict={
            "noise_encoder": {
                "start_factor": 0.001,
                "end_factor": 1.0,
                "total_iters": 50,
            }
        },
        model_lr={"noise_encoder": 1.0e-2},
    )

    def get_grad_norm():
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        try:
            norm = torch.cat(grads).norm()
        except:
            norm = 0
        return norm

    def get_grad_trans_norm():
        grads = [
            param.grad.detach().flatten()
            for param in model.transformer.parameters()
            if param.grad is not None
        ]
        try:
            norm = torch.cat(grads).norm()
        except:
            norm = 0
        return norm

    # random_noise = torch.randn((10, 1000, 100)).cuda()
    inference_kwargs = {
        "target_rms": 0.1,
        "cross_fade_duration": 0.15,
        "sway_sampling_coef": -1,
        "cfg_strength": 2,
        "nfe_step": 32,
    }
    # model.transformer.checkpoint_activations = False

    from f5_tts.infer.utils_infer import save_spectrogram

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
    from tqdm import tqdm

    for i in tqdm(range(100)):
        batch = next(data_iter)

        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = [v.cuda() if torch.is_tensor(v) else v for v in batch]
        else:
            batch = {
                k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()
            }
        optimizer.zero_grad()
        random_noise = model.get_initial_noise(**batch)
        batch_size, _, _ = random_noise.shape

        fake_logits = torch.ones(batch_size, dtype=torch.long, device="cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                # aug_noise, trajectory = forward_backward_sample_from_model(
                # aug_noise, trajectory = sample_from_model(
                aug_noise = sample_from_model(
                    fix_noise=random_noise,
                    model=model,
                    inference_kwargs=inference_kwargs,
                    forward_backward_step=32,
                    cache=False,
                    # use_grad_checkpoint=True,
                    **batch,
                )

                out = aug_noise  # trajectory[-1]#.permute(0, 2, 1)
                print(out.shape, random_noise.shape)

                loss = torch.abs(out - random_noise).mean()
                print(loss)
                model.transformer.clear_cache()
                wm_out = sampler(
                    fix_noise=out, use_grad_checkpoint=True, eval=True, **batch
                )
                inprint_latent = out - random_noise

                with torch.no_grad():
                    orig_out = sampler(
                        fix_noise=random_noise,
                        use_grad_checkpoint=False,
                        eval=True,
                        **batch,
                    )
                # wm_out = {("wm_" + k): v for k, v in wm_out.items()}
                wm_out = {}
                orig_out = {("rand_" + k): v for k, v in orig_out.items()}

                # fake_input = wm_out["wm_gr_wave"]
                # fake_input = aug_obj(fake_input.unsqueeze(1)).squeeze(1)
                # fake_detector_logits, fake_loss = detector.calculate_loss(
                #     fake_input.unsqueeze(1), fake_logits
                # )
                # total_loss = loss + fake_loss
                # total_loss.backward()
                # print(total_loss)
                # print("norm: ", get_grad_norm())
                # print("norm: ", get_grad_trans_norm())

                # optimizer.step("noise_encoder")
                # optimizer.scheduler()

                model.transformer.clear_cache()
                assert model.transformer.text_cond is None

            save_spectrogram(
                inprint_latent.float().cpu().permute(0, 2, 1).detach()[0],
                f"inprint_latent_{i}.png",
            )
            # torch.save(random_noise.float().cpu(), f"experiments/rand_noise_{i}_tensor.pt")
            # torch.save(aug_noise.float().cpu(), f"aug_noise_{i}_tensor.pt")
            # save_spectrogram(
            #     random_noise.float().cpu().permute(0, 2, 1).detach()[0],
            #     f"random_noise_{i}.png",
            # )

            # save_spectrogram(
            #     wm_out["wm_generated_rebatched_mel"].float().cpu().permute(0, 2, 1).detach()[0],
            #     f"wm_generated_rebatched_mel_{i}.png",
            # )
            # torch.save(orig_out["rand_generated_mel"].float().cpu(), f"experiments/rand_generated_rebatched_mel_tensor_{i}.pt")
            save_spectrogram(
                orig_out["rand_generated_rebatched_mel"]
                .float()
                .cpu()
                .permute(0, 2, 1)
                .detach()[0],
                f"rand_generated_rebatched_mel_{i}.png",
            )
            import torchaudio

            audio = orig_out["rand_g_wave"][0].detach().float().cpu().unsqueeze(0)
            print(audio.shape)
            break
            # torchaudio.save(f"experiments/rand_generated_audio_{i}.wav", audio, sample_rate=24000)
