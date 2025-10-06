import torch
import torch.nn as nn

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
    generated, _ = model._sample(
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
