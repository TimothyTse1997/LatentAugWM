from latent_aug_wm.f5_infer.infer import load_cfm, F5TTSBatchInferencer


class F5TTSBatchFBInferencer(F5TTSBatchInferencer):
    # add forward backward functionarity to the inferencer

    def forward_backward(
        self, cond, text, duration, lens, fix_noise=None, forward_backward_step=None
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
