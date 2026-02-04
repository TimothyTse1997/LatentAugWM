import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.stats import norm, percentileofscore
from tqdm import tqdm

import matplotlib.pyplot as plt
from f5_tts.infer.utils_infer import save_spectrogram

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# def cos(A, B):
#     print(A.shape, B.shape, "cos")
#     if len(A.shape) == 3:
#         a_aug = A[:, :, 10:60]
#     else:
#         a_aug = A[:, 10:60]
#     if len(B.shape) == 3:
#         b_aug = B[:, :, 10:60]
#     else:
#         b_aug = B[:, 10:60]

#     return t_cos(a_aug, b_aug)

mel_spec_kwargs = {
    "target_sample_rate": 24000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
    "mel_spec_type": "vocos",
}


def get_torchaudio_mel_spec(audio, mel_spec_kwargs):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_spec_kwargs["target_sample_rate"],
        n_fft=mel_spec_kwargs["n_fft"],
        win_length=mel_spec_kwargs["win_length"],
        hop_length=mel_spec_kwargs["hop_length"],
        n_mels=mel_spec_kwargs["n_mel_channels"],
        power=1,
        center=True,
    )
    mel = transform(audio)
    mel = mel.clamp(min=1e-5).log()
    print(mel.shape, "transformed mel_spec")
    # mel = mel[:, 20:50, :]
    return mel


from f5_tts.model.modules import MelSpec
from latent_aug_wm.data_augmentation.base import BaseBatchAugmentation


def log_spectrogram_distribution(spectrogram, fname):
    # Flatten the spectrogram values
    values = spectrogram.flatten().cpu().numpy()

    mu, std = 0, 1

    # Create the histogram
    fig = plt.figure(figsize=(8, 4))
    count, bins, ignored = plt.hist(
        values, bins=100, density=True, alpha=0.6, color="g", label="Spectrogram Values"
    )

    # Plot the normal distribution curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2, label=f"Gaussian (μ={mu:.2f}, σ={std:.2f})")

    plt.title(f"Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(fname, dpi=300)
    plt.close()


mel_spec = MelSpec(**mel_spec_kwargs)
noise_transform_configs = {
    # "HighPassFilter": {
    #     "mode": "per_example",
    #     "p": 1.0, "min_cutoff_freq": 400,
    #     "max_cutoff_freq": 600
    # },
    # "LowPassFilter": {
    #     "mode": "per_example",
    #     "p": 1.0, "min_cutoff_freq": 8000,
    #     "max_cutoff_freq": 8000
    # },
    "AddColoredNoise": {
        "mode": "per_example",
        "p": 1.0,
        "min_snr_in_db": 10.0,
        "max_snr_in_db": 10.0,
    }
}
highpass_transform_configs = {
    "HighPassFilter": {
        "mode": "per_example",
        "p": 1.0,
        "min_cutoff_freq": 400,
        "max_cutoff_freq": 600,
    }
}
lowpass_transform_configs = {
    "LowPassFilter": {
        "mode": "per_example",
        "p": 1.0,
        "min_cutoff_freq": 8000,
        "max_cutoff_freq": 8000,
    },
}

noise_aug_fn = BaseBatchAugmentation(
    sampling_rate=24000, transform_configs=noise_transform_configs, add_no_aug=False
)
hp_aug_fn = BaseBatchAugmentation(
    sampling_rate=24000, transform_configs=highpass_transform_configs, add_no_aug=False
)
lp_aug_fn = BaseBatchAugmentation(
    sampling_rate=24000, transform_configs=lowpass_transform_configs, add_no_aug=False
)


aug_fn_dict = {"noise": noise_aug_fn, "hp": hp_aug_fn, "lp": lp_aug_fn}


def main():
    source_noise_file = (
        "/home/tst000/projects/tst000/LatentAugWM/experiments/rand_noise_0_tensor.pt"
    )
    gen_mel_file = "/home/tst000/projects/tst000/LatentAugWM/experiments/rand_generated_rebatched_mel_tensor_0.pt"
    gen_wav_file = "/home/tst000/projects/tst000/LatentAugWM/experiments/rand_generated_audio_0.wav"
    gen_mp3_file = "/home/tst000/projects/tst000/LatentAugWM/experiments/rand_generated_audio_0.mp3"
    gen_bigvgan_file = "/home/tst000/projects/tst000/LatentAugWM/experiments/rand_generated_audio_0_hifigan.wav"

    source_noise = torch.load(source_noise_file)
    target_mel = torch.load(gen_mel_file)
    print(source_noise.shape, "source_noise")
    print(target_mel.shape, "target_mel")
    # source_noise = source_noise[:, :, 20:50]
    # target_mel = target_mel[:, :, 20:50]

    target_audio, sr = torchaudio.load(gen_wav_file)
    target_mp3_audio, sr = torchaudio.load(gen_mp3_file)

    rms = torch.sqrt(torch.mean(torch.square(target_mp3_audio)))
    if rms < 0.1:
        target_mp3_audio = target_mp3_audio * 0.1 / rms
    target_bigvgan_audio, sr = torchaudio.load(gen_bigvgan_file)
    rms = torch.sqrt(torch.mean(torch.square(target_bigvgan_audio)))
    if rms < 0.1:
        target_bigvgan_audio = target_bigvgan_audio * 0.1 / rms
    # target_mp3_audio[target_mp3_audio < -5] = 0

    log_spectrogram_distribution(source_noise, "source_noise_dist.png")
    aug_target_audio = noise_aug_fn(target_audio.unsqueeze(0)).squeeze(0)
    print(aug_target_audio.shape)
    print(aug_target_audio.shape)
    # mel_after_vocoder = mel_spec(target_audio)
    # mel_aug = mel_spec(aug_target_audio)

    mel_after_vocoder = get_torchaudio_mel_spec(target_audio, mel_spec_kwargs)
    mel_after_mp3 = get_torchaudio_mel_spec(target_mp3_audio, mel_spec_kwargs)
    mel_after_bigvgan = get_torchaudio_mel_spec(target_bigvgan_audio, mel_spec_kwargs)
    mel_aug = get_torchaudio_mel_spec(aug_target_audio, mel_spec_kwargs)

    mel_after_mp3[mel_after_mp3 < -5] = 0
    mel_after_bigvgan[mel_after_bigvgan < -5] = 0

    save_spectrogram(
        mel_after_mp3.float().cpu().detach()[0],
        "mel_after_mp3.png",
    )
    log_spectrogram_distribution(mel_after_mp3, "mel_after_mp3_dist.png")
    log_spectrogram_distribution(mel_after_vocoder, "mel_after_vocoder_dist.png")
    save_spectrogram(
        mel_after_vocoder.float().cpu().detach()[0],
        "mel_after_vocoder.png",
    )

    print(mel_after_vocoder.shape, sr)
    print(mel_after_vocoder.shape, sr)
    print(mel_aug.shape, sr)
    print(mel_aug.shape, sr)
    mel_after_vocoder = mel_after_vocoder.permute(0, 2, 1)
    mel_aug = mel_aug.permute(0, 2, 1)

    num_trial = 100
    log_spectrogram_distribution(target_mel, "target_mel_dist.png")
    print(target_mel.shape, source_noise.shape)
    print(source_noise.flatten().shape, target_mel.flatten().shape)
    source_cos = cos(
        source_noise.flatten().unsqueeze(0), target_mel.flatten().unsqueeze(0)
    )
    source_cos_vocod = cos(
        source_noise[0].flatten().unsqueeze(0),
        mel_after_vocoder[0].flatten().unsqueeze(0),
    )
    source_cos_mp3 = cos(
        source_noise[0].flatten().unsqueeze(0), mel_after_mp3[0].flatten().unsqueeze(0)
    )
    source_cos_aug = cos(
        source_noise[0].flatten().unsqueeze(0), mel_aug[0].flatten().unsqueeze(0)
    )
    source_cos_bigvgan = cos(
        source_noise[0].flatten().unsqueeze(0),
        mel_after_bigvgan[0].flatten().unsqueeze(0),
    )

    print("source", source_cos)
    print("source voc", source_cos_vocod)
    print("source aug", source_cos_aug)
    print("source mp3", source_cos_mp3)
    print("source bigvgan", source_cos_bigvgan)

    rand_cos_vals = []
    rand_cos_vals_vocod = []
    rand_cos_vals_mp3 = []
    rand_cos_vals_aug = []
    rand_cos_vals_bigvgan = []

    for i in range(num_trial):
        rand_noise = torch.randn_like(source_noise)
        rand_cos = cos(
            rand_noise.flatten().unsqueeze(0), target_mel.flatten().unsqueeze(0)
        )
        rand_cos_vocod = cos(
            rand_noise[0].flatten().unsqueeze(0),
            mel_after_vocoder[0].flatten().unsqueeze(0),
        )
        rand_cos_aug = cos(
            rand_noise[0].flatten().unsqueeze(0), mel_aug[0].flatten().unsqueeze(0)
        )
        rand_cos_mp3 = cos(
            rand_noise[0].flatten().unsqueeze(0),
            mel_after_mp3[0].flatten().unsqueeze(0),
        )
        rand_cos_bigvgan = cos(
            rand_noise[0].flatten().unsqueeze(0),
            mel_after_bigvgan[0].flatten().unsqueeze(0),
        )

        # print(rand_cos)
        rand_cos_vals.append(rand_cos.item())
        rand_cos_vals_vocod.append(rand_cos_vocod.item())
        rand_cos_vals_mp3.append(rand_cos_mp3.item())
        rand_cos_vals_aug.append(rand_cos_aug.item())
        rand_cos_vals_bigvgan.append(rand_cos_bigvgan.item())

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    mu, std = norm.fit(rand_cos_vals)

    # Generate x values for the Gaussian curve
    xmin, xmax = min(rand_cos_vals), max(rand_cos_vals)
    x = np.linspace(xmin, xmax, 200)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    print(mu, std, xmin, xmax)
    p = norm.pdf(x, mu, std)

    # Plot the fitted Gaussian
    plt.plot(
        x,
        p,
        "b-",
        linewidth=2,
        label=f"Gaussian Fit ($\mu$={mu:.3f}, $\sigma$={std:.3f})",
    )

    plt.hist(rand_cos_vals, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(
        source_cos.item(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Source Cosine = {source_cos.item():.4f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Cosine Similarities vs. Source Cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist.png")

    plt.figure(figsize=(8, 5))
    mu, std = norm.fit(rand_cos_vals_vocod)

    # Generate x values for the Gaussian curve
    xmin, xmax = min(rand_cos_vals_vocod), max(rand_cos_vals_vocod)
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)

    # Plot the fitted Gaussian
    plt.plot(
        x,
        p,
        "r-",
        linewidth=2,
        label=f"Gaussian Fit ($\mu$={mu:.3f}, $\sigma$={std:.3f})",
    )

    plt.hist(
        rand_cos_vals_vocod, bins=20, color="skyblue", edgecolor="black", alpha=0.7
    )
    plt.axvline(
        source_cos_vocod.item(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Source Cosine = {source_cos_vocod.item():.4f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Random Cosine Similarities vs. Source Cosine (after audio went through vocoder)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist_voc.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_voc.png")

    plt.figure(figsize=(8, 5))
    mu, std = norm.fit(rand_cos_vals_aug)

    # Generate x values for the Gaussian curve
    # xmin, xmax = plt.xlim()
    xmin, xmax = min(rand_cos_vals_aug), max(rand_cos_vals_aug)
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)

    # Plot the fitted Gaussian
    plt.plot(
        x,
        p,
        "r-",
        linewidth=2,
        label=f"Gaussian Fit ($\mu$={mu:.3f}, $\sigma$={std:.3f})",
    )

    plt.hist(rand_cos_vals_aug, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(
        source_cos_aug.item(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Source Cosine = {source_cos_aug.item():.4f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Random Cosine Similarities vs. Source Cosine (after noised)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist_noised.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_noised.png")

    plt.hist(rand_cos_vals_mp3, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(
        source_cos_mp3.item(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Source Cosine = {source_cos_mp3.item():.4f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Random Cosine Similarities vs. Source Cosine (after mp3)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist_mp3.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_mp3.png")

    plt.hist(
        rand_cos_vals_bigvgan, bins=20, color="skyblue", edgecolor="black", alpha=0.7
    )
    plt.axvline(
        source_cos_bigvgan.item(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Source Cosine = {source_cos_bigvgan.item():.4f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Random Cosine Similarities vs. Source Cosine (after bigvgan)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist_bigvgan.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_bigvgan.png")


from pathlib import Path


def main2():
    all_source_cos = []
    all_source_vocod = []
    all_source_mp3 = []
    all_source_bigvgan = []
    all_source_aug = {k: [] for k in aug_fn_dict.keys()}
    rand_cos_vals = []
    rand_cos_vals_vocod = []
    rand_cos_vals_mp3 = []
    rand_cos_vals_aug = {k: [] for k in aug_fn_dict.keys()}

    all_p_value = []
    all_p_value_vocod = []
    all_p_value_mp3 = []
    all_p_value_bigvgan = []

    all_p_value_aug = {k: [] for k in aug_fn_dict.keys()}

    all_paths = list(
        Path("/home/tst000/projects/tst000/LatentAugWM/experiments").glob(
            "rand_noise_*_tensor.pt"
        )
    )
    assert len(all_paths) > 0
    for data_path_index, p in enumerate(tqdm(all_paths)):
        print(p)
        index = p.name.split("_")[2]
        source_noise_file = p

        gen_mel_file = p.parent / f"rand_generated_rebatched_mel_tensor_{index}.pt"
        gen_wav_file = p.parent / f"rand_generated_audio_{index}.wav"
        gen_mp3_file = p.parent / f"rand_generated_audio_{index}.mp3"
        gen_bigvgan_file = p.parent / f"rand_generated_audio_{index}_hifigan.wav"

        source_noise = torch.load(source_noise_file)
        target_audio, sr = torchaudio.load(gen_wav_file)
        target_mp3_audio, sr = torchaudio.load(gen_mp3_file)
        target_bigvgan_audio, sr = torchaudio.load(gen_bigvgan_file)

        log_spectrogram_distribution(source_noise, f"source_noise_dist_index.png")
        aug_target_audios = {
            k: aug_fn(target_audio.unsqueeze(0)).squeeze(0)
            for k, aug_fn in aug_fn_dict.items()
        }
        # mel_after_vocoder = mel_spec(target_audio)
        # mel_aug = mel_spec(aug_target_audio)

        mel_after_vocoder = get_torchaudio_mel_spec(target_audio, mel_spec_kwargs)
        mel_after_mp3 = get_torchaudio_mel_spec(target_mp3_audio, mel_spec_kwargs)
        mel_after_bigvgan = get_torchaudio_mel_spec(
            target_bigvgan_audio, mel_spec_kwargs
        )
        mel_augs = {
            k: get_torchaudio_mel_spec(aug_target_audio, mel_spec_kwargs).permute(
                0, 2, 1
            )
            for k, aug_target_audio in aug_target_audios.items()
        }

        print(mel_after_vocoder.shape, sr)
        print(mel_after_vocoder.shape, sr)
        mel_after_vocoder = mel_after_vocoder.permute(0, 2, 1)

        num_trial = 100
        target_mel = torch.load(gen_mel_file)
        log_spectrogram_distribution(target_mel, "target_mel_dist.png")
        print(target_mel.shape, source_noise.shape)
        print(source_noise.flatten().shape, target_mel.flatten().shape)
        source_cos = cos(
            source_noise.flatten().unsqueeze(0), target_mel.flatten().unsqueeze(0)
        )
        source_cos_vocod = cos(
            source_noise[0].flatten().unsqueeze(0),
            mel_after_vocoder[0].flatten().unsqueeze(0),
        )
        source_cos_mp3 = cos(
            source_noise[0].flatten().unsqueeze(0),
            mel_after_mp3[0].flatten().unsqueeze(0),
        )
        source_cos_bigvgan = cos(
            source_noise[0].flatten().unsqueeze(0),
            mel_after_bigvgan[0].flatten().unsqueeze(0),
        )

        source_cos_aug = {
            k: cos(
                source_noise[0].flatten().unsqueeze(0),
                mel_aug[0].flatten().unsqueeze(0),
            )
            for k, mel_aug in mel_augs.items()
        }
        all_source_cos.append(source_cos.item())
        all_source_vocod.append(source_cos_vocod.item())
        all_source_mp3.append(source_cos_mp3.item())
        all_source_bigvgan.append(source_cos_bigvgan.item())

        for k, v in source_cos_aug.items():
            all_source_aug[k].append(v.item())

        print("source", source_cos)
        print("source voc", source_cos_vocod)
        print("source aug", source_cos_aug)
        print("source bigvgan", source_cos_bigvgan)

        current_rand_cos = []
        current_rand_cos_vocod = []
        current_rand_cos_mp3 = []
        current_rand_cos_bigvgan = []
        current_rand_cos_aug = {k: [] for k in mel_augs.keys()}

        for i in range(num_trial):
            rand_noise = torch.randn_like(source_noise)
            rand_cos = cos(
                rand_noise.flatten().unsqueeze(0), target_mel.flatten().unsqueeze(0)
            )
            rand_cos_vocod = cos(
                rand_noise[0].flatten().unsqueeze(0),
                mel_after_vocoder[0].flatten().unsqueeze(0),
            )
            rand_cos_mp3 = cos(
                rand_noise[0].flatten().unsqueeze(0),
                mel_after_mp3[0].flatten().unsqueeze(0),
            )
            rand_cos_bigvgan = cos(
                rand_noise[0].flatten().unsqueeze(0),
                mel_after_bigvgan[0].flatten().unsqueeze(0),
            )

            rand_cos_aug = {
                k: cos(
                    rand_noise[0].flatten().unsqueeze(0),
                    mel_aug[0].flatten().unsqueeze(0),
                ).item()
                for k, mel_aug in mel_augs.items()
            }

            current_rand_cos.append(rand_cos)
            current_rand_cos_vocod.append(rand_cos_vocod)
            current_rand_cos_mp3.append(rand_cos_mp3)
            current_rand_cos_bigvgan.append(rand_cos_bigvgan)

            for k, v in rand_cos_aug.items():
                current_rand_cos_aug[k].append(v)

            # print(rand_cos)
            rand_cos_vals.append(rand_cos.item())
            rand_cos_vals_vocod.append(rand_cos_vocod.item())
            for k, v in rand_cos_aug.items():
                rand_cos_vals_aug[k].append(v)

        print("source_cos", source_cos)
        current_rand_cos = torch.cat(current_rand_cos)
        current_rand_cos_vocod = torch.cat(current_rand_cos_vocod)
        current_rand_cos_mp3 = torch.cat(current_rand_cos_mp3)
        current_rand_cos_bigvgan = torch.cat(current_rand_cos_bigvgan)

        print("current_rand_cos", source_cos_mp3)
        print(
            "percentileofscore",
            percentileofscore(current_rand_cos_mp3, source_cos_mp3) / 100,
        )
        all_p_value.append(1 - percentileofscore(current_rand_cos, source_cos) / 100)
        all_p_value_vocod.append(
            1 - percentileofscore(current_rand_cos_vocod, source_cos_vocod) / 100
        )
        all_p_value_mp3.append(
            1 - percentileofscore(current_rand_cos_mp3, source_cos_mp3) / 100
        )
        all_p_value_bigvgan.append(
            1 - percentileofscore(current_rand_cos_bigvgan, source_cos_bigvgan) / 100
        )

        for k, v in current_rand_cos_aug.items():
            all_p_value_aug[k].append(1 - percentileofscore(v, source_cos_aug[k]) / 100)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    print(all_p_value)
    print(all_p_value)
    print(all_p_value)
    print(all_p_value)
    # plt.hist(np.concatenate(all_p_value), bins=20, color='red', edgecolor='black', alpha=0.7, density=True, label="original")
    # plt.hist(np.concatenate(all_p_value_vocod), bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True, label="after vocos")
    plt.hist(
        np.concatenate(all_p_value_mp3),
        bins=20,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
        label="after mp3",
    )
    plt.hist(
        np.concatenate(all_p_value_bigvgan),
        bins=20,
        color="red",
        edgecolor="black",
        alpha=0.7,
        density=True,
        label="afte bigvgan",
    )
    # for k, v in all_p_value_aug.items():
    #    plt.hist(np.concatenate(v), bins=20, color='purple', edgecolor='black', alpha=0.7, density=True, label=f"aug {k}")
    # plt.axvline(source_cos.item(), color='red', linestyle='dashed', linewidth=2, label=f"Source Cosine = {source_cos.item():.4f}")
    plt.xlabel("Cosine Similarity p value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Similarity p value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_p.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_p.png")

    plt.figure(figsize=(8, 5))
    plt.hist(
        rand_cos_vals,
        bins=20,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    # plt.axvline(source_cos.item(), color='red', linestyle='dashed', linewidth=2, label=f"Source Cosine = {source_cos.item():.4f}")
    plt.hist(
        all_source_cos, bins=20, color="red", edgecolor="black", alpha=0.7, density=True
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Cosine Similarities vs. Source Cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_100.png")

    plt.figure(figsize=(8, 5))
    plt.hist(
        rand_cos_vals_vocod,
        bins=20,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        all_source_vocod,
        bins=20,
        color="red",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    # plt.axvline(source_cos_vocod.item(), color='red', linestyle='dashed', linewidth=2, label=f"Source Cosine = {source_cos_vocod.item():.4f}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        "Distribution of Random Cosine Similarities vs. Source Cosine (after audio went through vocoder)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosine_hist_voc_100.png", dpi=300)
    plt.close()
    print("Saved histogram as cosine_hist_voc_100.png")

    for k, v in rand_cos_vals_aug.items():
        sag = all_source_aug[k]

        plt.figure(figsize=(8, 5))
        plt.hist(
            v, bins=20, color="skyblue", edgecolor="black", alpha=0.7, density=True
        )
        # plt.axvline(source_cos_aug.item(), color='red', linestyle='dashed', linewidth=2, label=f"Source Cosine = {source_cos_aug.item():.4f}")
        plt.hist(sag, bins=20, color="red", edgecolor="black", alpha=0.7, density=True)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.title(
            "Distribution of Random Cosine Similarities vs. Source Cosine (after noised)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cosine_hist_{k}_100.png", dpi=300)
        plt.close()
        print("Saved histogram as cosine_hist_noised.png")


main()
