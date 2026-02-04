import torch
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt


def log_spectrogram_distribution(writer, spectrogram, epoch, batch_id, name):
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

    plt.title(f"Value Distribution of {name}_{batch_id}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    # Log the figure
    writer.add_figure(f"{name}_dist_{batch_id}", fig, global_step=epoch)


def log_spectrogram(writer, spectrogram, epoch, batch_id, name):
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    writer.add_figure(f"{name}_{batch_id}", fig, global_step=epoch)


def log_mel_from_batch(writer, batch, epoch, batch_id, **kwargs):
    for k, v in batch.items():
        if not k.endswith("mel"):
            continue
        log_spectrogram(writer, v[0].permute(1, 0), epoch, batch_id, k)
        if "noise" not in k:
            continue
        log_spectrogram_distribution(writer, v[0], epoch, batch_id, k)


def log_audio_from_batch(writer, batch, epoch, batch_id, sampling_rate=24000, **kwargs):
    for k, v in batch.items():
        if not k.endswith("wave"):
            continue
        writer.add_audio(f"{k}_{batch_id}", v[0], epoch, sampling_rate)
