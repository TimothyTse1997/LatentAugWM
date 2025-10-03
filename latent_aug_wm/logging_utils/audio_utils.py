import matplotlib.pyplot as plt


def log_spectrogram(writer, spectrogram, epoch, batch_id, name):
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    writer.add_figure(f"{name}_{batch_id}", fig, global_step=epoch)


def log_mel_from_batch(writer, batch, epoch, batch_id, **kwargs):
    for k, v in batch.items():
        if not k.endswith("mel"):
            continue
        log_spectrogram(writer, v[0].permute(1, 0), epoch, batch_id, k)


def log_audio_from_batch(writer, batch, epoch, batch_id, sampling_rate=24000, **kwargs):
    for k, v in batch.items():
        if not k.endswith("wave"):
            continue
        writer.add_audio(f"{k}_{batch_id}", v[0], epoch, sampling_rate)
