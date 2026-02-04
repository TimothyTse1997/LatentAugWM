def simple_classification_loss(nets=None, step=None, scale=1.0, **kwargs):

    input, label = kwargs["audio_mel"], kwargs["label"]
    batch_size, seq_len, _ = input.shape

    detector_logits, loss = nets.detector.calculate_loss(input.unsqueeze(1), label)
    real_detector_logits = detector_logits[label == 1]
    fake_detector_logits = detector_logits[label == 0]
    # print("real:", real_detector_logits.shape)
    # print("fake:", fake_detector_logits.shape)

    return {
        "detector_loss": loss * scale,
        "real_detector_logits": real_detector_logits.detach().cpu(),
        "fake_detector_logits": fake_detector_logits.detach().cpu(),
    }, ["detector"]
