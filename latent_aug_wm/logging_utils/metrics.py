import re

import torch

from torcheval.metrics import MulticlassPrecision, MulticlassF1Score, BinaryF1Score
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_auprc,
    binary_f1_score,  # F1
    binary_precision,  # precision
)


class BaseMetric:
    useful_values = []

    def step_process(self, batch):
        return batch

    def reset(self):
        pass

    def __call__(self, writer, batches, epoch):
        return


class BinaryF1Metric(BaseMetric):
    useful_values = ["real_detector_logits", "fake_detector_logits"]

    def __init__(self, *args, **kwargs):
        self.metric = BinaryF1Score()

    def step_process(self, batch):

        real_detector_logits = batch["real_detector_logits"]
        fake_detector_logits = batch["fake_detector_logits"]

        real_detector_pred = torch.argmax(real_detector_logits, dim=-1).detach().cpu()
        fake_detector_pred = torch.argmax(fake_detector_logits, dim=-1).detach().cpu()
        full_pred_labels = (
            torch.cat((real_detector_pred, fake_detector_pred)).detach().cpu()
        )
        labels = torch.zeros(
            real_detector_logits.shape[0] + fake_detector_logits.shape[0],
            dtype=torch.long,
        )
        labels[: real_detector_logits.shape[0]] = 1
        self.metric.update(full_pred_labels, labels)

    def reset(self):
        self.metric.reset()

    def __call__(self, writer, epoch):
        f1_score = float(self.metric.compute().detach().cpu().numpy())

        writer.add_scalar("eval/f1", f1_score, epoch)
        self.reset()
        return {"f1": f1_score}


class BinaryPrecisionMetric(BinaryF1Metric):
    def __init__(self, *args, **kwargs):
        self.metric = MulticlassPrecision(num_classes=2)

    def __call__(self, writer, epoch):
        precision_score = float(self.metric.compute().detach().cpu().numpy())

        writer.add_scalar("eval/precision", precision_score, epoch)
        self.reset()
        return {"precision": precision_score}
