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
        self.start_update = False

    def step_process(self, batch):

        real_detector_logits = batch.get("real_detector_logits", None)
        fake_detector_logits = batch.get("fake_detector_logits", None)

        all_pred = []
        all_labels = []
        if real_detector_logits is not None:
            self.start_update = True
            real_detector_pred = (
                torch.argmax(real_detector_logits, dim=-1).detach().cpu()
            )
            real_labels = torch.ones(
                real_detector_logits.shape[0],
                dtype=torch.long,
            )
            all_pred.append(real_detector_pred)
            all_labels.append(real_labels)

        if fake_detector_logits is not None:
            self.start_update = True
            fake_detector_pred = (
                torch.argmax(fake_detector_logits, dim=-1).detach().cpu()
            )
            fake_labels = torch.zeros(
                fake_detector_pred.shape[0],
                dtype=torch.long,
            )
            all_pred.append(fake_detector_pred)
            all_labels.append(fake_labels)

        if len(all_pred) == 0:
            return
        if len(all_pred) == 1:
            full_pred_labels = all_pred[0].detach().cpu()
            labels = all_labels[0].detach().cpu()

        if len(all_pred) > 1:
            full_pred_labels = torch.cat(all_pred).detach().cpu()
            labels = torch.cat(all_labels).detach().cpu()

        self.metric.update(full_pred_labels, labels)

    def reset(self):
        self.start_update = False
        self.metric.reset()

    def __call__(self, writer, epoch):
        if self.start_update:
            f1_score = float(self.metric.compute().detach().cpu().numpy())
        else:
            f1_score = 0.0

        writer.add_scalar("eval/f1", f1_score, epoch)

        self.reset()
        return {"f1": f1_score}


class BinaryPrecisionMetric(BinaryF1Metric):
    def __init__(self, *args, **kwargs):
        self.metric = MulticlassPrecision(num_classes=2)
        self.start_update = False

    def __call__(self, writer, epoch):
        if self.start_update:
            precision_score = float(self.metric.compute().detach().cpu().numpy())
        else:
            precision_score = 0.0

        writer.add_scalar("eval/precision", precision_score, epoch)
        self.reset()
        return {"precision": precision_score}
