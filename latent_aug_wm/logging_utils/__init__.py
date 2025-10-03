import os
import time
import torch

from torch import nn

import torch.nn.functional as F
import numpy as np

# from munch import Munch
from loguru import logger as glogger
from .metrics import BaseMetric


def construct_logging_fn(list_of_fn, log_fn_time=False):
    def _log_fn(writer, batch, epoch, batch_id, **kwargs):
        # the batch here are result of the eval step
        logging_output = {}
        for fn in list_of_fn:
            start = time.time()
            result = fn(writer, batch, epoch, batch_id, **kwargs)
            # logging_output.update(result)
            end = time.time()
            if log_fn_time:
                glogger.debug(f"function {fn} tooks in total {end - start} sec")

        return  # logging_output

    return _log_fn


def construct_full_epoch_metrics(list_of_metric):
    # see .metrics.py
    class CombineMetrics:
        def step_process(self, batch):
            for metric in list_of_metric:
                metric.step_process(batch)

        def reset(self):
            for metric in list_of_metric:
                metric.reset()

        def write(self, writer, epoch):
            results = {}
            for metric in list_of_metric:
                results.update(metric(writer, epoch))
            return results

    return CombineMetrics()
