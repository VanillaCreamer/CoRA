"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import math

import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from minigpt4.common.dist_utils import is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.rec_base_task import RecBaseTask


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


# Function to gather tensors across processes
def gather_tensor(tensor, dst=0):
    if dist.is_available():
        world_size = dist.get_world_size()
        if world_size > 1:
            if not isinstance(tensor, list):
                tensor = [tensor]

            gathered_tensors = [torch.empty_like(t) for t in tensor]
            dist.gather(tensor, gathered_tensors, dst=dst)

            return gathered_tensors
        else:
            return tensor
    else:
        return tensor


@registry.register_task("rec_pretrain_rating")
class RecPretrainRatingTask(RecBaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loaders, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("rmse", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("mae", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        print_freq = len(data_loaders.loaders[0]) // 5  # 10

        results = []
        results_loss = []

        k = 0
        for data_loader in data_loaders.loaders:
            pred_ratings = []
            gt_ratings = []
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                # samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)
                # results_loss.append(eval_output['loss'].item())
                if 'ratings' in eval_output.keys():
                    pred_ratings.extend(eval_output['ratings'].detach().cpu().numpy())
                    gt_ratings.extend(samples['label'].detach().cpu().numpy())
                else:
                    metric_logger.update(acc=0)
                metric_logger.update(loss=eval_output['loss'].item())
            predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
                                p in [str(i / 10.0) for i in list(range(10, 50))]]
            rmse = root_mean_square_error(predicted_rating, 5.0, 1.0)
            mae = mean_absolute_error(predicted_rating, 5.0, 1.0)
            metric_logger.update(rmse=rmse)
            metric_logger.update(mae=mae)

            if is_dist_avail_and_initialized():
                dist.barrier()
                # dist.reduce()

            metric_logger.synchronize_between_processes()

            logging.info(
                "Averaged stats: " + str(metric_logger.global_avg()) + " ***rmse: " + str(rmse) + " ***mae:" + str(mae))

            results = {
                'epoch': 0,
                'agg_metrics': -metric_logger.meters['loss'].global_avg,
                'rmse': metric_logger.meters['rmse'].global_avg,
                'mae': metric_logger.meters['mae'].global_avg
            }

        return results
