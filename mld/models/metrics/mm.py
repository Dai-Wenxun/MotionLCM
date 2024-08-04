import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from .utils import calculate_multimodality_np


class MMMetrics(Metric):

    def __init__(self, mm_num_times: int = 10, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MultiModality scores"

        self.mm_num_times = mm_num_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.metrics = ["MultiModality"]
        self.add_state("MultiModality", default=torch.tensor(0.), dist_reduce_fx="sum")

        # cached batches
        self.add_state("mm_motion_embeddings", default=[], dist_reduce_fx='cat')

    def compute(self) -> dict:
        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # cat all embeddings
        all_mm_motions = dim_zero_cat(self.mm_motion_embeddings).cpu().numpy()
        metrics['MultiModality'] = calculate_multimodality_np(all_mm_motions, self.mm_num_times)
        return metrics

    def update(self, mm_motion_embeddings: torch.Tensor, lengths: list[int]) -> None:
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # store all mm motion embeddings
        self.mm_motion_embeddings.append(mm_motion_embeddings)
