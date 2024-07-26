import torch
from torchmetrics import Metric

from mld.utils.temos_utils import remove_padding
from .utils import calculate_mpjpe


class PosMetrics(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MPJPE"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mpjpe_sum", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self) -> dict:
        metric = dict(MPJPE=self.mpjpe_sum / self.count)
        return metric

    def update(self, joints_ref: torch.Tensor,
               joints_rst: torch.Tensor,
               lengths: list[int]) -> None:
        self.count += sum(lengths)
        joints_rst = remove_padding(joints_rst, lengths)
        joints_ref = remove_padding(joints_ref, lengths)
        for j1, j2 in zip(joints_ref, joints_rst):
            mpjpe = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_sum += mpjpe
