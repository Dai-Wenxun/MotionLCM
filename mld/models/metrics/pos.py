import torch
from torchmetrics import Metric

from mld.utils.temos_utils import remove_padding
from .utils import calculate_mpjpe


class PosMetrics(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MPJPE (aligned & unaligned), Feature l2 error"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mpjpe_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("mpjpe_unaligned_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("feature_error_sum", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self) -> dict:
        metric = dict(MPJPE=self.mpjpe_sum / self.count,
                      MPJPE_unaligned=self.mpjpe_unaligned_sum / self.count,
                      FeaError=self.feature_error_sum / self.count)
        return metric

    def update(self, joints_ref: torch.Tensor, joints_rst: torch.Tensor,
               feats_ref: torch.Tensor, feats_rst: torch.Tensor, lengths: list[int]) -> None:
        self.count += sum(lengths)
        joints_rst = remove_padding(joints_rst, lengths)
        joints_ref = remove_padding(joints_ref, lengths)
        feats_ref = remove_padding(feats_ref, lengths)
        feats_rst = remove_padding(feats_rst, lengths)

        for f1, f2 in zip(feats_ref, feats_rst):
            self.feature_error_sum += torch.norm(f1 - f2, p=2)

        for j1, j2 in zip(joints_ref, joints_rst):
            mpjpe = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_sum += mpjpe
            mpjpe_unaligned = torch.sum(calculate_mpjpe(j1, j2, align_root=False))
            self.mpjpe_unaligned_sum += mpjpe_unaligned
