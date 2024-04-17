import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from mld.utils.temos_utils import remove_padding
from .utils import calculate_skating_ratio, calculate_trajectory_error, control_l2


class ControlMetrics(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "control_metrics"

        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("skate_ratio_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("dist_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("traj_err", default=[], dist_reduce_fx="cat")
        self.traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]

    def compute(self) -> dict:
        count_seq = self.count_seq.item()

        metrics = dict()
        metrics['Skating Ratio'] = self.skate_ratio_sum / count_seq
        metrics['Control L2 dist'] = self.dist_sum / count_seq
        traj_err = dim_zero_cat(self.traj_err).mean(0)

        for (k, v) in zip(self.traj_err_key, traj_err):
            metrics[k] = v

        return {**metrics}

    def update(self, joints: torch.Tensor,  hint: torch.Tensor,
               mask_hint: torch.Tensor, lengths: list[int]) -> None:
        self.count_seq += len(lengths)

        joints_no_padding = remove_padding(joints, lengths)
        for j in joints_no_padding:
            skate_ratio, _ = calculate_skating_ratio(j.unsqueeze(0).permute(0, 2, 3, 1))
            self.skate_ratio_sum += skate_ratio[0]

        joints_np = joints.cpu().numpy()
        hint_np = hint.cpu().numpy()
        mask_hint_np = mask_hint.cpu().numpy()

        for j, h, m in zip(joints_np, hint_np, mask_hint_np):
            control_error = control_l2(j[None], h[None], m[None])
            mean_error = control_error.sum() / m.sum()
            self.dist_sum += mean_error
            control_error = control_error.reshape(-1)
            m = m.reshape(-1)
            err_np = calculate_trajectory_error(control_error, mean_error, m)
            self.traj_err.append(torch.tensor(err_np[None], device=joints.device))
