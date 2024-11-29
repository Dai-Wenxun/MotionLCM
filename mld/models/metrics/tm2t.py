import torch

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from .utils import (euclidean_distance_matrix, calculate_top_k, calculate_diversity_np,
                    calculate_activation_statistics_np, calculate_frechet_distance_np)


class TM2TMetrics(Metric):

    def __init__(self,
                 top_k: int = 3,
                 R_size: int = 32,
                 diversity_times: int = 300,
                 dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "Matching, FID, and Diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        self.add_state("Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # FID
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # cached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx='cat')
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx='cat')

    def compute(self) -> dict:
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        shuffle_idx = torch.randperm(count_seq)
        all_texts = dim_zero_cat(self.text_embeddings).cpu()[shuffle_idx, :]
        all_genmotions = dim_zero_cat(self.recmotion_embeddings).cpu()[shuffle_idx, :]
        all_gtmotions = dim_zero_cat(self.gtmotion_embeddings).cpu()[shuffle_idx, :]

        # Compute r-precision
        assert count_seq >= self.R_size
        top_k_mat = torch.zeros((self.top_k,))
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            group_motions = all_genmotions[i * self.R_size:(i + 1) * self.R_size]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            self.Matching_score += dist_mat.trace()
            argmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = self.Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq >= self.R_size
        top_k_mat = torch.zeros((self.top_k,))
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            group_motions = all_gtmotions[i * self.R_size:(i + 1) * self.R_size]
            dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
            self.gt_Matching_score += dist_mat.trace()
            argmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argmax, top_k=self.top_k).sum(axis=0)
        metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k + 1)}"] = top_k_mat[k] / R_count

        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq >= self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions, self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
            self,
            text_embeddings: torch.Tensor,
            recmotion_embeddings: torch.Tensor,
            gtmotion_embeddings: torch.Tensor,
            lengths: list[int]) -> None:
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # store all texts and motions
        self.text_embeddings.append(text_embeddings.detach())
        self.recmotion_embeddings.append(recmotion_embeddings.detach())
        self.gtmotion_embeddings.append(gtmotion_embeddings.detach())
