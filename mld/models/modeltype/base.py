from typing import Any
from collections import OrderedDict

import numpy as np

import torch.nn as nn

from mld.models.metrics import TM2TMetrics, MMMetrics, ControlMetrics


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.times = []
        self.text_encoder_times = []
        self.diffusion_times = []
        self.vae_decode_times = []
        self.all_lengths = []

    def test_step(self, batch: dict) -> None:
        test_batch_size = self.cfg.TEST.BATCH_SIZE
        if len(self.times) > 0:
            inference_aits = round(np.mean(self.times) / test_batch_size, 5)
            inference_aits_text = round(np.mean(self.text_encoder_times) / test_batch_size, 5)
            inference_aits_diff = round(np.mean(self.diffusion_times) / test_batch_size, 5)
            inference_aits_vae = round(np.mean(self.vae_decode_times) / test_batch_size, 5)
            print(f"\nAverage Inference Time per Sentence ({test_batch_size*len(self.times)}): {inference_aits}\n"
                  f"(Text: {inference_aits_text}, Diff: {inference_aits_diff}, VAE: {inference_aits_vae})")
            print(f"Average length: {round(np.mean(self.all_lengths), 5)}")
        return self.allsplit_step("test", batch)

    def allsplit_epoch_end(self) -> dict:
        res = dict()
        if self.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
            metrics_dicts = ['MMMetrics']
        else:
            metrics_dicts = self.metrics_dict
        for metric in metrics_dicts:
            metrics_dict = getattr(self, metric).compute()
            # reset metrics
            getattr(self, metric).reset()
            res.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })
        return res

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint['state_dict']
        clip_k = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                clip_k.append(k)
        for k in clip_k:
            del checkpoint['state_dict'][k]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> Any:
        clip_state_dict = self.text_encoder.state_dict()
        new_state_dict = OrderedDict()
        for k, v in clip_state_dict.items():
            new_state_dict['text_encoder.' + k] = v
        for k, v in state_dict.items():
            if 'text_encoder' not in k:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, strict)

    def configure_metrics(self) -> None:
        for metric in self.metrics_dict:
            if metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == 'ControlMetrics':
                self.ControlMetrics = ControlMetrics(dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
            else:
                raise NotImplementedError(f"Do not support Metric Type {metric}")

        if "TM2TMetrics" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
