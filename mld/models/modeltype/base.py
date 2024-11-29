import os
from typing import Any
from collections import OrderedDict

import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn as nn

from mld.models.metrics import TM2TMetrics, MMMetrics, ControlMetrics, PosMetrics
from mld.models.architectures import t2m_motionenc, t2m_textenc


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.times = []
        self.text_encoder_times = []
        self.traj_encoder_times = []
        self.diffusion_times = []
        self.vae_decode_times = []
        self.vae_encode_times = []
        self.frames = []

    def _get_t2m_evaluator(self, cfg: DictConfig) -> None:
        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent)

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent)

        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden)

        # load pretrained
        dataname = cfg.DATASET.NAME
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname, "text_mot_match/model/finest.tar"), map_location='cpu')
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def test_step(self, batch: dict) -> None:
        total_samples = len(self.frames)
        message = ''
        if len(self.times) > 0:
            inference_aits = round(np.sum(self.times) / total_samples, 5)
            message += f"\nAverage Inference Time per Sentence ({total_samples}): {inference_aits}\n"

        if len(self.text_encoder_times) > 0:
            inference_aits_text = round(np.sum(self.text_encoder_times) / total_samples, 5)
            message += f"Average Inference Time per Sentence [Text]: {inference_aits_text}\n"

        if len(self.traj_encoder_times) > 0:
            inference_aits_hint = round(np.sum(self.traj_encoder_times) / total_samples, 5)
            message += f"Average Inference Time per Sentence [Hint]: {inference_aits_hint}\n"

        if len(self.diffusion_times) > 0:
            inference_aits_diff = round(np.sum(self.diffusion_times) / total_samples, 5)
            message += f"Average Inference Time per Sentence [Diffusion]: {inference_aits_diff}\n"

        if len(self.vae_encode_times) > 0:
            inference_aits_vae_e = round(np.sum(self.vae_encode_times) / total_samples, 5)
            message += f"Average Inference Time per Sentence [VAE Encode]: {inference_aits_vae_e}\n"

        if len(self.vae_decode_times) > 0:
            inference_aits_vae_d = round(np.sum(self.vae_decode_times) / total_samples, 5)
            message += f"Average Inference Time per Sentence [VAE Decode]: {inference_aits_vae_d}\n"

        if len(self.frames) > 0:
            message += f"Average length: {round(np.mean(self.frames), 5)}\n"
            message += f"FPS: {np.sum(self.frames) / np.sum(self.times)}\n"

        if message:
            print(message)

        return self.allsplit_step("test", batch)

    def allsplit_epoch_end(self) -> dict:
        res = dict()
        if self.datamodule.is_mm and "TM2TMetrics" in self.metric_list:
            metric_list = ['MMMetrics']
        else:
            metric_list = self.metric_list
        for metric in metric_list:
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
        if hasattr(self, 'text_encoder'):
            clip_k = []
            for k, v in state_dict.items():
                if 'text_encoder' in k:
                    clip_k.append(k)
            for k in clip_k:
                del checkpoint['state_dict'][k]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> Any:
        if hasattr(self, 'text_encoder'):
            clip_state_dict = self.text_encoder.state_dict()
            new_state_dict = OrderedDict()
            for k, v in clip_state_dict.items():
                new_state_dict['text_encoder.' + k] = v
            for k, v in state_dict.items():
                if 'text_encoder' not in k:
                    new_state_dict[k] = v
            return super().load_state_dict(new_state_dict, strict)
        else:
            return super().load_state_dict(state_dict, strict)

    def configure_metrics(self) -> None:
        for metric in self.metric_list:
            if metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
            elif metric == 'ControlMetrics':
                self.ControlMetrics = ControlMetrics(self.datamodule.name,
                                                     dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
            elif metric == 'PosMetrics':
                self.PosMetrics = PosMetrics(dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
            else:
                raise NotImplementedError(f"Do not support Metric Type {metric}.")

        if "TM2TMetrics" in self.metric_list and self.cfg.TEST.DO_MM_TEST:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
