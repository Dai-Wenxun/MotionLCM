import time
import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, sum_flat
from .base import BaseModel

logger = logging.getLogger(__name__)


class VAE(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()

        self.cfg = cfg
        self.datamodule = datamodule

        self.vae = instantiate_from_config(cfg.model.motion_vae)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.rec_feats_ratio = cfg.model.rec_feats_ratio
        self.rec_joints_ratio = cfg.model.rec_joints_ratio
        self.kl_ratio = cfg.model.kl_ratio

        logger.info(f"latent_dim: {cfg.model.latent_dim}")
        logger.info(f"rec_feats_ratio: {self.rec_feats_ratio}, "
                    f"rec_joints_ratio: {self.rec_joints_ratio}, "
                    f"kl_ratio: {self.kl_ratio}")

        self.summarize_parameters()

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')

    def train_vae_forward(self, batch: dict) -> dict:
        lengths = batch["length"]
        feats_ref = batch['motion']

        z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(z, lengths)

        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        mu_ref = torch.zeros_like(dist_m.loc)
        scale_ref = torch.ones_like(dist_m.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)

        loss_dict = dict()
        rec_feats_loss = F.smooth_l1_loss(feats_ref, feats_rst)
        rec_joints_loss = F.smooth_l1_loss(joints_ref, joints_rst)
        kl_loss = torch.distributions.kl_divergence(dist_m, dist_ref).mean()

        loss_dict['rec_feats_loss'] = rec_feats_loss * self.rec_feats_ratio
        loss_dict['rec_joints_loss'] = rec_joints_loss * self.rec_joints_ratio
        loss_dict['kl_loss'] = kl_loss * self.kl_ratio
        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        start = time.time()

        vae_st_e = time.time()
        z, dist_m = self.vae.encode(feats_ref, lengths)
        vae_et_e = time.time()
        self.vae_encode_times.append(vae_et_e - vae_st_e)

        vae_st_d = time.time()
        feats_rst = self.vae.decode(z, lengths)
        vae_et_d = time.time()
        self.vae_decode_times.append(vae_et_d - vae_st_d)

        end = time.time()
        self.times.append(end - start)
        self.frames.extend(lengths)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        gt_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        gt_emb = self.t2m_motionencoder(gt_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": gt_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst
        }
        return rs_set

    def allsplit_step(self, split: str, batch: dict) -> Optional[dict]:
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)

            for metric in self.metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"])
                elif metric == "PosMetrics":
                    getattr(self, metric).update(rs_set["joints_ref"],
                                                 rs_set["joints_rst"],
                                                 batch["length"])
                else:
                    raise TypeError(f"Not support this metric {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_vae_forward(batch)
            return loss_dict
