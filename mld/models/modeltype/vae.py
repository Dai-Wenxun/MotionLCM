import time
import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, sum_flat
from .base import BaseModel

logger = logging.getLogger(__name__)


class VAE(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()

        self.cfg = cfg
        self.datamodule = datamodule
        self.njoints = cfg.DATASET.NJOINTS

        self.vae = instantiate_from_config(cfg.model.motion_vae)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.rec_feats_ratio = cfg.model.rec_feats_ratio
        self.rec_joints_ratio = cfg.model.rec_joints_ratio
        self.rec_velocity_ratio = cfg.model.rec_velocity_ratio
        self.kl_ratio = cfg.model.kl_ratio

        self.rec_feats_loss = cfg.model.rec_feats_loss
        self.rec_joints_loss = cfg.model.rec_joints_loss
        self.rec_velocity_loss = cfg.model.rec_velocity_loss
        self.mask_loss = cfg.model.mask_loss

        logger.info(f"latent_dim: {cfg.model.latent_dim}")
        logger.info(f"rec_feats_ratio: {self.rec_feats_ratio}, "
                    f"rec_joints_ratio: {self.rec_joints_ratio}, "
                    f"rec_velocity_ratio: {self.rec_velocity_ratio}, "
                    f"kl_ratio: {self.kl_ratio}")
        logger.info(f"rec_feats_loss: {self.rec_feats_loss}, "
                    f"rec_joints_loss: {self.rec_joints_loss}, "
                    f"rec_velocity_loss: {self.rec_velocity_loss}")
        logger.info(f"mask_loss: {cfg.model.mask_loss}")

        self.summarize_parameters()

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')

    def forward(self, batch: dict) -> tuple:
        feats_ref = batch['motion']
        lengths = batch["length"]
        mask = batch['mask']

        z, dist_m = self.vae.encode(feats_ref, mask)
        feats_rst = self.vae.decode(z, mask)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def loss_calculate(self, a: torch.Tensor, b: torch.Tensor, loss_type: str,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = None if not self.mask_loss else mask
        if loss_type == 'l1':
            loss = F.l1_loss(a, b, reduction='none')
        elif loss_type == 'l1_smooth':
            loss = F.smooth_l1_loss(a, b, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(a, b, reduction='none')
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

        if mask is not None:
            loss = (loss.mean(dim=-1) * mask).sum(-1) / mask.sum(-1)
        return loss.mean()

    def train_vae_forward(self, batch: dict) -> dict:
        feats_ref = batch['motion']
        mask = batch['mask']

        z, dist_m = self.vae.encode(feats_ref, mask)
        feats_rst = self.vae.decode(z, mask)

        loss_dict = dict(
            rec_feats_loss=torch.tensor(0., device=z.device),
            rec_joints_loss=torch.tensor(0., device=z.device),
            rec_velocity_loss=torch.tensor(0., device=z.device),
            kl_loss=torch.tensor(0., device=z.device))

        if self.rec_feats_ratio > 0:
            rec_feats_loss = self.loss_calculate(feats_ref, feats_rst, self.rec_feats_loss, mask)
            loss_dict['rec_feats_loss'] = rec_feats_loss * self.rec_feats_ratio

        if self.rec_joints_ratio > 0:
            joints_rst = self.feats2joints(feats_rst).reshape(mask.size(0), mask.size(1), -1)
            joints_ref = self.feats2joints(feats_ref).reshape(mask.size(0), mask.size(1), -1)
            rec_joints_loss = self.loss_calculate(joints_ref, joints_rst, self.rec_joints_loss, mask)
            loss_dict['rec_joints_loss'] = rec_joints_loss * self.rec_joints_ratio

        if self.rec_velocity_ratio > 0:
            rec_velocity_loss = self.loss_calculate(feats_ref[..., 4: (self.njoints - 1) * 3 + 4],
                                                    feats_rst[..., 4: (self.njoints - 1) * 3 + 4],
                                                    self.rec_velocity_loss, mask)
            loss_dict['rec_velocity_loss'] = rec_velocity_loss * self.rec_velocity_ratio

        if self.kl_ratio > 0:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = torch.distributions.kl_divergence(dist_m, dist_ref).mean()
            loss_dict['kl_loss'] = kl_loss * self.kl_ratio

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:
        feats_ref_ori = batch["motion"]
        mask = batch['mask']
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        start = time.time()

        vae_st_e = time.time()
        z, dist_m = self.vae.encode(feats_ref_ori, mask)
        vae_et_e = time.time()
        self.vae_encode_times.append(vae_et_e - vae_st_e)

        vae_st_d = time.time()
        feats_rst_ori = self.vae.decode(z, mask)
        vae_et_d = time.time()
        self.vae_decode_times.append(vae_et_d - vae_st_d)

        end = time.time()
        self.times.append(end - start)
        self.frames.extend(lengths)

        # joints recover
        joints_rst = self.feats2joints(feats_rst_ori)
        joints_ref = self.feats2joints(feats_ref_ori)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst_ori)
        feats_ref = self.datamodule.renorm4t2m(feats_ref_ori)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
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
            "m_ref_ori": feats_ref_ori,
            "m_rst_ori": feats_rst_ori,
            "lat_t": text_emb,
            "lat_m": gt_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst
        }
        return rs_set

    def allsplit_step(self, split: str, batch: dict) -> Optional[dict]:
        if split in ["test"]:
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
                                                 rs_set["m_ref_ori"],
                                                 rs_set["m_rst_ori"],
                                                 batch["length"])
                else:
                    raise TypeError(f"Not support this metric {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_vae_forward(batch)
            return loss_dict
