import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, sum_flat

from .base import BaseModel

logger = logging.getLogger(__name__)


class MLD(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()

        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncondp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)
        self.l2_loss = lambda a, b: (a - b) ** 2

        self.is_controlnet = cfg.model.is_controlnet
        if self.is_controlnet:
            c_cfg = self.cfg.model.denoiser.copy()
            c_cfg['params']['is_controlnet'] = True
            self.controlnet = instantiate_from_config(c_cfg)
            self.training_control_joint = cfg.model.training_control_joint
            self.testing_control_joint = cfg.model.testing_control_joint
            self.training_density = cfg.model.training_density
            self.testing_density = cfg.model.testing_density
            self.control_scale = cfg.model.control_scale
            self.vaeloss = cfg.model.vaeloss
            self.vaeloss_type = cfg.model.vaeloss_type
            self.cond_ratio = cfg.model.cond_ratio
            self.rot_ratio = cfg.model.rot_ratio

            self.is_controlnet_temporal = cfg.model.is_controlnet_temporal

            self.traj_encoder = instantiate_from_config(cfg.model.traj_encoder)

            logger.info(f"control_scale: {self.control_scale}, vaeloss: {self.vaeloss}, "
                        f"cond_ratio: {self.cond_ratio}, rot_ratio: {self.rot_ratio}, "
                        f"vaeloss_type: {self.vaeloss_type}")
            logger.info(f"is_controlnet_temporal: {self.is_controlnet_temporal}")
            logger.info(f"training_control_joint: {self.training_control_joint}")
            logger.info(f"testing_control_joint: {self.testing_control_joint}")
            logger.info(f"training_density: {self.training_density}")
            logger.info(f"testing_density: {self.testing_density}")

            time.sleep(2)

        self.summarize_parameters()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')

        if self.is_controlnet:
            vae = count_parameters(self.traj_encoder)
            controlnet = count_parameters(self.controlnet)
            logger.info(f'ControlNet: {controlnet}M')
            logger.info(f'Spatial VAE: {vae}M')

    def forward(self, batch: dict) -> tuple:
        texts = batch["text"]
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            texts = [""] * len(texts) + texts

        text_emb = self.text_encoder(texts)

        hint = batch['hint'] if 'hint' in batch else None  # control signals
        z = self._diffusion_reverse(text_emb, hint)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if 'motion' in batch:
            feats_ref = batch['motion']
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(self, model_output, timesteps, sample):
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
        return pred_x_0

    def _diffusion_reverse(self, encoder_hidden_states: torch.Tensor, hint: torch.Tensor = None) -> torch.Tensor:

        controlnet_cond = None
        if self.is_controlnet:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states.device,
            dtype=torch.float)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # reverse
        for i, t in tqdm.tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            if self.is_controlnet:
                if self.do_classifier_free_guidance:
                    controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[1]
                else:
                    controlnet_prompt_embeds = encoder_hidden_states

                controlnet_residuals = self.controlnet(
                    latents,
                    t,
                    timestep_cond=timestep_cond,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond)

                if self.do_classifier_free_guidance:
                    controlnet_residuals = [torch.cat([torch.zeros_like(d), d * self.control_scale], dim=1)
                                            for d in controlnet_residuals]
                else:
                    controlnet_residuals = [d * self.control_scale for d in controlnet_residuals]

            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_residuals=controlnet_residuals)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents

    def _diffusion_process(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor,
                           hint: torch.Tensor = None) -> dict:

        controlnet_cond = None
        if self.is_controlnet:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        controlnet_residuals = None
        if self.is_controlnet:
            controlnet_residuals = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond)

        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_residuals=controlnet_residuals)

        model_pred = self.predicted_origin(noise_pred, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "model_pred": model_pred,
            "model_gt": latents
        }
        return n_set

    def masked_l2(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[-1]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss = loss / non_zero_elements
        return mse_loss.mean()

    def train_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)

        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        # text encode
        cond_emb = self.text_encoder(text)

        # diffusion process return with noise and noise_pred
        hint = batch['hint'] if 'hint' in batch else None  # control signals
        n_set = self._diffusion_process(z, cond_emb, hint)

        loss_dict = dict()

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM
            model_pred = n_set['model_pred']
            target = n_set['model_gt']
            # Performance comparison: l2 loss > huber loss
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            model_pred = n_set['noise_pred']
            target = n_set['noise']
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        if self.is_controlnet and self.vaeloss:
            z_pred = n_set['model_pred']
            feats_rst = self.vae.decode(z_pred.transpose(0, 1), lengths)
            joints_rst = self.feats2joints(feats_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)
            hint = batch['hint']
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                if self.vaeloss_type == 'mean':
                    cond_loss = (self.l2_loss(joints_rst, hint) * mask_hint).mean()
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                elif self.vaeloss_type == 'sum':
                    cond_loss = (self.l2_loss(joints_rst, hint).sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                elif self.vaeloss_type == 'mask':
                    cond_loss = self.masked_l2(joints_rst, hint, mask_hint)
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                else:
                    raise ValueError(f'Unsupported vaeloss_type: {self.vaeloss_type}')
            else:
                loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)

            if self.rot_ratio != 0:
                mask_rot = lengths_to_mask(lengths, feats_rst.device).unsqueeze(-1)
                if self.vaeloss_type == 'mean':
                    rot_loss = (self.l2_loss(feats_rst, feats_ref) * mask_rot).mean()
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                elif self.vaeloss_type == 'sum':
                    rot_loss = (self.l2_loss(feats_rst, feats_ref).sum(-1, keepdims=True) * mask_rot).sum() / mask_rot.sum()
                    rot_loss = rot_loss / self.nfeats
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                elif self.vaeloss_type == 'mask':
                    rot_loss = self.masked_l2(feats_rst, feats_ref, mask_rot)
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                else:
                    raise ValueError(f'Unsupported vaeloss_type: {self.vaeloss_type}')
            else:
                loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        else:
            loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)
            loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        loss = sum([v for v in loss_dict.values()])
        loss_dict['loss'] = loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        start = time.time()

        if self.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.do_classifier_free_guidance:
            texts = [""] * len(texts) + texts

        text_st = time.time()
        text_emb = self.text_encoder(texts)
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)

        diff_st = time.time()
        hint = batch['hint'] if 'hint' in batch else None  # control signals
        z = self._diffusion_reverse(text_emb, hint)
        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        with torch.no_grad():
            vae_st = time.time()
            feats_rst = self.vae.decode(z, lengths)
            vae_et = time.time()
            self.vae_decode_times.append(vae_et - vae_st)

        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

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
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst
        }

        if 'hint' in batch:
            hint = batch['hint']
            mask_hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3).sum(dim=-1, keepdim=True) != 0
            hint = self.datamodule.denorm_spatial(hint)
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3) * mask_hint
            rs_set['hint'] = hint
            rs_set['mask_hint'] = mask_hint
        else:
            rs_set['hint'] = None

        return rs_set

    def allsplit_step(self, split: str, batch: dict) -> Optional[dict]:
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)

            if self.datamodule.is_mm:
                metric_list = ['MMMetrics']
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"])
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0), batch["length"])
                elif metric == 'ControlMetrics':
                    assert rs_set['hint'] is not None
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set['hint'],
                                                 rs_set['mask_hint'], batch['length'])
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
