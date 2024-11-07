import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, sum_flat
from mld.data.humanml.utils.plot_script import plot_3d_motion

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
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        logger.info(f"vae_scale_factor: {self.cfg.model.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")

        self.is_controlnet = cfg.model.is_controlnet
        if self.is_controlnet:
            c_cfg = self.cfg.model.denoiser.copy()
            c_cfg['params']['is_controlnet'] = True
            self.controlnet = instantiate_from_config(c_cfg)
            self.control_scale = cfg.model.control_scale
            self.vaeloss = cfg.model.vaeloss
            self.vaeloss_type = cfg.model.vaeloss_type
            self.cond_ratio = cfg.model.cond_ratio
            self.rot_ratio = cfg.model.rot_ratio
            self.traj_encoder = instantiate_from_config(cfg.model.traj_encoder)

            logger.info(f"control_scale: {self.control_scale}, vaeloss: {self.vaeloss}, "
                        f"cond_ratio: {self.cond_ratio}, rot_ratio: {self.rot_ratio}, "
                        f"vaeloss_type: {self.vaeloss_type}")
            time.sleep(2)

        self.dno = instantiate_from_config(cfg.model.noise_optimizer)

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

    def forward(self, batch: dict, optimize: bool = False) -> tuple:
        texts = batch["text"]
        lengths = batch["length"]

        # demo for [example]-[False] or [dataset]-[True]
        maybe_has_gt = 'motion' in batch

        if self.do_classifier_free_guidance:
            texts = [""] * len(texts) + texts

        text_emb = self.text_encoder(texts)

        if self.is_controlnet:
            assert 'hint' in batch, "Hint needed for motion ControlNet"
            hint_mask = batch['hint'].sum(-1) != 0
            controlnet_cond = self.traj_encoder(batch['hint'], mask=hint_mask)





        hint = batch['hint'] if 'hint' in batch else None  # control signals
        if optimize:
            pass
        z = self._diffusion_reverse(text_emb, hint)

        with torch.no_grad():
            if maybe_has_gt:
                padding_to_max_length = batch['motion'].shape[1] if self.cfg.DATASET.PADDING_TO_MAX else None
                mask = lengths_to_mask(lengths, text_emb.device, max_len=padding_to_max_length)
            else:
                mask = lengths_to_mask(lengths, text_emb.device)
            z = z / self.cfg.model.vae_scale_factor
            feats_rst = self.vae.decode(z, mask)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if maybe_has_gt:
            joints_ref = self.feats2joints(batch['motion'].detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon

    def _diffusion_reverse_with_optimize(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            texts: list[str], lengths: list[int], mask: torch.Tensor,
            hint: torch.Tensor, hint_mask: torch.Tensor,
            controlnet_cond: Optional[torch.Tensor] = None,
            feats_ref: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        current_latents = latents.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([current_latents], lr=self.dno.learning_rate)
        lr_scheduler = get_scheduler(
            self.dno.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.dno.lr_warmup_steps,
            num_training_steps=self.dno.max_train_steps)

        do_visualize = self.dno.do_visualize
        vis_id = self.dno.visualize_samples_done
        hint_3d = self.datamodule.denorm_spatial(hint) * hint_mask
        for step in tqdm.tqdm(range(1, self.dno.max_train_steps + 1)):
            z_pred = self._diffusion_reverse(current_latents, encoder_hidden_states, controlnet_cond)
            feats_rst = self.vae.decode(z_pred / self.cfg.model.vae_scale_factor, mask)
            joints_rst = self.feats2joints(feats_rst)

            loss_hint = F.smooth_l1_loss(joints_rst, hint_3d, reduction='none') * hint_mask
            loss_hint = loss_hint.sum(dim=[1, 2, 3]) / hint_mask.sum(dim=[1, 2, 3])
            loss_diff = (current_latents - latents).norm(p=2, dim=[1, 2])
            loss_correlate = self.dno.noise_regularize_1d(current_latents, dim=1)
            loss = loss_hint + self.dno.loss_correlate_penalty * loss_correlate + self.dno.loss_diff_penalty * loss_diff
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()

            grad_norm = current_latents.grad.norm(p=2, dim=[1, 2], keepdim=True)
            if self.dno.clip_grad:
                current_latents.grad.data /= grad_norm

            # Visualize
            if do_visualize:
                control_error = torch.norm((joints_rst - hint_3d) * hint_mask, p=2, dim=-1)
                control_error = control_error.sum(dim=[1, 2]) / hint_mask.mean(dim=-1).sum(dim=[1, 2])
                for batch_id in range(latents.shape[0]):
                    metrics = {
                        'loss': loss[batch_id].item(),
                        'loss_hint': loss_hint[batch_id].mean().item(),
                        'loss_diff': loss_diff[batch_id].item(),
                        'loss_correlate': loss_correlate[batch_id].item(),
                        'grad_norm': grad_norm[batch_id].item(),
                        'lr': lr_scheduler.get_last_lr()[0],
                        'control_error': control_error[batch_id].item()
                    }
                    for metric_name, metric_value in metrics.items():
                        self.dno.writer.add_scalar(f'Optimize_{vis_id+batch_id}/{metric_name}', metric_value, step)

                    if step in self.dno.visualize_ske_steps:
                        joints_rst_no_pad = joints_rst[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                        hint_3d_no_pad = hint_3d[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                        plot_3d_motion(f'{self.dno.vis_dir}/vis_id_{vis_id+batch_id}_step_{step}.mp4',
                                       joints=joints_rst_no_pad, title=texts[batch_id], hint=hint_3d_no_pad,
                                       fps=eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.FRAME_RATE"))

            optimizer.step()
            lr_scheduler.step()

        if feats_ref is not None and do_visualize and len(self.dno.visualize_ske_steps) > 0:
            joints_ref = self.feats2joints(feats_ref)
            for batch_id in range(latents.shape[0]):
                joints_ref_no_pad = joints_ref[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                hint_3d_no_pad = hint_3d[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                plot_3d_motion(f'{self.dno.vis_dir}/vis_id_{vis_id + batch_id}_ref.mp4',
                               joints=joints_ref_no_pad, title=texts[batch_id], hint=hint_3d_no_pad,
                               fps=eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.FRAME_RATE"))

        self.dno.visualize_samples_done += latents.shape[0]
        return current_latents.detach()

    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            controlnet_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

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
                    controlnet_cond=controlnet_cond)[0]

                if self.do_classifier_free_guidance:
                    controlnet_residuals = [torch.cat([torch.zeros_like(d), d * self.control_scale], dim=1)
                                            for d in controlnet_residuals]
                else:
                    controlnet_residuals = [d * self.control_scale for d in controlnet_residuals]

            # predict the noise residual
            model_output = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_residuals=controlnet_residuals)[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_text - model_output_uncond)

            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample

        return latents

    def _diffusion_process(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor,
                           hint: torch.Tensor = None) -> dict:

        controlnet_cond = None
        if self.is_controlnet:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)

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
        router_loss_controlnet = None
        if self.is_controlnet:
            controlnet_residuals, router_loss_controlnet = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond)

        model_output, router_loss = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_residuals=controlnet_residuals)

        latents_pred, noise_pred = self.predicted_origin(model_output, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "router_loss": router_loss_controlnet if self.is_controlnet else router_loss
        }
        return n_set

    @staticmethod
    def masked_l2(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(a, b, reduction='none')
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
            padding_to_max_length = feats_ref.shape[1] if self.cfg.DATASET.PADDING_TO_MAX else None
            mask = lengths_to_mask(lengths, feats_ref.device, max_len=padding_to_max_length)
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.cfg.model.vae_scale_factor

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
            # LCM (only used in motion ControlNet)
            model_pred = n_set['sample_pred']
            target = n_set['sample_gt']
            # Performance comparison: l2 loss > huber loss when training controlnet for LCM
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            if self.scheduler.config.prediction_type == "epsilon":
                model_pred = n_set['noise_pred']
                target = n_set['noise']
            elif self.scheduler.config.prediction_type == "sample":
                model_pred = n_set['sample_pred']
                target = n_set['sample_gt']
            else:
                raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        if n_set['router_loss'] is not None:
            loss_dict['router_loss'] = n_set['router_loss']
        else:
            loss_dict['router_loss'] = torch.tensor(0., device=diff_loss.device)

        if self.is_controlnet and self.vaeloss:
            z_pred = n_set['sample_pred'] / self.cfg.model.vae_scale_factor
            feats_rst = self.vae.decode(z_pred, mask)
            joints_rst = self.feats2joints(feats_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], -1)
            joints_rst = self.datamodule.norm_spatial(joints_rst)
            joints_rst = joints_rst.view(joints_rst.shape[0], joints_rst.shape[1], self.njoints, 3)
            hint = batch['hint']
            hint = hint.view(hint.shape[0], hint.shape[1], self.njoints, 3)
            mask_hint = hint.sum(dim=-1, keepdim=True) != 0

            if self.cond_ratio != 0:
                if self.vaeloss_type == 'mean':
                    cond_loss = (F.mse_loss(joints_rst, hint, reduction='none') * mask_hint).mean()
                    loss_dict['cond_loss'] = self.cond_ratio * cond_loss
                elif self.vaeloss_type == 'sum':
                    cond_loss = (F.mse_loss(joints_rst, hint, reduction='none').sum(-1, keepdims=True) * mask_hint).sum() / mask_hint.sum()
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
                    rot_loss = (F.mse_loss(feats_rst, feats_ref, reduction='none') * mask_rot).mean()
                    loss_dict['rot_loss'] = self.rot_ratio * rot_loss
                elif self.vaeloss_type == 'sum':
                    rot_loss = (F.mse_loss(feats_rst, feats_ref, reduction='none').sum(-1, keepdims=True) * mask_rot).sum() / mask_rot.sum()
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
        mask = batch['mask']
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
            texts = texts + [""] * len(texts)

        text_st = time.time()
        text_emb = self.text_encoder(texts)
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)

        controlnet_cond = None
        if self.is_controlnet:
            assert 'hint' in batch
            hint_st = time.time()
            hint, hint_mask = batch['hint'], batch['hint_mask']
            hint = hint.view(hint.shape[0], hint.shape[1], -1)
            hint_mask = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            hint_et = time.time()
            self.traj_encoder_times.append(hint_et - hint_st)

        diff_st = time.time()
        latents = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        if 'hint' in batch:
            hint, hint_mask = batch['hint'], batch['hint_mask']
            with torch.enable_grad():
                latents = self._diffusion_reverse_with_optimize(
                    latents, text_emb, texts, lengths, mask,
                    hint, hint_mask, controlnet_cond=controlnet_cond, feats_ref=feats_ref)

        z = self._diffusion_reverse(latents, text_emb, controlnet_cond=controlnet_cond)
        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        vae_st = time.time()
        feats_rst = self.vae.decode(z / self.cfg.model.vae_scale_factor, mask)
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
        m_lens = torch.div(m_lens, eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {"m_ref": feats_ref, "m_rst": feats_rst,
                  "lat_t": text_emb, "lat_m": motion_emb, "lat_rm": recons_emb,
                  "joints_ref": joints_ref, "joints_rst": joints_rst}

        if 'hint' in batch:
            hint_3d = self.datamodule.denorm_spatial(batch['hint']) * batch['hint_mask']
            rs_set['hint'] = hint_3d
            rs_set['hint_mask'] = batch['hint_mask']

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
                    getattr(self, metric).update(rs_set["joints_rst"], rs_set['hint'],
                                                 rs_set['hint_mask'], batch['length'])
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
