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
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor, control_loss_calculate
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
        self.datamodule = datamodule

        if cfg.model.guidance_scale == 'dynamic':
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f'Guidance Scale set as {self.guidance_scale}')

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

        self.vae_scale_factor = cfg.model.get("vae_scale_factor", 1.0)
        self.guidance_uncondp = cfg.model.get('guidance_uncondp', 0.0)

        logger.info(f"vae_scale_factor: {self.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")
        logger.info(f"guidance_scale: {self.guidance_scale}")
        logger.info(f"guidance_uncondp: {self.guidance_uncondp}")

        self.is_controlnet = cfg.model.get('is_controlnet', False)
        if self.is_controlnet:
            c_cfg = self.cfg.model.denoiser.copy()
            c_cfg['params']['is_controlnet'] = True
            self.controlnet = instantiate_from_config(c_cfg)
            self.traj_encoder = instantiate_from_config(cfg.model.traj_encoder)

            self.vaeloss = cfg.model.get('vaeloss', False)
            self.vaeloss_type = cfg.model.get('vaeloss_type', 'sum')
            self.cond_ratio = cfg.model.get('cond_ratio', 0.0)
            self.rot_ratio = cfg.model.get('rot_ratio', 0.0)
            self.control_loss_func = cfg.model.get('control_loss_func', 'l2')
            if self.vaeloss and self.cond_ratio == 0.0 and self.rot_ratio == 0.0:
                raise ValueError("Error: When 'vaeloss' is True, 'cond_ratio' and 'rot_ratio' cannot both be 0.")
            self.use_3d = cfg.model.get('use_3d', False)
            self.guess_mode = cfg.model.get('guess_mode', False)
            if self.guess_mode and not self.do_classifier_free_guidance:
                raise ValueError(
                    "Invalid configuration: 'guess_mode' is enabled, but 'do_classifier_free_guidance' is not. "
                    "Ensure that 'do_classifier_free_guidance' is True (MLD) when 'guess_mode' is active."
                )
            self.lcm_w_min_nax = cfg.model.get('lcm_w_min_nax')
            self.lcm_num_ddim_timesteps = cfg.model.get('lcm_num_ddim_timesteps')
            if (self.lcm_w_min_nax is not None or self.lcm_num_ddim_timesteps is not None) and self.denoiser.time_cond_proj_dim is None:
                raise ValueError(
                    "Invalid configuration: When either 'lcm_w_min_nax' or 'lcm_num_ddim_timesteps' is not None, "
                    "'denoiser.time_cond_proj_dim' must be None (MotionLCM)."
                )

            logger.info(f"vaeloss: {self.vaeloss}, "
                        f"vaeloss_type: {self.vaeloss_type}, "
                        f"cond_ratio: {self.cond_ratio}, "
                        f"rot_ratio: {self.rot_ratio}, "
                        f"control_loss_func: {self.control_loss_func}")
            logger.info(f"use_3d: {self.use_3d}, "
                        f"guess_mode: {self.guess_mode}")
            logger.info(f"lcm_w_min_nax: {self.lcm_w_min_nax}, "
                        f"lcm_num_ddim_timesteps: {self.lcm_num_ddim_timesteps}")

            time.sleep(2)  # 留个心眼

        self.dno = instantiate_from_config(cfg.model['noise_optimizer']) \
            if cfg.model.get('noise_optimizer') else None

        self.summarize_parameters()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')

        if self.is_controlnet:
            traj_encoder = count_parameters(self.traj_encoder)
            controlnet = count_parameters(self.controlnet)
            logger.info(f'ControlNet: {controlnet}M')
            logger.info(f'Trajectory Encoder: {traj_encoder}M')

    def forward(self, batch: dict) -> tuple:
        texts = batch["text"]
        feats_ref = batch.get("motion")
        lengths = batch["length"]
        hint = batch.get('hint')
        hint_mask = batch.get('hint_mask')

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        text_emb = self.text_encoder(texts)

        controlnet_cond = None
        if self.is_controlnet:
            assert hint is not None
            hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
            hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(dim=-1) != 0
            controlnet_cond = self.traj_encoder(hint_reshaped, hint_mask_reshaped)

        latents = torch.randn((len(lengths), *self.latent_dim), device=text_emb.device)
        mask = batch.get('mask', lengths_to_mask(lengths, text_emb.device))

        if hint is not None and self.dno and self.dno.optimize:
            latents = self._optimize_latents(
                latents, text_emb, texts, lengths, mask, hint, hint_mask,
                controlnet_cond=controlnet_cond, feats_ref=feats_ref)

        latents = self._diffusion_reverse(latents, text_emb, controlnet_cond=controlnet_cond)
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
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

    @torch.enable_grad()
    def _optimize_latents(
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
            z_pred = self._diffusion_reverse(current_latents, encoder_hidden_states, controlnet_cond=controlnet_cond)
            feats_rst = self.vae.decode(z_pred / self.vae_scale_factor, mask)
            joints_rst = self.feats2joints(feats_rst)

            loss_hint = self.dno.loss_hint_func(joints_rst, hint_3d, reduction='none') * hint_mask
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
                        self.dno.writer.add_scalar(f'Optimize_{vis_id + batch_id}/{metric_name}', metric_value, step)

                    if step in self.dno.visualize_ske_steps:
                        joints_rst_no_pad = joints_rst[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                        hint_3d_no_pad = hint_3d[batch_id][:lengths[batch_id]].detach().cpu().numpy()
                        plot_3d_motion(f'{self.dno.vis_dir}/vis_id_{vis_id + batch_id}_step_{step}.mp4',
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
            controlnet_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
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

        if self.is_controlnet and self.do_classifier_free_guidance and not self.guess_mode:
            controlnet_cond = torch.cat([controlnet_cond] * 2)

        for i, t in tqdm.tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            if self.is_controlnet:
                if self.do_classifier_free_guidance and self.guess_mode:
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[0]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = encoder_hidden_states

                controlnet_residuals = self.controlnet(
                    sample=control_model_input,
                    timestep=t,
                    timestep_cond=timestep_cond,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond)[0]

                if self.do_classifier_free_guidance and self.guess_mode:
                    controlnet_residuals = [torch.cat([d, torch.zeros_like(d)], dim=1) for d in controlnet_residuals]

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
                           hint: Optional[torch.Tensor] = None, hint_mask: Optional[torch.Tensor] = None) -> dict:

        controlnet_cond = None
        if self.is_controlnet:
            assert hint is not None
            hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
            hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint_reshaped, mask=hint_mask_reshaped)

        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if self.denoiser.time_cond_proj_dim is not None and self.lcm_num_ddim_timesteps is not None:
            step_size = self.scheduler.config.num_train_timesteps // self.lcm_num_ddim_timesteps
            candidate_timesteps = torch.arange(
                start=step_size - 1,
                end=self.scheduler.config.num_train_timesteps,
                step=step_size,
                device=latents.device
            )
            timesteps = candidate_timesteps[torch.randint(
                low=0,
                high=candidate_timesteps.size(0),
                size=(bsz,),
                device=latents.device
            )]
        else:
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device
            )
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            if self.lcm_w_min_nax is None:
                w = torch.tensor(self.guidance_scale - 1).repeat(latents.shape[0])
            else:
                w = (self.lcm_w_min_nax[1] - self.lcm_w_min_nax[0]) * torch.rand((bsz,)) + self.lcm_w_min_nax[0]
            timestep_cond = get_guidance_scale_embedding(
                w, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

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

    def train_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        mask = batch['mask']
        hint = batch.get('hint', None)
        hint_mask = batch.get('hint_mask', None)

        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.vae_scale_factor

        text = batch["text"]
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        text_emb = self.text_encoder(text)
        n_set = self._diffusion_process(z, text_emb, hint=hint, hint_mask=hint_mask)

        loss_dict = dict()

        if self.denoiser.time_cond_proj_dim is not None:
            # LCM (only used in motion ControlNet)
            model_pred, target = n_set['sample_pred'], n_set['sample_gt']
            # Performance comparison: l2 loss > huber loss when training controlnet for LCM
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # DM
            if self.scheduler.config.prediction_type == "epsilon":
                model_pred, target = n_set['noise_pred'], n_set['noise']
            elif self.scheduler.config.prediction_type == "sample":
                model_pred, target = n_set['sample_pred'], n_set['sample_gt']
            else:
                raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        # Router loss
        loss_dict['router_loss'] = n_set['router_loss'] if n_set['router_loss'] is not None \
            else torch.tensor(0., device=diff_loss.device)

        if self.is_controlnet and self.vaeloss:
            feats_rst = self.vae.decode(n_set['sample_pred'] / self.vae_scale_factor, mask)

            if self.cond_ratio != 0:
                joints_rst = self.feats2joints(feats_rst)
                if self.use_3d:
                    hint = self.datamodule.denorm_spatial(hint)
                else:
                    joints_rst = self.datamodule.norm_spatial(joints_rst)
                hint_mask = hint_mask.sum(-1, keepdim=True) != 0
                cond_loss = control_loss_calculate(self.vaeloss_type, self.control_loss_func, joints_rst, hint,
                                                   hint_mask)
                loss_dict['cond_loss'] = self.cond_ratio * cond_loss
            else:
                loss_dict['cond_loss'] = torch.tensor(0., device=diff_loss.device)

            if self.rot_ratio != 0:
                mask = mask.unsqueeze(-1)
                rot_loss = control_loss_calculate(self.vaeloss_type, self.control_loss_func, feats_rst, feats_ref, mask)
                loss_dict['rot_loss'] = self.rot_ratio * rot_loss
            else:
                loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        else:
            loss_dict['cond_loss'] = loss_dict['rot_loss'] = torch.tensor(0., device=diff_loss.device)

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:
        texts = batch["text"]
        feats_ref = batch["motion"]
        mask = batch['mask']
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]
        hint = batch.get('hint', None)
        hint_mask = batch.get('hint_mask', None)

        start = time.time()

        if self.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            mask = mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            hint = hint and hint.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            hint_mask = hint_mask and hint_mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        text_st = time.time()
        text_emb = self.text_encoder(texts)
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)

        controlnet_cond = None
        if self.is_controlnet:
            assert hint is not None
            hint_st = time.time()
            hint_reshaped = hint.view(hint.shape[0], hint.shape[1], -1)
            hint_mask_reshaped = hint_mask.view(hint_mask.shape[0], hint_mask.shape[1], -1).sum(dim=-1) != 0
            controlnet_cond = self.traj_encoder(hint_reshaped, hint_mask_reshaped)
            hint_et = time.time()
            self.traj_encoder_times.append(hint_et - hint_st)

        diff_st = time.time()

        latents = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)

        if hint is not None and self.dno and self.dno.optimize:
            latents = self._optimize_latents(
                latents, text_emb, texts, lengths, mask, hint, hint_mask,
                controlnet_cond=controlnet_cond, feats_ref=feats_ref)

        latents = self._diffusion_reverse(latents, text_emb, controlnet_cond=controlnet_cond)

        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        vae_st = time.time()
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)
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
