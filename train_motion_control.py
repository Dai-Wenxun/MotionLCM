import os
import sys
import logging
import datetime
import os.path as osp

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=False)
    
    if cfg.vis == "tb":
        writer = SummaryWriter(output_dir)
    elif cfg.vis == "swanlab":
        run = swanlab.init(project="MotionLCM", experiment_name=os.path.normpath(output_dir).replace(os.path.sep, "-"),
                           suffix=None, config=cfg, logdir=output_dir)
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(output_dir, 'output.log'))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=handlers)
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(output_dir, 'config.yaml'))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    assert cfg.model.is_controlnet, "cfg.model.is_controlnet must be true for controlling!"

    datasets = get_datasets(cfg)[0]
    train_dataloader = datasets.train_dataloader()
    val_dataloader = datasets.val_dataloader()

    logger.info(f"Loading pretrained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    lcm_key = 'denoiser.time_embedding.cond_proj.weight'
    is_lcm = False
    if lcm_key in state_dict:
        is_lcm = True
        time_cond_proj_dim = state_dict[lcm_key].shape[1]
        cfg.model.denoiser.params.time_cond_proj_dim = time_cond_proj_dim
    logger.info(f'Is LCM: {is_lcm}')

    model = MLD(cfg, datasets)
    logger.info(model.load_state_dict(state_dict, strict=False))
    logger.info(model.controlnet.load_state_dict(model.denoiser.state_dict(), strict=False))

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.denoiser.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.denoiser.eval()
    model.to(device)

    controlnet_params = list(model.controlnet.parameters())
    traj_encoder_params = list(model.traj_encoder.parameters())
    params = controlnet_params + traj_encoder_params
    params_to_optimize = [{'params': controlnet_params, 'lr': cfg.TRAIN.learning_rate},
                          {'params': traj_encoder_params, 'lr': cfg.TRAIN.learning_rate_spatial}]

    logger.info("learning_rate: {}, learning_rate_spatial: {}".
                format(cfg.TRAIN.learning_rate, cfg.TRAIN.learning_rate_spatial))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon)

    if cfg.TRAIN.max_train_steps == -1:
        assert cfg.TRAIN.max_train_epochs != -1
        cfg.TRAIN.max_train_steps = cfg.TRAIN.max_train_epochs * len(train_dataloader)

    if cfg.TRAIN.checkpointing_steps == -1:
        assert cfg.TRAIN.checkpointing_epochs != -1
        cfg.TRAIN.checkpointing_steps = cfg.TRAIN.checkpointing_epochs * len(train_dataloader)

    if cfg.TRAIN.validation_steps == -1:
        assert cfg.TRAIN.validation_epochs != -1
        cfg.TRAIN.validation_steps = cfg.TRAIN.validation_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_train_steps)

    # Train!
    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(0, cfg.TRAIN.max_train_steps), desc="Steps")

    @torch.no_grad()
    def validation():
        model.controlnet.eval()
        model.traj_encoder.eval()

        for val_batch in tqdm(val_dataloader):
            val_batch = move_batch_to_device(val_batch, device)
            model.allsplit_step('test', val_batch)
        metrics = model.allsplit_epoch_end()
        min_val_km = metrics['Metrics/kps_mean_err(m)']
        min_val_tj = metrics['Metrics/traj_fail_50cm']
        print_table(f'Metrics@Step-{global_step}', metrics)
        for k, v in metrics.items():
            if cfg.vis == "tb":
                writer.add_scalar(k, v, global_step=global_step)         
            elif cfg.vis == "swanlab":            
                run.log({k: v}, step=global_step)

        model.controlnet.train()
        model.traj_encoder.train()
        return min_val_km, min_val_tj

    min_km, min_tj = validation()

    for epoch in range(first_epoch, cfg.TRAIN.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            loss_dict = model.allsplit_step('train', batch)

            diff_loss = loss_dict['diff_loss']
            cond_loss = loss_dict['cond_loss']
            rot_loss = loss_dict['rot_loss']
            loss = diff_loss + cond_loss + rot_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.TRAIN.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            if global_step % cfg.TRAIN.checkpointing_steps == 0:
                save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}.ckpt")
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                logger.info(f"Saved state to {save_path}")

            if global_step % cfg.TRAIN.validation_steps == 0:
                cur_km, cur_tj = validation()
                if cur_km < min_km:
                    min_km = cur_km
                    save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}-km-{round(cur_km, 3)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with km:{round(cur_km, 3)}")

                if cur_tj < min_tj:
                    min_tj = cur_tj
                    save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}-tj-{round(cur_tj, 3)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with tj:{round(cur_tj, 3)}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "diff_loss": diff_loss.detach().item(), 'cond_loss': cond_loss.detach().item(), 'rot_loss': rot_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(k, v, global_step=global_step)         
                elif cfg.vis == "swanlab":            
                    run.log({k: v}, step=global_step)

            if global_step >= cfg.TRAIN.max_train_steps:
                break

    save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-last.ckpt")
    ckpt = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(ckpt)
    torch.save(ckpt, save_path)


if __name__ == "__main__":
    main()
