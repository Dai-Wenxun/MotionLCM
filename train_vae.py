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
from mld.data.get_data import get_dataset
from mld.models.modeltype.vae import VAE
from mld.utils.utils import print_table, set_seed, move_batch_to_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    cfg.output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(cfg.output_dir, exist_ok=False)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=False)

    if cfg.vis == "tb":
        writer = SummaryWriter(cfg.output_dir)
    elif cfg.vis == "swanlab":
        writer = swanlab.init(project="MotionLCM",
                              experiment_name=os.path.normpath(cfg.output_dir).replace(os.path.sep, "-"),
                              suffix=None, config=dict(**cfg), logdir=cfg.output_dir)
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=handlers)
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    dataset = get_dataset(cfg, motion_only=cfg.TRAIN.get('MOTION_ONLY', False))
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    dataset = get_dataset(cfg, motion_only=False)
    test_dataloader = dataset.test_dataloader()

    model = VAE(cfg, dataset)
    model.to(device)

    if cfg.TRAIN.PRETRAINED:
        logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
        state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
        logger.info(model.load_state_dict(state_dict))

    logger.info("learning_rate: {}".format(cfg.TRAIN.learning_rate))
    optimizer = torch.optim.AdamW(
        model.vae.parameters(),
        lr=cfg.TRAIN.learning_rate,
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

    @torch.no_grad()
    def validation():
        model.vae.eval()

        val_loss_list = []
        for val_batch in tqdm(val_dataloader):
            val_batch = move_batch_to_device(val_batch, device)
            val_loss_dict = model.allsplit_step(split='val', batch=val_batch)
            val_loss_list.append(val_loss_dict)

        for val_batch in tqdm(test_dataloader):
            val_batch = move_batch_to_device(val_batch, device)
            model.allsplit_step(split='test', batch=val_batch)
        metrics = model.allsplit_epoch_end()

        for loss_k in val_loss_list[0].keys():
            metrics[f"Val/{loss_k}"] = sum([d[loss_k] for d in val_loss_list]).item() / len(val_dataloader)

        max_val_mpjpe = metrics['Metrics/MPJPE']
        min_val_fid = metrics['Metrics/FID']
        print_table(f'Validation@Step-{global_step}', metrics)
        for mk, mv in metrics.items():
            if cfg.vis == "tb":
                writer.add_scalar(mk, mv, global_step=global_step)
            elif cfg.vis == "swanlab":
                writer.log({mk: mv}, step=global_step)

        model.vae.train()
        return max_val_mpjpe, min_val_fid

    min_mpjpe, min_fid = validation()

    progress_bar = tqdm(range(0, cfg.TRAIN.max_train_steps), desc="Steps")
    while True:
        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            loss_dict = model.allsplit_step('train', batch)

            rec_feats_loss = loss_dict['rec_feats_loss']
            rec_joints_loss = loss_dict['rec_joints_loss']
            rec_velocity_loss = loss_dict['rec_velocity_loss']
            kl_loss = loss_dict['kl_loss']
            loss = loss_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), cfg.TRAIN.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            if global_step % cfg.TRAIN.checkpointing_steps == 0:
                save_path = os.path.join(cfg.output_dir, 'checkpoints', f"checkpoint-{global_step}.ckpt")
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                logger.info(f"Saved state to {save_path}")

            if global_step % cfg.TRAIN.validation_steps == 0:
                cur_mpjpe, cur_fid = validation()
                if cur_mpjpe < min_mpjpe:
                    min_mpjpe = cur_mpjpe
                    save_path = os.path.join(cfg.output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-mpjpe-{round(cur_mpjpe, 5)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with mpjpe: {round(cur_mpjpe, 5)}")

                if cur_fid < min_fid:
                    min_fid = cur_fid
                    save_path = os.path.join(cfg.output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-fid-{round(cur_fid, 3)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with fid: {round(cur_fid, 3)}")

            logs = {"loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "rec_feats_loss": rec_feats_loss.item(),
                    'rec_joints_loss': rec_joints_loss.item(),
                    'rec_velocity_loss': rec_velocity_loss.item(),
                    'kl_loss': kl_loss.item()}

            progress_bar.set_postfix(**logs)
            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(f"Train/{k}", v, global_step=global_step)
                elif cfg.vis == "swanlab":
                    writer.log({f"Train/{k}": v}, step=global_step)

            if global_step >= cfg.TRAIN.max_train_steps:
                save_path = os.path.join(cfg.output_dir, 'checkpoints', "checkpoint-last.ckpt")
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                exit(0)


if __name__ == "__main__":
    main()
