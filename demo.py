import os
import pickle
import sys
import datetime
import logging
import os.path as osp

from omegaconf import OmegaConf

import torch

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.models.modeltype.vae import VAE
from mld.utils.utils import set_seed, move_batch_to_device
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.utils.temos_utils import remove_padding


def load_example_hint_input(text_path: str) -> tuple:
    with open(text_path, "r") as f:
        lines = f.readlines()

    n_frames, control_type_ids, control_hint_ids = [], [], []
    for line in lines:
        s = line.strip()
        n_frame, control_type_id, control_hint_id = s.split(' ')
        n_frames.append(int(n_frame))
        control_type_ids.append(int(control_type_id))
        control_hint_ids.append(int(control_hint_id))

    return n_frames, control_type_ids, control_hint_ids


def load_example_input(text_path: str) -> tuple:
    with open(text_path, "r") as f:
        lines = f.readlines()

    texts, lens = [], []
    for line in lines:
        s = line.strip()
        s_l, s_t = s.split(" ")
        lens.append(int(s_l))
        texts.append(s_t)
    return texts, lens


def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, "demo_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    cfg.output_dir = osp.join(cfg.TEST_FOLDER, name_time_str)
    vis_dir = osp.join(cfg.output_dir, 'samples')
    os.makedirs(cfg.output_dir, exist_ok=False)
    os.makedirs(vis_dir, exist_ok=False)

    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[steam_handler, file_handler])
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    # Step 1: Check if the checkpoint is VAE-based.
    is_vae = False
    vae_key = 'vae.skel_embedding.weight'
    if vae_key in state_dict:
        is_vae = True
    logger.info(f'Is VAE: {is_vae}')

    # Step 2: Check if the checkpoint is MLD-based.
    is_mld = False
    mld_key = 'denoiser.time_embedding.linear_1.weight'
    if mld_key in state_dict:
        is_mld = True
    logger.info(f'Is MLD: {is_mld}')

    # Step 3: Check if the checkpoint is LCM-based.
    is_lcm = False
    lcm_key = 'denoiser.time_embedding.cond_proj.weight'  # unique key for CFG
    if lcm_key in state_dict:
        is_lcm = True
        time_cond_proj_dim = state_dict[lcm_key].shape[1]
        cfg.model.denoiser.params.time_cond_proj_dim = time_cond_proj_dim
    logger.info(f'Is LCM: {is_lcm}')

    # Step 4: Check if the checkpoint is Controlnet-based.
    cn_key = "controlnet.controlnet_cond_embedding.0.weight"
    is_controlnet = True if cn_key in state_dict else False
    cfg.model.is_controlnet = is_controlnet
    logger.info(f'Is Controlnet: {is_controlnet}')

    if is_mld or is_lcm or is_controlnet:
        target_model_class = MLD
    else:
        target_model_class = VAE

    dataset = get_dataset(cfg)
    model = target_model_class(cfg, dataset)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info(model.load_state_dict(state_dict))

    FPS = eval(f"cfg.DATASET.{cfg.DATASET.NAME.upper()}.FRAME_RATE")

    if cfg.example is not None and not is_controlnet:
        text, length = load_example_input(cfg.example)
        for t, l in zip(text, length):
            logger.info(f"{l}: {t}")

        batch = {"length": length, "text": text}

        for rep_i in range(cfg.replication):
            with torch.no_grad():
                joints, _ = model(batch)

            num_samples = len(joints)
            batch_id = 0
            for i in range(num_samples):
                res = dict()
                pkl_path = osp.join(vis_dir, f"batch_id_{batch_id}_sample_id_{i}_length_{length[i]}_rep_{rep_i}.pkl")
                res['joints'] = joints[i].detach().cpu().numpy()
                res['text'] = text[i]
                res['length'] = length[i]
                res['hint'] = None
                with open(pkl_path, 'wb') as f:
                    pickle.dump(res, f)
                logger.info(f"Motions are generated here:\n{pkl_path}")

                if not cfg.no_plot:
                    plot_3d_motion(pkl_path.replace('.pkl', '.mp4'), joints[i].detach().cpu().numpy(), text[i], fps=FPS)

    else:
        test_dataloader = dataset.test_dataloader()
        for rep_i in range(cfg.replication):
            for batch_id, batch in enumerate(test_dataloader):
                batch = move_batch_to_device(batch, device)
                with torch.no_grad():
                    joints, joints_ref = model(batch)

                num_samples = len(joints)
                text = batch['text']
                length = batch['length']
                if 'hint' in batch:
                    hint, hint_mask = batch['hint'], batch['hint_mask']
                    hint = dataset.denorm_spatial(hint) * hint_mask
                    hint = remove_padding(hint, lengths=length)
                else:
                    hint = None

                for i in range(num_samples):
                    res = dict()
                    pkl_path = osp.join(vis_dir, f"batch_id_{batch_id}_sample_id_{i}_length_{length[i]}_rep_{rep_i}.pkl")
                    res['joints'] = joints[i].detach().cpu().numpy()
                    res['text'] = text[i]
                    res['length'] = length[i]
                    res['hint'] = hint[i].detach().cpu().numpy() if hint is not None else None
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(res, f)
                    logger.info(f"Motions are generated here:\n{pkl_path}")

                    if not cfg.no_plot:
                        plot_3d_motion(pkl_path.replace('.pkl', '.mp4'), joints[i].detach().cpu().numpy(),
                                       text[i], fps=FPS, hint=hint[i].detach().cpu().numpy() if hint is not None else None)

                    if rep_i == 0:
                        res['joints'] = joints_ref[i].detach().cpu().numpy()
                        with open(pkl_path.replace('.pkl', '_ref.pkl'), 'wb') as f:
                            pickle.dump(res, f)
                        logger.info(f"Motions are generated here:\n{pkl_path.replace('.pkl', '_ref.pkl')}")
                        if not cfg.no_plot:
                            plot_3d_motion(pkl_path.replace('.pkl', '_ref.mp4'), joints_ref[i].detach().cpu().numpy(),
                                           text[i], fps=FPS, hint=hint[i].detach().cpu().numpy() if hint is not None else None)


if __name__ == "__main__":
    main()
