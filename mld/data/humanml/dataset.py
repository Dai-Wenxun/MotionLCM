import random
import logging
import codecs as cs
from os.path import join as pjoin

import numpy as np
from rich.progress import track

import torch
from torch.utils import data

from mld.data.humanml.scripts.motion_process import recover_from_ric
from .utils.word_vectorizer import WordVectorizer

logger = logging.getLogger(__name__)


class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        split_file: str,
        w_vectorizer: WordVectorizer,
        max_motion_length: int,
        min_motion_length: int,
        max_text_len: int,
        unit_length: int,
        motion_dir: str,
        text_dir: str,
        fps: int,
        padding_to_max: bool,
        tiny: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.padding_to_max = padding_to_max

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 10 if tiny else 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if len(motion) < self.min_motion_length or len(motion) >= self.max_motion_length:
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps): int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or \
                                        len(n_motion) >= self.max_motion_length:
                                    continue
                                new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                while new_name in data_dict:
                                    new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(f"{bad_count} motion sequences not within the length range of "
                        f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out.")

        self.mean = mean
        self.std = std

        self.mode = None
        model_params = kwargs['model_kwargs']
        if 'is_controlnet' in model_params and model_params.is_controlnet is True:
            if 'test' in split_file or 'val' in split_file:
                self.mode = 'eval'
            else:
                self.mode = 'train'

            self.t_ctrl = model_params.is_controlnet_temporal
            spatial_norm_path = './datasets/humanml_spatial_norm'
            self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
            self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))

            self.training_control_joint = np.array(model_params.training_control_joint)
            self.testing_control_joint = np.array(model_params.testing_control_joint)

            self.training_density = model_params.training_density
            self.testing_density = model_params.testing_density

        self.data_dict = data_dict
        self.nfeats = data_dict[name_list[0]]["motion"].shape[1]
        self.name_list = name_list

    def __len__(self) -> int:
        return len(self.name_list)

    def random_mask(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        choose_joint = self.testing_control_joint

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train(self, joints: np.ndarray, n_joints: int = 22) -> np.ndarray:
        if self.t_ctrl:
            choose_joint = self.training_control_joint
        else:
            num_joints = len(self.training_control_joint)
            num_joints_control = 1
            choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
            choose_joint = self.training_control_joint[choose_joint]

        length = joints.shape[0]

        if self.training_density == 'random':
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(length * random.uniform(self.training_density[0], self.training_density[1]) / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, n_joints, 3)).astype(bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def __getitem__(self, idx: int) -> tuple:
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        hint = None
        if self.mode is not None:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

            # control any joints at any time
            if self.mode == 'train':
                hint = self.random_mask_train(joints, n_joints)
            else:
                hint = self.random_mask(joints, n_joints)

            hint = hint.reshape(hint.shape[0], -1)
            if self.padding_to_max and m_length < self.max_motion_length:
                hint = np.concatenate([hint,
                                       np.zeros((self.max_motion_length - m_length, hint.shape[1]))
                                       ], axis=0)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.padding_to_max and m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            hint
        )
