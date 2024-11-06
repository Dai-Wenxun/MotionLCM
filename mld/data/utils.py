import torch

from mld.utils.temos_utils import lengths_to_mask


def collate_tensors(batch: list) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate(batch: list) -> dict:
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        "word_embs":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text_len":
        collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches]
    }

    mask = lengths_to_mask(adapted_batch['length'], adapted_batch['motion'].device, adapted_batch['motion'].shape[1])
    adapted_batch['mask'] = mask

    # collate trajectory
    if notnone_batches[0][-1][0] is not None:
        adapted_batch['hint'] = collate_tensors([torch.tensor(b[-1][0]).float() for b in notnone_batches])
        adapted_batch['hint_mask'] = collate_tensors([torch.tensor(b[-1][1]).float() for b in notnone_batches])

    return adapted_batch


def mld_collate_motion_only(batch: list) -> dict:
    batch = {
        "motion": collate_tensors([torch.tensor(b[0]).float() for b in batch]),
        "length": [b[1] for b in batch]
    }
    return batch
