import torch


def lengths_to_mask(lengths: list[int],
                    device: torch.device,
                    max_len: int = None) -> torch.Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def remove_padding(tensors: torch.Tensor, lengths: list[int]) -> list:
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]
