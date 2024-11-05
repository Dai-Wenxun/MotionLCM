import torch


class DNO(object):
    def __init__(
            self,
            max_train_steps: int,
            learning_rate: float,
            lr_scheduler: str,
            lr_warmup_steps: int,
            loss_diff_penalty: float,
            loss_correlate_penalty: float) -> None:

        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.loss_diff_penalty = loss_diff_penalty
        self.loss_correlate_penalty = loss_correlate_penalty

    @staticmethod
    def noise_regularize_1d(noise: torch.Tensor, stop_at: int = 2, dim: int = 1) -> torch.Tensor:
        size = noise.shape[dim]
        if size & (size - 1) != 0:
            new_size = 2 ** (size - 1).bit_length()
            pad = new_size - size
            pad_shape = list(noise.shape)
            pad_shape[dim] = pad
            pad_noise = torch.randn(*pad_shape, device=noise.device)
            noise = torch.cat([noise, pad_noise], dim=dim)
            size = noise.shape[dim]

        loss = torch.zeros(noise.shape[0], device=noise.device)
        while size > stop_at:
            rolled_noise = torch.roll(noise, shifts=1, dims=dim)
            loss += (noise * rolled_noise).mean(dim=tuple(range(1, noise.ndim))).pow(2)
            noise = noise.view(*noise.shape[:dim], size // 2, 2, *noise.shape[dim + 1:]).mean(dim=dim + 1)
            size //= 2
        return loss
