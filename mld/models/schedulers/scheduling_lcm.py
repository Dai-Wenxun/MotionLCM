from typing import Optional, Union

import torch
import diffusers


class LCMScheduler(diffusers.schedulers.LCMScheduler):
    def __init__(self, timesteps_step_map: Optional[dict] = None, **kwargs) -> None:
        super(LCMScheduler, self).__init__(**kwargs)
        self.timesteps_step_map = timesteps_step_map

    def set_timesteps(self, num_inference_steps: Optional[int] = None,
                      device: Union[str, torch.device] = None, **kwargs) -> None:
        if self.timesteps_step_map is None:
            super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        else:
            assert num_inference_steps is not None
            self.num_inference_steps = num_inference_steps
            timesteps = self.timesteps_step_map[num_inference_steps]
            assert all([timestep < self.config.num_train_timesteps for timestep in timesteps])
            self.timesteps = torch.tensor(timesteps).to(device=device, dtype=torch.long)
            self._step_index = None
