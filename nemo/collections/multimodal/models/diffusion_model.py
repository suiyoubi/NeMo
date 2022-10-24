from abc import ABC, abstractclassmethod
from typing import Any, Optional
import torch

from nemo.core.classes import ModelPT

class DiffusionModel(ModelPT, ABC):
    @abstractclassmethod
    def get_conditioning(self, c: Any) -> Any:
        """
        Encode conditioning c.
        For txt2img use-case, the input conditioning would be the plain text,
        and output would be the encoded embedding for the corresponding text;
        For img2img use-case, the input conditioning would be the raw image,
        and output would be the corresponding image embedding

        Args:
            c: conditioning
        
        Returns:
            encoded conditioning
        """
        pass

    @abstractclassmethod
    def apply_model(self, x_t: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply Diffusion model.
        If c is not given, the model acts as an unconditional diffusion model.
        For diffusion model that applies on the pixel space, x_t should be in the pixel space;
        for diffusion model that applies on the latent space, x_t is in latent space.

        Args:
            x_t: noisy input x at timestamp t
            t: timestamp
            c: conditioning
        
        Returns:
            Predicted result that has the same shape as x_t
        """