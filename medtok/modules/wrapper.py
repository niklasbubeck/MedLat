import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from medtok.utils import init_from_ckpt


class GenWrapper(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        scale_factor: float = None,
        is_autoregressive: bool = False,
        ckpt_path: Optional[str] = None,
        scale_steps: int = 100,
    ):
        super().__init__()
        self.generator = generator
        self.first_stage = first_stage
        self.is_autoregressive = is_autoregressive
        self.scale_steps = scale_steps

        # Determine if we should do automatic scale_factor estimation
        self._auto_scale_factor = scale_factor is None
        
        # Register scale factor as buffer
        # If None, initialize to 1.0 (will be updated automatically)
        # If provided, use the given value (will not be updated)
        initial_scale = 1.0 if scale_factor is None else scale_factor
        self.register_buffer("scale_factor", torch.tensor(initial_scale))
        
        # Track step counter and running statistics for automatic scale_factor determination
        # Only used when _auto_scale_factor is True
        self._scale_step_counter = 0
        self._running_std_sum = 0.0

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

        # Freeze first stage
        if self.first_stage is not None:
            for p in self.first_stage.parameters():
                p.requires_grad = False
            self.first_stage.eval()

    # ---------------------------------------------------------------------
    # Training mode handling
    # ---------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.generator is not None:
            self.generator.train(mode)

        if self.first_stage is not None:
            self.first_stage.eval()
            for p in self.first_stage.parameters():
                p.requires_grad = False
        return self

    # ---------------------------------------------------------------------
    # Scale factor determination
    # ---------------------------------------------------------------------
    def _update_scale_factor(self, quant: torch.Tensor) -> None:
        """
        Automatically determine scale_factor during the first scale_steps steps.
        Updates scale_factor based on the standard deviation of quantized latents.
        After scale_steps steps, the scale_factor remains fixed.
        Only updates if scale_factor was initially None.
        """
        # Only do automatic estimation if scale_factor was None
        if not self._auto_scale_factor:
            return
        
        if self._scale_step_counter > self.scale_steps:
            return
        
        if self._scale_step_counter == self.scale_steps:
            print(f"Scale factor fixed at {self.scale_factor.item()}")
            self._scale_step_counter +=1
            return
        
        with torch.no_grad():
            # Compute standard deviation of quantized latents
            quant_std = quant.std().item()
            
            # Accumulate std values
            self._running_std_sum += quant_std
            self._scale_step_counter += 1
            
            # Compute average std and update scale_factor
            avg_std = self._running_std_sum / self._scale_step_counter
            self.scale_factor.data = torch.tensor(1.0 / (avg_std + 1e-8), device=self.scale_factor.device)

    # ---------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------
    def vae_encode(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if self.first_stage is None:
            return image
        
        quant, loss, info = self.first_stage.encode(image)
        self.quant_shape = quant.permute(0, 2, 3, 1).shape  # (B, H, W, C) for later

        # ---- diffusion / continuous path ----
        if not self.is_autoregressive:
            # Automatically determine scale_factor during the first scale_steps steps
            self._update_scale_factor(quant)
            quant = quant * self.scale_factor

        # ---- autoregressive path ----
        if info is None:
            return quant
        else:
            if self.is_autoregressive:
                _, _, indices = info
                return indices.reshape(image.shape[0], -1)
            else:
                return quant

    # ---------------------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------------------
    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.first_stage is None:
            return z
        
        if self.is_autoregressive:
            return self.first_stage.decode_code(z, out_shape=self.quant_shape)
        else:
            return self.first_stage.decode(z / self.scale_factor)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x, *args, **kwargs):
        return self.generator.forward(x, *args, **kwargs)
