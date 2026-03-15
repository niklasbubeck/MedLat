import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from medlat.utils import init_from_ckpt, get_model_type


class GenWrapper(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        scale_factor: float = None,
        ckpt_path: Optional[str] = None,
        scale_steps: int = 100,
    ):
        super().__init__()
        self.generator = generator
        self.generator_type = get_model_type(generator)
        self.first_stage = first_stage
        self.first_stage_type = get_model_type(first_stage)
        print(f"generator_type: {self.generator_type}, first_stage_type: {self.first_stage_type}")
        self.scale_steps = scale_steps

        if self.first_stage_type == "discrete" and self.generator_type == "non-autoregressive":
            self.fcn_encode = self.first_stage.encode_to_prequant
            self.fcn_decode = self.first_stage.decode_from_prequant

        elif self.first_stage_type == "continuous" and self.generator_type == "non-autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode

        elif self.first_stage_type == "discrete" and self.generator_type == "autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode_code

        elif self.first_stage_type == "continuous" and self.generator_type == "autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode

        else:
            raise ValueError(f"Unsupported combination of generator and first stage types: {self.generator_type} and {self.first_stage_type}")

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
        
        quant, loss, info = self.fcn_encode(image)
        self.quant_shape = quant.permute(0, 2, 3, 1).shape  # (B, H, W, C) for indices decoding later


        # Automatically determine scale_factor during the first scale_steps steps
        if self.training:
            self._update_scale_factor(quant)
        quant = quant * self.scale_factor

        # ---- autoregressive path ----
        if info is None:
            return quant
        else:
            _, _, indices = info
            if type(indices) == torch.Tensor: ## normal
                return indices.reshape(image.shape[0], -1)
            elif isinstance(indices, (list, tuple)):  # residual quantizer
                indices = [ind.reshape(image.shape[0], -1) for ind in indices]
                return torch.cat(indices, dim=1)
            return quant


    # ---------------------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------------------
    def vae_decode(self, z: torch.Tensor, out_shape = None) -> torch.Tensor:
        if self.first_stage is None:
            return z
        
        if self.generator_type == "autoregressive" and self.first_stage_type == "discrete":
            return self.fcn_decode(z, out_shape=out_shape if out_shape is not None else self.quant_shape)
        else:
            return self.fcn_decode(z / self.scale_factor)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x, *args, **kwargs):
        return self.generator.forward(x, *args, **kwargs)
