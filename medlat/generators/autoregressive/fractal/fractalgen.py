from functools import partial

import torch
import torch.nn as nn

from .ar import AR
from .mar import MAR
from .pixelloss import PixelLoss


class FractalGen(nn.Module):
    """ Fractal Generative Model"""

    def __init__(self,
                 img_size_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 guiding_pixel=False,
                 num_conds=1,
                 r_weight=1.0,
                 grad_checkpointing=False,
                 fractal_level=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(img_size_list)

        # --------------------------------------------------------------------------
        # Class embedding for the first fractal level
        if self.fractal_level == 0:
            self.num_classes = class_num
            self.class_emb = nn.Embedding(class_num, embed_dim_list[0])
            self.label_drop_prob = label_drop_prob
            self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            torch.nn.init.normal_(self.class_emb.weight, std=0.02)
            torch.nn.init.normal_(self.fake_latent, std=0.02)

        # --------------------------------------------------------------------------
        # Generator for the current level
        if generator_type_list[fractal_level] == "ar":
            generator = AR
        elif generator_type_list[fractal_level] == "mar":
            generator = MAR
        else:
            raise NotImplementedError
        self.generator = generator(
            seq_len=(img_size_list[fractal_level] // img_size_list[fractal_level+1]) ** 2,
            patch_size=img_size_list[fractal_level+1],
            cond_embed_dim=embed_dim_list[fractal_level-1] if fractal_level > 0 else embed_dim_list[0],
            embed_dim=embed_dim_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=guiding_pixel if fractal_level > 0 else False,
            num_conds=num_conds,
            grad_checkpointing=grad_checkpointing,
        )

        # --------------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                label_drop_prob=label_drop_prob,
                class_num=class_num,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                guiding_pixel=guiding_pixel,
                num_conds=num_conds,
                r_weight=r_weight,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level+1
            )
        else:
            # The final fractal level uses PixelLoss.
            self.next_fractal = PixelLoss(
                c_channels=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level+1],
                width=embed_dim_list[fractal_level+1],
                num_heads=num_heads_list[fractal_level+1],
                r_weight=r_weight,
            )

    def forward(self, imgs, cond_list):
        """
        Forward pass to get loss recursively.
        """
        if self.fractal_level == 0:
            # Compute class embedding conditions.
            class_embedding = self.class_emb(cond_list)
            if self.training:
                # Randomly drop labels according to label_drop_prob.
                drop_latent_mask = (torch.rand(cond_list.size(0)) < self.label_drop_prob).unsqueeze(-1).to(class_embedding.device).to(class_embedding.dtype)
                class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
            else:
                # For evaluation (unconditional NLL), use a constant mask.
                drop_latent_mask = torch.ones(cond_list.size(0)).unsqueeze(-1).to(class_embedding.device).to(class_embedding.dtype)
                class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
            cond_list = [class_embedding for _ in range(5)]

        # Get image patches and conditions for the next level
        imgs, cond_list, guiding_pixel_loss = self.generator(imgs, cond_list)
        # Compute loss recursively from the next fractal level.
        loss = self.next_fractal(imgs, cond_list)
        return loss + guiding_pixel_loss

    def sample(self, cond_list, num_iter_list, cfg, cfg_schedule, temperature, filter_threshold, fractal_level):
        """
        Generate samples recursively.
        """
        if fractal_level < self.num_fractal_levels - 2:
            next_level_sample_function = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                cfg_schedule="constant",
                fractal_level=fractal_level + 1
            )
        else:
            next_level_sample_function = self.next_fractal.sample

        # Recursively sample using the current generator.
        return self.generator.sample(
            cond_list, num_iter_list[fractal_level], cfg, cfg_schedule,
            temperature, filter_threshold, next_level_sample_function
        )
