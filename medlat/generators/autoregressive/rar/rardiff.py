import logging
from functools import partial

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from medlat.diffusion.diffloss import DiffLoss
from .modules import Block, Transformer, modulate
import math
from medlat.modules.in_and_out import PatchEmbed, ToPixel


__all__ = ["RAR", "RAR_B", "RAR_L", "RAR_XL", "RAR_H"]


class FinalLayer(nn.Module):
    """final layer with adaptive layer normalization."""

    def __init__(self, in_features) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(in_features, 2 * in_features))

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return x


class RAR(nn.Module):
    """decoder-only autoregressive diffusion model."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        depth=12,
        width=768,
        heads=12,
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        # diffloss parameters
        noise_schedule="cosine",
        diffloss_d=3,
        diffloss_w=1024,
        diffusion_batch_mul=4,
        # sampling parameters
        num_sampling_steps="100",
        grad_checkpointing=False,
        force_one_d_seq=False,
        order="raster",
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.token_channels = token_channels
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.grad_checkpointing = grad_checkpointing
        self.force_one_d_seq = force_one_d_seq
        self.order = order

        # sequence dimensions
        self.grid_size = [img_size // tokenizer_patch_size // patch_size for (img_size, tokenizer_patch_size, patch_size) in zip(img_size, tokenizer_patch_size, patch_size)]
        self.seq_len = math.prod(self.grid_size) + 1  # +1 for BOS token
        self.token_embed_dim = token_channels * math.prod(patch_size)

        self.patch_embed = PatchEmbed(to_embed='identity', img_size=img_size, patch_size=patch_size, in_chans=token_channels, embed_dim=token_channels)
        self.to_pixel = ToPixel(to_pixel='identity', img_size=img_size, out_channels=token_channels, in_dim=token_channels, patch_size=tokenizer_patch_size)

        if force_one_d_seq:
            self.seq_len = force_one_d_seq + 1

        # model architecture configuration
        self.depth = depth
        self.width = width
        self.heads = heads

        self.label_drop_prob = label_drop_prob

        scale = width**-0.5

        # class and null token embeddings
        self.class_emb = nn.Embedding(self.num_classes, self.width)
        self.fake_latent = nn.Parameter(scale * torch.randn(1, self.width))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, self.width))

        # input and positional embeddings
        self.x_embedder = nn.Linear(self.token_embed_dim, self.width)
        self.pos_embed = nn.Parameter(scale * torch.randn((1, self.seq_len, self.width)))
        self.target_pos_embed = nn.Parameter(scale * torch.randn((1, self.seq_len - 1, self.width)))
        self.timesteps_embeddings = nn.Parameter(scale * torch.randn((1, self.seq_len, self.width)))

        # training mask for causal attention
        self.train_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool))

        # --------------------------------------------------------------------------
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_pre = norm_layer(self.width)
        self.transformer = Transformer(
            self.width,
            self.depth,
            self.heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=norm_layer,
            force_causal=True,
            grad_checkpointing=self.grad_checkpointing,
        )
        self.final_layer = FinalLayer(self.width)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=self.width,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
            noise_schedule=noise_schedule,
            diffusion_batch_mul=diffusion_batch_mul,
        )

    def initialize_weights(self):
        """initialize model weights."""
        # parameter initialization
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.bos_token, std=0.02)
        torch.nn.init.normal_(self.target_pos_embed, std=0.02)
        torch.nn.init.normal_(self.timesteps_embeddings, std=0.02)
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)

        # apply standard initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """standard weight initialization for layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

        # zero-out adaptive modulation layers
        for block in self.transformer.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out final layer modulation
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

    def enable_kv_cache(self):
        for block in self.transformer.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.transformer.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def get_random_orders(self, x):
        """generate random token ordering."""
        batch_size = x.shape[0]
        random_noise = torch.randn(batch_size, self.seq_len - 1, device=x.device)
        shuffled_orders = torch.argsort(random_noise, dim=1)
        return shuffled_orders

    def get_raster_orders(self, x):
        """generate raster (sequential) token ordering."""
        batch_size = x.shape[0]
        raster_orders = torch.arange(self.seq_len - 1, device=x.device)
        shuffled_orders = torch.stack([raster_orders for _ in range(batch_size)])
        return shuffled_orders

    def shuffle(self, x, orders):
        """shuffle tokens according to given orders."""
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        """unshuffle tokens to restore original ordering."""
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def forward_transformer(self, x, class_embedding, orders=None):
        """forward pass through the transformer."""
        x = self.x_embedder(x)
        bsz = x.shape[0]

        # add BOS token
        bos_token = self.bos_token.expand(bsz, 1, -1)
        x = torch.cat([bos_token, x], dim=1)
        current_seq_len = x.shape[1]

        # add positional embeddings
        pos_embed = self.pos_embed.expand(bsz, -1, -1)
        if orders is not None:
            pos_embed = torch.cat([pos_embed[:, :1], self.shuffle(pos_embed[:, 1:], orders)], dim=1)
        x = x + pos_embed[:, :current_seq_len]

        # add target positional embeddings
        target_pos_embed = self.target_pos_embed.expand(bsz, -1, -1)
        embed_dim = target_pos_embed.shape[-1]
        if orders is not None:
            target_pos_embed = self.shuffle(target_pos_embed, orders)
        target_pos_embed = torch.cat([target_pos_embed, torch.zeros(bsz, 1, embed_dim).to(x.device)], dim=1)
        x = x + target_pos_embed[:, :current_seq_len]

        x = self.ln_pre(x)

        # prepare condition tokens
        condition_token = class_embedding.repeat(1, current_seq_len, 1)
        timestep_embed = self.timesteps_embeddings.expand(bsz, -1, -1)
        condition_token = condition_token + timestep_embed[:, :current_seq_len]

        # handle kv cache for inference
        if self.transformer.blocks[0].attn.kv_cache:
            x = x[:, -1:]
            condition_token = condition_token[:, -1:]

        # transformer forward pass
        for block in self.transformer.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, None, None, condition_token)
            else:
                x = block(x, condition=condition_token)

        x = self.final_layer(x, condition=class_embedding)
        return x

    def forward_loss(self, z, target):
        """compute diffusion loss."""
        return self.diffloss(z=z, target=target)

    def forward(self, x, labels):
        logger.debug("forward x.shape=%s", x.shape)
        """forward pass for training."""
        # get token ordering
        if self.order == "raster":
            orders = self.get_raster_orders(x)
        elif self.order == "random":
            orders = self.get_random_orders(x)
        else:
            raise NotImplementedError(f"Order '{self.order}' not implemented")

        # prepare class embeddings
        class_embedding = self.class_emb(labels)
        if self.training:
            # randomly drop class embedding during training
            drop_mask = torch.rand(x.shape[0]) < self.label_drop_prob
            drop_mask = drop_mask.unsqueeze(-1).to(x.device).to(x.dtype)
            class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding
        class_embedding = class_embedding.unsqueeze(1)

        # prepare input tokens
        x = self.patch_embed(x) if not self.force_one_d_seq else x
        logger.debug("patchify x.shape=%s", x.shape)
        x = self.shuffle(x, orders)
        gt_latents = x.clone().detach()
        # forward pass and loss computation
        z = self.forward_transformer(x[:, :-1], class_embedding, orders=orders)
        return self.forward_loss(z=z, target=gt_latents)

    def sample(
        self,
        bsz,
        device,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
        kv_cache=False,
    ):
        """sample tokens autoregressively."""
        tokens = torch.zeros(bsz, 0, self.token_embed_dim, device=device)
        indices = list(range(self.seq_len - 1))

        # setup kv cache if requested
        if kv_cache:
            self.enable_kv_cache()

        if progress:
            indices = tqdm(indices)

        # get token ordering
        if self.order == "raster":
            orders = self.get_raster_orders(torch.zeros(bsz, self.seq_len - 1, self.token_embed_dim, device=device))
        elif self.order == "random":
            orders = self.get_random_orders(torch.zeros(bsz, self.seq_len - 1, self.token_embed_dim, device=device))
        else:
            raise NotImplementedError(f"Order '{self.order}' not implemented")

        # prepare for classifier-free guidance
        if cfg != 1.0:
            orders = torch.cat([orders, orders], dim=0)

        # generate tokens step by step
        for step in indices:
            cur_tokens = tokens.clone()

            # prepare class embeddings and CFG
            cls_embd = self.fake_latent.repeat(bsz, 1) if labels is None else self.class_emb(labels)

            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                cls_embd = torch.cat([cls_embd, self.fake_latent.repeat(bsz, 1)], dim=0)
            cls_embd = cls_embd.unsqueeze(1)
            z = self.forward_transformer(tokens, cls_embd, orders=orders)[:, -1]

            # apply CFG schedule
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * step / len(indices)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError(f"CFG schedule '{cfg_schedule}' not implemented")

            # sample next token
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)

            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)

            cur_tokens = torch.cat([cur_tokens, sampled_token_latent.unsqueeze(1)], dim=1)
            tokens = cur_tokens.clone()

        # cleanup
        if kv_cache:
            self.disable_kv_cache()

        if cfg != 1.0:
            orders, _ = orders.chunk(2, dim=0)

        # restore original ordering and convert back to image format
        tokens = self.unshuffle(tokens, orders)
        if not self.force_one_d_seq:
            tokens = self.to_pixel(tokens)

        return tokens

    def generate(self, n_samples, cfg, labels, args):
        """generate samples using the model."""
        return self.sample_tokens(
            n_samples,
            cfg=cfg,
            labels=labels,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            progress=True,
            kv_cache=False,
        )


# model size variants
def RAR_B(**kwargs):
    return RAR(width=768, depth=12, heads=12, **kwargs)


def RAR_L(**kwargs):
    return RAR(width=1024, depth=24, heads=16, **kwargs)


def RAR_XL(**kwargs):
    return RAR(width=1152, depth=28, heads=16, **kwargs)


def RAR_H(**kwargs):
    return RAR(width=1280, depth=32, heads=16, **kwargs)
