import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from timm.models.vision_transformer import DropPath, Mlp

from medlat.modules.pos_embed import get_2d_sincos_pos_embed, to_ntuple

from omegaconf import OmegaConf
import numpy as np
import math
import scipy.stats as stats
from tqdm import tqdm


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).to(probs.device)
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):
    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits



class MaskGIT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, num_tokens=8192, vae_stride=16, in_channels=3,
                 embed_dim=1024, depth=24, num_heads=16, num_classes=1000,
                 dataset_num=None,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.05, mask_ratio_max=0.95, mask_ratio_mu=0.5, mask_ratio_std=0.25,
                 label_smoothing=0.1,
                 dataset_label_drop_prob=0.0,
                 label_drop_prob=0.1,  # Classifier-free guidance: probability of dropping class label
                 seq_len=None,
                 dims=2,
                ):
        super().__init__()

        self.dims = dims
        self.num_classes = num_classes                                             # Number of classes
        if seq_len is None and vae_stride is not None:
            _img_size   = to_ntuple(img_size,   dims)
            _vae_stride = to_ntuple(vae_stride, dims)
            self.seq_len = int(np.prod([i // s for i, s in zip(_img_size, _vae_stride)]))
        elif seq_len is not None:
            self.seq_len = seq_len
        else:
            raise ValueError("Either seq_len or vae_stride must be provided")
        self.codebook_size = num_tokens
        vocab_size = self.codebook_size + num_classes + 1 + 1                       # +1 fake class, +1 mask token
        self.fake_class_label = vocab_size - 2
        self.mask_token_label = vocab_size - 1
        self.label_drop_prob = label_drop_prob
        self.dataset_label_drop_prob = dataset_label_drop_prob

        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=self.seq_len+1,   # seq len + 1 cls token
                                        dropout=0.1)

        # --------------------------------------------------------------------------
        # Dataset ID Embedding (optional)
        self.use_dataset_conditioning = dataset_num is not None
        if self.use_dataset_conditioning:
            self.num_datasets = dataset_num
            self.dataset_emb = nn.Embedding(dataset_num, embed_dim)
            # Fake dataset embedding for unconditional generation
            self.fake_dataset_latent = nn.Parameter(torch.zeros(1, embed_dim))

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MaskGIT encoder specifics
        dropout_rate = 0.1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim)) #learnable

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # initialize dataset embedding if used
        if self.use_dataset_conditioning:
            torch.nn.init.normal_(self.dataset_emb.weight, std=.02)
            torch.nn.init.normal_(self.fake_dataset_latent, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, y=None, dataset_id=None):
        """
        Args:
            x: (bsz, seq_len) token indices
            y: (bsz,) class labels, optional. If None, uses fake_class_label
            dataset_id: (bsz,) dataset IDs, optional. Only used if dataset_num was provided in __init__
        """
        token_indices = x
        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break
            else:
                logger.debug("Rerandom the noise!")
        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label

        # Determine class token labels
        # Class labels are in range [codebook_size, codebook_size + num_classes)
        if y is not None:
            # Apply label dropout for classifier-free guidance during training
            if self.training and self.label_drop_prob > 0:
                drop_mask = torch.rand(bsz, device=x.device) < self.label_drop_prob
                class_labels = torch.where(
                    drop_mask,
                    torch.full_like(y, self.fake_class_label),
                    self.codebook_size + y  # Map class label to vocabulary index
                )
            else:
                class_labels = self.codebook_size + y
        else:
            class_labels = torch.full((bsz,), self.fake_class_label, device=x.device, dtype=torch.long)

        # concate class token
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).to(token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = class_labels
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).to(token_indices.device), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).to(token_indices.device), token_all_mask], dim=1)
        token_indices = token_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(token_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # Add dataset embedding to class token (position 0) if dataset conditioning is enabled
        if self.use_dataset_conditioning:
            zeros_ds = torch.zeros(
                input_embeddings.size(0), 1, input_embeddings.size(-1),
                device=input_embeddings.device, dtype=input_embeddings.dtype
            )
            input_embeddings = torch.cat([zeros_ds, input_embeddings], dim=1)
            drop_prob = self.dataset_label_drop_prob
            if dataset_id is not None:
                if drop_prob > 0:
                    drop_mask = torch.rand(bsz, device=x.device) < drop_prob
                    dataset_embedding = torch.where(
                        drop_mask.unsqueeze(-1),
                        self.fake_dataset_latent.expand(bsz, -1),
                        self.dataset_emb(dataset_id)
                    )
                else:
                    dataset_embedding = self.dataset_emb(dataset_id)
            else:
                dataset_embedding = self.fake_dataset_latent.expand(bsz, -1)

            # Add dataset embedding
            input_embeddings[:, 0] = dataset_embedding
        logger.debug(f"input_embeddings.shape: {input_embeddings.shape}")
        # No dropping needed because we use BEiT style architecture
        # dropping
        # token_keep_mask = 1 - token_drop_mask
        # input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        # apply Transformer blocks
        x = input_embeddings
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        if self.use_dataset_conditioning:
            logits = logits[:, 2:, :self.codebook_size].reshape(bsz*seq_len, -1)
            gt_indices = gt_indices.reshape(bsz*seq_len)
        else:
            logits = logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1)
            gt_indices = gt_indices.reshape(bsz*seq_len)


        loss = self.criterion(logits, gt_indices)
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, y=None, dataset_id=None):
        """
        Args:
            imgs: (bsz, seq_len) token indices
            y: (bsz,) class labels, optional. If provided, uses class conditioning.
            dataset_id: (bsz,) dataset IDs, optional. Only used if dataset_num was provided in __init__
        """
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs, y=y, dataset_id=dataset_id)
        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        logits = self.mlm_layer(latent, word_embeddings)
        loss = self.forward_loss(gt_indices, logits, token_all_mask)
        return loss


    def sample(self, bsz, device, input_token_indices=None, mask=None, y=None, dataset_id=None, num_iter=12, choice_temperature=4.5, verbose=False, cfg=1.0):
        """
        Sample tokens using iterative refinement (MaskGIT sampling).
        
        Supports two modes:
        1. Inpainting: Provide input_token_indices and mask
        2. Generation from scratch: Provide bsz and seq_len (or they will be inferred)
        
        Args:
            input_token_indices: (bsz, seq_len) LongTensor of input tokens, optional.
                If None, generation from scratch mode is used.
            mask: (bsz, seq_len) ByteTensor or BoolTensor where 1 = to inpaint (masked), 0 = keep as is.
                If None and input_token_indices is None, all tokens are masked (generation from scratch).
                If None but input_token_indices is provided, assumes all tokens are masked.
            y: (bsz,) class labels, optional. If None, uses fake_class_label (unconditional)
            dataset_id: (bsz,) dataset IDs, optional. Only used if dataset_num was provided in __init__
            num_iter: number of iterative refinement steps
            choice_temperature: temperature for token selection
            verbose: whether to show progress bar
            cfg: classifier-free guidance scale. If > 1.0, performs CFG by doubling batch size
            bsz: batch size, required if input_token_indices is None
            seq_len: sequence length, required if input_token_indices is None. If None, uses self.seq_len
        """        
        # Determine mode and initialize accordingly
        if input_token_indices is None:
            # Generation from scratch mode
            # Initialize all tokens as masked
            token_indices = torch.full((bsz, self.seq_len), self.mask_token_label, device=device, dtype=torch.long)
            mask = torch.ones((bsz, self.seq_len), device=device, dtype=torch.bool)
        else:
            # Inpainting mode
            bsz, seq_len = input_token_indices.shape
            input_token_indices = input_token_indices.to(device)
            
            if mask is None:
                # If mask not provided, assume all tokens are masked (generation from scratch with given shape)
                mask = torch.ones((bsz, seq_len), device=device, dtype=torch.bool)
            else:
                mask = mask.bool().to(device)
            
            # Initialize token indices: masked positions get mask_token_label, others get given input
            token_indices = torch.where(
                mask,
                torch.full_like(input_token_indices, self.mask_token_label),
                input_token_indices
            )
        bsz, seq_len = token_indices.shape
        use_cfg = cfg > 1.0 and y is not None
        original_bsz = bsz  # Keep track of original batch size
        
        # Prepare class labels
        if y is not None:
            class_labels = (self.codebook_size + y).to(device)  # Map class to vocabulary index
        else:
            class_labels = torch.full((bsz,), self.fake_class_label, device=device, dtype=torch.long)
        
        # Prepare dataset embedding (optional)
        dataset_embedding = None
        if self.use_dataset_conditioning:
            drop_prob = self.dataset_label_drop_prob
            if dataset_id is not None:
                dataset_embedding = self.dataset_emb(dataset_id.to(device))
                if drop_prob > 0:
                    drop_mask = torch.rand(bsz, device=device) < drop_prob
                    dataset_embedding = torch.where(
                        drop_mask.unsqueeze(-1),
                        self.fake_dataset_latent.expand(bsz, -1).to(device),
                        dataset_embedding
                    )
            else:
                dataset_embedding = self.fake_dataset_latent.expand(bsz, -1).to(device)
        
        # Classifier-free guidance: duplicate batch ONCE before the loop
        if use_cfg:
            token_indices = torch.cat([token_indices, token_indices], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            class_labels = torch.cat([
                class_labels,
                torch.full((bsz,), self.fake_class_label, device=device, dtype=torch.long)
            ], dim=0)
            if dataset_embedding is not None:
                dataset_embedding = torch.cat([
                    dataset_embedding,
                    self.fake_dataset_latent.expand(bsz, -1).to(device)
                ], dim=0)
            bsz = bsz * 2

        _CONFIDENCE_OF_KNOWN_TOKENS = float("inf")

        for step in tqdm(range(num_iter), disable=not verbose):
            # 2. Prepend class token
            token_indices_with_cls = torch.cat(
                [class_labels.unsqueeze(1), token_indices], dim=1
            )

            # 3. Token embedding → transformer → norm
            input_embeddings = self.token_emb(token_indices_with_cls)
            
            # Add dataset embedding to class token (position 0) if dataset conditioning is enabled
            if dataset_embedding is not None:
                zeros_ds = torch.zeros(
                    input_embeddings.size(0), 1, input_embeddings.size(-1),
                    device=input_embeddings.device, dtype=input_embeddings.dtype
                )
                input_embeddings = torch.cat([zeros_ds, input_embeddings], dim=1)
                input_embeddings[:, 0] = input_embeddings[:, 0] + dataset_embedding
            
            x = input_embeddings
            for blk in self.blocks:
                x = blk(x)
            latent = self.norm(x)

            # 4. Remove class token from output
            if self.use_dataset_conditioning:
                latent = latent[:, 2:, :]  # [bsz, seq_len, hidden_dim]
            else:
                latent = latent[:, 1:, :]  # [bsz, seq_len, hidden_dim]

            # 5. Get logits from MLM head
            word_embeddings = self.token_emb.word_embeddings.weight.detach()
            logits = self.mlm_layer(latent, word_embeddings)
            logits = logits[:, :, :self.codebook_size]  # [bsz, seq_len, vocab_size]

            # 6. Apply classifier-free guidance if enabled
            if use_cfg:
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                # CFG: logits = uncond_logits + cfg * (cond_logits - uncond_logits)
                logits = uncond_logits + cfg * (cond_logits - uncond_logits)
                # After CFG, logits have original batch size, so reduce mask, token_indices, and dataset_embedding to match
                mask = mask[:original_bsz]
                token_indices = token_indices[:original_bsz]
                if dataset_embedding is not None:
                    dataset_embedding = dataset_embedding[:original_bsz]
                bsz = original_bsz

            # 7. Sample tokens
            sample_dist = torch.distributions.Categorical(logits=logits)
            sampled_ids = sample_dist.sample()  # [bsz, seq_len] (original batch size after CFG)

            # 8. Keep known tokens fixed (using original batch size mask and token_indices)
            sampled_ids = torch.where(mask, sampled_ids, token_indices)

            # 9. Compute mask ratio for next iteration (cosine schedule)
            ratio = (step + 1) / num_iter
            mask_ratio = np.cos(np.pi / 2. * ratio)
            mask_ratio = torch.tensor(mask_ratio, device=device)

            # 10. Confidence scores for predicted tokens
            probs = torch.nn.functional.softmax(logits, dim=-1)
            selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)

            # 11. Force known tokens to have high confidence so they're never remasked
            selected_probs = torch.where(mask, selected_probs, torch.full_like(selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS))

            # 12. Determine how many tokens to re-mask
            unknown_counts = mask.sum(dim=1, keepdim=True)  # how many masked tokens per example
            mask_len = torch.floor(self.seq_len * mask_ratio).to(device)
            mask_len = torch.clamp(mask_len, min=1)
            mask_len = torch.min(mask_len, unknown_counts - 1)
            mask_len = torch.clamp(mask_len, min=1)

            # 13. Create new mask for next iteration (low-confidence tokens get re-masked)
            masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
            # Combine with original mask so we don't touch known tokens
            new_mask = torch.where(mask, masking, torch.zeros_like(masking, dtype=torch.bool))

            # 14. Update token indices: re-mask low-confidence tokens
            token_indices = torch.where(new_mask, self.mask_token_label, sampled_ids)

            mask = new_mask  # update for next round
            
            # 15. If CFG was used, duplicate for next iteration (after all computations with original batch size)
            if use_cfg:
                token_indices = torch.cat([token_indices, token_indices], dim=0)
                mask = torch.cat([mask, mask], dim=0)
                if dataset_embedding is not None:
                    dataset_embedding = torch.cat([
                        dataset_embedding[:original_bsz],
                        self.fake_dataset_latent.expand(original_bsz, -1).to(device)
                    ], dim=0)
                bsz = bsz * 2
        
        # If CFG was used, return only the first half (conditional samples)
        if use_cfg:
            sampled_ids = sampled_ids[:original_bsz]
        
        return sampled_ids



