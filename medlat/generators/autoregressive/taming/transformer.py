import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ['GPT', 'GPT_B', 'GPT_L', 'GPT_H']

logger = logging.getLogger(__name__)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop, n_unmasked=0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

        mask = torch.tril(torch.ones(block_size, block_size))
        if n_unmasked > 0:
            mask[:n_unmasked, :n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=(layer_past is None),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop, n_unmasked=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop, n_unmasked)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present:
            assert not self.training
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, self.block_size, attn_pdrop, resid_pdrop, n_unmasked) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        # Initialize with slightly higher temperature and moderate top-k
        self.default_temperature = 1.0
        self.default_top_k = 50

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: {t}  {self.block_size}"
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def sample(self, bsz, x=None, cond=None, steps=None, temperature=None, sample=True, top_k=None, debug=True):
        """Sample from the model with optional conditioning.
        Args:
            bsz: Batch size
            x: Optional starting sequence (B, T)
            cond: Optional conditioning sequence (B, T_cond)
            steps: Number of tokens to generate
            temperature: Sampling temperature (default: self.default_temperature)
            sample: Whether to sample or take argmax (default: True)
            top_k: Optional top-k filtering (default: self.default_top_k)
            debug: Whether to print debugging information
        """
        self.eval()
        
        # Use default parameters if not specified
        temperature = temperature if temperature is not None else self.default_temperature
        top_k = top_k if top_k is not None else self.default_top_k
        
        # Set default steps if not specified
        if steps is None:
            steps = self.block_size - 1  # -1 for sos token

        # Initialize sequence
        if x is None and cond is None:
            x = torch.full((bsz, 1), fill_value=self.sos_token, dtype=torch.long, device=self.pos_emb.device)
        elif x is None and cond is not None:
            x = cond
        elif cond is not None:
            x = torch.cat((cond, x), dim=1)

        if debug:
            logger.debug(f"Sampling Configuration: temperature={temperature}, top_k={top_k}, "
                         f"initial_shape={x.shape}, steps={steps}")

        # Generate tokens
        for step in range(steps):
            # Ensure we don't exceed block size
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
            
            # Get next token prediction
            logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :] / temperature

            if debug and step == 0:
                logger.debug(f"Step {step} logits — range: [{logits.min().item():.2f}, {logits.max().item():.2f}], "
                             f"mean: {logits.mean().item():.2f}, std: {logits.std().item():.2f}")

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            if debug and step == 0:
                top_probs, top_tokens = torch.topk(probs[0], k=5)
                logger.debug("Top 5 token probabilities: %s",
                             {t.item(): round(p.item(), 4) for p, t in zip(top_probs, top_tokens)})

            # Sample from the distribution
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            # Append new token
            x = torch.cat((x, ix), dim=1)

        return x

    @torch.no_grad()
    def sample_with_past(self, x, steps, temperature=1.0, sample=False, top_k=None):
        """Sample from the model using cached key/value pairs for faster generation.
        This method is more efficient for generating long sequences as it avoids
        recomputing the entire sequence's attention for each new token.
        
        Args:
            x: Starting sequence (B, T)
            steps: Number of tokens to generate
            temperature: Sampling temperature
            sample: Whether to sample or take argmax
            top_k: Optional top-k filtering
        """
        self.eval()
        block_size = self.get_block_size()
        
        for k in range(steps):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            # Run through each transformer block, collecting the past
            presents = []
            x_emb = self.tok_emb(x_cond)
            x_pos = x_emb + self.pos_emb[:, :x_emb.shape[1], :]
            x_in = self.drop(x_pos)
            
            # Manual forward through blocks to collect presents
            for block in self.blocks:
                x_in = block.ln1(x_in)
                attn, present = block.attn(x_in)
                presents.append(present)
                x_in = x_in + attn
                x_in = x_in + block.mlp(block.ln2(x_in))
            
            x_in = self.ln_f(x_in)
            logits = self.head(x_in)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
                
            probs = F.softmax(logits, dim=-1)
            
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
                
            x = torch.cat((x, ix), dim=1)
            
        return x[:, 1:]  # remove SOS token

    def forward_with_past(self, x, past=None):
        """Forward pass that can use cached past key/value pairs.
        Args:
            x: Input tokens (B, T)
            past: Optional tuple of past key/value pairs
        """
        token_embeddings = self.tok_emb(x)
        t = token_embeddings.shape[1]
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: {t} > {self.block_size}"
        
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        presents = []
        if past is None:
            past = [None] * len(self.blocks)
            
        for i, (block, past_i) in enumerate(zip(self.blocks, past)):
            x = block.ln1(x)
            attn, present = block.attn(x, past_i)
            presents.append(present)
            x = x + attn
            x = x + block.mlp(block.ln2(x))
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, presents

    def top_k_logits(self, logits, k):
        """Apply top-k filtering to logits."""
        return top_k_logits(logits, k)

def GPT_B(**kwargs):
    model = GPT(
        n_embd=768, n_layer=12, n_head=12, **kwargs)
    return model

def GPT_L(**kwargs):
    model = GPT(
        n_embd=1024, n_layer=16, n_head=16, **kwargs)
    return model

def GPT_H(**kwargs):
    model = GPT(
        n_embd=1280, n_layer=20, n_head=16, **kwargs)
    return model

class CodeGPT(GPT):
    def __init__(self, vocab_size, block_size, in_channels, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__(vocab_size, block_size, n_layer, n_head, n_embd,
                         embd_pdrop, resid_pdrop, attn_pdrop, n_unmasked)
        self.tok_emb = nn.Linear(in_channels, n_embd)  # override with linear embedding

    def forward(self, idx, embeddings=None, targets=None):
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.drop(token_embeddings + self.pos_emb[:, :t, :])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
