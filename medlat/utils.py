from __future__ import annotations

import hashlib
import importlib
import logging
import os
import urllib.request
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_type(model: nn.Module) -> Literal["continuous", "discrete", "token", "autoregressive", "non-autoregressive"]:
    from medlat.base import (
        ContinuousFirstStage, DiscreteFirstStage, TokenFirstStage,
        AutoregressiveGenerator, NonAutoregressiveGenerator,
    )
    if isinstance(model, ContinuousFirstStage):
        return "continuous"
    if isinstance(model, DiscreteFirstStage):
        return "discrete"
    if isinstance(model, TokenFirstStage):
        return "token"
    if isinstance(model, AutoregressiveGenerator):
        return "autoregressive"
    if isinstance(model, NonAutoregressiveGenerator):
        return "non-autoregressive"
    # Fallback: legacy string-matching for models not yet migrated
    module_path = model.__class__.__module__
    if "first_stage.continuous" in module_path:
        return "continuous"
    if "first_stage.discrete" in module_path:
        return "discrete"
    if "first_stage.token" in module_path:
        return "token"
    if "generators.autoregressive" in module_path:
        return "autoregressive"
    if "generators.non_autoregressive" in module_path:
        return "non-autoregressive"
    raise ValueError(
        f"Cannot determine model type for {model.__class__.__qualname__}. "
        f"Inherit from one of the base classes in medlat.base."
    )


def _resolve_ckpt_path(path: str) -> str:
    """
    Resolve ``path`` to a local file path. If ``path`` is an http(s) URL, the
    file is downloaded to a cache directory and the local path is returned.
    Works with any direct download link (Hugging Face, Google Drive, etc.).
    """
    if os.path.isfile(path):
        return path
    if not path.startswith(("http://", "https://")):
        return path

    # Download from URL to cache
    parsed = urlparse(path)
    ext = os.path.splitext(parsed.path)[1] or ".bin"
    cache_key = hashlib.sha256(path.encode()).hexdigest()
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "medlat", "downloads")
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, cache_key + ext)

    if os.path.isfile(cached_path):
        return cached_path

    try:
        request = urllib.request.Request(path, headers={"User-Agent": "medlat/1.0"})
        with urllib.request.urlopen(request) as resp:
            with open(cached_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        if os.path.isfile(cached_path):
            try:
                os.remove(cached_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download checkpoint from {path!r}: {e}") from e

    if not os.path.isfile(cached_path):
        raise FileNotFoundError(f"Download completed but file missing: {cached_path!r}")
    return cached_path


def init_from_ckpt(model, path: str, weights_only: bool = False, strict: bool = True) -> None:
    """
    Load a checkpoint into ``model``.
    Supports .ckpt/.pt and .safetensors files. If ``path`` is an http(s) URL, the
    checkpoint is downloaded and cached under ~/.cache/medlat/downloads.

    Args:
        strict: If True (default), all keys must match exactly. If False, missing or
                unexpected keys are tolerated — useful for 2D→3D weight transfer.
                Prefer keeping strict=True; passing strict=False is an explicit opt-in.
    """
    path = _resolve_ckpt_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint path is not an existing file: {path!r}. "
            "Use a local path or an http(s) download URL."
        )

    is_safetensors = "safetensor" in path

    def _load_torch() -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=weights_only)

    def _load_safetensors() -> dict[str, Any]:
        from safetensors.torch import load_file
        return load_file(path, device="cpu")

    def _load_candidate(candidate_key: str) -> dict[str, Any] | None:
        if is_safetensors:
            # safetensors are always flat state_dicts
            return None
        try:
            return _load_torch()[candidate_key]
        except Exception:
            return None

    # --- Load state_dict ---
    if is_safetensors:
        state_dict = _load_safetensors()
    else:
        state_dict = (
            _load_candidate("state_dict")
            or _load_candidate("model")
            or _load_torch()
        )

    # --- Clean keys ---
    cleaned = {}
    for key, value in state_dict.items():
        if "loss" in key:
            continue
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value

    # --- Load into model ---
    if strict:
        msg = model.load_state_dict(cleaned, strict=True)
    else:
        try:
            msg = model.load_state_dict(cleaned, strict=True)
        except RuntimeError as err:
            logger.warning(f"Strict load failed (strict=False was requested, falling back): {err}")
            msg = model.load_state_dict(cleaned, strict=False)

    torch.cuda.empty_cache()

    logger.info(f"Loading pre-trained {model.__class__.__name__}")
    logger.debug("Missing keys: %s", msg.missing_keys)
    logger.debug("Unexpected keys: %s", msg.unexpected_keys)
    logger.info(f"Restored from {path}")


def suggest_generator_params(tokenizer: nn.Module) -> dict:
    """Return the keyword-arguments a compatible generator needs.

    Given an instantiated first-stage tokenizer, this function inspects its
    attributes and returns a dict you can ``**``-unpack directly into
    :func:`~medlat.get_model` when building the paired generator.

    Example::

        tok = get_model("continuous.aekl.f8_d16", img_size=224)
        gen = get_model("dit.xl_2", img_size=224, num_classes=1000,
                        **suggest_generator_params(tok))
        # No need to look up embed_dim / vae_stride manually.

    Returns:
        A dict that may contain any subset of the following keys:
        ``vae_stride`` (int, spatial compression factor, e.g. 8),
        ``in_channels`` (int, latent channel count for diffusion / continuous-AR generators),
        ``codebook_size`` (int, for AR generators that use this param name),
        ``num_tokens`` (int, alias for ``codebook_size`` used by some AR models).
    """
    fs_type = get_model_type(tokenizer)
    params: dict = {}

    if hasattr(tokenizer, "vae_stride"):
        stride = tokenizer.vae_stride
        params["vae_stride"] = stride[0] if isinstance(stride, (tuple, list)) else stride

    if fs_type in ("continuous", "token"):
        if hasattr(tokenizer, "embed_dim"):
            params["in_channels"] = tokenizer.embed_dim
    elif fs_type == "discrete":
        if hasattr(tokenizer, "embed_dim"):
            params["in_channels"] = tokenizer.embed_dim
        if hasattr(tokenizer, "n_embed"):
            params["codebook_size"] = tokenizer.n_embed
            params["num_tokens"] = tokenizer.n_embed

    return params


def validate_compatibility(first_stage: nn.Module, generator: nn.Module) -> None:
    """
    Validate that a first-stage tokenizer and a generator are compatible for use with GenWrapper.

    Checks:
    - continuous/discrete + non-autoregressive: tokenizer.embed_dim must match generator.in_channels
    - continuous + autoregressive: tokenizer.embed_dim must match generator.in_channels
    - discrete + autoregressive: tokenizer.n_embed must match generator.codebook_size
    - when both models expose vae_stride: the values must match

    Raises:
        ValueError: with a descriptive message listing all detected mismatches.
    """
    fs_type = get_model_type(first_stage)
    gen_type = get_model_type(generator)

    errors: list[str] = []

    # ── channel / codebook size checks ──────────────────────────────────────
    if gen_type == "non-autoregressive":
        embed_dim = getattr(first_stage, "embed_dim", None)
        in_channels = getattr(generator, "in_channels", None)
        if embed_dim is not None and in_channels is not None and embed_dim != in_channels:
            errors.append(
                f"tokenizer.embed_dim={embed_dim} != generator.in_channels={in_channels}; "
                "pass in_channels=tokenizer.embed_dim when building the generator"
            )

    elif gen_type == "autoregressive" and fs_type == "continuous":
        embed_dim = getattr(first_stage, "embed_dim", None)
        in_channels = getattr(generator, "in_channels", None)
        if embed_dim is not None and in_channels is not None and embed_dim != in_channels:
            errors.append(
                f"tokenizer.embed_dim={embed_dim} != generator.in_channels={in_channels}; "
                "pass in_channels=tokenizer.embed_dim when building the generator"
            )

    elif gen_type == "autoregressive" and fs_type == "discrete":
        n_embed = getattr(first_stage, "n_embed", None)
        codebook_size = getattr(generator, "codebook_size", None)
        if n_embed is not None and codebook_size is not None and n_embed != codebook_size:
            errors.append(
                f"tokenizer.n_embed={n_embed} != generator.codebook_size={codebook_size}; "
                "pass num_tokens=tokenizer.n_embed when building the generator"
            )

    # ── vae_stride check (when both sides expose it) ─────────────────────────
    tok_stride = getattr(first_stage, "vae_stride", None)
    gen_stride = getattr(generator, "vae_stride", None)
    if tok_stride is not None and gen_stride is not None:
        # Normalise tuples to a single representative value for comparison
        def _scalar(s):
            return s[0] if isinstance(s, (tuple, list)) else s
        if _scalar(tok_stride) != _scalar(gen_stride):
            errors.append(
                f"tokenizer.vae_stride={tok_stride} != generator.vae_stride={gen_stride}; "
                "pass vae_stride=tokenizer.vae_stride when building the generator"
            )

    if errors:
        raise ValueError(
            f"Incompatible tokenizer ({type(first_stage).__name__}) and "
            f"generator ({type(generator).__name__}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    """
    Instantiate an object from a Hydra/OmegaConf-style configuration dict.

    The dictionary must contain a ``_target_`` entry with the fully qualified
    import path to the callable. Any additional key/value pairs are forwarded
    as keyword arguments.
    """

    if "_target_" not in config:
        raise KeyError("Configuration dictionary must contain a '_target_' entry.")

    target = config["_target_"]
    module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attr_name)
    kwargs = {k: v for k, v in config.items() if k != "_target_"}
    return callable_obj(**kwargs)
