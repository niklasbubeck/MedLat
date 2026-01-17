from __future__ import annotations

import importlib
from typing import Any, Mapping, Literal

import torch
import torch.nn as nn


def get_model_type(model: nn.Module) -> Literal["continuous", "discrete", "token", "autoregressive", "non-autoregressive"]:
    module_path = model.__class__.__module__
    if "first_stage.continuous" in module_path:
        return "continuous"
    elif "first_stage.discrete" in module_path:
        return "discrete"
    elif "first_stage.token" in module_path:
        return "token"
    elif "generators.autoregressive" in module_path:
        return "autoregressive"
    elif "generators.non_autoregressive" in module_path:
        return "non-autoregressive"
    else:
        raise ValueError(f"Unknown model type: {module_path}")

def init_from_ckpt(model, path: str, weights_only: bool = False) -> None:
    """
    Load a checkpoint into ``model`` while remaining tolerant to shape mismatches.
    """

    def _load_candidate(candidate_key: str) -> dict[str, Any] | None:
        try:
            return torch.load(path, map_location="cpu", weights_only=weights_only)[candidate_key]
        except Exception:
            return None

    state_dict = (
        _load_candidate("state_dict")
        or _load_candidate("model")
        or torch.load(path, map_location="cpu", weights_only=weights_only)
    )

    cleaned = {}
    for key, value in state_dict.items():
        if "loss" in key:
            continue
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value

    try:
        msg = model.load_state_dict(cleaned, strict=True)
    except RuntimeError as err:
        print(f"Warning: {err}")
        print("Falling back to non-strict loading (often happens when mixing 2D/3D weights).")
        msg = model.load_state_dict(cleaned, strict=False)

    torch.cuda.empty_cache()

    print(f"Loading pre-trained {model.__class__.__name__}")
    print("Missing keys:")
    print(msg.missing_keys)
    print("Unexpected keys:")
    print(msg.unexpected_keys)
    print(f"Restored from {path}")


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
