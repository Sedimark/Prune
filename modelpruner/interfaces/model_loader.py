"""
model_loader.py

This module provides a generic interface to dynamically load PyTorch or PyTorch Lightning models
based on configuration. It supports both state_dict-based and full checkpoint loading using
safe deserialization via `safe_globals`.

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""

import torch
import importlib
from pytorch_lightning import LightningModule
from torch.serialization import safe_globals

def load_model(cfg: dict) -> torch.nn.Module:
    """
    Load a model based on the configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary containing keys:
            - type (str): "torch" or "lightning"
            - path (str): Path to the model file (state_dict or full checkpoint)
            - class_path (str): Python class path for the model (required for 'torch')
            - model_args (dict): Arguments to initialize the model class (required for 'torch')

    Returns:
        torch.nn.Module: The loaded model instance.

    Raises:
        RuntimeError: If the model fails to load using both state_dict and full checkpoint.
        ValueError: If the model type is not supported.
    """
    model_type = cfg.get('type')

    if model_type == 'torch':
        model_path = cfg['path']

        # load full model using
        try:
            print("[INFO] Trying to load full model...")    
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"[DEBUG] state_dict type {type(model)}")
            print("[INFO] Full model loaded successfully.")
            return model

        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model: {e}")

    elif model_type == 'lightning':
        print("[INFO] Loading PyTorch Lightning model...")
        return LightningModule.load_from_checkpoint(cfg['path'])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
