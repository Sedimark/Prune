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
        model_args = cfg['model_args']
        model_path = cfg['path']
        class_path = cfg['class_path']

        # Dynamically import the model class
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)

        # Attempt to load model from state_dict
        try:
            print("[INFO] Trying to load model from state_dict...")
            model = ModelClass(**model_args)
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            print("[INFO] Model loaded successfully from state_dict.")
            return model

        except Exception as e:
            print(f"[WARNING] Failed to load state_dict: {e}")
            print("[INFO] Falling back to full model load using safe_globals...")

        # Fallback: load full model using trusted class
        try:
            with safe_globals([ModelClass]):
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                print("[INFO] Full model loaded successfully with safe_globals.")
                return model

        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model even with safe_globals: {e}")

    elif model_type == 'lightning':
        print("[INFO] Loading PyTorch Lightning model...")
        return LightningModule.load_from_checkpoint(cfg['path'])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
