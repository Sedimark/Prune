"""
Implements channel pruning logic for 
Linear and LayerNorm layers in PyTorch models.
"""

import copy
from typing import List, Union

import torch
import torch.nn as nn


@torch.no_grad()
def channel_prune(model: nn.Module, prune_ratio: Union[List[float], float]) -> nn.Module:
    """
    Applies channel pruning to each layer in the model, including `nn.Parameter` layers.

    Args:
        model (nn.Module): The PyTorch model to be pruned.
        prune_ratio (Union[List[float], float]): The pruning ratio(s). 
            - If a float, the same ratio is applied to all layers.
            - If a list, it must match the number of prunable layers.

    Returns:
        nn.Module: A new pruned model.
    """
    if not isinstance(prune_ratio, (float, list)):
        raise TypeError("prune_ratio must be a float or a list of floats.")

    # Count the number of layers
    layers = [m for m in model.modules()]
    n_layers = len(layers) - 1  # Exclude the root model module

    # Convert single float prune_ratio to a list
    if isinstance(prune_ratio, float):
        prune_ratio = [prune_ratio] * n_layers
    elif len(prune_ratio) != n_layers:
        raise ValueError("Prune ratio list length must match the number of layers.")

    # Create a deep copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)

    # Get all named layers
    all_named_layers = list(pruned_model.named_modules())

    # Define dimension mappings for pruning
    dim_mappings = {
        256: 176,
        512: 352,
        1024: 704,
    }

    # Iterate through layers and apply pruning
    for i, p_ratio in enumerate(prune_ratio):
        layer_name, layer = all_named_layers[i]
        _, next_layer = all_named_layers[i + 1] if i + 1 < len(all_named_layers) else (None, None)

        # Prune Linear layers
        if isinstance(layer, nn.Linear):
            _prune_linear_layer(layer, next_layer, dim_mappings)

        # Prune LayerNorm layers
        elif isinstance(layer, nn.LayerNorm):
            _prune_layernorm_layer(layer, next_layer, dim_mappings)

    # Handle `nn.Parameter` layers (enc_pos, dec_pos_embedding, router)
    _prune_named_parameters(pruned_model, dim_mappings)

    return pruned_model


def _prune_linear_layer(layer: nn.Linear, next_layer: nn.Module, dim_mappings: dict) -> None:
    """Applies pruning to a Linear layer and updates the subsequent layer if necessary."""
    original_out_features = layer.out_features
    original_in_features = layer.in_features

    # Apply dimension mappings
    new_out_features = dim_mappings.get(original_out_features, original_out_features)
    new_in_features = dim_mappings.get(original_in_features, original_in_features)

    # Update the current Linear layer
    layer.out_features = new_out_features
    layer.in_features = new_in_features
    layer.weight = nn.Parameter(layer.weight[:new_out_features, :new_in_features])
    if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias[:new_out_features])

    # Update the next layer if it is a Linear layer
    if isinstance(next_layer, nn.Linear):
        next_in_features = next_layer.in_features
        new_next_in_features = dim_mappings.get(next_in_features, next_in_features)
        next_layer.in_features = new_next_in_features
        next_layer.weight = nn.Parameter(next_layer.weight[:, :new_next_in_features])


def _prune_layernorm_layer(layer: nn.LayerNorm, next_layer: nn.Module, dim_mappings: dict) -> None:
    """Applies pruning to a LayerNorm layer and updates the subsequent Linear layer if necessary."""
    original_normalized_shape = layer.normalized_shape[0]

    # Apply dimension mappings
    new_normalized_shape = dim_mappings.get(original_normalized_shape, original_normalized_shape)

    # Update the current LayerNorm layer
    layer.normalized_shape = (new_normalized_shape,)
    layer.weight = nn.Parameter(layer.weight[:new_normalized_shape])
    if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias[:new_normalized_shape])

    # Update the next layer if it is a Linear layer
    if isinstance(next_layer, nn.Linear):
        next_in_features = next_layer.in_features
        new_next_in_features = dim_mappings.get(next_in_features, next_in_features)
        next_layer.in_features = new_next_in_features
        next_layer.weight = nn.Parameter(next_layer.weight[:, :new_next_in_features])


def _prune_named_parameters(model: nn.Module, dim_mappings: dict) -> None:
    """Prunes specific nn.Parameter tensors (e.g., enc_pos, dec_pos_embedding, router) in the model."""
    for name, param in model.named_parameters():
        if any(key in name for key in ["enc_pos", "dec_pos_embedding", "router"]):
            original_shape = param.shape
            if len(original_shape) > 1:  # Only resize multi-dimensional parameters
                target_shape = list(original_shape)
                target_shape[-1] = dim_mappings.get(target_shape[-1], target_shape[-1])  # Resize last dimension
                new_param = param[..., :target_shape[-1]]

                # Update the parameter in the correct module
                parts = name.split(".")
                current_module = model
                for part in parts[:-1]:  # Traverse module hierarchy
                    current_module = getattr(current_module, part)
                setattr(current_module, parts[-1], nn.Parameter(new_param))