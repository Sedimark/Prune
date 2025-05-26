"""
Model Inspector Module

This module provides tools for analyzing and inspecting PyTorch models. It includes utilities to
fetch model-level statistics such as size, sparsity, and number of parameters, as well as
layer-wise information and summaries.

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""
from typing import Dict
import torch.nn as nn
import pandas as pd
from modelpruner.utils.model_profiling import ( 
    get_model_size,
    get_num_parameters,
    get_model_size_parameters,
    profile_model,
    get_model_sparsity
)

# Memory unit constants
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB



def get_model_info(model: nn.Module) -> Dict[str, str]:
    """
    Returns the initial information of the model including its size, number of parameters, and sparsity.
    
    Args:
        model (nn.Module): The model to analyze.
    
    Returns:
        dict: A dictionary containing the model's size, number of parameters, and sparsity.
    """
    model_size = get_model_size(model)
    num_params = get_num_parameters(model)
    sparsity = get_model_sparsity(model)
    
    return {
        "Model Size (bits)": model_size,
        "Number of Parameters": num_params,
        "Sparsity": sparsity
    }

def get_pruned_model_info(model: nn.Module, pruned_model: nn.Module) -> Dict[str, str]:
    sparse_model_size = get_model_size(pruned_model, count_nonzero_only=True)
    sparse_model_parameters = get_num_parameters(pruned_model, count_nonzero_only=True)
    model_sparsity = get_model_sparsity(model)
    model_size = get_model_size(model)
    print(f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
        model_sparsity - {model_sparsity}")
    print(f"Sparse model has size={sparse_model_size / MiB:.2f} \
        MiB = {sparse_model_size / model_size * 100:.2f}% of dense model size")

    print("with counting zeroes")
    sparse_model_size = get_model_size(pruned_model)
    sparse_model_parameters = get_num_parameters(pruned_model)
    model_sparsity = get_model_sparsity(model)
    print(f"sparse - model size: {sparse_model_size / MiB:.2f}, number of parameters: {sparse_model_parameters} \
        model_sparsity - {model_sparsity}")
    print(f"Sparse model has size={sparse_model_size / MiB:.2f} \
        MiB = {sparse_model_size / model_size * 100:.2f}% of dense model size")
    
    return {
        "Sparse Model Size (bits)": sparse_model_size,
        "Sparse Number of Parameters": sparse_model_parameters,
        "Sparse Model Sparsity": model_sparsity
        }
    
def inspect_model(model: nn.Module) -> pd.DataFrame:
    layers = []
    for name, param in model.named_parameters():
        layers.append({
            "Layer": name,
            "Shape": list(param.shape),
            "Params": param.numel(),
        })
    return pd.DataFrame(layers)


def get_layer_details(state_dict: dict) -> None:
    """
    Prints the statistical details (mean, std, min, max, count) of each layer in the state dictionary.
    
    Args:
        state_dict (dict): The model's state dictionary containing layer parameters.
    """
    layer_stats = []
    for layer_name, param in state_dict.items():
        param_data = param.cpu().numpy()
        layer_stats.append([layer_name, param_data.mean(), param_data.std(), param_data.min(), param_data.max(), param_data.size])
    
    df = pd.DataFrame(layer_stats, columns=["Layer Name", "Mean", "Std Dev", "Min", "Max", "Count"])
    print(df)

def print_layer_details(model: nn.Module) -> None:
    """
    Prints details of each layer in the model.
    
    Args:
        model (nn.Module): The model whose layers are to be printed.
    """
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            print(f"Layer: {name}\nType: {module.__class__.__name__}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"  Weight shape: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  Bias shape: {module.bias.shape}")
            print()
