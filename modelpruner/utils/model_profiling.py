from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
from thop import profile

def profile_model(model: nn.Module, input_shape: tuple) -> None:
    """
    Profiles the FLOPs and Parameters of the given model.

    Args:
        model (nn.Module): The model to be profiled.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, sequence_length, etc.)

    Returns:
        None (Prints the results)
    """
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"Model FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.3f} MParams")

def get_num_parameters(model: nn.Module, count_nonzero_only: bool = False) -> int:
    """
    Calculate the total number of parameters of the model.
    
    Args:
        model (nn.Module): The model to analyze.
        count_nonzero_only (bool): If True, only count nonzero weights.

    Returns:
        int: Number of parameters.
    """
    return sum(param.count_nonzero() if count_nonzero_only else param.numel() for param in model.parameters())

def get_model_size(model: nn.Module, data_width: int = 32, count_nonzero_only: bool = False) -> int:
    """
    Calculate the model size in bits.
    
    Args:
        model (nn.Module): The model to analyze.
        data_width (int): Number of bits per element.
        count_nonzero_only (bool): If True, only count nonzero weights.

    Returns:
        int: Model size in bits.
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def get_model_size_parameters(model: nn.Module) -> Tuple[int, float]:
    """
    Compute the total number of parameters and model size in MB.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        tuple[int, float]: (Number of parameters, Model size in MB)
    """
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)

    print(f"Number of Parameters: {num_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

    return num_params, model_size_mb

def get_model_sparsity(model: nn.Module) -> float:
    """
    Calculate the sparsity of the given model.
    
    Sparsity is calculated as:
        sparsity = 1 - (#nonzeros / #elements)
    
    Args:
        model (nn.Module): The model to analyze.

    Returns:
        float: Model sparsity value.
    """
    num_nonzeros = sum(param.count_nonzero() for param in model.parameters())
    num_elements = sum(param.numel() for param in model.parameters())
    return 1 - (num_nonzeros / num_elements)

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