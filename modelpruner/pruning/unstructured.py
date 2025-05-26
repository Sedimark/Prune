"""
Implements unstructured (fine-grained) pruning 
using magnitude-based pruning and CoV-based heuristics.
"""

from typing import Dict

import pandas as pd
import torch
import torch.nn.utils.prune as prune
from torch import nn

def calculate_pruning_ratio(mean: float, std_dev: float) -> tuple:
    """
    Compute the Coefficient of Variation (CoV) and return the pruning ratio based on thresholds.
    
    Args:
        mean (float): Mean of the layer parameters.
        std_dev (float): Standard deviation of the layer parameters.
    
    Returns:
        tuple: (CoV, pruning ratio)
    """
    cov = float('inf') if abs(mean) == 0 else std_dev / abs(mean)
    if cov > 600:
        return cov, 0.7
    elif 150 < cov <= 600:
        return cov, 0.5
    return cov, 0.3

def get_pruning_ratios(model: nn.Module) -> dict:
    """
    Compute and return the pruning ratio for each weight layer in the model.
    
    Args:
        model (nn.Module): The model to analyze.
    
    Returns:
        dict: Dictionary mapping layer names to their pruning ratios.
    """
    state_dict = model.state_dict()
    pruning_ratios = {}
    layer_stats = []
    
    for layer_name, param in state_dict.items():
        if 'weight' not in layer_name:
            continue
        param_data = param.cpu().numpy()
        mean, std_dev = param_data.mean(), param_data.std()
        min_val, max_val, count = param_data.min(), param_data.max(), param_data.size
        cov, pruning_ratio = calculate_pruning_ratio(mean, std_dev)
        pruning_ratios[layer_name] = pruning_ratio
        layer_stats.append([layer_name, mean, std_dev, min_val, max_val, count, cov, pruning_ratio])
    
    df = pd.DataFrame(layer_stats, columns=["Layer Name", "Mean", "Std Dev", "Min", "Max", "Count", "CoV", "Pruning Ratio"])
    #print(df)
    return pruning_ratios

def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Applies magnitude-based fine-grained pruning to a given tensor.

    :param tensor: torch.Tensor, weight tensor of a layer (conv/fc).
    :param sparsity: float, pruning sparsity ratio (0 to 1).
                     sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return: torch.Tensor, binary mask (1 for retained weights, 0 for pruned weights).
    """
    # Ensure sparsity is within a valid range
    sparsity = min(max(0.0, sparsity), 1.0)

    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)

    if sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(sparsity * num_elements)

    # Compute importance (absolute values of weights)
    importance = torch.abs(tensor)

    # Determine threshold for pruning
    if num_zeros > 0:
        threshold, _ = torch.kthvalue(importance.view(-1), num_zeros)
    else:
        threshold = torch.tensor(0.0, device=tensor.device)

    # Generate binary mask (1 for retained, 0 for pruned)
    mask = torch.gt(importance, threshold).float()

    # Apply pruning mask
    tensor.mul_(mask)

    return mask


class FineGrainedPruner:
    """
    Fine-grained pruning class for applying weight pruning to a model.

    Attributes:
        masks (dict): Dictionary storing pruning masks for each layer.
    """

    def __init__(self, model: torch.nn.Module, sparsity_dict: Dict[str, float]):
        """
        Initializes the FineGrainedPruner and applies pruning.

        :param model: torch.nn.Module, the model to be pruned.
        :param sparsity_dict: Dict[str, float], dictionary mapping layer names to sparsity levels.
        """
        self.masks = self.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module):
        """
        Applies stored pruning masks to the model.

        :param model: torch.nn.Module, the model to apply pruning masks to.
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model: torch.nn.Module, sparsity_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Applies pruning to specified layers of the model.

        :param model: torch.nn.Module, the model to prune.
        :param sparsity_dict: Dict[str, float], dictionary specifying sparsity per layer.
        :return: Dict[str, torch.Tensor], dictionary of pruning masks.
        """
        masks = {}
        for name, param in model.named_parameters():
            if name in sparsity_dict and param.dim() > 1:
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

    @staticmethod
    @torch.no_grad()
    def create_sparse_model(model: torch.nn.Module):
        """
        Deletes zeroed parameters to create a smaller model.

        :param model: torch.nn.Module, the pruned model.
        """
        for name, param in model.named_parameters():
            if torch.all(param.data == 0):
                print(f"Removing zeroed parameter: {name}")
                del param

    @staticmethod
    @torch.no_grad()
    def remove_pruning(model: torch.nn.Module):
        """
        Removes pruning reparameterization, making the pruning permanent.

        :param model: torch.nn.Module, the pruned model.
        """
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.remove(module, "weight")
                if hasattr(module, "bias") and module.bias is not None:
                    prune.remove(module, "bias")