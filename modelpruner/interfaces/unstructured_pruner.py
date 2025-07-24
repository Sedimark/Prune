"""
Unstructured (Fine-Grained) Pruning Interface

Wraps the fine-grained pruning logic, using dynamic sparsity ratios per layer.

Functions:
    apply_unstructured_pruning: Applies unstructured pruning to a model based on learned ratios.

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""

from modelpruner.pruning.unstructured import get_pruning_ratios, FineGrainedPruner

def apply_unstructured_pruning(model):
    """
    Applies unstructured (fine-grained) pruning to the model using dynamic sparsity ratios.

    Args:
        model (nn.Module): The PyTorch model to prune.
        cfg (dict): Configuration dictionary (currently unused but kept for future extensibility).

    Returns:
        nn.Module: The pruned model.
    """
    sparsity_dict = get_pruning_ratios(model)
    pruner = FineGrainedPruner(model, sparsity_dict)
    pruner.apply(model)
    return model

