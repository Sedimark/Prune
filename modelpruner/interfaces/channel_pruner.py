"""
Channel Pruning Interface

Provides a wrapper around the core channel pruning logic to apply structured pruning
to a model based on a specified pruning ratio.

Functions:
    apply_channel_pruning: Applies channel pruning to a given model using configuration settings.

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""

from modelpruner.pruning.channel import channel_prune

def apply_channel_pruning(model, cfg: dict):
    """
    Applies channel pruning to the given model using the specified configuration.

    Args:
        model (nn.Module): The PyTorch model to prune.
        cfg (dict): Configuration dictionary with key 'prune_ratio' indicating pruning strength.

    Returns:
        nn.Module: The pruned model.
    """
    pruner = channel_prune(model, cfg['prune_ratio'])
    pruner.apply(model)
    return model
