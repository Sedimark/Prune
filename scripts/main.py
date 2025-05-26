"""
Main pruning pipeline using configuration.

Steps:
1. Load a model from configuration.
2. Inspect and log model details.
3. Apply unstructured pruning.
4. Save the pruned model.

Author: Sneha H - Surrey
Maintainer: Sneha H - Surrey
"""

import argparse
import logging
import yaml
import torch

from modelpruner.interfaces import model_loader, model_inspector, unstructured_pruner

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(cfg_path: str):
    """
    Executes the model pruning pipeline from configuration.

    Args:
        cfg_path (str): Path to YAML configuration file.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    logger.info("Loaded configuration.")

    # Step 1: Load model
    model = model_loader.load_model(cfg['model'])
    logger.info("Model loaded.")

    # Step 2: Inspect model
    info = model_inspector.get_model_info(model)
    logger.info("Original Model Info:\n%s", info)

    # Step 3: Apply unstructured pruning
    pruned_model = unstructured_pruner.apply_unstructured_pruning(model, cfg['unstructured'])
    logger.info("Pruning applied.")

    # Step 4: Inspect pruned model
    pruned_info = model_inspector.get_pruned_model_info(model, pruned_model)
    logger.info("Pruned Model Info:\n%s", pruned_info)

    # Step 5: Save pruned model
    torch.save(pruned_model.state_dict(), cfg['output_path'])
    logger.info("Pruned model saved to %s", cfg['output_path'])


def parse_args():
    parser = argparse.ArgumentParser(description="Run model pruning pipeline.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/pruning_config.yaml",
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
