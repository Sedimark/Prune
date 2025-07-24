import yaml
import torch
from modelpruner.interfaces import model_loader, model_inspector, unstructured_pruner

def test_unstructured_pruning_pipeline():
    with open("config/pruning_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = model_loader.load_model(cfg["model"])
    original_info = model_inspector.get_model_info(model)

    pruned_model = unstructured_pruner.apply_unstructured_pruning(model)
    pruned_info = model_inspector.get_pruned_model_info(model, pruned_model, model)

    print(f"Original Model Info: {original_info} \
        Pruned Model Info: {pruned_info}")
    assert pruned_model is not None
    assert pruned_info["Sparse Number of Parameters"] <= original_info["Number of Parameters"]


def test_config_loading():
    with open("config/pruning_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    assert "model" in cfg
    assert "output_path" in cfg

def test_model_loading():
    '''cfg = {
        "type": "torch",
        "path": "original_models/model.pth",
        "class_path": "crossformer.model.crossformer.Crossformer",
        "model_args": {
            "data_dim": 8,
            "in_len": 24,
            "out_len": 24,
            "seg_len": 2,
            "window_size": 4,
            "factor": 10,
            "model_dim": 256,
            "feedforward_dim": 512,
            "head_num": 4,
            "layer_num": 6,
            "dropout": 0.2,
            "baseline": False
        }
    }
    '''
    cfg = {
        "type": "torch",
        "path": "original_models/model.pth",
    }

    model = model_loader.load_model(cfg)
    assert model is not None


if __name__ == "__main__":
    test_unstructured_pruning_pipeline()
    test_config_loading()
    test_model_loading()
    print("All tests passed.")