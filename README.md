> 📄 This repository supports the paper "**Weight Statistics Aware Network Pruning**" (submitted to IEEE CSCN 2025).  
> Further documentation, code cleanup, and reproducibility scripts will be added shortly.

 ## Related Repositories

This repository provides a generic, model-agnostic pruning framework. For specific use cases, see the following repositories where this framework is applied:

- 🔬 **DOVER-Mobile (VQA Task)** – [https://github.com/sneha-h/Efficient-DOVER-Mobile](https://github.com/sneha-h/Efficient-DOVER-Mobile)
- 📈 **CrossFormer (Time-Series Forecasting Task)** – [https://github.com/Sedimark/Crossformer/tree/feature_pruning](https://github.com/Sedimark/Crossformer/tree/feature_pruning)


# ModelPruner

## What is ModelPruner?

ModelPruner is a lightweight, modular Python framework designed to efficiently apply **structured** (channel) and **unstructured** pruning techniques to deep learning models, primarily PyTorch-based. It aims to reduce model size and computational cost while preserving accuracy and performance.

While initially applied and tested on transformer-based models like Crossformer for multivariate time series forecasting, ModelPruner is designed to be **model-agnostic** and easily extensible to support a wide range of architectures.

---

## Key Features

* Structured pruning (channel pruning) and unstructured pruning support
* Config-driven pipeline for loading, inspecting, pruning, and saving models
* Support for complex models with minimal code change
* Integrated model inspection and profiling tools
* Lightweight and easy to extend
* Compatible with PyTorch models and Lightning training framework

---

## Installation

You can clone the repository and install ModelPruner in editable mode:

```bash
git clone git@github.com:Sedimark/Prune.git
cd Prune
conda create --name modelpruner_env python=3.9
conda activate modelpruner_env
pip install -e .
```

## Project Structure

```text
.
├── modelpruner 
│   ├── interfaces
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── model_inspector.py
│   │   ├── channel_pruner.py
│   │   └── unstructured_pruner.py
├── pruning
│   │   ├── __init__.py
│   │   ├── channel.py
│   │   ├── unstructured.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── model_profiling.py
│   └── __init__.py
├── scripts
│   ├── main.py
│   └── main_prune.py
├── tests
│   ├── __init__.py
│   └── test_pruning.py
├── config
│   └── pruning_config.yaml
├── LICENSE
├── README.md
├── pyproject.toml
└── setup.py
```

---

## Getting Started

ModelPruner uses a **config-driven pipeline** to control the pruning process. The config file defines your model parameters, pruning method, and other settings.

### Example Configuration (`pruning_config.yaml`)

```yaml
model:
  type: "torch"
  path: "original_models/model.pth"
  class_path: "crossformer.model.crossformer.Crossformer"
  model_args:
    data_dim: 8
    in_len: 24
    out_len: 24
    seg_len: 2
    window_size: 4
    factor: 10
    model_dim: 256
    feedforward_dim: 512
    head_num: 4
    layer_num: 6
    dropout: 0.2
    baseline: false

unstructured:
  method: "magnitude"
  thresholds:
    default: 0.5

channels:
  prune_ratio: 0.5

output_path: "pruned_models/pruned_model.pth"
```

You can modify this config for your own model and pruning strategy.

---

### Running the Pruning Pipeline

Use the provided main script as an example of running the full pruning pipeline:

```bash
python scripts/main_prune.py --config config/pruning_config.yaml
```

This will:

* Load the model using your config
* Inspect and profile the model
* Apply unstructured and/or channel pruning as specified
* Save the pruned model to the specified path

---

### Code Usage Example

```python
from modelpruner.interfaces import model_loader, model_inspector, unstructured_pruner
import yaml

def run_pipeline(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = model_loader.load_model(cfg["model"])
    info = model_inspector.get_model_info(model)
    print("Original Model Info:", info)

    pruned_model = unstructured_pruner.apply_unstructured_pruning(model, cfg["unstructured"])
    pruned_info = model_inspector.get_pruned_model_info(model, pruned_model)
    print("Pruned Model Info:", pruned_info)

    torch.save(pruned_model.state_dict(), cfg["output_path"])
```

---

## Testing

Tests are located in the `tests` folder and can be run with:

```bash
pytest tests/test_pruning_pipeline.py
```

## Acknowledgement


This software has been developed by the [University of Surrey](https://www.surrey.ac.uk/) under the [SEDIMARK(SEcure Decentralised Intelligent Data MARKetplace)](https://sedimark.eu/) project. SEDIMARK is funded by the European Union under the Horizon Europe framework programme [grant no. 101070074]. This project is also partly funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant no. 10043699].
