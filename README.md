# ModelPruner

**ModelPruner** is a lightweight, modular Python framework designed to apply **Weight Statistics Aware Pruning (WSAP)** and support efficient, config-driven pruning pipelines for deep learning models — particularly targeting **AI edge deployments**.

This repository generalizes the pruning methodology developed and validated in two real-world applications:

- **DOVER-Mobile**: Video Quality Assessment (VQA)
- **CrossFormer**: Time-Series Forecasting

The code here abstracts and refactors the original pruning logic from those models, aiming to make it **model-agnostic**, reusable, and extendable for broader use cases.

---

## 🚀 Purpose

- Provide a reusable framework for unstructured pruning using WSAP
- Enable rapid experimentation via YAML-based configuration
- Optimize PyTorch models for inference in edge environments
- Serve as a foundation for structured and hybrid pruning extensions

---

## ⚙️ Prerequisites / Dependencies

Requirements are defined in `pyproject.toml`. Main dependencies:

```toml
requires-python = ">=3.9"

dependencies = [
  "torch>=1.9",
  "pytorch_lightning",
  "einops",
  "pandas",
  "onnx",
  "thop"  # For model profiling / FLOP counting
]
```

To install:

```bash
pip install -e .
```

---

## 🧩 Installation

```bash
git clone https://github.com/Sedimark/Prune.git
cd Prune
conda create --name modelpruner_env python=3.9
conda activate modelpruner_env
pip install -e .
```

---

## 📁 Project Structure

```text
.
├── modelpruner/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── model_loader.py         # Load full models from .pth
│   │   ├── model_inspector.py      # Print/model info, FLOPs, params
│   │   ├── channel_pruner.py       # Placeholder for structured pruning interface
│   │   └── unstructured_pruner.py  # Core WSAP unstructured pruning interface
│   │
│   ├── pruning/
│   │   ├── __init__.py
│   │   ├── channel.py              # (Placeholder) Channel pruning logic (to be extended)
│   │   └── unstructured.py         # WSAP implementation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_profiling.py      # FLOPs and model inspection utilities
│
├── scripts/
│   ├── main.py                     # Main pruning runner (WSAP)
│
├── config/
│   └── pruning_config.yaml         # Sample config with model/input/output paths
│
├── tests/
│   ├── __init__.py
│   └── test_pruning.py             # Unit test for pruning validation
│
├── LICENSE
├── README.md
├── pyproject.toml                  # Project metadata and dependencies
└── setup.py                        # Install script for legacy compatibility
```

---

## 🔧 Configuration Parameters

### Example (`config/pruning_config.yaml`):

```yaml
model:
  type: "torch"
  path: "original_models/model.pth"

output_path: "pruned_models/pruned_model.pth"
```

| Key            | Description |
|----------------|-------------|
| `model.path`   | Full model file saved with `torch.save(model)` |
| `output_path`  | Where the pruned model is saved |

---

## 🔄 Running the Pipeline

```bash
python scripts/main.py --config config/pruning_config.yaml
```

This will:
- Load the model
- Analyze and profile it
- Apply WSAP-based pruning
- Save the pruned model

---

## 🧠 Code Integration Example

```python
from modelpruner.interfaces import model_loader, model_inspector, unstructured_pruner
import yaml
import torch

def run_pipeline(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = model_loader.load_model(cfg["model"])
    model_inspector.get_model_info(model)

    pruned_model = unstructured_pruner.apply_unstructured_pruning(model)
    model_inspector.get_pruned_model_info(model, pruned_model)

    torch.save(pruned_model, cfg["output_path"])
```

---

## ✅ Test Suite

Run:

```bash
python tests/test_pruning_pipeline.py
```

Test cases validate:
- Model loading
- WSAP pruning logic
- Pruned model structure

---

## 🐞 Debug / Console Messages

Console outputs include:
- Pre- and post-pruning model stats (params, FLOPs, model size)
- Layer-wise sparsity if profiling is added
- Save path confirmation

---

## 📌 Best Practices

### Saving Your Model

Ensure you use:
```python
torch.save(model, "original_models/model.pth")
```

### Output Location

```python
torch.save(pruned_model, "pruned_models/pruned_model.pth")
```

> After pruning, finetune your model to restore accuracy.

### Evaluating Results

Use `model_inspector` logs to compare pre/post model stats:
- Parameters
- FLOPs
- Model Size
- Layer-specific sparsity 

---

## 📚 From Paper to General Tooling

This framework is the **generalized version** of the pruning method from:

> 📄 "*Weight Statistics Aware Network Pruning*" (submitted to IEEE CSCN 2025)

Originally implemented for:

- 🎥 **DOVER-Mobile (VQA)**: [Efficient-DOVER-Mobile](https://github.com/sneha-h/Efficient-DOVER-Mobile)
- 📈 **CrossFormer (Forecasting)**: [CrossFormer - feature_pruning](https://github.com/Sedimark/Crossformer/tree/feature_pruning)

This repo abstracts the pruning logic into a **clean, model-agnostic framework**, to help reuse it for **other models and domains**.

**Note:** Channel pruning shown in the paper is implemented in the task-specific repos above. This repository currently supports only unstructured WSAP pruning.

---

## 🎓 Acknowledgement

This software has been developed by the [University of Surrey](https://www.surrey.ac.uk/) under the [SEDIMARK (SEcure Decentralised Intelligent Data MARKetplace)](https://sedimark.eu/) project. SEDIMARK is funded by the European Union under the Horizon Europe framework programme [grant no. 101070074]. This project is also partly funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant no. 10043699].