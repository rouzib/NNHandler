# Neural Network Handler (nn_handler)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add license -->
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) <!-- Optional: Add Python version -->
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/) <!-- Optional: Add PyTorch version -->

The `nn_handler` repository provides a comprehensive and flexible Python framework designed to streamline the development, training, evaluation, and management of PyTorch neural network models. It aims to abstract away boilerplate code, allowing researchers and developers to focus on model architecture and experimentation.

NNHandler offers a unified interface supporting:
*   Standard training and validation loops.
*   Advanced features like Automatic Mixed Precision (AMP), gradient accumulation, and Exponential Moving Average (EMA).
*   Seamless integration with **Distributed Data Parallel (DDP)** for multi-GPU and multi-node training.
*   A rich, extensible **callback system** for monitoring, checkpointing, visualization, and custom logic.
*   Built-in support for **generative models**, including score-based models (SDEs) and custom samplers.
*   Comprehensive **model saving and loading**, including full training state resumption.
*   Integrated logging and metric tracking with plotting capabilities.
*   Support for `torch.compile` for potential performance boosts.

## Installation

### Using `pip`

```bash
pip install --upgrade git+https://github.com/rouzib/NNHandler.git
```

### Alternative method

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rouzib/NNHandler.git
    cd NNHandler # Or your repository's root directory
    ```

2.  **Install dependencies:** Ensure you have Python 3.10+ and PyTorch 1.10+ installed. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For specific features like AMP, EMA, or plotting, ensure the corresponding libraries (`torch_ema`, `matplotlib`, `tqdm`) are installed.*

## Getting Started

For a quick introduction and basic usage examples, please refer to the [Getting Started Guide](doc/getting_started.md).

## Core Components

The framework revolves around the central `NNHandler` class and its supporting modules:

*   **`NNHandler` Class ([nn_handler/nn_handler.md](doc/nn_handler/nn_handler.md))**: The main orchestrator for model training, evaluation, and management. Implemented in `nn_handler_distributed.py`.
*   **Distributed Training ([nn_handler/distributed.md](doc/nn_handler/distributed.md))**: Details on how `NNHandler` integrates with PyTorch DDP.
*   **AutoSaver ([nn_handler/autosaver.md](doc/nn_handler/autosaver.md))**: Functionality for automatic checkpoint saving during training (integrated within `NNHandler`).
*   **Sampler ([nn_handler/sampler.md](doc/nn_handler/sampler.md))**: Support for custom sampling algorithms via the `Sampler` base class.
*   **Callbacks ([nn_handler/callbacks/README.md](doc/nn_handler/callbacks/README.md))**: A powerful system for customizing the training loop with various hooks.
*   **Utilities ([nn_handler/utils/README.md](doc/nn_handler/utils/README.md))**: A collection of utility functions and classes, including DDP utilities for working with PyTorch's Distributed Data Parallel functionality.

## Documentation Structure

*   [Getting Started](doc/getting_started.md): Quick-start tutorial.
*   [NNHandler Module](doc/nn_handler/README.md): Overview of the core module.
    *   [NNHandler Class](doc/nn_handler/nn_handler.md): Detailed API reference for `NNHandler`.
    *   [Distributed Training (DDP)](doc/nn_handler/distributed.md): Guide to using DDP features.
    *   [AutoSaver Feature](doc/nn_handler/autosaver.md): Auto-saving configuration.
    *   [Sampler Integration](doc/nn_handler/sampler.md): Using custom samplers.
    *   [Callbacks System](doc/nn_handler/callbacks/README.md): Introduction to callbacks and available implementations.
    *   [Utilities](doc/nn_handler/utils/README.md): Documentation for utility functions and classes, including DDP utilities.

## Basic Usage Example

```python
import torch
from src.nn_handler import NNHandler # Assuming src layout
# from your_model_file import YourModel, your_loss_fn
# from your_dataset_file import your_train_dataset, your_val_dataset

# Dummy components for illustration
class YourModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.linear = torch.nn.Linear(10, 1)
    def forward(self, x): return self.linear(x)
def your_loss_fn(pred, target): return torch.nn.functional.mse_loss(pred, target)
your_train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
your_val_dataset = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 1))

# --- Main NNHandler Workflow ---

# 1. Initialize NNHandler (DDP auto-detected if environment is set)
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu", # DDP will assign specific cuda device
    logger_mode=NNHandler.LoggingMode.CONSOLE, # Log to console (Rank 0 only)
    model_type=NNHandler.ModelType.REGRESSION # Specify model type
    # Add model_kwargs if needed: hidden_units=128
)

# 2. Configure Components
handler.set_optimizer(torch.optim.Adam, lr=1e-3)
handler.set_loss_fn(your_loss_fn)
handler.set_train_loader(your_train_dataset, batch_size=16)
handler.set_val_loader(your_val_dataset, batch_size=16)

# Optional: Add Metrics & Callbacks
# def your_metric(pred, target): return torch.abs(pred - target).mean().item()
# handler.add_metric("mae", your_metric)
# from src.nn_handler.callbacks import ModelCheckpoint
# handler.add_callback(ModelCheckpoint(filepath="models/best_model.pth", monitor="val_loss"))

# 3. Train
handler.train(
    epochs=10,
    validate_every=1,
    use_amp=True, # Enable AMP if available and on CUDA
    gradient_accumulation_steps=2, # Example: Accumulate gradients
    ema_decay=0.99 # Example: Use EMA
)

# 4. Save Final State (Rank 0 saves)
handler.save("models/final_handler_state.pth")

# 5. Load and Predict
# To run prediction/loading, ensure the environment matches (e.g., DDP or single process)
# loaded_handler = NNHandler.load("models/final_handler_state.pth")
# predictions = loaded_handler.predict(some_data_loader) # Predict gathers on Rank 0
```
*(Remember to run DDP examples using `torchrun`)*
## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
