# Checkpointing Module

The checkpointing module in the `nn_handler` package provides functionality for saving, loading, and managing model states during training and inference.

## Overview

Effective model checkpointing is crucial for deep learning workflows. The checkpointing module offers a comprehensive set of tools for:

- Saving complete model states to files
- Loading model states for resuming training or inference
- Automatically saving checkpoints during training
- Managing model state across distributed training environments

## Components

The checkpointing module consists of several submodules, each handling specific aspects of the checkpointing process:

### [AutoSaver](../autosaver.md)

The AutoSaver component provides automatic saving functionality during training. It allows you to configure:
- Save intervals (every N epochs)
- Custom filename formats based on metrics
- Overwrite options to manage disk space
- Model code saving for reproducibility

### [Loading](loading.md)

The loading module provides functions for loading saved model states:
- `load()`: Loads a complete NNHandler state from a file
- `initialize_from_checkpoint()`: Initializes a new NNHandler with only model weights

### [Saving](saving.md)

The saving module provides functions for saving model states:
- `save_single_file()`: Saves the complete state to a single file
- `save_multi_files()`: Saves components to separate files
- `save_multi_from_single()`: Converts a single file checkpoint to multiple files

### [Trainer Saving](trainer_saving.md)

The trainer_saving module handles the logic for automatic saving during training:
- Determines when to save based on configured intervals
- Formats filenames using current metrics
- Manages previous checkpoints when overwrite is enabled

## Usage in Distributed Training

All checkpointing components are designed to work seamlessly in distributed training environments:
- Save operations are typically performed only by Rank 0
- Load operations are performed by all ranks, with tensors mapped to the appropriate devices
- Barriers ensure synchronization across processes during save/load operations

## Integration with NNHandler

The checkpointing functionality is integrated into the NNHandler class through methods like:
- `handler.save(path)`: Saves the complete handler state
- `NNHandler.load(path)`: Loads a handler state from a file
- `handler.auto_save(interval, save_path, name, overwrite)`: Configures automatic saving

## Example Workflow

A typical workflow using the checkpointing module might look like:

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# 1. Initialize the handler
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Configure the handler
handler.set_optimizer(torch.optim.Adam, lr=0.001)
handler.set_loss_fn(torch.nn.CrossEntropyLoss())
handler.set_train_loader(train_dataset, batch_size=32)
handler.set_val_loader(val_dataset, batch_size=32)

# 3. Configure auto-saving
handler.auto_save(
    interval=5,                          # Save every 5 epochs
    save_path="checkpoints",             # Directory to save checkpoints
    name="model_epoch{epoch}_valloss{val_loss:.4f}",  # Filename format
    overwrite=False                      # Keep all checkpoints
)

# 4. Train the model
handler.train(epochs=30, validate_every=1)

# 5. Save the final model state
handler.save("models/final_model.pth")

# Later: Load the model for inference
loaded_handler = NNHandler.load(
    path="models/final_model.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

For more detailed information on each component, refer to the individual documentation pages linked above.