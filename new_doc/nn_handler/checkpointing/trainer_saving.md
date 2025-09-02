# Trainer Saving Module

The trainer_saving module in the `nn_handler.checkpointing` package provides functionality for automatically saving model states during the training process.

## Overview

During long training runs, it's important to periodically save the model state to:
- Prevent data loss in case of unexpected interruptions
- Track model progress over time
- Create checkpoints for later analysis
- Enable resuming training from specific points

The trainer_saving module handles the logic for determining when to save the model state during training, based on configurable intervals and conditions.

## Functions

### `auto_save_epoch`

```python
@on_rank(0, barrier=True)
def auto_save_epoch(nn_handler: 'NNHandler', epoch: int, total_epochs: int, 
                    save_on_last_epoch: bool, logs: Dict[str, Any])
```

Handles the logic for auto-saving the model state during training. This function is only executed on rank 0 in a distributed training environment, with a barrier to ensure synchronization across processes.

#### Parameters:

* `nn_handler` (NNHandler):
  * The NNHandler instance containing the model and state to be saved.
* `epoch` (int):
  * The current epoch number (0-based).
* `total_epochs` (int):
  * The total number of epochs in the training run.
* `save_on_last_epoch` (bool):
  * Whether to save on the last epoch regardless of the save interval.
* `logs` (Dict[str, Any]):
  * Dictionary containing metrics and other information from the current epoch, used for filename formatting.

#### Returns:

* None

## Behavior

The `auto_save_epoch` function determines whether to save the model state based on the following conditions:

1. **Regular interval saving**: If the save interval is positive (e.g., 5), the model is saved every N epochs (e.g., epochs 5, 10, 15, etc.).
2. **Last epoch saving**: If `save_on_last_epoch` is True, the model is saved on the final epoch regardless of the interval.
3. **Only last epoch saving**: If the save interval is -1, the model is saved only on the last epoch.

When a save is triggered, the function:
1. Formats the filename using the metrics in the `logs` dictionary
2. Saves the complete handler state to the specified path
3. Optionally removes the previously auto-saved file if overwrite is enabled
4. Updates the record of the last saved model path

## Usage

The `auto_save_epoch` function is primarily used internally by the NNHandler's training loop and is not typically called directly by users. Instead, users configure auto-saving through the NNHandler's `auto_save` method before starting training.

### Configuration Example

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# Initialize the handler
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set up the handler with optimizer, loss function, etc.
handler.set_optimizer(torch.optim.Adam, lr=0.001)
handler.set_loss_fn(torch.nn.CrossEntropyLoss())

# Configure auto-saving
handler.auto_save(
    interval=5,                          # Save every 5 epochs
    save_path="checkpoints",             # Directory to save checkpoints
    name="model_epoch{epoch}_loss{loss:.4f}",  # Filename format using metrics
    overwrite=True                       # Only keep the latest auto-saved file
)

# Start training - auto_save_epoch will be called internally after each epoch
handler.train(epochs=30, validate_every=1)
```

## Filename Formatting

The auto-save filename can include placeholders for metrics and epoch numbers using Python's string formatting syntax:

- `{epoch}`: The current epoch number (1-based)
- `{loss}`: The training loss for the current epoch
- `{val_loss}`: The validation loss (if validation is performed)
- Any other metric in the `logs` dictionary (e.g., `{accuracy}`, `{val_accuracy}`)

For example, with the format `"model_epoch{epoch}_valloss{val_loss:.4f}"`, the saved file might be named `model_epoch10_valloss0.3421.pth`.

If a formatting key is not available in the logs (e.g., requesting `{val_loss}` when validation wasn't run), the function falls back to a simpler format like `model_name_epoch10.pth`.

## Notes on Distributed Training

In distributed training environments:

1. The `@on_rank(0, barrier=True)` decorator ensures that:
   - Only rank 0 executes the save operation to prevent file conflicts
   - Other ranks wait at a barrier until the save is complete
   - All ranks continue training together after the save completes

2. The barrier synchronization is important to maintain consistent timing across processes, especially when saving large models.

3. The auto-save configuration should be identical across all ranks to ensure consistent behavior.