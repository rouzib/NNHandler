# AutoSaver Feature

The automatic saving functionality, integrated within the `NNHandler`, provides a mechanism to automatically save the complete handler state at regular intervals during the training process.

## Overview

Training deep learning models can be time-consuming. Unexpected interruptions (e.g., crashes, power outages) can lead to the loss of significant training progress. The auto-save feature mitigates this risk by periodically saving the complete state of the `NNHandler`, including:

*   Model weights (unwrapped)
*   Optimizer state
*   Scheduler state
*   Training history (losses, metrics - aggregated on Rank 0)
*   EMA state (if used)
*   Gradient Scaler state (if AMP used)
*   Callback states (if implemented by callbacks)
*   Handler configuration (model class/kwargs, optimizer class/kwargs, etc.)
*   SDE/Sampler state (if applicable)

In Distributed Data Parallel (DDP) mode, the save operation is performed **only by Rank 0** to ensure consistency and avoid race conditions. Other ranks wait at a barrier during the save operation.

## Enabling and Configuring Auto-Save

The auto-save feature is configured using the `auto_save` method of the `NNHandler` instance *before* starting the training process. The configuration is applied on all ranks for consistency, but saving only occurs on Rank 0.

```python
def auto_save(self, interval: Optional[int], save_path: str = '.', 
              name: str = "model_epoch{epoch:02d}", overwrite: bool = False)
```

### Parameters:

*   `interval` (Optional[int]):
    *   The frequency (in epochs) at which to save the handler state.
    *   If set to `None` or `0`, auto-saving is disabled.
    *   An interval of `5` means a checkpoint is saved after epoch 5, 10, 15, etc.
    *   An interval of `-1` means save only on the very last epoch of the `train` call.
*   `save_path` (str, default='.'):
    *   The directory where the automatic checkpoints will be saved. Rank 0 will create this directory if it doesn't exist.
*   `name` (str, default="model_epoch{epoch:02d}"):
    *   The base name format for the checkpoint files. It uses Python's f-string formatting.
    *   The `logs` dictionary from the epoch end (containing aggregated metrics like `loss`, `val_loss`, `accuracy`, etc., plus `epoch`) is available for formatting.
    *   Example: `"ckpt_ep{epoch:03d}_valloss{val_loss:.4f}"` would produce filenames like `ckpt_ep010_valloss0.1234.pth` (if `val_loss` is in logs).
    *   The `.pth` extension is automatically appended if not present in the formatted name.
*   `overwrite` (bool, default=False):
    *   If `False`, a new checkpoint file is created according to the `name` format for each save interval.
    *   If `True`, the checkpoint file is overwritten at each save interval. Rank 0 will attempt to remove the previously auto-saved file before saving the new one. This saves disk space but only keeps the latest auto-saved checkpoint.

## Usage Example

```python
from src.nn_handler import NNHandler
# from your_project.models import YourModel
# from your_project.loss import your_loss_fn
import torch

# Dummy Model for example
class YourModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.layer = torch.nn.Linear(10,1)
    def forward(self, x): return self.layer(x)
def your_loss_fn(y, t): return torch.mean((y-t)**2)
dummy_ds = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))

# Initialize the handler (DDP auto-detected if launched via torchrun)
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu",
    logger_mode=NNHandler.LoggingMode.CONSOLE
)

# Configure optimizer, loss, data loaders...
handler.set_optimizer(torch.optim.Adam, lr=1e-4)
handler.set_loss_fn(your_loss_fn)
handler.set_train_loader(dummy_ds, batch_size=16)

# Enable auto-saving every 10 epochs in the 'checkpoints' directory
handler.auto_save(
    interval=10, 
    save_path='checkpoints', 
    name='my_model_autosave_ep{epoch:03d}', # Format example
    overwrite=False 
)

# Start training - checkpoints will be saved automatically by Rank 0
# handler.train(epochs=100)
print("Auto-save configured. Run handler.train(epochs=...) to start training.")
```

In this example (if training runs), checkpoints like `checkpoints/my_model_autosave_ep010.pth`, `checkpoints/my_model_autosave_ep020.pth`, etc., would be created by Rank 0 during training.

## Resuming from Auto-Saved Checkpoints

You can resume training from any auto-saved checkpoint using the standard `NNHandler.load` class method. This should be done on all ranks if running in DDP mode.

```python
from src.nn_handler import NNHandler
import torch

checkpoint_path = "checkpoints/my_model_autosave_ep050.pth" # Example path

# Load the handler state from the checkpoint
# All ranks load the state, mapping tensors to their local device
loaded_handler = NNHandler.load(
    path=checkpoint_path,
    device="cuda" if torch.cuda.is_available() else "cpu" 
    # Optional: skip parts if needed, e.g., skip_optimizer=True
)

# Optionally adjust training parameters (e.g., learning rate) if needed
# loaded_handler.set_scheduler(...) 

# Resume training from the loaded state (e.g., starting from epoch 51)
print(f"Resuming training from loaded state (Epoch {len(loaded_handler.train_losses)})...")
# loaded_handler.train(epochs=100) # Training will continue
```

Remember to launch the resuming script using the same DDP configuration (e.g., the same number of processes via `torchrun`) as the original training run if applicable.