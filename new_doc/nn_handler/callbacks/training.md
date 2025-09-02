# Training Control Callbacks

These callbacks are designed to modify or control aspects of the training process itself, beyond simple monitoring or saving.

## `EarlyStopping`

Stops training when a monitored quantity has stopped improving for a specified number of epochs.

**Purpose:** Prevent overfitting by stopping training when performance on a validation metric stops improving.

**Key Parameters:**

*   `monitor` (str, default='val_loss'): Quantity to be monitored.
*   `min_delta` (float, default=0.0): Minimum change in the monitored quantity to qualify as an improvement.
*   `patience` (int, default=10): Number of epochs with no improvement after which training will be stopped.
*   `mode` (str, default='min'): One of {"min", "max"}. Determines whether the goal is to minimize or maximize the monitored quantity.
*   `verbose` (int, default=0): Verbosity mode.
*   `restore_best_weights` (bool, default=False): Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.

**Actual Implementation:**

```python
import copy
import warnings
from typing import Any, Dict, Optional

import torch

from .base import Callback


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): Quantity to be monitored.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        mode (str): One of {"min", "max"}.
        verbose (int): Verbosity mode.
        restore_best_weights (bool): Whether to restore model weights from the epoch
            with the best value of the monitored quantity. If False, the model weights
            obtained at the last step of training are used. Requires the handler state
            to be saved (e.g., via ModelCheckpoint).
    """

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 10,
                 mode: str = 'min', verbose: int = 0, restore_best_weights: bool = False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None  # To store weights if restore_best_weights is True

        if mode not in ['min', 'max']:
            warnings.warn(f"EarlyStopping mode '{mode}' is unknown, fallback to 'min'.", RuntimeWarning)
            mode = 'min'
        self.mode = mode
        if self.mode == 'min':
            self.monitor_op = torch.lt
            self.min_delta *= -1  # Adjust delta for minimization
            self.best = torch.tensor(torch.inf)
        else:
            self.monitor_op = torch.gt
            self.best = torch.tensor(-torch.inf)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.wait = 0  # Reset wait counter at the beginning of training
        self.stopped_epoch = 0
        self.best = torch.tensor(torch.inf) if self.mode == 'min' else torch.tensor(-torch.inf)
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} available!, skipping.", RuntimeWarning)
            return

        current_tensor = torch.tensor(current)
        if self.monitor_op(current_tensor - self.min_delta, self.best):
            self.best = current_tensor
            self.wait = 0
            if self.restore_best_weights:
                # Need deep copy if model can be modified later
                self.best_weights = copy.deepcopy(self.handler.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1  # Store 1-based epoch
                self.handler._stop_training = True  # Signal handler to stop
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print("Restoring model weights from the end of the best epoch.")
                    self.handler.model.load_state_dict(self.best_weights)
                if self.verbose > 0:
                    print(f"Epoch {self.stopped_epoch}: early stopping")

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'best': self.best.item(),
            # Don't save best_weights here, rely on ModelCheckpoint or handler save
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.wait = state_dict.get('wait', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)
        self.best = torch.tensor(state_dict.get('best', self.best.item()))
        # best_weights are restored via handler load or ModelCheckpoint
```

**Usage Examples:**

```python
# Basic early stopping on validation loss
handler.add_callback(EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
))

# Early stopping on accuracy (maximizing) with weight restoration
handler.add_callback(EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=15,
    restore_best_weights=True,
    verbose=1
))

# Early stopping with a minimum improvement threshold
handler.add_callback(EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,  # Require at least 0.001 improvement
    patience=5,
    verbose=1
))
```

## Key Features

1. **Flexible Monitoring**: Can monitor any metric available in the logs, such as 'val_loss', 'val_accuracy', or custom metrics.

2. **Improvement Detection**: Uses the `min_delta` parameter to determine what constitutes a significant improvement, avoiding stopping due to minor fluctuations.

3. **Weight Restoration**: When `restore_best_weights=True`, automatically restores the model to its best state when stopping, rather than using the final weights.

4. **State Serialization**: Implements `state_dict()` and `load_state_dict()` methods, allowing its state to be saved and restored when checkpointing the training process.

## Implementation Notes

- The callback uses PyTorch tensor operations (`torch.lt`, `torch.gt`) for comparing metric values, ensuring consistent behavior with tensor metrics.
- The `min_delta` is adjusted for minimization when `mode='min'` to ensure proper comparison.
- The callback signals the handler to stop training by setting `handler._stop_training = True`.
- Best weights are stored using a deep copy to ensure they aren't modified during subsequent training.
