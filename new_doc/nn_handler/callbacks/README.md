# Callbacks System

The `nn_handler` framework includes a flexible callback system that allows you to hook into various stages of the training and evaluation process to execute custom logic.

## Overview

Callbacks provide a powerful mechanism for:

*   Monitoring training progress (e.g., logging metrics, displaying progress bars).
*   Saving model checkpoints (periodically, based on performance, best only).
*   Implementing early stopping based on metric performance.
*   Adjusting hyperparameters during training (e.g., learning rate adjustments beyond standard schedulers).
*   Visualizing model predictions or internal states.
*   Integrating with experiment tracking tools (e.g., TensorBoard, WandB - potentially via custom callbacks).
*   Performing custom validation logic.

## The Base `Callback` Class

All callbacks must inherit from the base `Callback` class defined in `callbacks/base.py`.

```python
# From src/nn_handler/callbacks/base.py
import abc

class Callback(abc.ABC):
    """Abstract base class used to build new callbacks.

    Properties:
        handler: The NNHandler instance this callback is registered with.
    """
    def __init__(self):
        self.handler = None # Set by NNHandler upon registration

    def set_handler(self, handler):
        self.handler = handler

    # --- Training Hooks ---
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_train_batch_begin(self, batch, logs=None): pass
    def on_train_batch_end(self, batch, logs=None): pass

    # --- Validation Hooks ---
    def on_val_begin(self, logs=None): pass
    def on_val_end(self, logs=None): pass
    def on_val_batch_begin(self, batch, logs=None): pass
    def on_val_batch_end(self, batch, logs=None): pass

    # --- State Management (Optional but recommended) ---
    def state_dict(self) -> dict:
        # Return internal state to be saved
        return {}

    def load_state_dict(self, state_dict: dict):
        # Load internal state
        pass
```

You only need to implement the methods corresponding to the events you want to react to.

*   **`logs` dictionary**: The `logs` dictionary passed to most hooks contains metrics and information relevant to the current stage (e.g., `loss`, `val_loss`, custom metrics, `lr`, `epoch`). In DDP mode, logs passed to epoch/train end hooks on Rank 0 contain *aggregated* results from all ranks.
*   **Accessing the Handler**: Within a callback, `self.handler` provides access to the `NNHandler` instance, allowing you to read its state (e.g., `self.handler.model`, `self.handler.optimizer`, `self.handler.train_losses`) or even modify certain aspects (use with caution).
*   **State Management**: Implementing `state_dict` and `load_state_dict` allows your callback's internal state (e.g., best metric value seen so far in `ModelCheckpoint`) to be saved and restored with the main `NNHandler` state via `handler.save()` and `handler.load()`.
*   **DDP Awareness**: Callbacks run on **all** ranks by default. If a callback performs actions that should only happen once globally (like writing files, printing unique logs, saving models), it needs to check the rank: `if self.handler._rank == 0: ...`.

## Adding Callbacks to the Handler

Instantiate your callback class and add it to the handler using `add_callback`:

```python
from src.nn_handler import NNHandler
from src.nn_handler.callbacks import ModelCheckpoint # Example import

handler = NNHandler(...) 

# Instantiate and add callbacks
checkpoint_cb = ModelCheckpoint(filepath="models/best.pth", monitor="val_loss")
handler.add_callback(checkpoint_cb)

# ... configure handler ...
handler.train(...)
```

## Provided Callbacks

This framework likely includes several pre-built callbacks (check the `.md` files in this directory):

*   **[Monitor Callbacks](monitor.md)**: For logging progress and metrics (e.g., `ProgressMonitor`, `EarlyStopping`).
*   **[Saving Callbacks](saving.md)**: For saving model checkpoints (e.g., `ModelCheckpoint`).
*   **[Training Callbacks](training.md)**: For controlling the training process (specific examples depend on implementation).
*   **[Visualization Callbacks](visualisation.md)**: For visualizing results during training (e.g., reconstructing images, plotting attention maps).
*   **[Utility Callbacks](utils.md)**: Helper utilities for callbacks (if applicable).

Refer to the specific documentation for each callback module for details on available classes and their options.

## Creating Custom Callbacks

1.  Create a new Python file (e.g., `my_custom_callbacks.py`).
2.  Define a class inheriting from `src.nn_handler.callbacks.base.Callback`.
3.  Implement the desired `on_*` methods.
4.  Implement `state_dict` and `load_state_dict` if your callback has internal state.
5.  Remember DDP rank checks if performing rank-specific actions.
6.  Import and add an instance of your custom callback to the `NNHandler`.