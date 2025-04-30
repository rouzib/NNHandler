# Base Callback Class

All custom callbacks within the `nn_handler` framework must inherit from the abstract base class `Callback` defined in `src/nn_handler/callbacks/base.py`.

## Class Definition

```python
import abc
# Forward declaration or import if needed
# from src.nn_handler import NNHandler 

class Callback(abc.ABC):
    """Abstract base class used to build new callbacks.

    This class provides the structure and hooks that NNHandler calls during its
    training and evaluation loops.

    Properties:
        handler: The NNHandler instance this callback is registered with. 
                 This is set automatically when the callback is added via `handler.add_callback()`.
    """
    def __init__(self):
        # Initialized to None, will be set by NNHandler
        self.handler: 'NNHandler' | None = None 

    def set_handler(self, handler: 'NNHandler'):
        """Sets the NNHandler instance associated with this callback."""
        self.handler = handler

    # --- Training Hooks ---
    def on_train_begin(self, logs: dict | None = None):
        """Called at the beginning of the `train` method."""
        pass

    def on_train_end(self, logs: dict | None = None):
        """Called at the end of the `train` method."""
        pass

    def on_epoch_begin(self, epoch: int, logs: dict | None = None):
        """Called at the beginning of each training epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        """Called at the end of each training epoch.
        
        In DDP mode, `logs` on Rank 0 contains aggregated metrics.
        """
        pass

    def on_train_batch_begin(self, batch: int, logs: dict | None = None):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(self, batch: int, logs: dict | None = None):
        """Called at the end of each training batch.
        
        `logs` contains local loss/metrics for the batch on the current rank.
        """
        pass

    # --- Validation Hooks ---
    def on_val_begin(self, logs: dict | None = None):
        """Called at the beginning of the validation phase within an epoch."""
        pass

    def on_val_end(self, logs: dict | None = None):
        """Called at the end of the validation phase within an epoch.
        
        In DDP mode, `logs` on Rank 0 contains aggregated validation metrics.
        """
        pass

    def on_val_batch_begin(self, batch: int, logs: dict | None = None):
        """Called at the beginning of each validation batch."""
        pass

    def on_val_batch_end(self, batch: int, logs: dict | None = None):
        """Called at the end of each validation batch.
        
        `logs` contains local validation loss/metrics for the batch on the current rank.
        """
        pass

    # --- State Management Hooks (Optional but Recommended) ---
    def state_dict(self) -> dict:
        """Returns the state of the callback as a dictionary.
        
        Implement this method to save any internal state of the callback 
        (e.g., best score, patience counter) when `handler.save()` is called.
        """
        return {}

    def load_state_dict(self, state_dict: dict):
        """Restores the callback state from a dictionary.
        
        Implement this method to load the internal state saved by `state_dict`
        when `handler.load()` is called.
        """
        pass

```

## Implementing Custom Callbacks

1.  **Inherit**: Create a class that inherits from `Callback`.
2.  **Override Hooks**: Implement the `on_*` methods corresponding to the events you need to handle. You don't need to implement all of them.
3.  **Access Handler**: Use `self.handler` within your methods to access the `NNHandler` instance and its properties (e.g., `self.handler.module`, `self.handler.optimizer`, `self.handler.train_losses`, `self.handler._rank`).
4.  **Use Logs**: Inspect the `logs` dictionary passed to hooks to get current metric values (e.g., `logs['loss']`, `logs['val_accuracy']`). Remember that epoch-level logs are aggregated on Rank 0 in DDP.
5.  **DDP Awareness**: If your callback performs I/O or other actions that should only happen once globally (per epoch/batch), add a rank check: `if self.handler._rank == 0: ...`.
6.  **Statefulness (Optional)**: If your callback needs to maintain state across epochs or batches (e.g., the best validation score seen so far), implement `state_dict` to return this state and `load_state_dict` to restore it.