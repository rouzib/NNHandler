# Callback Utilities

This file potentially contains helper functions or utility classes used by various callbacks.

*(Note: The content depends on your `callbacks/utils.py` file. This is a placeholder.)*

## Potential Utilities

*   **Metric Smoothing:** Functions to apply exponential moving average or other smoothing techniques to noisy metrics before evaluation (e.g., in `EarlyStopping` or `ModelCheckpoint`).
*   **File System Helpers:** Functions for safe file writing, directory creation, or path manipulation used by saving callbacks.
*   **DDP Helpers:** Utility functions to simplify common DDP operations within callbacks, like checking the rank or broadcasting simple data.
*   **Log Formatting:** Functions to consistently format log messages or metric dictionaries.

## Example: DDP Rank Check

A simple utility function might be:

```python
# Potentially in src/nn_handler/callbacks/utils.py

def is_rank_zero(callback_instance) -> bool:
    """Checks if the callback is running on Rank 0 in a DDP setup."""
    if hasattr(callback_instance, 'handler') and \ 
       callback_instance.handler is not None and \ 
       hasattr(callback_instance.handler, '_rank'):
        return callback_instance.handler._rank == 0
    # Default to True if not DDP or handler not set (behaves like single process)
    return True 
```

Usage within a callback:

```python
from .base import Callback
from .utils import is_rank_zero # Assuming utils is in the same directory

class MyFileWritingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if is_rank_zero(self):
            # Only write the file on Rank 0
            with open(f"output_epoch_{epoch+1}.txt", "w") as f:
                f.write(str(logs))
```

*(Please add documentation for the actual utilities present in your `utils.py` file, if it exists.)*