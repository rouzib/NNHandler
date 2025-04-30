# Saving Callbacks

Saving callbacks automate the process of saving model checkpoints during training, often based on performance monitoring.

*(Note: The exact implementation details depend on your `callbacks/saving.py` file. This documentation assumes a common `ModelCheckpoint` callback might exist.)*

## `ModelCheckpoint` (Example)

Saves the handler state (including model weights, optimizer state, etc.) during training.

**Purpose:** Save the best performing model, save models periodically, or save the latest model.

**Key Parameters:**

*   `filepath` (str): Path/filename format for saving the checkpoint. Can include formatting options like `{epoch:02d}` or metric names available in the `logs` dict (e.g., `{val_loss:.4f}`).
*   `monitor` (str, optional): Metric name to monitor (e.g., `'val_loss'`). If provided, saving decisions (`save_best_only`, `save_weights_only` related to best) are based on this metric.
*   `save_best_only` (bool, default=False): If `True`, only saves when the monitored metric improves compared to the previous best. Overwrites the previous best checkpoint file.
*   `save_weights_only` (bool, default=False): If `True`, only the model's weights (`handler.module.state_dict()`) are saved. If `False`, the entire handler state is saved using `handler.save()`.
*   `mode` (str, default='auto'): One of `{'auto', 'min', 'max'}`. Used with `monitor`. Determines whether improvement means decreasing ('min') or increasing ('max'). 'auto' infers based on common metric names (e.g., 'loss' -> min, 'accuracy' -> max).
*   `save_freq` (str or int, default='epoch'):
    *   `'epoch'`: Saves at the end of each epoch (subject to `monitor` and `save_best_only`).
    *   `int`: Saves every `save_freq` *batches*.
*   `initial_value_threshold` (float, optional): Only start saving after the monitored metric has reached this value.

**DDP Awareness:**
*   Checkpoint saving (`handler.save()`) is performed **only by Rank 0**.
*   Decisions based on monitored metrics use the **aggregated** values available in `logs` on Rank 0.

**Example Implementation Sketch:**

```python
# Simplified sketch - requires careful handling of state, paths, and DDP
from .base import Callback
import os
import numpy as np
import warnings

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor=None, save_best_only=False, 
                 save_weights_only=False, mode='auto', save_freq='epoch',
                 initial_value_threshold=None):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.initial_value_threshold = initial_value_threshold

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(f"ModelCheckpoint mode '{mode}' is invalid, fallback to 'auto'.", RuntimeWarning)
            mode = 'auto'
            
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_score = -np.Inf
        else: # Auto mode
            if monitor and ('acc' in monitor or monitor.startswith('fmeasure')):
                self.monitor_op = np.greater
                self.best_score = -np.Inf
            else:
                self.monitor_op = np.less
                self.best_score = np.Inf
                
        self.last_batch_saved = -1 # For batch frequency saving
        self.last_epoch_saved = -1 # Track last epoch saved to avoid double saving

    def _save_model(self, epoch, logs):
        # Only Rank 0 performs the save
        if self.handler._rank != 0:
            return
            
        try:
            # Format filepath with epoch and aggregated logs from Rank 0
            format_dict = logs.copy()
            format_dict['epoch'] = epoch + 1 # Use 1-based epoch
            filepath = self.filepath.format(**format_dict)
        except KeyError as e:
            # Fallback if key not found in logs
            filepath = f"{self.filepath}_epoch{epoch+1}.pth"
            print(f"Warning: ModelCheckpoint filepath key {e} not found in logs. Using fallback: {filepath}")
        except Exception as e:
            filepath = f"{self.filepath}_epoch{epoch+1}.pth"
            print(f"Warning: ModelCheckpoint filepath formatting error: {e}. Using fallback: {filepath}")

        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except OSError as e:
             print(f"Error creating directory for checkpoint {filepath}: {e}")
             return # Abort save if directory fails

        # Perform save
        if self.save_weights_only:
            # Save only model weights (implementation needed in NNHandler or here)
            # self.handler.save_weights(filepath) # Assumes such a method exists
            torch.save(self.handler.module.state_dict(), filepath)
            print(f"Epoch {epoch+1}: saving model weights to {filepath}")
        else:
            # Save full handler state via NNHandler's save method
            self.handler.save(filepath) # NNHandler.save handles rank 0 logic
            # The logger inside handler.save will print the message
        
        self.last_epoch_saved = epoch # Mark this epoch as saved

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq != 'epoch':
            return # Only handle epoch saving here

        # Prevent saving again if batch freq already saved this epoch
        if self.last_epoch_saved == epoch: 
            return

        # Decision logic (run on Rank 0 only)
        if self.handler._rank == 0:
            self._save_decision(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if not isinstance(self.save_freq, int) or self.save_freq <= 0:
            return # Only handle batch frequency saving here
        
        # Check if it's time to save based on batch frequency
        # Need global step, NNHandler might need to track this
        # Simplified: save every N batches within the current epoch
        # Note: This doesn't track global batches across epochs accurately without more state
        current_epoch_batch = batch + 1
        if current_epoch_batch % self.save_freq == 0:
             # Save based on batch freq - potentially without monitoring best score
             # Rank 0 performs the save
             # Pass local batch logs, might not be ideal for naming
             self._save_model(self.handler._current_epoch, logs or {})
             self.last_batch_saved = batch
             # Mark epoch saved to maybe skip epoch end save if desired?
             self.last_epoch_saved = self.handler._current_epoch 

    def _save_decision(self, epoch, logs):
        # Assumes running on Rank 0
        logs = logs or {}
        
        if not self.monitor: # Save regardless of metrics if no monitor
            if not self.save_best_only:
                self._save_model(epoch, logs)
            return

        current_score = logs.get(self.monitor)
        if current_score is None:
            if not self.save_best_only: # Save if not monitoring best, despite missing metric
                warnings.warn(f"ModelCheckpoint monitored metric '{self.monitor}' not found in logs. Saving anyway.", RuntimeWarning)
                self._save_model(epoch, logs)
            return

        # Check threshold if set
        if self.initial_value_threshold is not None:
            if not self.monitor_op(current_score, self.initial_value_threshold):
                return # Don't save until threshold is met

        # Check for improvement
        if self.monitor_op(current_score, self.best_score):
            print(f"
Epoch {epoch+1}: {self.monitor} improved from {self.best_score:.5f} to {current_score:.5f}")
            self.best_score = current_score
            self._save_model(epoch, logs)
        elif not self.save_best_only:
             # Save even if not best
             self._save_model(epoch, logs)

    def state_dict(self) -> dict:
        return {
            'best_score': self.best_score,
            'last_epoch_saved': self.last_epoch_saved,
            # Add last_batch_saved if tracking global step
        }

    def load_state_dict(self, state_dict: dict):
        default_best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_score = state_dict.get('best_score', default_best)
        self.last_epoch_saved = state_dict.get('last_epoch_saved', -1)
```

**Usage:**

```python
# Save best model based on validation loss
handler.add_callback(ModelCheckpoint(
    filepath="checkpoints/best_val_loss_model.pth", 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False, # Save full state
    mode='min'
))

# Save checkpoint every 5 epochs
handler.add_callback(ModelCheckpoint(
    filepath="checkpoints/epoch_{epoch:03d}.pth", 
    save_best_only=False, 
    save_freq='epoch' # Default, but explicit here
))

# Save weights only, every 1000 batches (requires NNHandler to track global step)
# handler.add_callback(ModelCheckpoint(
#    filepath="checkpoints/batch_{global_step:06d}_weights.pth", 
#    save_weights_only=True, 
#    save_freq=1000 
# ))
```

*(Please adapt the examples above based on the actual code in your `saving.py` file.)*