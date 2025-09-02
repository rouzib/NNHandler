# Monitor Callbacks

Monitor callbacks are designed to observe the training process, log progress, and potentially stop training based on metric performance.

*(Note: The exact implementation details depend on your `callbacks/monitor.py` file. This documentation assumes common callbacks like Progress Monitoring and Early Stopping might exist.)*

## `ProgressMonitor` (Example)

Logs basic progress information at the end of each epoch.

**Purpose:** Provide simple feedback on training progress.

**Example Implementation Idea:**

```python
from .base import Callback
import time

class ProgressMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Only log on Rank 0 in DDP
        if self.handler._rank == 0:
            epoch_time = time.time() - self.epoch_start_time
            log_msg = f"Epoch {epoch+1} finished in {epoch_time:.2f}s."
            # Extract aggregated loss/metrics from logs (already aggregated on Rank 0)
            train_loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            log_msg += f" Train Loss: {train_loss:.4e}" if isinstance(train_loss, float) else f" Train Loss: {train_loss}"
            log_msg += f" | Val Loss: {val_loss:.4e}" if isinstance(val_loss, float) else f" | Val Loss: {val_loss}"
            
            # Add primary learning rate
            lr = logs.get('lr', 'N/A')
            log_msg += f" | LR: {lr:.2e}" if isinstance(lr, float) else f" | LR: {lr}"
            
            # Print or log the message
            print(log_msg) # Or use self.handler.logger if available
```

**Usage:**

```python
handler.add_callback(ProgressMonitor())
```

## `EarlyStopping` (Example)

Stops training early if a monitored metric stops improving.

**Purpose:** Prevent overfitting and save computation time when performance plateaus.

**Key Parameters:**

*   `monitor` (str): The name of the metric to monitor (e.g., `'val_loss'`, `'val_accuracy'`). Must match a key in the `logs` dictionary available at `on_epoch_end`.
*   `patience` (int): Number of epochs with no improvement after which training will be stopped.
*   `min_delta` (float): Minimum change in the monitored quantity to qualify as an improvement.
*   `mode` (str): One of `{'min', 'max'}`. In `min` mode, training stops when the quantity monitored has stopped decreasing; in `max` mode it stops when the quantity monitored has stopped increasing.
*   `baseline` (float, optional): Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
*   `restore_best_weights` (bool): Whether to restore model weights from the epoch with the best value of the monitored quantity. If `False`, the model weights obtained at the last step of training are used.

**DDP Awareness:**
*   The decision to stop should be made based on **aggregated** metrics available on Rank 0.
*   The stopping signal needs to be broadcast from Rank 0 to all other ranks.
*   Restoring best weights should happen synchronously across all ranks.

**Example Implementation Sketch:**

```python
# Simplified sketch - requires careful handling of state and DDP
from .base import Callback
import numpy as np
import torch
import torch.distributed as dist

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=10, 
                 mode='min', baseline=None, restore_best_weights=False):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        self.wait_count = 0
        self.best_score = np.Inf if mode == 'min' else -np.Inf
        self.best_epoch = 0
        self.best_weights = None # Store weights only on Rank 0 if restoring

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= 1 if mode == 'min' else -1

    def on_epoch_end(self, epoch, logs=None):
        stop_training_signal = 0 # 0 = continue, 1 = stop
        current_score = None

        # Rank 0 evaluates stopping condition based on aggregated logs
        if self.handler._rank == 0:
            logs = logs or {}
            current_score = logs.get(self.monitor)

            if current_score is None:
                print(f"Warning: EarlyStopping monitored metric '{self.monitor}' not found in logs.")
            else:
                # Check against baseline if provided
                if self.baseline is not None and not self.monitor_op(current_score, self.baseline - self.min_delta):
                    print(f"Epoch {epoch+1}: Metric {self.monitor} did not improve over baseline {self.baseline}.")
                    stop_training_signal = 1
                
                # Check for improvement
                elif self.monitor_op(current_score, self.best_score - self.min_delta):
                    self.best_score = current_score
                    self.wait_count = 0
                    self.best_epoch = epoch
                    if self.restore_best_weights:
                        # Save current best weights (only Rank 0 needs the copy initially)
                        self.best_weights = {k: v.cpu().clone() for k, v in self.handler.module.state_dict().items()}
                else:
                    self.wait_count += 1
                    if self.wait_count >= self.patience:
                        print(f"Epoch {epoch+1}: Early stopping triggered after {self.patience} epochs of no improvement.")
                        stop_training_signal = 1
        
        # Broadcast decision from Rank 0
        if self.handler._distributed:
            stop_tensor = torch.tensor(stop_training_signal, device=self.handler.device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            stop_training_signal = stop_tensor.item()

        # Set stop flag on all ranks if needed
        if stop_training_signal == 1:
            self.handler._stop_training = True # Signal NNHandler to stop
            # If restoring weights, Rank 0 broadcasts weights, others receive
            if self.restore_best_weights:
                self._restore_weights_ddp()

    def _restore_weights_ddp(self):
        # Logic for Rank 0 to broadcast self.best_weights and all ranks to load them
        # Requires careful synchronization and object broadcasting/loading
        if self.handler._rank == 0:
            print(f"Restoring model weights from epoch {self.best_epoch + 1}")
            # Rank 0 prepares the state dict list to broadcast
            weights_list = [self.best_weights]
        else:
            weights_list = [None]
        
        if self.handler._distributed:
            dist.broadcast_object_list(weights_list, src=0)
            
        # All ranks load the received weights
        if weights_list[0] is not None:
             # Ensure weights are loaded onto the correct device for the rank
             weights_on_device = {k: v.to(self.handler.device) for k, v in weights_list[0].items()}
             self.handler.module.load_state_dict(weights_on_device)
        
        if self.handler._distributed:
            dist.barrier() # Ensure all ranks finished loading
            
    def state_dict(self) -> dict:
        # Exclude weights from saved state, save counters/best values
        return {
            'wait_count': self.wait_count,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            # Don't save best_weights here, maybe save path if using ModelCheckpoint
        }

    def load_state_dict(self, state_dict: dict):
        self.wait_count = state_dict.get('wait_count', 0)
        default_best = np.Inf if self.mode == 'min' else -np.Inf
        self.best_score = state_dict.get('best_score', default_best)
        self.best_epoch = state_dict.get('best_epoch', 0)
```

**Usage:**

```python
# Requires validation loader to be set
handler.add_callback(EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
))
```

*(Please adapt the examples above based on the actual code in your `monitor.py` file.)*