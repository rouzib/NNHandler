# Training Control Callbacks

These callbacks are designed to modify or control aspects of the training process itself, beyond simple monitoring or saving.

*(Note: The specific callbacks available depend on your `callbacks/training.py` file. This is a placeholder for potential examples.)*

## Potential Callback Examples

Here are some ideas for callbacks that might fit into this category:

*   **Learning Rate Finder:** A callback that runs a short training segment at the beginning with increasing learning rates to help find an optimal starting point (similar to fastai's `LRFinder`).
*   **Hyperparameter Scheduler:** A callback that modifies hyperparameters (beyond the learning rate managed by standard schedulers) according to a predefined schedule or based on metric performance (e.g., adjusting dropout rate, weight decay, or loss function parameters like `beta` in a VAE loss).
*   **Gradient Accumulation Scheduler:** Dynamically adjust the number of gradient accumulation steps during training, perhaps increasing it as training progresses to simulate larger batch sizes later on.
*   **Custom Stopping Condition:** Implement stopping logic based on criteria other than simple metric patience (e.g., stop if loss gradient becomes too small, or if training time exceeds a limit).
*   **Data Augmentation Scheduler:** Modify data augmentation parameters based on the current epoch or training progress.

## Example: `LossParameterScheduler` (Hypothetical)

Adjusts a keyword argument passed to the loss function based on the epoch.

```python
from .base import Callback
import numpy as np

class LossParameterScheduler(Callback):
    """Linearly interpolates a kwarg for the loss function between epochs."""
    def __init__(self, kwarg_name: str, start_epoch: int, end_epoch: int, 
                 start_value: float, end_value: float):
        super().__init__()
        self.kwarg_name = kwarg_name
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = start_value
        self.end_value = end_value
        
        if start_epoch >= end_epoch:
            raise ValueError("start_epoch must be less than end_epoch")

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate the current value based on epoch progress
        if epoch < self.start_epoch:
            current_value = self.start_value
        elif epoch >= self.end_epoch:
            current_value = self.end_value
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            current_value = np.interp(progress, [0, 1], [self.start_value, self.end_value])
        
        # Update the loss function kwargs in the handler
        # This requires the handler's loss_fn_kwargs to be mutable (dict)
        if self.handler._loss_fn_kwargs is not None:
            self.handler._loss_fn_kwargs[self.kwarg_name] = current_value
            
            # Log the change (Rank 0)
            if self.handler._rank == 0 and epoch >= self.start_epoch and epoch <= self.end_epoch:
                print(f"Epoch {epoch+1}: Set loss kwarg '{self.kwarg_name}' to {current_value:.4f}")
        elif self.handler._rank == 0:
             print(f"Warning: LossParameterScheduler cannot set '{self.kwarg_name}', handler.loss_fn_kwargs is None.")

    # No state needs saving for this simple scheduler
```

**Usage:**

```python
# Assume VAE loss takes a 'beta' kwarg
def vae_loss(recon, target, mu, logvar, beta=1.0):
    # ... loss calculation ...
    return total_loss

handler.set_loss_fn(vae_loss, pass_epoch_to_loss=False, beta=0.0) # Initial beta

# Schedule beta from 0.0 to 1.0 between epochs 10 and 50
handler.add_callback(LossParameterScheduler(
    kwarg_name='beta',
    start_epoch=10, 
    end_epoch=50,
    start_value=0.0,
    end_value=1.0
))

handler.train(epochs=100)
```

*(Please add documentation for the actual callbacks present in your `training.py` file.)*