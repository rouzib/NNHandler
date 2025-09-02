# Loading Module

The loading module in the `nn_handler.checkpointing` package provides functions for loading saved NNHandler states and initializing new NNHandler instances from checkpoints.

## Overview

When training deep learning models, it's often necessary to save and load model states for various purposes:
- Resuming training after interruption
- Deploying trained models for inference
- Transfer learning from pre-trained models
- Evaluating model performance

The loading module offers robust functionality to handle these scenarios, with special consideration for distributed training environments.

## Functions

### `load`

```python
def load(NNHandler, path: str, device: Optional[Union[str, torch.device]] = None, 
         strict_load: bool = False, skip_optimizer: bool = False, 
         skip_scheduler: bool = False, skip_history: bool = False, 
         skip_callbacks: bool = False, skip_sampler_sde: bool = False, 
         skip_ema: bool = False) -> 'NNHandler'
```

Loads a saved NNHandler state from a file.

In Distributed Data Parallel (DDP) mode, all ranks load the checkpoint state, mapping tensors to their assigned local device. The handler is initialized within the DDP environment before loading the state dictionary components.

#### Parameters:

* `path` (str):
  * Path to the saved state file.
* `device` (Optional[Union[str, torch.device]]):
  * Explicitly specify the device to load onto. 
  * If None, uses the DDP-assigned device (cuda:local_rank or cpu) or defaults for non-DDP mode.
* `strict_load` (bool, default=False):
  * Passed to model.load_state_dict. If True, keys must match exactly.
* `skip_optimizer` (bool, default=False):
  * Don't load optimizer state.
* `skip_scheduler` (bool, default=False):
  * Don't load scheduler state.
* `skip_history` (bool, default=False):
  * Don't load loss/metric history (rank 0).
* `skip_callbacks` (bool, default=False):
  * Don't load callback states.
* `skip_sampler_sde` (bool, default=False):
  * Don't load sampler/SDE state.
* `skip_ema` (bool, default=False):
  * Don't load EMA state.

#### Returns:

* `NNHandler`: An instance loaded with the saved state.

### `initialize_from_checkpoint`

```python
def initialize_from_checkpoint(NNHandler, checkpoint_path: str, 
                              model_class: type[nn.Module], 
                              model_type: Optional[Union[ModelType, str]] = None, 
                              device: Optional[Union[str, torch.device]] = None, 
                              strict_load: bool = True, **model_kwargs) -> 'NNHandler'
```

Initializes a new NNHandler instance loading ONLY model weights from a checkpoint file.

This is useful for inference or transfer learning. Optimizer, scheduler, history, etc., are *not* loaded. Assumes the checkpoint file contains *only* the model's state_dict, or a dictionary where the weights are under the key "model_state_dict".

In DDP mode, all ranks perform this initialization and load the weights onto their assigned device.

#### Parameters:

* `checkpoint_path` (str):
  * Path to the checkpoint file (model state_dict).
* `model_class` (type[nn.Module]):
  * The model class to instantiate.
* `model_type` (Optional[Union[ModelType, str]]):
  * The type of the model. Defaults to CLASSIFICATION.
* `device` (Optional[Union[str, torch.device]]):
  * Target device. If None, uses DDP default or auto-detect.
* `strict_load` (bool, default=True):
  * Whether to strictly enforce state_dict key matching.
* `**model_kwargs`:
  * Keyword arguments for the model constructor.

#### Returns:

* `NNHandler`: An instance with the specified model and loaded weights.

## Usage Examples

### Loading a Complete Handler State

```python
import torch
from src.nn_handler import NNHandler

# Path to the saved handler state
checkpoint_path = "models/my_model_checkpoint.pth"

# Load the complete handler state
loaded_handler = NNHandler.load(
    path=checkpoint_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Resume training from the loaded state
print(f"Resuming training from epoch {len(loaded_handler.train_losses)}...")
loaded_handler.train(epochs=20)  # Continue training for 20 more epochs
```

### Loading Only Model Weights for Inference

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# Path to the model weights checkpoint
weights_path = "models/pretrained_weights.pth"

# Initialize a new handler with only the model weights loaded
inference_handler = NNHandler.initialize_from_checkpoint(
    checkpoint_path=weights_path,
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # Pass any required model constructor arguments
    input_size=784,
    hidden_size=256,
    output_size=10
)

# Set to evaluation mode for inference
inference_handler.eval()

# Use the model for inference
test_input = torch.randn(1, 784).to(inference_handler.device)
with torch.no_grad():
    output = inference_handler.model(test_input)
    prediction = output.argmax(dim=1).item()
    print(f"Predicted class: {prediction}")
```

### Selective Loading for Fine-tuning

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# Path to the saved handler state
checkpoint_path = "models/pretrained_model.pth"

# Load the handler state, but skip optimizer and scheduler
# This is useful for fine-tuning with different optimization settings
fine_tune_handler = NNHandler.load(
    path=checkpoint_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    skip_optimizer=True,
    skip_scheduler=True
)

# Set up a new optimizer with a lower learning rate for fine-tuning
fine_tune_handler.set_optimizer(
    optimizer_class=torch.optim.Adam,
    lr=1e-5  # Lower learning rate for fine-tuning
)

# Set up a new scheduler if needed
fine_tune_handler.set_scheduler(
    scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
    mode='min',
    factor=0.5,
    patience=5
)

# Continue training with the new optimization settings
fine_tune_handler.train(epochs=10)
```

## Notes on Distributed Training

When using these functions in a distributed training environment:

1. All ranks must call the loading functions to ensure consistent state across processes.
2. The device mapping is automatically handled based on the current DDP environment.
3. In `load()`, history and some other components are only loaded on Rank 0 to avoid redundancy.
4. Barriers are used to ensure all ranks complete the loading process before continuing.

Always launch your loading script with the same DDP configuration (e.g., same number of processes via `torchrun`) as the original training run when applicable.