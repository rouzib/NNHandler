# Saving Module

The saving module in the `nn_handler.checkpointing` package provides functions for saving NNHandler states to files, with options for saving to a single file or multiple files.

## Overview

Saving model states is a critical part of the deep learning workflow. It allows you to:
- Checkpoint training progress to resume later
- Save trained models for deployment
- Share models with others
- Create model archives for reproducibility

The saving module offers flexible options for saving model states, with robust error handling and support for distributed training environments.

## Functions

### `save_single_file`

```python
def save_single_file(nn_handler: 'NNHandler', state, path)
```

Saves the provided state to a single file at the specified path. Ensures the directory exists before attempting to save the file. If saving fails, logs the error and raises an exception.

#### Parameters:

* `nn_handler` (NNHandler):
  * The NNHandler object that manages error handling and logging operations.
* `state` (dict):
  * The state object to be serialized and saved. Typically includes model weights or other relevant data.
* `path` (str):
  * The file path where the state should be saved. Includes the directory and file name.

#### Returns:

* None

### `save_multi_files`

```python
def save_multi_files(nn_handler: 'NNHandler', state, path, keys_to_save=None)
```

Saves specific components of the neural network state to separate files.

Each component of the state, as determined by `keys_to_save`, is saved to a separate file with `_key` appended to the base filename. The function also ensures that directories in the file path exist and logs the success or failure of the save operation. If keys like 'seed' or 'sde_class' are present in the state, they are used to update the `model_kwargs` attribute of the state.

#### Parameters:

* `nn_handler` (NNHandler):
  * The neural network handler object, responsible for managing the operation of the neural network and its states.
* `state` (dict):
  * A dictionary containing the state of the neural network. This includes data such as model parameters, optimizer state, and additional context-specific information.
* `path` (str):
  * The base path where the state components will be saved. The component keys will be appended to the base path for saving individual files.
* `keys_to_save` (list, optional):
  * A list of keys in the state dictionary to be saved as individual files. Defaults to `["model_state_dict", "optimizer_state_dict", "model_kwargs"]`.

#### Returns:

* None

### `save_multi_from_single`

```python
def save_multi_from_single(path, keys_to_save=None)
```

Saves multiple components of a PyTorch model state to separate files, extracting and organizing crucial components such as model state, optimizer state, and additional metadata. The function allows flexibility in selecting which parts of the state to save while ensuring compatibility and directory safety.

#### Parameters:

* `path` (str):
  * The file path from which the state will be loaded and also serves as the base for saving separated components.
* `keys_to_save` (list or None, optional):
  * List of specific keys in the state to save. Defaults to `["model_state_dict", "optimizer_state_dict", "model_kwargs"]` if not provided.

#### Returns:

* None

## Usage Examples

### Saving to a Single File

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# Create a simple model for demonstration
model = YourModel()
loss_fn = torch.nn.CrossEntropyLoss()

# Initialize the handler
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set up the handler with optimizer, loss function, etc.
handler.set_optimizer(torch.optim.Adam, lr=0.001)
handler.set_loss_fn(loss_fn)

# Train the model (simplified for example)
# handler.train(epochs=10)

# Prepare the state to save
state = handler.get_state_dict()

# Save the state to a single file
save_path = "models/my_model.pth"
from src.nn_handler.checkpointing.saving import save_single_file
save_single_file(handler, state, save_path)

print(f"Model saved to {save_path}")
```

### Saving to Multiple Files

```python
import torch
from src.nn_handler import NNHandler
from your_project.models import YourModel

# Create and train a model as in the previous example
# ...

# Prepare the state to save
state = handler.get_state_dict()

# Save specific components to separate files
save_path = "models/my_model.pth"
from src.nn_handler.checkpointing.saving import save_multi_files
save_multi_files(
    handler, 
    state, 
    save_path, 
    keys_to_save=["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "model_kwargs"]
)

print(f"Model components saved to separate files with base path {save_path}")
```

### Converting a Single File to Multiple Files

```python
import torch
from src.nn_handler.checkpointing.saving import save_multi_from_single

# Path to an existing single-file checkpoint
checkpoint_path = "models/my_model_checkpoint.pth"

# Convert to multiple files
save_multi_from_single(
    checkpoint_path,
    keys_to_save=["model_state_dict", "optimizer_state_dict", "model_kwargs"]
)

print(f"Checkpoint at {checkpoint_path} split into multiple files")
```

## Notes on Distributed Training

When using these functions in a distributed training environment:

1. Typically, only Rank 0 should perform the save operation to avoid file conflicts.
2. The `save_single_file` and `save_multi_files` functions include the rank in their logging messages to help with debugging.
3. If you need to save from multiple ranks (e.g., for different shards of a model), ensure each rank uses a unique file path.

Example of saving only from Rank 0:

```python
import torch
import torch.distributed as dist
from src.nn_handler import NNHandler
from src.nn_handler.checkpointing.saving import save_single_file

# Assuming DDP is initialized
if dist.get_rank() == 0:
    # Only Rank 0 saves the model
    state = handler.get_state_dict()
    save_path = "models/my_ddp_model.pth"
    save_single_file(handler, state, save_path)
    print(f"Model saved by Rank 0 to {save_path}")

# Wait for Rank 0 to finish saving
dist.barrier()
```

This ensures that only one process attempts to write to the file, preventing potential corruption or race conditions.