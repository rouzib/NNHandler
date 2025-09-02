# Utilities

The `nn_handler.utils` package provides a collection of utility functions and classes that support the core functionality of the `nn_handler` framework. These utilities are designed to be used both within the framework and independently in your own PyTorch code.

## Available Utilities

### Distributed Data Parallel (DDP) Utilities

The DDP utilities provide tools for working with PyTorch's Distributed Data Parallel functionality:

* **[DDP Utilities](ddp_utils.md)**: Documentation for utilities that simplify working with DDP, including:
  * Core DDP functions for initialization and environment detection
  * Decorators for rank-specific execution
  * Tools for parallel execution across multiple GPUs

## Usage

Most utilities can be imported directly from their respective modules:

```python
# Import DDP utilities
from nn_handler.utils.ddp import initialize_ddp
from nn_handler.utils.ddp_decorators import parallelize_on_gpus, on_rank

# Example: Initialize DDP
distributed, rank, local_rank, world_size, device = initialize_ddp()

# Example: Use the parallelize_on_gpus decorator
@parallelize_on_gpus()
def process_on_gpu(device, data):
    return data.to(device) * 2

# Example: Use the on_rank decorator
@on_rank(0)
def save_model(model, path):
    torch.save(model.state_dict(), path)
```

Refer to the specific documentation pages for detailed information about each utility.