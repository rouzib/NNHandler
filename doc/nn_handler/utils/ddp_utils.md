# DDP Utilities

The `nn_handler` framework provides a set of utilities for working with PyTorch's Distributed Data Parallel (DDP) functionality. These utilities are designed to simplify the process of using DDP for multi-GPU and multi-node training, as well as provide additional functionality for parallel execution on multiple GPUs.

## Overview

The DDP utilities are divided into two main categories:

1. **DDP Core Utilities** (`ddp.py`): Functions for initializing and managing DDP processes, resolving devices, and determining whether to use distributed training.
2. **DDP Decorators** (`ddp_decorators.py`): Decorators and classes for executing functions in parallel across multiple GPUs, with robust error handling and synchronization.

These utilities can be used independently of the main `NNHandler` class, allowing for flexible integration into existing PyTorch code.

## DDP Core Utilities

### Initialization and Environment Detection

```python
from nn_handler.utils.ddp import initialize_ddp, _should_use_distributed, _resolve_device
```

#### `initialize_ddp(timeout=None)`

Initializes the Distributed Data Parallel (DDP) process group if not already done.

```python
# Initialize DDP and get distributed information
distributed, rank, local_rank, world_size, device = initialize_ddp()

# Use the returned values to configure your training
if distributed:
    print(f"Running on rank {rank} of {world_size} with device {device}")
else:
    print(f"Running in non-distributed mode on device {device}")
```

**Parameters:**
- `timeout` (Optional[timedelta]): Timeout for operations. If None, defaults to 60 minutes.

**Returns:**
- If DDP was not initialized before: Returns a 5-tuple containing:
  - `distributed` (bool): Whether distributed mode is enabled
  - `rank` (int): Global rank of this process
  - `local_rank` (int): Local rank of this process (for device selection)
  - `world_size` (int): Total number of processes
  - `device` (torch.device): The device to use for this process
- If DDP was already initialized: Returns None

#### `_should_use_distributed(use_distributed_flag)`

Determines if Distributed Data Parallel (DDP) should be used based on flag and environment.

**Parameters:**
- `use_distributed_flag` (Optional[bool]): Flag to control DDP usage.
  - True: Explicitly enable DDP (if possible)
  - False: Explicitly disable DDP
  - None: Auto-detect based on environment

**Returns:**
- bool: True if DDP should be used, False otherwise.

#### `_resolve_device(device)`

Resolves a device specification to a torch.device object, handling CUDA availability.

**Parameters:**
- `device` (Union[torch.device, str]): The device specification to resolve.
  - If a torch.device: Validates it and returns it (or falls back to CPU if needed)
  - If a string: Converts it to a torch.device (with special handling for 'cuda' and 'gpu')

**Returns:**
- torch.device: The resolved device object.

## DDP Decorators

### Rank-Specific Execution

```python
from nn_handler.utils.ddp_decorators import on_rank
```

#### `on_rank(rank, barrier=False)`

A decorator to execute a function only on a specific rank or ranks, with robust error handling.

```python
import torch.distributed as dist
from nn_handler.utils.ddp_decorators import on_rank

# Initialize DDP (not shown)

@on_rank(0)  # Only execute on rank 0
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# This will only execute on rank 0, other ranks will wait
save_model(model, "model.pth")

@on_rank([0, 1])  # Execute on ranks 0 and 1
def process_data(data):
    # Process data on ranks 0 and 1
    return processed_data

# This will execute on ranks 0 and 1, other ranks will wait
process_data(data)
```

**Parameters:**
- `rank` (Union[int, List[int]]): The rank or list of ranks to execute on.
- `barrier` (bool): If True, a barrier is called after the function execution (only if no errors occurred).

### Multi-GPU Parallel Execution

```python
from nn_handler.utils.ddp_decorators import parallelize_on_gpus, parallel_on_all_devices, ParallelExecutor
```

#### `parallelize_on_gpus(num_gpus=None, pass_device=True)`

A decorator to run a function in parallel on multiple GPUs.

```python
import torch
from nn_handler.utils.ddp_decorators import parallelize_on_gpus

@parallelize_on_gpus()
def process_on_gpu(device, data):
    # This function will be executed on each GPU
    return data.to(device) * 2

# Create some data
data = torch.tensor([1, 2, 3])

# Run the function on all available GPUs
results = process_on_gpu(data)
# results is a list of tensors, one from each GPU

# You can also specify the number of GPUs to use
@parallelize_on_gpus(num_gpus=2)
def another_gpu_function(device, x, y):
    return (x + y).to(device)

# This will run only on the first 2 GPUs
results = another_gpu_function(torch.tensor([1, 2]), torch.tensor([3, 4]))
```

**Parameters:**
- `num_gpus` (int, optional): Number of GPUs to use. If None, uses all available GPUs.
- `pass_device` (bool, optional): If True, passes the device as an argument to the function. The device will be passed as a named argument 'device'. Defaults to True.

**Returns:**
- callable: A wrapped function that, when called, executes the original function in parallel across multiple GPUs and returns a list of results.

#### `parallel_on_all_devices(func)`

A decorator to run a function in parallel on all available CUDA devices using torch.nn.DataParallel.

```python
import torch
from nn_handler.utils.ddp_decorators import parallel_on_all_devices

@parallel_on_all_devices
def my_model_forward(data_chunk):
    # data_chunk is a slice of the batch on a specific GPU
    device = data_chunk.device
    # ... perform operations on data_chunk ...
    new_tensor = torch.ones(1, device=device)
    # ... more operations ...
    return new_tensor

# Move your entire input batch to the primary CUDA device
full_batch = torch.randn(128, 10).to('cuda:0')

# Call the function. DataParallel will automatically split the batch,
# move chunks to other GPUs, execute the function, and gather results
results = my_model_forward(full_batch)
```

**Note:** This is suitable for single-node, multi-GPU data parallelism. It is simpler than DDP but often less performant due to factors like GIL contention and unbalanced workload on the primary GPU.

#### `ParallelExecutor`

A class for executing functions in parallel across multiple GPUs.

```python
import torch
from nn_handler.utils.ddp_decorators import ParallelExecutor

def heavy_computation(device, data):
    # Move data to the specified device
    data = data.to(device)
    # Perform computation
    result = data * 2
    return result

# Create some data
data = torch.tensor([1, 2, 3])

# Use ParallelExecutor as a context manager
with ParallelExecutor(num_gpus=2) as executor:
    # Run the function on 2 GPUs
    results = executor.run(heavy_computation, data)
    # results is a list of tensors, one from each GPU
```

**Constructor Parameters:**
- `num_gpus` (int, optional): Number of GPUs to use. If None, uses all available GPUs.
- `pass_device` (bool, optional): If True, passes the device as an argument to the function. Defaults to True.

**Methods:**
- `run(func, *args, **kwargs)`: Execute a function in parallel across multiple GPUs.
  - `func` (callable): The function to execute in parallel.
  - `*args`: Positional arguments to pass to the function.
  - `**kwargs`: Keyword arguments to pass to the function.
  - Returns: list: A list of results from each GPU, in order of GPU index.

## Examples

### Basic DDP Initialization

```python
import torch
import torch.distributed as dist
from nn_handler.utils.ddp import initialize_ddp

# Initialize DDP
distributed, rank, local_rank, world_size, device = initialize_ddp()

# Create a model and move it to the correct device
model = MyModel().to(device)

# If using DDP, wrap the model
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == 'cuda' else None
    )

# Now you can use the model for training
```

### Parallel GPU Computation

```python
import torch
import numpy as np
from nn_handler.utils.ddp_decorators import parallelize_on_gpus

@parallelize_on_gpus()
def matrix_multiply(device, size=1000, iterations=10):
    # Create random matrices on the specified device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Perform multiple matrix multiplications
    for _ in range(iterations):
        c = torch.matmul(a, b)
        a = c
    
    # Return the result as a numpy array
    return c.cpu().numpy()

# Run the function on all available GPUs
results = matrix_multiply(size=2000, iterations=20)
print(f"Received {len(results)} results, one from each GPU")
```

### Rank-Specific Operations

```python
import torch
import torch.distributed as dist
from nn_handler.utils.ddp import initialize_ddp
from nn_handler.utils.ddp_decorators import on_rank

# Initialize DDP
initialize_ddp()

# Define a function that should only run on rank 0
@on_rank(0)
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

# This will only execute on rank 0, other ranks will wait
save_checkpoint(model, optimizer, epoch, "checkpoint.pth")
```

## Best Practices

1. **Error Handling**: The `on_rank` decorator includes robust error handling that propagates errors to all ranks, preventing hangs when one rank fails.

2. **Device Management**: Always use the device provided by the decorators or initialization functions, rather than hardcoding device indices.

3. **Data Transfer**: When using `parallelize_on_gpus`, be mindful of data transfer costs. It's often best to create data directly on the target device or use CPU tensors that will be moved to the appropriate device by the worker function.

4. **Synchronization**: The `on_rank` decorator with `barrier=True` ensures all ranks are synchronized after the function execution, which can be important for maintaining consistency across ranks.

5. **Serialization**: If encountering an EOFError with `parallelize_on_gpus`, try converting the results of your function to a list or a numpy array before returning, as this can help with serialization between processes.