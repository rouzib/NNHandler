# DDP Decorators

The `ddp_decorators` module provides decorators and utilities for working with distributed and parallel processing in PyTorch. These tools simplify common patterns like rank-specific execution, multi-GPU parallelism, and error handling in distributed environments.

## Overview

This module offers three main types of functionality:

1. **Rank-Specific Execution**: Execute code only on specific ranks in a distributed setting.
2. **Data Parallel Execution**: Run functions in parallel across multiple GPUs using PyTorch's DataParallel.
3. **Multi-GPU Process Parallelism**: Execute functions in separate processes, one per GPU, with robust error handling.

These utilities help manage the complexity of distributed and parallel processing while ensuring proper error propagation and synchronization.

## Decorators and Classes

### Rank-Specific Execution

#### `on_rank(rank: Union[int, List[int]], barrier: bool = False)`

A decorator to execute a function only on a specific rank or ranks, with robust error handling.

**Parameters:**
- `rank`: The rank or list of ranks to execute on.
- `barrier`: If True, a barrier is called after the function execution (only if no errors occurred).

**Error Handling:**
If the decorated function raises an exception on any of the target ranks, the decorator will:
1. Catch the exception.
2. Communicate the failure to all other ranks in the process group.
3. Raise a RuntimeError on all ranks to ensure a clean, synchronized shutdown of the DDP job, preventing hangs.

**Example:**
```python
import torch.distributed as dist
from src.nn_handler.utils.ddp_decorators import on_rank

@on_rank(0)  # Only execute on rank 0
def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

# This will only execute on rank 0, other ranks will wait
save_checkpoint(model, "checkpoint.pth")

@on_rank([0, 1])  # Execute on ranks 0 and 1
def process_data(data):
    # Process data on ranks 0 and 1
    return processed_data

# This will execute on ranks 0 and 1, other ranks will wait
process_data(data)
```

### Data Parallel Execution

#### `parallel_on_all_devices(func)`

A decorator to run a function in parallel on all available CUDA devices using `torch.nn.DataParallel`.

**Requirements:**
The decorated function MUST be device-aware. It should:
1. Accept at least one tensor as input.
2. Infer the correct device from its input tensors (e.g., `device = my_tensor.device`).
3. Create any new tensors on that same device.

**Example:**
```python
import torch
from src.nn_handler.utils.ddp_decorators import parallel_on_all_devices

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

### Multi-GPU Process Parallelism

#### `ParallelExecutor`

A class for executing functions in parallel across multiple GPUs using separate processes.

**Constructor Parameters:**
- `num_gpus` (int, optional): Number of GPUs to use. If None, uses all available GPUs.
- `pass_device` (bool, optional): If True, passes the device as an argument to the function. Defaults to True.

**Methods:**
- `run(func, *args, **kwargs)`: Execute a function in parallel across multiple GPUs.
  - `func` (callable): The function to execute in parallel.
  - `*args`: Positional arguments to pass to the function.
  - `**kwargs`: Keyword arguments to pass to the function.
  - Returns: list: A list of results from each GPU, in order of GPU index.

**Example:**
```python
import torch
from src.nn_handler.utils.ddp_decorators import ParallelExecutor

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

#### `parallelize_on_gpus(num_gpus: int = None, pass_device: bool = True)`

A decorator to run a function in parallel on multiple GPUs using separate processes.

**Parameters:**
- `num_gpus` (int, optional): Number of GPUs to use. If None, uses all available GPUs.
- `pass_device` (bool, optional): If True, passes the device as an argument to the function. The device will be passed as a named argument 'device'. Defaults to True.

**Returns:**
- callable: A wrapped function that, when called, executes the original function in parallel across multiple GPUs and returns a list of results.

**Example:**
```python
import torch
from src.nn_handler.utils.ddp_decorators import parallelize_on_gpus

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

## Helper Functions

### `_as_cpu(obj: Any) -> Any`

Recursively moves PyTorch tensors to CPU, handling nested data structures.

**Parameters:**
- `obj` (Any): The object to process. Can be a tensor, list, tuple, dict, or any other type.

**Returns:**
- Any: The processed object with all tensors moved to CPU.

### `_parallel_worker(rank, pickled_user_func, result_queue, pass_device)`

Internal worker function for parallel execution on a specific GPU.

## Usage Notes

- The `on_rank` decorator is particularly useful for operations that should only happen once in a distributed setting, like saving checkpoints or logging.
- The `parallel_on_all_devices` decorator is simpler to use than `parallelize_on_gpus` but less flexible and potentially less efficient.
- The `parallelize_on_gpus` decorator and `ParallelExecutor` class use separate processes for true parallelism, which can be more efficient for compute-intensive operations but has higher overhead for data transfer.
- When using `parallelize_on_gpus`, if you encounter an EOFError, try converting the results of your function to a list or a numpy array before returning.