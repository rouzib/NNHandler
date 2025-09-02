# DDP Initialization

The `ddp_init` module provides functions for initializing and managing PyTorch's Distributed Data Parallel (DDP) environment. It handles the complexities of setting up distributed training across multiple GPUs and nodes, including environment detection, device selection, and process group initialization.

## Overview

Distributed training requires careful setup and coordination between multiple processes. This module simplifies the process by:

1. **Detecting the distributed environment**: Checking for environment variables set by tools like `torchrun` or Slurm.
2. **Selecting appropriate devices**: Mapping processes to the correct GPU based on their local rank.
3. **Initializing the process group**: Setting up communication between processes with the appropriate backend.
4. **Resolving device specifications**: Converting device strings to PyTorch device objects with fallbacks.

These functions are designed to work in various environments, from single-machine multi-GPU setups to multi-node clusters managed by Slurm.

## Functions

### Core Initialization

#### `initialize_ddp(timeout: Optional[timedelta] = None)`

The main entry point for initializing DDP. This function checks if the process group is already initialized, and if not, attempts to initialize it with the appropriate settings.

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

**Example:**
```python
import torch.distributed as dist
from src.nn_handler.utils.ddp_init import initialize_ddp

# Initialize DDP and get distributed information
distributed, rank, local_rank, world_size, device = initialize_ddp()

# Use the returned values to configure your training
if distributed:
    print(f"Running on rank {rank} of {world_size} with device {device}")
else:
    print(f"Running in non-distributed mode on device {device}")
```

### Environment Detection

#### `_should_use_distributed(use_distributed_flag: Optional[bool]) -> bool`

Determines if DDP should be used based on the provided flag and the current environment.

**Parameters:**
- `use_distributed_flag` (Optional[bool]): Flag to control DDP usage.
  - True: Explicitly enable DDP (if possible)
  - False: Explicitly disable DDP
  - None: Auto-detect based on environment

**Returns:**
- bool: True if DDP should be used, False otherwise.

#### `_is_env_distributed() -> bool`

Checks whether the current process has the information it needs to start DDP, coming either from torchrun-style variables (RANK, LOCAL_RANK, WORLD_SIZE) or from Slurm.

**Returns:**
- bool: True if the environment is configured for distributed training, False otherwise.

### Device Management

#### `_resolve_device(device: Union[torch.device, str]) -> torch.device`

Resolves a device specification to a torch.device object, handling CUDA availability.

**Parameters:**
- `device` (Union[torch.device, str]): The device specification to resolve.
  - If a torch.device: Validates it and returns it (or falls back to CPU if needed)
  - If a string: Converts it to a torch.device (with special handling for 'cuda' and 'gpu')

**Returns:**
- torch.device: The resolved device object.

**Example:**
```python
from src.nn_handler.utils.ddp_init import _resolve_device

# Resolve a device string
device = _resolve_device("cuda")  # Returns torch.device("cuda") if CUDA is available, else CPU

# Resolve a specific GPU
device = _resolve_device("cuda:1")  # Returns torch.device("cuda:1") if available, else CPU

# Resolve a device object
device_obj = torch.device("cuda")
resolved_device = _resolve_device(device_obj)  # Validates and returns the device
```

### Internal Functions

#### `_initialize_distributed(timeout: Optional[timedelta] = None)`

Initializes the distributed process group for PyTorch DDP.

**Parameters:**
- `timeout` (Optional[timedelta]): Timeout for operations. If None, defaults to 60 minutes.

**Returns:**
- tuple: A 5-tuple containing distributed status, rank, local_rank, world_size, and device.

#### `_pick_device(local_rank: int) -> torch.device`

Selects the appropriate CUDA device based on the local rank.

**Parameters:**
- `local_rank` (int): The local rank of the current process.

**Returns:**
- torch.device: The selected device.

#### `_first_host_from_slurm_nodelist(nodelist: str) -> str`

Extracts the first hostname from a Slurm nodelist string.

**Parameters:**
- `nodelist` (str): The Slurm nodelist string (e.g., 'node[001-004,007]').

**Returns:**
- str: The first hostname in the list.

## Usage Notes

- The module automatically detects whether to use DDP based on environment variables set by tools like `torchrun` or Slurm.
- It handles both standard PyTorch DDP environment variables (RANK, LOCAL_RANK, WORLD_SIZE) and Slurm variables (SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS).
- If CUDA is requested but not available, the module will issue a warning and fall back to CPU.
- The module attempts to determine the master address (MASTER_ADDR) from Slurm variables if not explicitly set.
- The initialization process includes a barrier to ensure all processes are synchronized before proceeding.

## Example: Manual DDP Setup

```python
import torch
import torch.distributed as dist
from src.nn_handler.utils.ddp_init import initialize_ddp

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

This module is typically used internally by the `NNHandler` class, which provides a higher-level interface for distributed training.