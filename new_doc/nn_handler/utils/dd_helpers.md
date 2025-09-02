# Distributed Data Helpers

The `dd_helpers` module provides utility functions for working with distributed data parallel (DDP) operations in PyTorch. These functions simplify common tasks like metric aggregation, loss reduction, data broadcasting, and creating distributed data loaders.

## Overview

When training models across multiple GPUs or nodes using PyTorch's Distributed Data Parallel (DDP), several operations need special handling to ensure correct behavior:

1. **Metric Aggregation**: Metrics calculated on each process need to be combined.
2. **Loss Reduction**: Loss values need to be averaged across all processes.
3. **Data Broadcasting**: Some data needs to be consistent across all processes.
4. **Distributed Data Loading**: Data loaders need special configuration for distributed training.

This module provides helper functions to handle these operations efficiently and correctly.

## Functions

### Metric and Loss Aggregation

#### `aggregate_metrics(metrics_dict: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]`

Aggregates a dictionary of metrics across all processes in a distributed environment.

**Parameters:**
- `metrics_dict`: Dictionary of metric names and their corresponding values.
- `world_size`: Number of processes in the distributed group.
- `device`: Device where tensor operations will be performed.

**Returns:**
- Dictionary with the same keys but values averaged across all processes.

**Example:**
```python
import torch.distributed as dist
from src.nn_handler.utils.dd_helpers import aggregate_metrics

# Calculate metrics on each process
local_metrics = {
    'accuracy': 0.85,
    'f1_score': 0.78
}

# Aggregate metrics across all processes
if dist.is_initialized():
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{dist.get_rank()}')
    global_metrics = aggregate_metrics(local_metrics, world_size, device)
    # global_metrics now contains the average accuracy and f1_score across all processes
```

#### `aggregate_loss(loss_value: float, world_size: int, device: torch.device) -> float`

Aggregates a loss value across all processes in a distributed environment.

**Parameters:**
- `loss_value`: The loss value to aggregate.
- `world_size`: Number of processes in the distributed group.
- `device`: Device where tensor operations will be performed.

**Returns:**
- The loss value averaged across all processes.

**Example:**
```python
import torch.distributed as dist
from src.nn_handler.utils.dd_helpers import aggregate_loss

# Calculate loss on each process
local_loss = model_loss.item()  # e.g., 0.342

# Aggregate loss across all processes
if dist.is_initialized():
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{dist.get_rank()}')
    global_loss = aggregate_loss(local_loss, world_size, device)
    # global_loss now contains the average loss across all processes
```

### Data Broadcasting

#### `broadcast_data(data, src: int = 0)`

Broadcasts any picklable data from the source rank to all processes in a distributed group.

**Parameters:**
- `data`: Data to broadcast (sent from source rank, received by all other ranks).
- `src`: Rank from which to broadcast (default: 0).

**Returns:**
- The broadcasted data, consistent across all ranks.

#### `broadcast_if_ddp(data, src: int = 0)`

Conditionally broadcasts data if DDP is initialized, otherwise returns the data unchanged.

**Parameters:**
- `data`: Data to potentially broadcast.
- `src`: Source rank for broadcasting (default: 0).

**Returns:**
- The broadcasted data if in DDP mode, otherwise the original data.

**Example:**
```python
import torch.distributed as dist
from src.nn_handler.utils.dd_helpers import broadcast_if_ddp

# Only rank 0 loads the configuration
config = None
if dist.get_rank() == 0:
    config = load_config_from_file('config.json')

# Broadcast to all ranks
config = broadcast_if_ddp(config)
# Now all ranks have the same config
```

### Distributed Data Loaders

#### `_create_distributed_loader(dataset: Dataset, loader_kwargs: Dict[str, Any], device: torch.device, log_fn: Callable, is_eval: bool = False) -> Tuple[DataLoader, DistributedSampler]`

Creates a DataLoader with a DistributedSampler for distributed training or evaluation.

**Parameters:**
- `dataset`: The dataset to load.
- `loader_kwargs`: Keyword arguments for the DataLoader.
- `device`: Device for computation.
- `log_fn`: Function for logging messages.
- `is_eval`: Whether this is an evaluation loader (affects shuffling and drop_last).

**Returns:**
- A tuple containing the DataLoader and its DistributedSampler.

#### `_create_rank_cached_dataloader(dataset: Dataset, loader_kwargs: Dict[str, Any], device: torch.device, log_fn: Callable, is_eval: bool = False)`

Creates a DataLoader that caches data per rank, optimized for certain distributed scenarios.

**Parameters:**
- `dataset`: The dataset to load.
- `loader_kwargs`: Keyword arguments for the DataLoader.
- `device`: Device for computation.
- `log_fn`: Function for logging messages.
- `is_eval`: Whether this is an evaluation loader.

**Returns:**
- A tuple containing the DataLoader and its sampler.

**Note:** These loader creation functions are typically used internally by the NNHandler class rather than called directly.

## Usage Notes

- These functions are designed to work with PyTorch's distributed package (`torch.distributed`).
- Most functions require that the distributed process group is already initialized.
- The aggregation functions handle potential NaN values gracefully.
- The distributed loader functions set sensible defaults for parameters like `num_workers` and `pin_memory` based on the environment.