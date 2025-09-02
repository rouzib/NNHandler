# Rank-Cached H5 Dataloader Module

The rank_cached_h5_dataloader module provides specialized PyTorch Dataset and Sampler implementations for efficiently loading and processing HDF5 data in distributed training environments.

## Overview

When training deep learning models in a distributed environment, efficient data loading is crucial for performance. This module addresses the challenges of using HDF5 files in distributed training by:

- Preloading each rank's data shard into memory once, eliminating repeated I/O
- Supporting different sharding strategies for balanced data distribution
- Providing memory optimization options (pinning, sharing)
- Implementing epoch-based shuffling for training

The module is specifically designed for distributed training with PyTorch's Distributed Data Parallel (DDP) and requires torch.distributed to be initialized.

## Helper Functions

### `_balanced_contiguous_shard`

```python
def _balanced_contiguous_shard(total: int, world_size: int, rank: int) -> Tuple[int, int]
```

Returns the start and end indices for a balanced contiguous shard of data for the current rank.

#### Parameters:

* `total` (int):
  * Total number of samples in the dataset.
* `world_size` (int):
  * Number of processes in the distributed training.
* `rank` (int):
  * Current process rank.

#### Returns:

* `Tuple[int, int]`: A tuple containing the start (inclusive) and end (exclusive) indices for the current rank's shard.

### `_interleaved_indices`

```python
def _interleaved_indices(total: int, world_size: int, rank: int) -> np.ndarray
```

Returns interleaved indices for the current rank's shard (rank, rank+world_size, rank+2*world_size, etc.).

#### Parameters:

* `total` (int):
  * Total number of samples in the dataset.
* `world_size` (int):
  * Number of processes in the distributed training.
* `rank` (int):
  * Current process rank.

#### Returns:

* `np.ndarray`: An array of indices for the current rank's shard.

## Classes

### `RankMemCachedH5Dataset`

```python
class RankMemCachedH5Dataset(Dataset):
    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            *,
            mode: str = "contiguous",
            dtype_x: Optional[np.dtype] = np.float32,
            dtype_y: Optional[np.dtype] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pin_host_memory: bool = True,
            share_memory: bool = False,
            rdcc_nbytes: int = 256 * 1024 ** 2,
            rdcc_nslots: int = 1_000_003,
            rdcc_w0: float = 0.75,
            swmr: bool = True,
            log_fn: Callable[[str], None] = print,
    )
```

A PyTorch Dataset that preloads the current rank's shard of an HDF5 dataset into host memory once, eliminating repeated I/O during training.

#### Parameters:

* `path` (str):
  * Path to the HDF5 file.
* `x_key` (str):
  * Key for the input data in the HDF5 file.
* `y_key` (Optional[str]):
  * Key for the target data in the HDF5 file. If None, only input data is returned.
* `mode` (str):
  * Sharding mode. Options:
    * "contiguous" (default): Each rank gets a contiguous slice of data (I/O efficient).
    * "interleave": Each rank gets interleaved indices (better for class balance).
* `dtype_x` (Optional[np.dtype]):
  * NumPy dtype to cast input data to. Default is np.float32.
* `dtype_y` (Optional[np.dtype]):
  * NumPy dtype to cast target data to. If None, no casting is performed.
* `transform` (Optional[Callable]):
  * Function to apply to input data.
* `target_transform` (Optional[Callable]):
  * Function to apply to target data.
* `pin_host_memory` (bool):
  * Whether to pin the cached tensors in host memory for faster GPU transfer. Default is True.
* `share_memory` (bool):
  * Whether to place cached tensors in shared memory. Default is False.
* `rdcc_nbytes` (int):
  * Size of HDF5 chunk cache in bytes. Default is 256 MB.
* `rdcc_nslots` (int):
  * Number of slots in the HDF5 chunk cache. Default is 1,000,003 (a large prime).
* `rdcc_w0` (float):
  * HDF5 chunk cache write strategy. Default is 0.75.
* `swmr` (bool):
  * Whether to use Single-Writer/Multiple-Reader mode. Default is True.
* `log_fn` (Callable[[str], None]):
  * Function for logging messages. Default is print.

#### Methods:

* `__len__()`: Returns the number of samples in the current rank's shard.
* `__getitem__(idx)`: Returns the sample at the given index from the cached shard.

### `EpochShuffleSampler`

```python
class EpochShuffleSampler(Sampler[int]):
    def __init__(self, data_len: int, seed: int = 0, shuffle: bool = True)
```

A PyTorch Sampler for a local dataset (already sharded) that reshuffles indices at each epoch.

#### Parameters:

* `data_len` (int):
  * Number of samples in the local dataset shard.
* `seed` (int):
  * Base seed for the random number generator. Default is 0.
* `shuffle` (bool):
  * Whether to shuffle the indices. Default is True.

#### Methods:

* `set_epoch(epoch)`: Sets the current epoch number for deterministic shuffling.
* `__iter__()`: Returns an iterator over the shuffled indices.
* `__len__()`: Returns the number of samples in the dataset shard.

## Usage Examples

### Basic Distributed Training Setup

```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from src.nn_handler.dataloaders.rank_cached_h5_dataloader import RankMemCachedH5Dataset, EpochShuffleSampler

# Initialize distributed environment (typically done with torch.distributed.launch or torchrun)
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Create dataset with rank-specific sharding
dataset = RankMemCachedH5Dataset(
    path="data/large_dataset.h5",
    x_key="images",
    y_key="labels",
    mode="contiguous",  # Each rank gets a contiguous slice
    pin_host_memory=True,  # Pin memory for faster GPU transfer
)

# Create sampler with epoch-based shuffling
sampler = EpochShuffleSampler(
    data_len=len(dataset),
    seed=42,  # Ensure reproducible shuffling across runs
    shuffle=True
)

# Create DataLoader (no need for DistributedSampler as sharding is handled by the dataset)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Use our custom sampler
    num_workers=4,  # Can use multiple workers safely
    pin_memory=False,  # Already pinned in dataset
)

# Model setup
model = YourModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop
for epoch in range(num_epochs):
    # Set epoch for reproducible shuffling
    sampler.set_epoch(epoch)
    
    for inputs, targets in dataloader:
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Training step
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # ... rest of training loop
```

### Using Interleaved Sharding for Class Balance

```python
# Create dataset with interleaved sharding for better class balance
dataset = RankMemCachedH5Dataset(
    path="data/imbalanced_dataset.h5",
    x_key="images",
    y_key="labels",
    mode="interleave",  # Each rank gets samples at stride=world_size
    dtype_x=np.float32,
    dtype_y=np.int64,
)

# Rest of setup is the same as the basic example
```

### Input-Only Dataset for Feature Extraction

```python
# Create dataset with only input data (no targets)
dataset = RankMemCachedH5Dataset(
    path="data/features.h5",
    x_key="embeddings",
    y_key=None,  # No targets
    mode="contiguous",
    dtype_x=np.float32,
)

# Create sampler without shuffling for deterministic order
sampler = EpochShuffleSampler(
    data_len=len(dataset),
    shuffle=False  # No shuffling for feature extraction
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=4,
)

# Feature extraction loop
features_list = []
for features in dataloader:
    features = features.cuda(non_blocking=True)
    # Process or store features
    processed_features = model(features)
    features_list.append(processed_features.cpu())
```

## Memory Optimization

The `RankMemCachedH5Dataset` provides several options for memory optimization:

1. **Pinned Host Memory**: Setting `pin_host_memory=True` (default) pins the cached tensors in host memory, which can significantly speed up transfers to GPU memory.

2. **Shared Memory**: Setting `share_memory=True` places the cached tensors in shared memory, which can be useful when using multiple processes that need access to the same data.

3. **Data Type Casting**: The `dtype_x` and `dtype_y` parameters allow you to control the precision of the cached data, potentially reducing memory usage.

## Notes for Distributed Training

1. **Initialization Requirement**: This module requires torch.distributed to be initialized before creating the dataset.

2. **Sharding Strategies**: 
   - "contiguous" mode is more I/O efficient as each rank reads a single contiguous chunk of data.
   - "interleave" mode can provide better class balance but may be less I/O efficient.

3. **Epoch-Based Shuffling**: Always call `sampler.set_epoch(epoch)` at the beginning of each epoch to ensure proper shuffling.

4. **Memory Usage**: Be mindful of memory usage when preloading large datasets. Each rank will cache its entire shard in memory.