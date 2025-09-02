# H5 Dataloader Module

The h5_dataloader module provides a PyTorch Dataset implementation for efficiently loading data from HDF5 files, with special consideration for distributed training and multiprocessing environments.

## Overview

HDF5 is a popular format for storing large datasets, especially in scientific computing. However, using HDF5 files with PyTorch's DataLoader in a distributed or multiprocessing environment can be challenging due to issues with file handle sharing across processes.

This module addresses these challenges by providing:
- Safe handling of HDF5 file handles in multiprocessing environments
- Lazy loading of data to minimize memory usage
- Support for input-target pairs or single inputs
- Optional data transformations
- HDF5 chunk cache tuning for performance optimization

## Classes and Functions

### `H5LazyDataset`

```python
class H5LazyDataset(Dataset):
    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            x_dtype: Optional[np.dtype] = None,
            y_dtype: Optional[np.dtype] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            rdcc_nbytes: int = 256 * 1024 ** 2,
            rdcc_nslots: int = 1_000_003,
            rdcc_w0: float = 0.75,
            swmr: bool = True,
    )
```

A PyTorch Dataset implementation for HDF5 files that:
- Opens the HDF5 file lazily in each worker/process (no shared handles)
- Supports optional (x_key, y_key) for (input, target) pairs
- Provides optional transforms for inputs and targets

#### Parameters:

* `path` (str):
  * Path to the HDF5 file.
* `x_key` (str):
  * Key for the input data in the HDF5 file.
* `y_key` (Optional[str]):
  * Key for the target data in the HDF5 file. If None, only input data is returned.
* `x_dtype` (Optional[np.dtype]):
  * NumPy dtype to cast input data to. If None, no casting is performed.
* `y_dtype` (Optional[np.dtype]):
  * NumPy dtype to cast target data to. If None, no casting is performed.
* `transform` (Optional[Callable]):
  * Function to apply to input data.
* `target_transform` (Optional[Callable]):
  * Function to apply to target data.
* `rdcc_nbytes` (int):
  * Size of HDF5 chunk cache in bytes. Default is 256 MB.
* `rdcc_nslots` (int):
  * Number of slots in the HDF5 chunk cache. Default is 1,000,003 (a large prime).
* `rdcc_w0` (float):
  * HDF5 chunk cache write strategy. Default is 0.75.
* `swmr` (bool):
  * Whether to use Single-Writer/Multiple-Reader mode. Default is True.

#### Methods:

* `__len__()`: Returns the number of samples in the dataset.
* `__getitem__(idx)`: Returns the sample at the given index.
* `close()`: Closes the HDF5 file handle.

### `h5_worker_init_fn`

```python
def h5_worker_init_fn(_worker_id: int)
```

A worker initialization function for PyTorch's DataLoader that ensures no accidentally inherited HDF5 handles remain when using multiple workers.

#### Parameters:

* `_worker_id` (int):
  * The ID of the worker process.

## Usage Examples

### Basic Usage

```python
import torch
from torch.utils.data import DataLoader
from src.nn_handler.dataloaders.h5_dataloader import H5LazyDataset, h5_worker_init_fn

# Create a dataset from an HDF5 file with input-target pairs
dataset = H5LazyDataset(
    path="data/my_dataset.h5",
    x_key="images",
    y_key="labels",
    x_dtype=np.float32,
    y_dtype=np.int64
)

# Create a DataLoader with worker initialization
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=h5_worker_init_fn
)

# Use the dataloader in a training loop
for inputs, targets in dataloader:
    # Training code here
    pass
```

### With Transformations

```python
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.nn_handler.dataloaders.h5_dataloader import H5LazyDataset, h5_worker_init_fn

# Define transformations
transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = lambda y: torch.tensor(y, dtype=torch.long)

# Create a dataset with transformations
dataset = H5LazyDataset(
    path="data/my_dataset.h5",
    x_key="images",
    y_key="labels",
    transform=transform,
    target_transform=target_transform
)

# Create a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=h5_worker_init_fn
)
```

### Input-Only Dataset

```python
import torch
from torch.utils.data import DataLoader
from src.nn_handler.dataloaders.h5_dataloader import H5LazyDataset, h5_worker_init_fn

# Create a dataset with only input data (no targets)
dataset = H5LazyDataset(
    path="data/features.h5",
    x_key="embeddings",
    y_key=None,  # No targets
    x_dtype=np.float32
)

# Create a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    worker_init_fn=h5_worker_init_fn
)

# Use for inference or feature extraction
for features in dataloader:
    # Process features
    pass
```

## Performance Considerations

1. **HDF5 Chunk Cache Tuning**: The parameters `rdcc_nbytes`, `rdcc_nslots`, and `rdcc_w0` control the HDF5 chunk cache. Tuning these can significantly improve performance, especially for large files on network file systems.

2. **Worker Initialization**: Always use the provided `h5_worker_init_fn` with DataLoader's `worker_init_fn` parameter to ensure proper cleanup of file handles.

3. **File Closing**: The dataset automatically closes the file handle when the object is deleted, but you can explicitly call `dataset.close()` if needed.

4. **SWMR Mode**: Single-Writer/Multiple-Reader mode (`swmr=True`) allows safe concurrent reading of the HDF5 file, which is useful in distributed environments.

## Notes for Distributed Training

When using this dataset in a distributed training environment:

1. Each process will open its own file handle, which is safe for concurrent reading.
2. For more efficient distributed training with HDF5 files, consider using the `RankMemCachedH5Dataset` from the `rank_cached_h5_dataloader` module, which preloads data shards into memory.