# Dataloaders Module

The dataloaders module in the `nn_handler` package provides specialized PyTorch Dataset implementations for efficiently loading and processing data, with particular focus on HDF5 files and distributed training environments.

## Overview

Efficient data loading is a critical component of deep learning workflows, especially when working with large datasets or in distributed training environments. The dataloaders module addresses these challenges by providing:

- Optimized Dataset implementations for HDF5 files
- Support for distributed training with data sharding
- Memory-efficient loading strategies
- Tools for handling multiprocessing safely

## Components

The dataloaders module consists of several components, each addressing specific data loading scenarios:

### [H5 Dataloader](h5_dataloader.md)

The H5 Dataloader provides a PyTorch Dataset implementation for efficiently loading data from HDF5 files:

- `H5LazyDataset`: A Dataset that opens HDF5 files lazily in each worker/process
- `h5_worker_init_fn`: A worker initialization function for safe multiprocessing

Key features:
- Safe handling of HDF5 file handles in multiprocessing environments
- Lazy loading of data to minimize memory usage
- Support for input-target pairs or single inputs
- Optional data transformations
- HDF5 chunk cache tuning for performance optimization

### [Rank-Cached H5 Dataloader](rank_cached_h5_dataloader.md)

The Rank-Cached H5 Dataloader provides specialized Dataset and Sampler implementations for distributed training:

- `RankMemCachedH5Dataset`: A Dataset that preloads each rank's shard of an HDF5 dataset into memory
- `EpochShuffleSampler`: A Sampler that reshuffles indices at each epoch
- Helper functions for data sharding

Key features:
- Preloading each rank's data shard into memory once, eliminating repeated I/O
- Supporting different sharding strategies for balanced data distribution
- Providing memory optimization options (pinning, sharing)
- Implementing epoch-based shuffling for training

## Usage in Distributed Training

The dataloaders module is designed to work seamlessly with PyTorch's Distributed Data Parallel (DDP) framework:

1. **H5LazyDataset**: Safe for use in distributed environments, but each process loads data independently.

2. **RankMemCachedH5Dataset**: Specifically designed for distributed training, with built-in data sharding and memory optimization.

## Integration with NNHandler

The dataloaders can be used with the NNHandler class through methods like:

```python
from src.nn_handler import NNHandler
from src.nn_handler.dataloaders.h5_dataloader import H5LazyDataset, h5_worker_init_fn

# Create a dataset
dataset = H5LazyDataset(
    path="data/my_dataset.h5",
    x_key="images",
    y_key="labels"
)

# Set up the handler with the dataset
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Set the train loader with the dataset and worker initialization function
handler.set_train_loader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=h5_worker_init_fn
)
```

For distributed training with RankMemCachedH5Dataset:

```python
from src.nn_handler import NNHandler
from src.nn_handler.dataloaders.rank_cached_h5_dataloader import RankMemCachedH5Dataset, EpochShuffleSampler

# Create a dataset with rank-specific sharding
dataset = RankMemCachedH5Dataset(
    path="data/large_dataset.h5",
    x_key="images",
    y_key="labels",
    mode="contiguous"
)

# Create a sampler
sampler = EpochShuffleSampler(
    data_len=len(dataset),
    seed=42,
    shuffle=True
)

# Set the train loader with the dataset and sampler
handler = NNHandler(
    model_class=YourModel,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_distributed=True  # Enable distributed mode
)

handler.set_train_loader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4,
    pin_memory=False  # Already pinned in dataset
)

# Don't forget to set the epoch for the sampler before each epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    # Training code...
```

## Performance Considerations

1. **HDF5 Chunk Cache Tuning**: Both dataset implementations allow tuning the HDF5 chunk cache parameters, which can significantly improve performance, especially for large files on network file systems.

2. **Memory Usage**: The `RankMemCachedH5Dataset` preloads data into memory, which improves training speed but increases memory usage. Be mindful of this when working with very large datasets.

3. **Pinned Memory**: Pinning memory can speed up CPU-to-GPU transfers, but it also increases host memory usage. The `RankMemCachedH5Dataset` allows controlling this with the `pin_host_memory` parameter.

4. **Sharding Strategies**: The choice between "contiguous" and "interleave" sharding modes in `RankMemCachedH5Dataset` affects both I/O efficiency and class balance. Choose based on your specific dataset characteristics.

For more detailed information on each component, refer to the individual documentation pages linked above.