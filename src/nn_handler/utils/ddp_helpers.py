import os
from typing import Dict, Any, Tuple, Callable
import math

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader, Dataset, SequentialSampler

from ..dataloaders.rank_cached_h5_dataloader import EpochShuffleSampler


def _collective_device_for_backend() -> torch.device:
    """
    Choose a tensor device compatible with the current process group backend.
    - NCCL / UCC -> CUDA tensor on the current device
    - GLOO / (others that support CPU) -> CPU tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        # default to CPU if not in DDP (won't be used by collectives anyway)
        return torch.device("cpu")

    backend_name = str(dist.get_backend()).lower()
    if backend_name in ("nccl", "ucc"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{backend_name.upper()} backend requires CUDA available")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    # gloo or anything else defaulting to CPU tensors
    return torch.device("cpu")


def aggregate_metrics(metrics_dict: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
    """
    Aggregates metrics across distributed processes. This function reduces a
    dictionary of metrics from all processes into a single dictionary of aggregated
    metrics on each process. It assumes that all ranks have the same keys in their
    `metrics_dict`.

    :param metrics_dict:
        A dictionary where keys represent metric names and values represent their
        corresponding scalar values to be aggregated. The keys are expected to
        match across all ranks.
    :param world_size:
        An integer indicating the number of processes in the distributed world.
        Aggregation is performed only if `world_size > 1`.
    :param device:
        A PyTorch device specifying where the tensor reductions should be performed.

    :return:
        A dictionary containing the aggregated metrics. Each key corresponds
        to the input metric name, and its value is the average of the metrics
        across all processes in the distributed group.
    """
    if world_size <= 1:
        return metrics_dict

    aggregated_metrics = {}
    for name, value in metrics_dict.items():
        # Handle potential NaN values - don't reduce them or use a placeholder?
        # Option 1: Skip NaN reduction (might lead to missing metric on rank 0)
        # Option 2: Reduce count of non-NaNs and sum, then divide. More complex.
        # Option 3: Reduce as is, hoping NaN doesn't propagate badly (risky).
        # Let's stick to simple average, assuming NaNs are infrequent errors handled elsewhere.
        # If a value is NaN on one rank, the average might become NaN.
        metric_tensor = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
        aggregated_metrics[name] = metric_tensor.item()
    return aggregated_metrics


def aggregate_loss(loss_value: float, world_size: int, device: torch.device) -> float:
    """
    Aggregates a loss value across multiple processes in a distributed environment when necessary.
    This function reduces the loss value by computing the average of the loss values across
    all participating processes. If the world is not distributed (i.e., `world_size <= 1`) or the
    loss value is NaN, the function simply returns the original loss value without performing
    any aggregation.

    :param loss_value: The loss value to be potentially aggregated.
    :type loss_value: float
    :param world_size: The number of processes participating in the distributed environment.
    :type world_size: int
    :param device: The torch device on which the operation is performed
                   and where the loss tensor is allocated.
    :type device: torch.device
    :return: The aggregated loss value if in a distributed environment; otherwise, the input
             loss value unchanged.
    :rtype: float
    """
    if world_size <= 1 or math.isnan(loss_value):  # Don't aggregate if not distributed or NaN
        return loss_value
    loss_tensor = torch.tensor(loss_value, device=device, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def broadcast_data(data, src: int = 0):
    """
    Broadcasts any picklable data from the source rank to all processes in a distributed group.

    :param data: Data/object to broadcast (will be sent from src rank and received by all other ranks).
                 On non-src ranks, pass a dummy value (will be overwritten).
    :param src: Rank from which the data will be broadcast.
    :return: The broadcasted data, consistent across all ranks after the call.
    """
    obj_list = [data]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def broadcast_tensor(tensor: torch.Tensor, src: int = 0):
    """
    Broadcasts a tensor from the source rank to all processes in a distributed group.

    :param tensor: Tensor to broadcast (will be sent from src rank and received by all other ranks).
                   On non-src ranks, pass a tensor of the same shape (will be overwritten).
    :param src: Rank from which the tensor will be broadcast.
    :return: The broadcasted tensor, consistent across all ranks after the call.
    """
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_if_ddp(data, src: int = 0):
    """
    Broadcasts data across distributed processes if Distributed Data Parallel (DDP)
    is initialized. If DDP is not initialized, the function returns the input data
    unchanged. This function is particularly useful in environments where data
    needs to be consistent across multiple processes.

    :param data: The input data to be broadcasted. It can be of any type
        depending on the specific application context.
    :param src: The source rank from which the data will be broadcasted.
        Defaults to 0.
    :return: The broadcasted data if DDP is initialized. Otherwise, returns
        the input data unchanged.
    """
    if dist.is_initialized():
        if isinstance(data, torch.Tensor):
            return broadcast_tensor(data, src=src)
        else:
            return broadcast_data(data, src=src)
    else:
        return data


def _create_distributed_loader(dataset: Dataset, loader_kwargs: Dict[str, Any], device: torch.device, log_fn: Callable,
                               is_eval: bool = False) -> \
        Tuple[DataLoader, DistributedSampler]:
    """
    Creates a distributed `DataLoader` and corresponding `DistributedSampler` for use in a distributed
    data parallel (DDP) environment. The function is designed to handle DDP-specific requirements like
    synchronizing data shuffling and ensuring proper batch handling across multiple processes.

    The `DataLoader` is created based on the provided dataset and loader configurations while ensuring
    the appropriate defaults are applied for distributed training or evaluation. It handles the
    selection of the `DistributedSampler`, configuration of shuffle and drop_last behavior, and other
    important parameters like `num_workers` and `pin_memory`.

    :param dataset: The dataset object to be used for the `DataLoader`.
    :type dataset: Dataset
    :param loader_kwargs: A dictionary of keyword arguments for configuring the `DataLoader`.
    :type loader_kwargs: Dict[str, Any]
    :param device: The device where computations will be performed. Used to set options like `pin_memory`.
    :type device: torch.device
    :param log_fn: Function to log messages with a single string argument.
    :type log_fn: Callable
    :param is_eval: A boolean flag indicating whether the loader is being created for evaluation.
    :type is_eval: bool, optional
    :return: A tuple containing the created `DataLoader` and its associated `DistributedSampler`.
    :rtype: Tuple[DataLoader, DistributedSampler]
    """
    if not dist.is_initialized():
        # This should not be called in non-distributed mode, but check defensively
        raise RuntimeError("Internal Error: _create_distributed_loader called in non-DDP mode.")

    # Determine shuffle and drop_last for the sampler
    # Training: usually shuffle=True, drop_last=True (recommended for DDP)
    # Eval: shuffle=False, drop_last=False (usually)
    default_shuffle = not is_eval
    shuffle = loader_kwargs.get('shuffle', default_shuffle)  # User can override default

    # DDP often benefits from drop_last=True during training to avoid hangs
    # if the last batch is smaller and syncs incorrectly. For eval, usually False.
    default_drop_last = not is_eval
    drop_last = loader_kwargs.get('drop_last', default_drop_last)  # User can override

    # Create the DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=drop_last
    )

    # Prepare DataLoader kwargs
    new_loader_kwargs = loader_kwargs.copy()
    # Sampler replaces shuffle and batch_sampler
    new_loader_kwargs['sampler'] = sampler
    new_loader_kwargs['shuffle'] = False  # Sampler handles shuffling
    if 'batch_sampler' in new_loader_kwargs:
        del new_loader_kwargs['batch_sampler']  # Sampler is mutually exclusive
    new_loader_kwargs['drop_last'] = drop_last  # Ensure DataLoader knows drop_last

    # Set sensible defaults if not provided
    new_loader_kwargs.setdefault('batch_size', 1)  # Need a default batch size
    # Default num_workers based on SLURM env var or 0
    new_loader_kwargs.setdefault('num_workers',
                                 int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_PER_GPU", 0))))
    # Enable pin_memory if using CUDA device
    new_loader_kwargs.setdefault('pin_memory', device.type == 'cuda')
    # Set persistent_workers based on num_workers
    new_loader_kwargs.setdefault('persistent_workers', new_loader_kwargs['num_workers'] > 0)

    # Create the DataLoader
    new_loader = DataLoader(dataset, **new_loader_kwargs)

    log_fn(
        f"Created DDP DataLoader ({'Eval' if is_eval else 'Train'}) for {type(dataset).__name__}: "
        f"Shuffle={shuffle}, DropLast={drop_last}, BatchSize={loader_kwargs['batch_size']}, "
        f"Workers={new_loader_kwargs['num_workers']}, PinMem={new_loader_kwargs['pin_memory']}, "
        f"PersistWrk={new_loader_kwargs['persistent_workers']}")

    return new_loader, sampler


def _create_rank_cached_dataloader(dataset: Dataset, loader_kwargs: Dict[str, Any], device: torch.device,
                                   log_fn: Callable, is_eval: bool = False):
    if is_eval:
        sampler = SequentialSampler(dataset)
        epoch_sampler = None
    else:
        seed = loader_kwargs.get("seed", torch.random.default_generator.seed())
        epoch_sampler = EpochShuffleSampler(len(dataset), seed=seed, shuffle=True)
        sampler = epoch_sampler

    # sampler drives ordering
    loader_kwargs.setdefault("shuffle", False)
    loader_kwargs["shuffle"] = False
    # NOTE: num_workers=0 to avoid duplicating the large cached tensor.
    loader_kwargs.setdefault("num_workers", 0)
    loader_kwargs["num_workers"] = 0
    loader_kwargs.setdefault('pin_memory', device.type == 'cuda')

    loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)

    log_fn(
        f"[rank {dist.get_rank()}] Cached Loader ({'Eval' if is_eval else 'Train'}) "
        f"BatchSize={loader_kwargs.get('batch_size', 1)}, PinHost={loader_kwargs.get('pin_memory', False)}, "
        f"NumWorkers=0, Seed={epoch_sampler.base_seed if epoch_sampler is not None else 'N/A'}"
    )

    return loader, epoch_sampler
