from typing import Optional, Tuple, Callable, Iterator, Union, Dict

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


def _balanced_contiguous_shard(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    """
    Calculate a balanced, contiguous range (shard) for a given total size, evenly
    distributed across multiple workers (world size). Each rank (worker) will
    receive its respective shard, ensuring minimal difference in sizes between
    the shards.

    :param total: The total size to divide into shards.
    :type total: int
    :param world_size: The total number of workers among which the shards will
        be distributed.
    :type world_size: int
    :param rank: The rank of the current worker for which the shard is to be
        calculated (zero-based index).
    :type rank: int
    :return: A tuple representing the start (inclusive) and end (exclusive)
        indices of the shard assigned to the given rank.
    :rtype: Tuple[int, int]
    """
    per = total // world_size
    rem = total % world_size
    if rank < rem:
        start = rank * (per + 1)
        end = start + (per + 1)
    else:
        start = rem * (per + 1) + (rank - rem) * per
        end = start + per
    return start, end


def _interleaved_indices(total: int, world_size: int, rank: int) -> np.ndarray:
    """
    Generate interleaved indices for a distributed system, where indices are distributed
    across multiple ranks. The function calculates indices that are evenly spaced
    based on the rank of the current process and the total number of processes (world size).

    :param total: Total number of elements to be indexed.
    :type total: int
    :param world_size: The number of distributed workers or ranks.
    :type world_size: int
    :param rank: The rank of the current worker or process.
    :type rank: int
    :return: Numpy array containing interleaved indices for the specified rank.
    :rtype: np.ndarray
    """
    return np.arange(rank, total, world_size, dtype=np.int64)


class RankMemCachedH5Dataset(Dataset):
    """
    Provides a distributed HDF5 dataset with caching and preloading capabilities, optimized for
    multi-process environments and large-scale data handling.

    This dataset is designed for efficient use in distributed training by assigning
    shards of data to each rank in a contiguous or interleaved manner. Additionally,
    it supports optional caching of the assigned shard in RAM for faster access.
    Custom transformations can be applied to both the input data and target labels.

    :ivar path: Path to the HDF5 file containing the dataset.
    :vartype path: str
    :ivar x_key: HDF5 key identifying the dataset for the input features.
    :vartype x_key: str
    :ivar y_key: HDF5 key identifying the dataset for the target labels, if applicable.
    :vartype y_key: Optional[str]
    :ivar mode: Determines shard assignment strategy; either "contiguous" or "interleave".
    :vartype mode: str
    :ivar x_dtype: Desired data type for input features after loading from HDF5.
    :vartype x_dtype: Optional[np.dtype]
    :ivar y_dtype: Desired data type for target labels after loading from HDF5.
    :vartype y_dtype: Optional[np.dtype]
    :ivar transform: Callable transformation applied to input features.
    :vartype transform: Optional[Callable]
    :ivar target_transform: Callable transformation applied to target labels.
    :vartype target_transform: Optional[Callable]
    :ivar pin_host_memory: Whether to pin the cached shard in host memory for faster I/O.
    :vartype pin_host_memory: bool
    :ivar share_memory: Whether to place cached tensors in shared memory for multi-process access.
    :vartype share_memory: bool
    :ivar log: Logging function for status updates and warnings.
    :vartype log: Callable[[str], None]
    :ivar rank: The current rank of the distributed process.
    :vartype rank: int
    :ivar world_size: Total number of processes in the distributed setup.
    :vartype world_size: int
    :ivar _N: Total number of samples in the dataset.
    :vartype _N: int
    :ivar _global_idx: Global indices assigned to the current rank.
    :vartype _global_idx: np.ndarray
    :ivar _len: Number of samples assigned to the current rank.
    :vartype _len: int
    :ivar _preload_kwargs: Arguments used for preloading HDF5 shards.
    :vartype _preload_kwargs: dict
    :ivar _x_cache: Cached input features for the current rank's shard.
    :vartype _x_cache: Optional[torch.Tensor]
    :ivar _y_cache: Cached target labels for the current rank's shard, if applicable.
    :vartype _y_cache: Optional[torch.Tensor]
    """

    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            aux_keys: Optional[Union[str, Tuple[str, ...]]] = None,
            *,
            mode: str = "contiguous",  # "contiguous" | "interleave"
            x_dtype: Optional[np.dtype] = np.float32,
            y_dtype: Optional[np.dtype] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pin_host_memory: bool = True,  # pin the whole shard in host RAM
            share_memory: bool = False,  # place cached tensors in torch shared memory
            rdcc_nbytes: int = 256 * 1024 ** 2,
            rdcc_nslots: int = 1_000_003,
            rdcc_w0: float = 0.75,
            swmr: bool = True,
            log_fn: Callable[[str], None] = print,
            restrict_to_indices: Optional[Union[np.ndarray, "Sequence[int]"]] = None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("RankMemCachedH5Dataset requires torch.distributed to be initialized.")

        self.path = path
        self.x_key = x_key
        self.y_key = y_key
        self.aux_keys = aux_keys
        self.mode = mode
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.transform = transform
        self.target_transform = target_transform
        self.pin_host_memory = pin_host_memory
        self.share_memory = share_memory
        self.log = log_fn

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Determine global length
        with h5py.File(self.path, "r") as f:
            self._N = int(f[self.x_key].shape[0])

        # Base index set (either full range or provided subset)
        if restrict_to_indices is None:
            base = np.arange(self._N, dtype=np.int64)
        else:
            base = np.asarray(restrict_to_indices, dtype=np.int64)
            # sanity: clip to valid range and sort to keep I/O friendly for contiguous mode
            base = base[(base >= 0) & (base < self._N)]
            base.sort()

        # Compute this-rank indices from 'base'
        if self.mode == "contiguous":
            start, end = _balanced_contiguous_shard(base.size, self.world_size, self.rank)
            self._global_idx = base[start:end]
        elif self.mode == "interleave":
            local = _interleaved_indices(base.size, self.world_size, self.rank)
            self._global_idx = base[local]
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        self._len = int(self._global_idx.size)

        # Preload shard once
        self._preload_kwargs = dict(mode="r", libver="latest", swmr=swmr,
                                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=rdcc_w0)
        self._x_cache: Optional[torch.Tensor] = None
        self._y_cache: Optional[torch.Tensor] = None
        self._aux_caches: Optional[Dict[str, torch.tensor]] = {key: None for key in self.aux_keys}

        self._preload_shard()

    def _preload_shard(self) -> None:
        """Read this rank's indices from HDF5 and hold a single big cached tensor in RAM."""
        with h5py.File(self.path, **self._preload_kwargs) as f:
            X = f[self.x_key]
            if self.y_key is not None:
                Y = f[self.y_key]
            # Read in one go (fastest if dataset is chunked sensibly)
            x_np = X[self._global_idx]
            if self.x_dtype is not None:
                x_np = x_np.astype(self.x_dtype, copy=False)
            x_t = torch.from_numpy(x_np)

            if self.y_key is not None:
                y_np = Y[self._global_idx]
                if self.y_dtype is not None:
                    y_np = y_np.astype(self.y_dtype, copy=False)
                y_t = torch.from_numpy(y_np)
            else:
                y_t = None

            for key in self.aux_keys:
                aux_np = f[key][self._global_idx]
                aux_t = torch.from_numpy(aux_np)
                self._aux_caches[key] = aux_t

        # Optional memory placement tweaks
        if self.share_memory:
            x_t = x_t.share_memory_()
            if y_t is not None:
                y_t = y_t.share_memory_()
            for key in self.aux_keys:
                if self._aux_caches[key] is not None:
                    self._aux_caches[key] = self._aux_caches[key].share_memory_()

        if self.pin_host_memory:
            try:
                x_t = x_t.pin_memory()
                if y_t is not None:
                    y_t = y_t.pin_memory()
                for key in self.aux_keys:
                    if self._aux_caches[key] is not None:
                        self._aux_caches[key] = self._aux_caches[key].pin_memory()
            except RuntimeError:
                # Pinning can fail on some systems; fall back gracefully
                self.log(f"[rank {self.rank}] Warning: pin_memory() failed; continuing unpinned.")

        self._x_cache = x_t
        self._y_cache = y_t

        # Log memory footprint (rough)
        bytes_x = self._x_cache.element_size() * self._x_cache.numel()
        bytes_y = 0 if self._y_cache is None else self._y_cache.element_size() * self._y_cache.numel()
        gb = (bytes_x + bytes_y) / (1024 ** 3)
        self.log(f"[rank {self.rank}] Cached shard: {self._len} samples ({gb:.2f} GB)")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        x = self._x_cache[idx]
        if self.aux_keys is not None:
            aux = {key: self._aux_caches[key][idx] for key in self.aux_keys}
        else:
            aux = None
        if self._y_cache is None:
            if aux:
                return self.transform(x, aux) if self.transform else x
            else:
                return self.transform(x) if self.transform else x
        y = self._y_cache[idx]
        if self.transform:
            x = self.transform(x, aux) if aux else self.transform(x)
        if self.target_transform:
            y = self.target_transform(y, aux) if aux else self.target_transform(y)
        return x, y


class EpochShuffleSampler(Sampler[int]):
    """
    A sampler class for shuffling datasets with per-epoch control.

    Provides functionality to shuffle dataset indices deterministically using a
    base seed and epoch number. The class allows setting the epoch to alter the
    shuffling sequence for each epoch, ensuring reproducibility across different
    epochs.

    :ivar data_len: The total number of items in the dataset.
    :type data_len: int
    :ivar base_seed: The base seed used for initializing the random number generator.
    :type base_seed: int
    :ivar shuffle: Flag to determine whether shuffling of indices is enabled or not.
    :type shuffle: bool
    :ivar epoch: The current epoch number used to modify the shuffling sequence.
    :type epoch: int
    """

    def __init__(self, data_len: int, seed: int = 0, shuffle: bool = True):
        super().__init__()
        self.data_len = int(data_len)
        self.base_seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        idx = np.arange(self.data_len, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.base_seed + self.epoch)
            rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.data_len
