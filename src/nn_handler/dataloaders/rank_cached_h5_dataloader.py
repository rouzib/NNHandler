from typing import Optional, Tuple, Callable, Iterator, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


def _balanced_contiguous_shard(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    """Return [start, end) for a balanced contiguous shard."""
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
    """Interleaved shard: indices rank, rank+world_size, ... (good for class balance if needed)."""
    return np.arange(rank, total, world_size, dtype=np.int64)


class RankMemCachedH5Dataset(Dataset):
    """
    Preloads the current rank's shard of an HDF5 dataset into host memory ONCE.
    - No shared HDF5 handles during iteration.
    - Optionally pin memory for faster H2D copies.
    - Supports X only or (X, Y).

    Sharding modes:
      - mode='contiguous' (default): single contiguous slice per rank (I/O efficient).
      - mode='interleave': interleaved indices rank, rank+world_size, ... (class-balance friendly).
    """

    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            *,
            mode: str = "contiguous",  # "contiguous" | "interleave"
            dtype_x: Optional[np.dtype] = np.float32,
            dtype_y: Optional[np.dtype] = None,
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
        self.mode = mode
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
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
        self._preload_shard()

    def _preload_shard(self) -> None:
        """Read this rank's indices from HDF5 and hold a single big cached tensor in RAM."""
        with h5py.File(self.path, **self._preload_kwargs) as f:
            X = f[self.x_key]
            if self.y_key is not None:
                Y = f[self.y_key]
            # Read in one go (fastest if dataset is chunked sensibly)
            x_np = X[self._global_idx]
            if self.dtype_x is not None:
                x_np = x_np.astype(self.dtype_x, copy=False)
            x_t = torch.from_numpy(x_np)

            if self.y_key is not None:
                y_np = Y[self._global_idx]
                if self.dtype_y is not None:
                    y_np = y_np.astype(self.dtype_y, copy=False)
                y_t = torch.from_numpy(y_np)
            else:
                y_t = None

        # Optional memory placement tweaks
        if self.share_memory:
            x_t = x_t.share_memory_()
            if y_t is not None:
                y_t = y_t.share_memory_()

        if self.pin_host_memory:
            try:
                x_t = x_t.pin_memory()
                if y_t is not None:
                    y_t = y_t.pin_memory()
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
        if self._y_cache is None:
            return self.transform(x) if self.transform else x
        y = self._y_cache[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class EpochShuffleSampler(Sampler[int]):
    """
    Sampler for a LOCAL dataset (already sharded) that reshuffles each epoch.
    Call .set_epoch(epoch) before each epoch.
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
