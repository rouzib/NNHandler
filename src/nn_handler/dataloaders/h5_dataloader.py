from typing import Optional, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


class H5LazyDataset(Dataset):
    """
    Provides a lazy-loading dataset interface for HDF5-based datasets.

    This class is designed to handle HDF5 datasets with lazy loading, which
    avoids loading the entire dataset into memory. It supports optional
    transformation of inputs and targets, as well as optimized file access
    via specific HDF5 cache settings. The dataset design ensures compatibility
    with multi-process data loaders through processes like PyTorch's DataLoader.

    :ivar path: Path to the HDF5 file.
    :type path: str
    :ivar x_key: Key corresponding to the input dataset within the HDF5 file.
    :type x_key: str
    :ivar y_key: Key corresponding to the target dataset within the HDF5 file, optional.
    :type y_key: Optional[str]
    :ivar x_dtype: Desired dtype for input data after loading, optional.
    :type x_dtype: Optional[numpy.dtype]
    :ivar y_dtype: Desired dtype for target data after loading, optional.
    :type y_dtype: Optional[numpy.dtype]
    :ivar transform: Transformations applied to the input data, optional.
    :type transform: Optional[Callable]
    :ivar target_transform: Transformations applied to the target data, optional.
    :type target_transform: Optional[Callable]
    :ivar _h5_args: Dictionary of settings for HDF5 file access (including caching and
                    file locking strategies).
    :type _h5_args: dict
    :ivar _length: Number of samples in the dataset, lazily determined from the HDF5 file.
    :type _length: int
    :ivar _file: Internal handle to the open HDF5 file, initialized as None.
    :type _file: Optional[h5py.File]
    :ivar _x: Internal handle to the input dataset within the HDF5 file, initialized as None.
    :type _x: Optional[h5py.Dataset]
    :ivar _y: Internal handle to the target dataset within the HDF5 file, initialized as None.
    :type _y: Optional[h5py.Dataset]
    """

    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            x_dtype: Optional[np.dtype] = None,
            y_dtype: Optional[np.dtype] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            # HDF5 chunk cache tuning (can help a lot for large files on network FS)
            rdcc_nbytes: int = 256 * 1024 ** 2,  # 256 MB
            rdcc_nslots: int = 1_000_003,  # large prime
            rdcc_w0: float = 0.75,
            swmr: bool = True,  # read-only single-writer-multi-reader mode
    ):
        self.path = path
        self.x_key = x_key
        self.y_key = y_key
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.transform = transform
        self.target_transform = target_transform

        # HDF5 caching/opening settings
        self._h5_args = dict(mode="r", libver="latest", swmr=swmr,
                             rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=rdcc_w0)

        # Determine length without keeping the file handle open
        with h5py.File(self.path, "r") as f:
            self._length = f[self.x_key].shape[0]

        # File/dataset handles are per-process/worker; start closed
        self._file = None
        self._x = None
        self._y = None

    def _ensure_open(self):
        if self._file is None:
            # (Some HPCs require this env to avoid stale file lock errors on shared FS)
            # os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            self._file = h5py.File(self.path, **self._h5_args)
            self._x = self._file[self.x_key]
            self._y = self._file[self.y_key] if self.y_key is not None else None

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._x = None
            self._y = None

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        # Ensure we open the file handle lazily in this worker/process
        self._ensure_open()

        x = self._x[idx]
        if self.x_dtype is not None:
            x = x.astype(self.x_dtype, copy=False)
        x = torch.from_numpy(x)

        if self.y_key is None:
            if self.transform is not None:
                x = self.transform(x)
            return x

        y = self._y[idx]
        if self.y_dtype is not None:
            y = y.astype(self.y_dtype, copy=False)
        y = torch.from_numpy(y)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    # Make the object picklable without leaking open handles to subprocesses
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_x"] = None
        state["_y"] = None
        return state

    def __del__(self):
        self.close()


def h5_worker_init_fn(_worker_id: int):
    """
    Initializes the worker for the dataset in a multiprocessing DataLoader setting. This function ensures that each worker
    is properly initialized and closes any open dataset connections if applicable. It is particularly useful for managing
    datasets with file-based resources, such as HDF5 files, that require proper handling in a multi-worker environment.

    :param _worker_id: The ID of the worker process. Used to differentiate between multiple workers in cases where
                       specific worker initialization is necessary.
    :type _worker_id: int

    :return: None
    """
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "close"):
        info.dataset.close()
