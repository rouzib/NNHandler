from typing import Optional, Callable, Union, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


class H5LazyDataset(Dataset):
    """
    Handles the lazy loading of datasets stored in HDF5 files.

    This class manages the efficient retrieval, transformation, and usage of large datasets
    stored in HDF5 files. It supports optional transformation of both inputs and targets,
    single-writer-multi-reader mode, caching options for HDF5 files, and auxiliary data retrieval.

    :ivar path: Path to the HDF5 file containing the dataset.
    :type path: str
    :ivar x_key: Key for the main dataset (input data) within the HDF5 file.
    :type x_key: str
    :ivar y_key: Optional key for the target dataset (label data) within the HDF5 file.
    :type y_key: Optional[str]
    :ivar aux_keys: Optional key or tuple of keys for auxiliary datasets within the HDF5 file.
    :type aux_keys: Optional[Union[str, Tuple[str, ...]]]
    :ivar x_dtype: Optional data type for the input dataset if casting is required.
    :type x_dtype: Optional[np.dtype]
    :ivar y_dtype: Optional data type for the target dataset if casting is required.
    :type y_dtype: Optional[np.dtype]
    :ivar transform: Function or callable to apply transformations to the input data and auxiliary data.
    :type transform: Optional[Callable]
    :ivar target_transform: Function or callable to apply transformations to the target data and auxiliary data.
    :type target_transform: Optional[Callable]
    :ivar _length: Length of the main dataset (number of samples).
    :type _length: int
    :ivar _h5_args: Internal dictionary of arguments for configuring HDF5 file access (includes caching options).
    :type _h5_args: dict
    :ivar _file: Internal handle to the opened HDF5 file, initialized as None and managed lazily.
    :type _file: Optional[h5py.File]
    :ivar _x: Internal reference to the input dataset within the opened HDF5 file, initialized as None.
    :type _x: Optional[h5py.Dataset]
    :ivar _y: Internal reference to the target dataset within the opened HDF5 file, initialized as None if not provided.
    :type _y: Optional[h5py.Dataset]
    :ivar _aux: Internal reference to auxiliary dataset handles in the HDF5 file, initialized as None if not provided.
    :type _aux: Optional[dict]
    """

    def __init__(
            self,
            path: str,
            x_key: str,
            y_key: Optional[str] = None,
            aux_keys: Optional[Union[str, Tuple[str, ...]]] = None,
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
        self.aux_keys = aux_keys
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
        self._aux = {}

    def _ensure_open(self):
        if self._file is None:
            # (Some HPCs require this env to avoid stale file lock errors on shared FS)
            # os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            self._file = h5py.File(self.path, **self._h5_args)
            self._x = self._file[self.x_key]
            self._y = self._file[self.y_key] if self.y_key is not None else None
            if self.aux_keys is not None:
                for key in self.aux_keys:
                    self._aux[key] = self._file[key]

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._x = None
            self._y = None
            self._aux = None

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        # Ensure we open the file handle lazily in this worker/process
        self._ensure_open()

        x = self._x[idx]
        if self.x_dtype is not None:
            x = x.astype(self.x_dtype, copy=False)
        x = torch.from_numpy(x)

        if self.aux_keys is not None:
            aux = {}
            for key in self.aux_keys:
                aux[key] = self._aux[key][idx]
        else:
            aux = None

        if self.y_key is None:
            if self.transform is not None:
                x = self.transform(x, aux) if aux else self.transform(x)
            return x


        y = self._y[idx]
        if self.y_dtype is not None:
            y = y.astype(self.y_dtype, copy=False)
        y = torch.from_numpy(y)

        if self.transform is not None:
            x = self.transform(x, aux) if aux else self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y, aux) if aux else self.target_transform(y)
        return x, y

    # Make the object picklable without leaking open handles to subprocesses
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_x"] = None
        state["_y"] = None
        state["_aux"] = None
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
