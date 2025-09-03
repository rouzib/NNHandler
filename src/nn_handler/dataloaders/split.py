import os

import numpy as np
import torch.distributed as dist


def make_train_val_indices(n_items: int,
                           val_fraction: float = 0.1,
                           seed: int = 1234,
                           save_path: str | None = None,
                           overwrite: bool = False):
    """
    Generates train-validation indices by splitting a given dataset into training
    and validation sets. Optionally, the generated indices can be saved to or
    loaded from a file to ensure reproducibility. Supports distributed environments
    where the indices are broadcasted to all ranks.

    :param n_items: Number of items in the dataset.
    :type n_items: int
    :param val_fraction: Fraction of items to include in the validation set. Default is 0.1.
    :type val_fraction: float
    :param seed: Random seed for reproducibility. Default is 1234.
    :type seed: int
    :param save_path: Path to save or load the train-validation indices. If None,
        the split is not saved. Default is None.
    :type save_path: str or None
    :param overwrite: If True, overwrite existing split in the save path. If False,
        reuse existing split if available. Default is False.
    :type overwrite: bool
    :return: Tuple containing training indices as the first element and validation
        indices as the second element.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    train_idx = None
    val_idx = None

    if rank == 0:
        if save_path and os.path.exists(save_path) and not overwrite:
            # Reuse existing split
            data = np.load(save_path, allow_pickle=True)
            train_idx = data["train_idx"]
            val_idx = data["val_idx"]
        else:
            g = np.random.default_rng(seed)
            perm = g.permutation(n_items)
            n_val = int(round(val_fraction * n_items))
            val_idx = np.sort(perm[:n_val])
            train_idx = np.sort(perm[n_val:])
            if save_path:
                np.savez_compressed(save_path, train_idx=train_idx, val_idx=val_idx)

    # Broadcast to all ranks
    if world > 1:
        obj = [train_idx, val_idx]
        dist.broadcast_object_list(obj, src=0)
        train_idx, val_idx = obj

    return train_idx, val_idx
