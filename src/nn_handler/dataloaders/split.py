import os

import numpy as np
import torch.distributed as dist


def make_train_val_indices(n_items: int,
                           val_fraction: float = 0.1,
                           seed: int = 1234,
                           save_path: str | None = None):
    """
    Create a deterministic train/val index split on rank 0 and broadcast to all ranks.
    Optionally save/load to/from disk so future runs reuse the same split.
    """
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    train_idx = None
    val_idx = None

    if rank == 0:
        if save_path and os.path.exists(save_path):
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
