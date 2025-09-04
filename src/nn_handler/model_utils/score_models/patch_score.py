import torch
import torch.nn.functional as F


def make_pos_grid(x0: int, y0: int, p: int, H: int, device: torch.device):
    """Return a (1,2,p,p) tensor with x/y in [-1,1]."""
    xs = torch.arange(x0, x0 + p, device=device)
    ys = torch.arange(y0, y0 + p, device=device)
    xg = ((xs.view(1, -1).repeat(p, 1) / (H - 1)) - 0.5) * 2
    yg = ((ys.view(-1, 1).repeat(1, p) / (H - 1)) - 0.5) * 2
    return torch.stack([xg, yg], 0).unsqueeze(0)  # (1,2,p,p)


def patch_score_vectorized(nn_handler, t, x, patch_size=16, stride=6, patch_chunk=None, *conditions):
    """
    Computes a sliding-window score for patches extracted from an image tensor, evaluates them
    using the provided neural network handler, and reassembles the scores back into the image
    space. The computation is optionally chunked to handle memory constraints for large numbers
    of patches.

    The function relies on PyTorch's `F.unfold` and `F.fold` for efficient patch extraction and
    reassembly, respectively. It supports operations such as patch extraction, grid computation,
    model forward pass, and overlap normalization to ensure smooth blending of patch-based results.

    Args:
        nn_handler: A neural network handler providing the `score` method.
        t: A tensor representing the time step or index used as input to `nn_handler`.
        x: Input image tensor with shape (B, C, H, W) where B, C, H, and W are batch size,
            number of channels, height, and width, respectively.
        patch_size (int): Size of the sliding window (patch), which determines the patch height and width.
            Default is 16.
        stride (int): Stride used for sliding-window extraction, determining the distance between
            successive patches. Default is 6.
        patch_chunk (Optional[int]): Number of patches that are chunked together for processing when
            memory constraints are an issue. Default is None, indicating single-pass processing for
            all patches.
        *conditions: Additional arguments passed to the neural network handler, if supported.

    Returns:
        torch.Tensor: Output score tensor with shape (B, C, H, W), containing the normalized
        predictions from the neural network handler for the extracted patches.

    Raises:
        ValueError: If `patch_size` or `stride` does not allow valid patch extraction.
        RuntimeError: If the input tensor `x` has an unsupported shape or any device mismatch occurs.
    """
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # Number of sliding-window positions
    patch_H = (H - patch_size) // stride + 1
    patch_W = (W - patch_size) // stride + 1
    num_patches = patch_H * patch_W

    # Unfold image into sliding windows (view-like layout)
    # x_unf: (B, C*patch_size*patch_size, num_patches)
    x_unf = F.unfold(x, kernel_size=patch_size, stride=stride)

    # Helper: top-left coordinates (y, x) for each patch index [0..num_patches)
    rows = torch.arange(0, H - patch_size + 1, stride, device=device)
    cols = torch.arange(0, W - patch_size + 1, stride, device=device)
    ii, jj = torch.meshgrid(rows, cols, indexing='ij')  # (patch_H, patch_W)
    top_y = ii.reshape(-1)  # (num_patches,)
    left_x = jj.reshape(-1)  # (num_patches,)
    if t.ndim == 0:
        t = t.unsqueeze(0)

    if patch_chunk is None:
        # --- Single pass ---
        # Build all patch tensors
        x_patches = x_unf.permute(0, 2, 1).reshape(B * num_patches, C, patch_size, patch_size)

        # Build all position grids
        pos_list = [
            make_pos_grid(int(x0), int(y0), patch_size, H, nn_handler.device)  # (1,2,p,p)
            for y0, x0 in zip(top_y.tolist(), left_x.tolist())
        ]
        pos_grids = torch.cat(pos_list, dim=0)  # (num_patches, 2, p, p)
        pos_grids = pos_grids.unsqueeze(0).repeat(B, 1, 1, 1, 1).reshape(
            B * num_patches, 2, patch_size, patch_size
        )

        # Repeat t for all patches
        t_exp = t.unsqueeze(1).expand(B, num_patches).reshape(-1)

        # Model prediction: (B*num_patches, C, p, p)
        s = nn_handler.score(t_exp, x_patches, pos_grids, *conditions)

        # Prepare for fold: (B, C*p*p, num_patches)
        s_cols = s.reshape(B, num_patches, C * patch_size * patch_size).transpose(1, 2)

    else:
        # --- Chunked pass ---
        patch_chunk = max(1, int(patch_chunk))

        # We accumulate per-patch columns and do a single fold at the end.
        s_cols = torch.empty(
            B, C * patch_size * patch_size, num_patches, device=device, dtype=dtype
        )

        # Pre-expand time in a streaming-friendly way
        # (we materialize only per chunk below)
        for start in range(0, num_patches, patch_chunk):
            end = min(start + patch_chunk, num_patches)
            npi = end - start  # number of patches in this chunk

            # Take the corresponding patch columns (view), then reshape to (B*npi, C, p, p)
            x_chunk = x_unf[:, :, start:end].permute(0, 2, 1).reshape(
                B * npi, C, patch_size, patch_size
            )

            # Build only the needed position grids for this chunk
            pos_list = [
                make_pos_grid(int(x0), int(y0), patch_size, H, nn_handler.device)
                for y0, x0 in zip(top_y[start:end].tolist(), left_x[start:end].tolist())
            ]
            pos_chunk = torch.cat(pos_list, dim=0)  # (npi, 2, p, p)
            pos_chunk = pos_chunk.unsqueeze(0).repeat(B, 1, 1, 1, 1).reshape(
                B * npi, 2, patch_size, patch_size
            )

            # Time for this chunk
            t_chunk = t.unsqueeze(1).expand(B, npi).reshape(-1)

            # Score this chunk
            s_chunk = nn_handler.score(t_chunk, x_chunk, pos_chunk, *conditions)  # (B*npi, C, p, p)

            # Write into the correct columns so we can fold once at the end
            s_chunk_cols = s_chunk.reshape(B, npi, C * patch_size * patch_size).transpose(1, 2).contiguous()
            s_cols[:, :, start:end] = s_chunk_cols

    # Fold all patches back to image space (linear; equivalent to summing overlaps)
    score = F.fold(
        s_cols,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride,
    )

    # Overlap normalization (how many times each pixel was covered)
    ones = torch.ones((B, C, H, W), device=device, dtype=dtype)
    overlap = F.fold(
        F.unfold(ones, kernel_size=patch_size, stride=stride),
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride,
    )

    return score / overlap