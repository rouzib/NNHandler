import torch


def patchify(images, patch_size, return_pos=True, image_res=None):
    """
    Extracts randomly selected patches from input images and optionally computes positional encodings for the patches.
    Supports both unpadded and padded image scenarios, accounting for any central cropping applied to the images
    before patch extraction.

    :param images: Tensor of shape `(batch_size, channels, height, width)` containing image data.
    :param patch_size: Integer specifying the size of the square patch to be cropped from the images.
    :param return_pos: Boolean indicating whether positional encodings for the patches should be computed
        and returned. Default is True.
    :param image_res: Optional tuple `(height, width)` specifying the resolution of the original images,
        used if the input images are padded. Default is None.
    :return: Tuple containing:
        - **patches**: Tensor of shape `(batch_size, channels, patch_size, patch_size)` containing the cropped patches.
        - **pos_grids** (optional): Tensor of shape `(batch_size, 2, patch_size, patch_size)` containing positional
          encodings for the patches if return_pos is True.
    """
    device = images.device
    batch_size, _, res_h, res_w = images.size()

    if image_res is not None:
        h, w = image_res
    else:
        h, w = res_h, res_w
    th, tw = patch_size, patch_size

    # Compute offsets if image has been padded: find top-left corner of original image area
    offset_i = max(0, (res_h - h) // 2)
    offset_j = max(0, (res_w - w) // 2)

    if w == tw and h == th:
        i = torch.zeros((batch_size,), device=device, dtype=torch.long) + offset_i
        j = torch.zeros((batch_size,), device=device, dtype=torch.long) + offset_j
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=device) + offset_i
        j = torch.randint(0, w - tw + 1, (batch_size,), device=device) + offset_j

    # [batch_size, C, th, tw]
    patches = []
    pos_grids = []
    for n in range(batch_size):
        patch = images[n, :, i[n]:i[n] + th, j[n]:j[n] + tw].unsqueeze(0)
        patches.append(patch)
        # Positional encoding
        if return_pos:
            x_pos = torch.arange(tw, dtype=torch.float32, device=device) / (w - 1) * 2 - 1
            y_pos = torch.arange(th, dtype=torch.float32, device=device) / (h - 1) * 2 - 1
            mesh_y, mesh_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
            pos_grid = torch.stack([mesh_x, mesh_y], dim=0).unsqueeze(0)
            pos_grids.append(pos_grid)
    patches = torch.cat(patches, dim=0)
    if return_pos:
        pos_grids = torch.cat(pos_grids, dim=0)
        return patches, pos_grids
    else:
        return patches