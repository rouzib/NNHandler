import torch


def denoising_score_matching(samples, sde, model, device, *args):
    """
    Compute the Denoising Score Matching (DSM) loss for the given samples.

    This function calculates the DSM loss by comparing the noise in the perturbed
    samples with the predicted noise from the provided model. The DSM loss is a
    standard approach in denoising score matching, where the goal is to learn the
    score function of a probability distribution.

    Args:
        samples: A tensor of shape [B, *D], where B is the batch size and *D denotes
            the spatial dimensions of the samples. These represent the data samples
            to compute the DSM loss on.
        sde: An object representing the stochastic differential equation (SDE) model.
            It should contain methods for obtaining the marginal mean and noise
            standard deviation given a time point.
        model: A neural network model that predicts the noise in the perturbed samples.
        device: The device (e.g., 'cuda' or 'cpu') where the computation will take place.
        *args: Optional additional arguments to be passed to the model.

    Returns:
        A scalar tensor representing the DSM loss, reduced over the batch size.
    """
    B, *D = samples.shape
    z = torch.randn_like(samples)
    t = torch.rand(B).to(device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, samples)
    return torch.sum((z + model(t, mean + sigma * z, *args)) ** 2) / B


def patch_denoising_score_matching(
    samples: torch.Tensor,
    sde,
    model,
    device: torch.device,
    patch_sizes,
    patches_per_sample: int = 1,
    *extra_conditions
) -> torch.Tensor:
    """
    Computes the DSM loss using patches. Allows batching & multiple patches per input.

    Args:
        samples (torch.Tensor): [B, C, H, W] input batch.
        sde: SDE object.
        model: Score model.
        device (torch.device)
        patch_sizes: List of possible patch sizes.
        patches_per_sample (int): How many patches to generate per sample in this batch.
        *extra_conditions: Additional model conditions.

    Returns:
        torch.Tensor: Scalar loss.
    """

    def pachify(images, patch_size):
        device = images.device
        batch_size, _, res_h, res_w = images.size()

        h, w = res_h, res_w
        th, tw = patch_size, patch_size

        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device, dtype=torch.long)
            j = torch.zeros((batch_size,), device=device, dtype=torch.long)
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        # [batch_size, C, th, tw]
        patches = []
        pos_grids = []
        for n in range(batch_size):
            patch = images[n, :, i[n]:i[n]+th, j[n]:j[n]+tw].unsqueeze(0)
            patches.append(patch)
            # Positional encoding
            x_pos = torch.arange(tw, dtype=torch.float32, device=device) / (w - 1) * 2 - 1
            y_pos = torch.arange(th, dtype=torch.float32, device=device) / (h - 1) * 2 - 1
            mesh_y, mesh_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
            pos_grid = torch.stack([mesh_x, mesh_y], dim=0).unsqueeze(0)
            pos_grids.append(pos_grid)
        patches = torch.cat(patches, dim=0)
        pos_grids = torch.cat(pos_grids, dim=0)
        return patches, pos_grids

    B, C, H, W = samples.shape
    total_patches = B * patches_per_sample

    # Repeat each sample for `patches_per_sample` to get N patches per image
    expanded_samples = samples.repeat_interleave(patches_per_sample, dim=0)
    expanded_conditions = [
        cond.repeat_interleave(patches_per_sample, dim=0)
        if isinstance(cond, torch.Tensor) and cond.size(0) == B else cond
        for cond in extra_conditions
    ]

    # Randomly choose a patch size for each patch
    patch_size_choices = torch.randint(0, len(patch_sizes), (total_patches,), device=device)
    patch_sizes_for_each = [patch_sizes[i.item()] for i in patch_size_choices]

    patches = []
    pos_grids = []
    for idx, p in enumerate(patch_sizes_for_each):
        patch, pos = pachify(expanded_samples[idx:idx+1], p)  # Shape: [1, C, p, p], [1, 2, p, p]
        patches.append(patch)
        pos_grids.append(pos)
    patches = torch.cat(patches, dim=0)
    pos_grids = torch.cat(pos_grids, dim=0)

    z = torch.randn_like(patches)
    t = torch.rand(total_patches, device=device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, patches)

    noisy = mean + sigma * z
    score = model(t, noisy, pos_grids, *expanded_conditions)
    loss = torch.sum((z + score) ** 2, dim=[1, 2, 3]).mean()
    return loss