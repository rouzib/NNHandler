import torch

from .patches import patchify


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
    pass_grid: bool = True,
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
        pass_grid (bool): Whether to pass positional encoding to the model.
        *extra_conditions: Additional model conditions.

    Returns:
        torch.Tensor: Scalar loss.
    """

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
        if pass_grid:
            patch, pos = patchify(expanded_samples[idx:idx + 1], p, pass_grid)
            pos_grids.append(pos)
        else:
            patch = patchify(expanded_samples[idx:idx + 1], p, pass_grid)
        patches.append(patch)
    patches = torch.cat(patches, dim=0)
    if pass_grid:
        pos_grids = torch.cat(pos_grids, dim=0)

    z = torch.randn_like(patches)
    t = torch.rand(total_patches, device=device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, patches)

    noisy = mean + sigma * z
    if pass_grid:
        score = model(t, noisy, pos_grids, *expanded_conditions)
    else:
        score = model(t, noisy, *expanded_conditions)
    loss = torch.sum((z + score) ** 2, dim=[1, 2, 3]).mean()
    return loss