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


def patch_denoising_score_matching(samples: torch.Tensor, sde, model, device: torch.device, patch_sizes,
                                   *extra_conditions) -> torch.Tensor:
    """
    Computes the loss for denoising score matching using patch-based sampling. The function divides
    input images into random patches of varied sizes, introduces noise in each patch, and calculates
    the score matching loss along with position encodings.

    Args:
        samples (torch.Tensor): A batch of input images with shape [B, C, H, W], where B is the batch
            size, C is the number of channels, H is the height, and W is the width.
        sde: An object implementing the Stochastic Differential Equation (SDE) interface, which defines
            `T` (end time), `epsilon` (minimum time), and `marginal_prob(t, x)` methods for denoising.
        model: A model that computes the score estimate, typically a neural network accepting the noisy
            input, time step, positional encodings, and extra conditions.
        device (torch.device): The device on which computations are run (e.g., CPU or CUDA device).
        patch_sizes: A list or iterable of integers representing various patch sizes to be sampled
            randomly during the process.
        *extra_conditions: Additional conditional input(s) for the score model, if required.

    Returns:
        torch.Tensor: A scalar loss value representing the mean computed loss across the batch for
            denoising score matching.
    """

    def pachify(images, patch_size, padding=None):
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)

        if padding is not None:
            padded = torch.zeros((images.size(0), images.size(1), images.size(2) + padding * 2,
                                  images.size(3) + padding * 2), dtype=images.dtype, device=device)
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images

        h, w = padded.size(2), padded.size(3)
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                 columns[:, None]]
        padded = padded.permute(1, 0, 2, 3)

        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(
            0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(
            0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        images_pos = torch.cat((x_pos, y_pos), dim=1)
        return padded, images_pos

    B, C, H, W = samples.shape

    p = patch_sizes[torch.randint(0, len(patch_sizes), ()).item()]

    patches, pos_grid = pachify(samples, p)

    z = torch.randn_like(patches)  # standard normal in patch space
    t = torch.rand(B, device=device) * (sde.T - sde.epsilon) + sde.epsilon  # [B]
    mean, sigma = sde.marginal_prob(t, patches)  # both [B, ...] broadcastable

    noisy = mean + sigma * z  # x̃

    score = model(t, noisy, pos_grid, *extra_conditions)  # should be same shape as patches

    loss = torch.sum((z + score) ** 2, dim=[1, 2, 3])  # [B]
    loss = loss.mean()  # scalar

    return loss
