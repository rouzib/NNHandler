import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from tqdm.auto import tqdm


def create_latent_dataset(model, dataset, batch_size=128, device=None, return_labels=False, pbar=True, obtain_latent_fn=None):
    """
    Runs dataset through the model's encoder to obtain latent vectors.

    Args:
        model: Your trained autoencoder (should have an 'encode' or similar method).
        dataset: PyTorch Dataset (e.g., TensorDataset or MNIST).
        batch_size: Batch size for DataLoader.
        device: Device to run model on (if None, tries to use model's device if possible).
        return_labels: If True and dataset provides (data, label) tuples, also returns labels.

    Returns:
        If return_labels: (latents, labels)
        else: latents (TensorDataset)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    device = device if device is not None else model.device
    latents = []
    labels = []

    if obtain_latent_fn is None:
        obtain_latent_fn = lambda x: x

    with torch.no_grad():
        for data_tuple in tqdm(loader, disable=not pbar):
            # Handle datasets yielding (img,) or (img, label)
            if isinstance(data_tuple, (tuple, list)):
                imgs = data_tuple[0].to(device)
                if return_labels and len(data_tuple) > 1:
                    labels.append(data_tuple[1])
            else:
                imgs = data_tuple.to(device)

            encoded = model.model.encode(imgs)
            z = obtain_latent_fn(encoded)
            latents.append(z.cpu())

    model.eval(True)

    latent_tensor = torch.cat(latents, dim=0)
    if return_labels and labels:
        labels_tensor = torch.cat(labels, dim=0)
        return TensorDataset(latent_tensor, labels_tensor)
    else:
        return TensorDataset(latent_tensor)


def compute_vesde_sigma_max_robust(data, batch_size=256, device='cpu', percentile=99.9):
    """
    Computes a robust sigma_max for a VESDE scheduler by calculating a high
    percentile of the L2 norms of the latent vectors in the dataset.

    This method is robust to outliers and provides a more efficient noise schedule.

    Args:
        data (Dataset or DataLoader): The dataset or dataloader containing the latent vectors.
        batch_size (int): The batch size to use if a Dataset is provided.
        device (str or torch.device): The device to use for computation.
        percentile (float): The percentile of data magnitudes to use for sigma_max.

    Returns:
        float: The calculated robust sigma_max.
    """
    # If a Dataset is provided, create a DataLoader for it
    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif isinstance(data, DataLoader):
        loader = data
    else:
        raise TypeError("Input 'data' must be a PyTorch Dataset or DataLoader.")

    all_norms = []
    print(f"Calculating robust sigma_max (using {percentile}th percentile): Iterating through dataset...")
    with torch.no_grad():
        for batch in loader:
            # Handle datasets that return (data,) or (data, label)
            if isinstance(batch, (list, tuple)):
                latents_batch = batch[0]
            else:
                latents_batch = batch

            latents_batch = latents_batch.to(device)

            # Flatten each latent vector and compute its L2 norm
            # Shape of latents_batch: (B, C, H, W)
            # Flatten to (B, C*H*W) to compute norm per vector in the batch
            norms = torch.norm(latents_batch.view(latents_batch.size(0), -1), dim=1)
            all_norms.append(norms)

    # Concatenate all norm values into a single tensor
    full_norms_tensor = torch.cat(all_norms, dim=0)

    # Compute the specified percentile of the norms
    sigma_max = torch.quantile(full_norms_tensor, q=percentile / 100.0)

    print(f"Calculation complete. Found robust sigma_max = {sigma_max.item():.4f}")
    return sigma_max.item()
