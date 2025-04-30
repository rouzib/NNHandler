#%%
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

import score_models

from src.nn_handler.nn_handler_distributed import NNHandler
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
def denoising_score_matching(samples, sde, model, device, *args):
    B, *D = samples.shape
    z = torch.randn_like(samples)
    t = torch.rand(B).to(device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, samples)
    return torch.sum((z + model(t, mean + sigma * z, *args)) ** 2) / B
#%%
dataset = torchvision.datasets.MNIST('notebooks/.', train=True, transform=transforms.ToTensor(), download=True)
dataset = torch.utils.data.TensorDataset(dataset.data[:10000].unsqueeze(1) / 255.)
print(len(dataset))
#%%
score_models.NCSNpp
#%%
hyperparameters = {
    "channels": 1,
    "dimensions": 2,
    "nf": 64,
    "activation_type": "swish",
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "resample_with_conv": True,
    "dropout": 0.,
    "attention": True,
}

model = NNHandler(model_class=score_models.NCSNpp, model_type=NNHandler.ModelType.SCORE_BASED, device=device,
                  **hyperparameters, logger_mode=NNHandler.LoggingMode.FILE, logger_level=10,
                  logger_filename="notebooks/models/MNIST_score_models.log")
#%%
model.set_sde(sde_class=score_models.sde.VESDE, sigma_min=1e-2, sigma_max=10)
#%%
model.set_loss_fn(denoising_score_matching)
#%%
model.set_optimizer(torch.optim.Adam, lr=1e-3)
model.set_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.9)
#%%
model.auto_save(10, "notebooks/models", "MNIST_score_models", overwrite=True)
#%%
model.set_train_loader(dataset, batch_size=128)
#%%
model.train(150, validate_every=0, ema_decay=0.99, epoch_train_and_val_pbar=False)
#%%
N = 144

samples = model.sample(shape=[N, 1, 28, 28], steps=100)

sample_grid = make_grid(samples, nrow=int(np.sqrt(N)), value_range=(0, 1), normalize=True)

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0, vmax=1)
plt.show()
#%%
for i in range(10, 51, 10):
    model = NNHandler.load(f"models/MNIST_score_models_{i}.pth", device=device)

    N = 144
    samples = model.sample(shape=[N, 1, 28, 28], steps=100)

    sample_grid = make_grid(samples, nrow=int(np.sqrt(N)), value_range=(0, 1), normalize=False)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0, vmax=1)
    plt.show()
#%%
