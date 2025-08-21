import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
from torchvision import transforms

import score_models

from src.nn_handler.nn_handler_distributed import NNHandler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

import os
from src.nn_handler.utils.ddp import _is_env_distributed

print(f"{_is_env_distributed()=}")
print(f"{os.environ['RANK'] = }")
print(f"{os.environ['LOCAL_RANK'] = }")
print(f"{os.environ['WORLD_SIZE'] = }")


def denoising_score_matching(samples, sde, model, device, *args):
    B, *D = samples.shape
    z = torch.randn_like(samples)
    t = torch.rand(B).to(device) * (sde.T - sde.epsilon) + sde.epsilon
    mean, sigma = sde.marginal_prob(t, samples)
    return torch.sum((z + model(t, mean + sigma * z, *args)) ** 2) / B

print("downloading MNIST dataset...")
dataset = torchvision.datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
dataset = torch.utils.data.TensorDataset(dataset.data.unsqueeze(1) / 255.)
print(len(dataset))

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
                  logger_filename="models/MNIST_score_models_8.log")

print(model)

model.set_sde(sde_class=score_models.sde.VESDE, sigma_min=1e-2, sigma_max=10)
model.set_loss_fn(denoising_score_matching)
model.set_optimizer(torch.optim.Adam, lr=1e-3)
model.set_scheduler(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.9)
model.auto_save(10, "models", "MNIST_score_models_8", overwrite=True)
model.set_train_loader(dataset, batch_size=128)
model.train(150, validate_every=0, ema_decay=0.99)