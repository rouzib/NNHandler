from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.nn_handler.utils.ddp import _initialize_distributed, _is_env_distributed

print(os.environ["CUDA_VISIBLE_DEVICES"])

print(f"{os.environ['SLURM_PROCID'] = }")
print(f"{os.environ['SLURM_LOCALID'] = }")
print(f"{os.environ['SLURM_NTASKS'] = }")

print(f"{torch.cuda.device_count()} GPUs available. {torch.cuda.is_available()=}")
print(f"{_is_env_distributed()=}")
_initialize_distributed()

print(f"{torch.cuda.device_count()} GPUs visible. {torch.cuda.is_available()=}")
print(f"Finished")