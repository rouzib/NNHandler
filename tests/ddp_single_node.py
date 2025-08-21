from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.nn_handler.utils.ddp import _initialize_distributed, _is_env_distributed

print(os.environ["CUDA_VISIBLE_DEVICES"])

print(os.environ['SLURM_PROCID'])
print(os.environ['SLURM_LOCALID'])
print(os.environ['SLURM_NTASKS'])

print(os.environ['SLURM_JOB_GPUS'])
print(os.environ['SLURM_GPUS_PER_NODE'])
print(os.environ['SLURM_TRES_PER_TASK'])
print(os.environ['SLURM_STEP_GPUS'])

print(f"{torch.cuda.device_count()} GPUs available. {torch.cuda.is_available()=}")
print(f"{_is_env_distributed()=}")
_initialize_distributed()

print([k for k in os.environ.keys() if "SLURM" in k])