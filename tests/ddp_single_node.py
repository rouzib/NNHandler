from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.nn_handler import NNHandler, initialize_ddp

print(os.environ["CUDA_VISIBLE_DEVICES"])

print(f"{torch.cuda.device_count()} GPUs available. {torch.cuda.is_available()=}")
initialize_ddp()
