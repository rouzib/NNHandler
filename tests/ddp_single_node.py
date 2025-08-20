from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.nn_handler import NNHandler, initialize_ddp

initialize_ddp()
