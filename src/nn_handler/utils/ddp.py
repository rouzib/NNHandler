import os
import warnings
from datetime import timedelta
from typing import Union, Optional

import torch
import torch.distributed as dist


def _should_use_distributed(use_distributed_flag: Optional[bool]) -> bool:
    """
    Determines if Distributed Data Parallel (DDP) should be used based on flag and environment.

    This function checks if DDP should be enabled based on the provided flag and the current
    environment. It handles three cases:
    1. If use_distributed_flag is False: DDP is disabled
    2. If use_distributed_flag is True: DDP is enabled if available and environment is valid
    3. If use_distributed_flag is None (auto-detect): DDP is enabled if available and environment is valid

    Args:
        use_distributed_flag (Optional[bool]): Flag to control DDP usage.
            - True: Explicitly enable DDP (if possible)
            - False: Explicitly disable DDP
            - None: Auto-detect based on environment

    Returns:
        bool: True if DDP should be used, False otherwise.
    """
    env_is_distributed = _is_env_distributed()

    if use_distributed_flag is False:
        if env_is_distributed:
            print("INFO: DDP environment detected, but use_distributed=False. Running non-DDP.")
        return False

    if use_distributed_flag is True:
        if not dist.is_available():
            print("WARNING: DDP explicitly requested but torch.distributed is not available. Running non-DDP.")
            return False
        if not env_is_distributed:
            print(
                "WARNING: DDP explicitly requested but environment variables ('RANK', 'LOCAL_RANK', 'WORLD_SIZE') not set or WORLD_SIZE <= 1. Running non-DDP.")
            return False
        # Explicitly requested and environment is valid
        print("INFO: DDP explicitly enabled and environment seems valid.")
        return True

    # If use_distributed is None (auto-detect)
    if dist.is_available() and env_is_distributed:
        print("INFO: DDP environment detected automatically. Enabling DDP.")
        return True
    else:
        if use_distributed_flag is None and not env_is_distributed:
            print("INFO: No DDP environment detected and use_distributed=None. Running non-DDP.")
        # Fallback to non-distributed
        return False


def _is_env_distributed() -> bool:
    """
    Checks whether the current process has the information it needs to start
    DistributedDataParallel, coming either from torchrun-style variables
    (RANK, LOCAL_RANK, WORLD_SIZE) *or* from Slurm.
    """
    # -------- 1. Standard torchrun / torch.distributed.launch --------------
    if all(v in os.environ for v in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')):
        try:
            rank, local_rank, world_size = (
                int(os.environ['RANK']),
                int(os.environ['LOCAL_RANK']),
                int(os.environ['WORLD_SIZE']),
            )
        except ValueError:
            print("WARN: RANK / LOCAL_RANK / WORLD_SIZE are not integers.")
            return False

    # -------- 2. Slurm fallback --------------------------------------------
    elif all(v in os.environ for v in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS')):
        try:
            rank = int(os.environ['SLURM_PROCID'])
            local_rank = int(os.environ['SLURM_LOCALID'])
            world_size = int(os.environ['SLURM_NTASKS'])
        except ValueError:
            print("WARN: Slurm env vars could not be parsed as integers.")
            return False

        # Mirror into the regular names so the rest of the pipeline is agnostic
        os.environ.setdefault('RANK', str(rank))
        os.environ.setdefault('LOCAL_RANK', str(local_rank))
        os.environ.setdefault('WORLD_SIZE', str(world_size))

    # -------- 3. Nothing useful found --------------------------------------
    else:
        return False

    # -------- 4. Sanity checks ---------------------------------------------
    if rank < 0 or local_rank < 0 or world_size <= 0 or rank >= world_size:
        print(f"WARN: Invalid DDP values: rank={rank}, local_rank={local_rank}, "
              f"world_size={world_size}")
        return False

    return world_size > 1  # DDP only makes sense if >1 process


def _initialize_distributed(timeout: Optional[timedelta] = None):
    """
    Initialize the distributed process group for PyTorch Distributed Data Parallel (DDP).

    This function reads rank, local_rank, and world_size from the environment variables
    (set by either torchrun, torch.distributed.launch, or Slurm) and initializes the
    distributed process group. It handles device selection based on the local_rank
    and available CUDA devices, and ensures that MASTER_ADDR and MASTER_PORT are set.

    Args:
        timeout (Optional[timedelta]): Timeout for operations. If None, defaults to 60 minutes.
            This is particularly important for large jobs that may take time to start up.

    Returns:
        tuple: A 5-tuple containing:
            - _distributed (bool): Whether distributed mode is enabled
            - _rank (int): Global rank of this process
            - _local_rank (int): Local rank of this process (for device selection)
            - _world_size (int): Total number of processes
            - _device (torch.device): The device to use for this process

    Raises:
        ValueError: If the MASTER_ADDR cannot be determined from the environment.
    """
    if not dist.is_available():
        print("ERROR: torch.distributed is not available. Disabling DDP.")
        _distributed = False
        return _distributed, 0, -1, 1, _resolve_device("cuda")
    else:
        _distributed = True

    # --- Consume the environment ---
    _rank = int(os.environ['RANK'])
    _local_rank = int(os.environ['LOCAL_RANK'])
    _world_size = int(os.environ['WORLD_SIZE'])

    # --------------------------------- device selection --------------------
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            _device = torch.device(f"cuda:0")
        else:
            _device = torch.device(f"cuda:{_local_rank}")
        torch.cuda.set_device(_device)
        backend = "nccl"
    else:
        if torch.cuda.is_available():
            print(f"WARN (Rank {_rank}): LOCAL_RANK {_local_rank} "
                  f"is out of range for {torch.cuda.device_count()} visible GPU(s).")
        _device = torch.device("cpu")
        backend = "gloo"

    # -------------- Ensure MASTER_ADDR / MASTER_PORT exist -----------------
    # Slurm jobs often export MASTER_ADDR automatically when using `srun`.
    os.environ.setdefault("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        # Fallback: use the first hostname in the node list, if present.
        nodelist = os.environ.get("SLURM_NODELIST")
        if nodelist:
            # e.g. "node[001-004]" -> "node001"
            import re
            match = re.match(r"^([^\[]+)\[(\d+)", nodelist)
            if match:
                prefix = match.group(1)
                first_number = match.group(2)
                first_node = prefix + first_number
                os.environ["MASTER_ADDR"] = first_node
            else:
                message = ("No host name in the node list. Please set the host name of the node to be used as a "
                           "host, i.e., in MASTER_ADDR environment variable.")
                print(f"ERROR (Rank {_rank}): {message}")
                raise ValueError(message)
        else:
            # As a last resort, use the current host name.
            import socket
            os.environ["MASTER_ADDR"] = socket.gethostname()

    # ---------------------------- init -------------------------------------
    print(f"INFO (Rank {_rank}): Initializing process group | "
          f"backend={backend}, world_size={_world_size}, "
          f"local_rank={_local_rank}, device={_device}")

    # A generous timeout helps with slow start-ups on large jobs.
    dist.init_process_group(
        backend=backend,
        rank=_rank,
        world_size=_world_size,
        timeout=timedelta(minutes=60) if timeout is None else timeout,
    )

    dist.barrier()  # safety sync
    print(f"INFO (Rank {_rank}): DDP initialised and barrier passed.")

    return _distributed, _rank, _local_rank, _world_size, _device


def initialize_ddp(timeout: Optional[timedelta] = None):
    """
    Initialize the Distributed Data Parallel (DDP) process group if not already done.

    This function serves as a public entry point for initializing DDP. It checks if
    the process group is already initialized, and if not, attempts to initialize it
    with the appropriate settings. It's designed to be called directly without needing
    an instance of any class.

    Args:
        timeout (Optional[timedelta]): Timeout for operations. If None, defaults to 60 minutes.
            This is particularly important for large jobs that may take time to start up.

    Returns:
        Union[tuple, None]: 
            - If DDP was not initialized before: Returns a 5-tuple containing:
                - distributed (bool): Whether distributed mode is enabled
                - rank (int): Global rank of this process
                - local_rank (int): Local rank of this process (for device selection)
                - world_size (int): Total number of processes
                - device (torch.device): The device to use for this process
            - If DDP was already initialized: Returns None

    Note:
        This function always attempts to use DDP (by passing True to _should_use_distributed)
        unless the process group is already initialized.
    """
    if not dist.is_initialized():
        if _should_use_distributed(True):
            return _initialize_distributed(timeout)
        else:
            return False, 0, -1, 1, _resolve_device("cuda")
    else:
        print("INFO: DDP process group already initialized.")
        return None


def _resolve_device(device: Union[torch.device, str]) -> torch.device:
    """
    Resolve a device specification to a torch.device object, handling CUDA availability.

    This function takes a device specification (either a string like 'cuda', 'cuda:0', 'cpu',
    or a torch.device object) and resolves it to a valid torch.device object. It handles
    cases where CUDA is requested but not available by falling back to CPU.

    Args:
        device (Union[torch.device, str]): The device specification to resolve.
            - If a torch.device: Validates it and returns it (or falls back to CPU if needed)
            - If a string: Converts it to a torch.device (with special handling for 'cuda' and 'gpu')

    Returns:
        torch.device: The resolved device object.

    Raises:
        TypeError: If the device is neither a string nor a torch.device.
        ValueError: If the device string is invalid.

    Note:
        If CUDA is requested but not available, this function will issue a warning
        and fall back to CPU instead of raising an error.
    """
    if isinstance(device, torch.device):
        # If CUDA specified but not available, warn and use CPU
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn(f"Requested CUDA device {device} but CUDA is not available. Using CPU instead.",
                          RuntimeWarning)
            return torch.device("cpu")
        return device
    elif isinstance(device, str):
        dev_str = device.lower()
        if dev_str == "cuda" or dev_str == "gpu":
            if not torch.cuda.is_available():
                warnings.warn("Requested CUDA device but CUDA is not available. Using CPU instead.", RuntimeWarning)
                return torch.device("cpu")
            # If CUDA is available, return torch.device("cuda")
            # The specific GPU index will be handled by DDP or DataParallel later if needed
            return torch.device("cuda")
        elif dev_str.startswith("cuda:"):
            if not torch.cuda.is_available():
                warnings.warn(f"Requested CUDA device {dev_str} but CUDA is not available. Using CPU instead.",
                              RuntimeWarning)
                return torch.device("cpu")
            # Check if the specific index is valid? - Let PyTorch handle this error usually
            return torch.device(dev_str)
        elif dev_str == "cpu":
            return torch.device("cpu")
        else:
            try:
                # Allow other device types like 'mps' if supported
                return torch.device(dev_str)
            except RuntimeError as e:
                raise ValueError(f"Invalid device string '{device}': {e}") from e
    else:
        raise TypeError(f"Device must be a string or torch.device, got {type(device)}.")
