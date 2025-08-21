import os
import re
import sys
import socket
import subprocess
import warnings
from datetime import timedelta
from typing import Union, Optional

import torch
import torch.distributed as dist


# ---------------------------- Helpers for Slurm/env ----------------------------

def _first_host_from_slurm_nodelist(nodelist: str) -> str:
    """
    Return the first hostname from SLURM_NODELIST.
      Examples:
        'gra123' -> 'gra123'
        'gra[123-126]' -> 'gra123'
        'nodeA[01,03,07]' -> 'nodeA01'
        'nodeA[001-003,007]' -> 'nodeA001'
    """
    if "[" not in nodelist:
        return nodelist
    m = re.match(r"^([^\[]+)\[(.+)\]$", nodelist)
    if not m:
        return nodelist
    prefix, body = m.groups()
    first_group = body.split(",")[0].strip()
    start = first_group.split("-", 1)[0]
    return f"{prefix}{start}"


def _ensure_master_env():
    """Ensure MASTER_ADDR/MASTER_PORT exist (works for 1+ nodes)."""
    os.environ.setdefault("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        if nodelist:
            os.environ["MASTER_ADDR"] = _first_host_from_slurm_nodelist(nodelist)
        else:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURMD_NODENAME", socket.gethostname())


def _maybe_spawn_local_workers():
    """
    If we're running one Slurm task per node (all GPUs visible in this process),
    spawn one child process per GPU with RANK/LOCAL_RANK/WORLD_SIZE set, then exit parent.

    Children re-run the same script and will *skip* spawning (NH_SPAWNED=1),
    then run the normal code path:
        print(f"{_is_env_distributed()=}")
        _initialize_distributed()
        # training code ...
    """
    # Don't recurse; children mark themselves as spawned.
    print(f"{os.environ.get('NH_SPAWNED')=}")
    if os.environ.get("NH_SPAWNED") == "1":
        return
    # If a rank is already set (e.g., torchrun), do nothing.
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return

    # Only worth spawning if multiple GPUs are visible in this process
    if not torch.cuda.is_available():
        return
    nvis = torch.cuda.device_count()
    if nvis <= 1:
        return

    # Derive world size from Slurm (task-per-node) + nproc_per_node = num visible GPUs
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    nproc_per_node = int(os.environ.get("NH_NPROC_PER_NODE", str(nvis)))
    world_size = nnodes * nproc_per_node

    print(f"{nnodes = }, {node_rank = }, {nproc_per_node = }, {world_size = }")

    _ensure_master_env()

    # Prepare base environment for children
    base_env = os.environ.copy()
    base_env["NH_SPAWNED"] = "1"
    base_env["WORLD_SIZE"] = str(world_size)
    base_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # stable GPU ordering
    # Modern PyTorch NCCL controls (optional but helpful)
    # base_env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    # base_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    procs = []
    for local_rank in range(nproc_per_node):
        child_env = base_env.copy()
        child_env["LOCAL_RANK"] = str(local_rank)
        child_env["RANK"] = str(node_rank * nproc_per_node + local_rank)
        cmd = [sys.executable] + sys.argv
        procs.append(subprocess.Popen(cmd, env=child_env))

        print(f"{child_env = }, {cmd = }")

    # Parent waits for all children and then exits with aggregated return code.
    rc = 0
    for p in procs:
        p.wait()
        rc = rc or p.returncode
    sys.exit(rc)


# ---------------------------- Public-facing toggles ----------------------------

def _is_env_distributed() -> bool:
    """
    Return True iff standard DDP env vars exist and are sane:
      RANK, LOCAL_RANK, WORLD_SIZE with WORLD_SIZE > 1 and 0 <= RANK < WORLD_SIZE.

    NOTE: We intentionally DO NOT synthesize these from Slurm here.
          The parent process (one task per node) will spawn per-GPU children
          and set these variables explicitly.
    """
    if not all(v in os.environ for v in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')):
        return False
    try:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    except ValueError:
        print("WARN: RANK / LOCAL_RANK / WORLD_SIZE are not integers.")
        return False

    if rank < 0 or local_rank < 0 or world_size <= 0 or rank >= world_size:
        print(f"WARN: Invalid DDP values: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return False
    return world_size > 1


def _should_use_distributed(use_distributed_flag: Optional[bool]) -> bool:
    """
    Should we run with DDP?
    - False     -> never
    - True      -> only if torch.distributed is available AND env looks valid
    - None/auto -> enable if torch.distributed is available AND env looks valid
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
            print("WARNING: DDP explicitly requested but DDP env (RANK/LOCAL_RANK/WORLD_SIZE) is not set or WORLD_SIZE <= 1. Running non-DDP.")
            return False
        print("INFO: DDP explicitly enabled and environment seems valid.")
        return True

    # Auto
    if dist.is_available() and env_is_distributed:
        print("INFO: DDP environment detected automatically. Enabling DDP.")
        return True
    else:
        if use_distributed_flag is None and not env_is_distributed:
            print("INFO: No DDP environment detected and use_distributed=None. Running non-DDP.")
        return False


# ---------------------------- Device selection & init ----------------------------

def _pick_device(local_rank: int) -> torch.device:
    """
    Map LOCAL_RANK -> cuda:{LOCAL_RANK} when multiple GPUs are visible;
    if only one GPU is visible, use cuda:0; else fall back to CPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    nvis = torch.cuda.device_count()
    dev = torch.device(f"cuda:{local_rank % nvis}" if nvis > 1 else "cuda:0")
    torch.cuda.set_device(dev)
    return dev


def _initialize_distributed(timeout: Optional[timedelta] = None):
    """
    Initialize the PyTorch DDP process group. If we're in a single Slurm task
    with multiple visible GPUs, this will auto-spawn one child per GPU and exit
    the parent. Children then run the normal training code.

    Returns:
        tuple(distributed: bool, rank: int, local_rank: int, world_size: int, device: torch.device)
    """
    # NEW: if we're the parent (task-per-node), spawn children and exit.
    _maybe_spawn_local_workers()

    return

    if not dist.is_available():
        print("ERROR: torch.distributed is not available. Disabling DDP.")
        _distributed = False
        return _distributed, 0, -1, 1, _resolve_device("cuda")
    else:
        _distributed = True

    # Expect RANK/LOCAL_RANK/WORLD_SIZE to be set now (by torchrun or our spawner)
    _rank = int(os.environ['RANK'])
    _local_rank = int(os.environ['LOCAL_RANK'])
    _world_size = int(os.environ['WORLD_SIZE'])

    print(f"{_rank = }, {_local_rank = }, {_world_size = }")

    # Device & backend
    _device = _pick_device(_local_rank)
    backend = "nccl" if _device.type == "cuda" else "gloo"

    # Rendezvous
    _ensure_master_env()

    print(f"INFO (Rank {_rank}): Initializing process group | "
          f"backend={backend}, world_size={_world_size}, "
          f"local_rank={_local_rank}, device={_device}")

    # Generous timeout helps slow start-ups on large jobs
    pg_kwargs = dict(
        backend=backend,
        rank=_rank,
        world_size=_world_size,
        init_method="env://",
        timeout=timedelta(minutes=60) if timeout is None else timeout,
    )

    # PyTorch ≥ 2.4: pass device_id for NCCL to silence mapping warnings
    try:
        dist.init_process_group(device_id=_device, **pg_kwargs)
    except TypeError:
        dist.init_process_group(**pg_kwargs)

    # Barrier (device_ids supported on newer PyTorch)
    if backend == "nccl":
        try:
            dist.barrier(device_ids=[_device.index])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()

    print(f"INFO (Rank {_rank}): DDP initialised and barrier passed.")
    return _distributed, _rank, _local_rank, _world_size, _device


def initialize_ddp(timeout: Optional[timedelta] = None):
    """
    Optional public entry point that avoids re-initialising if already initialized.
    """
    if not dist.is_initialized():
        if _should_use_distributed(True):
            return _initialize_distributed(timeout)
        else:
            return False, 0, -1, 1, _resolve_device("cuda")
    else:
        print("INFO: DDP process group already initialized.")
        return None


# ---------------------------- Misc device resolver ----------------------------

def _resolve_device(device: Union[torch.device, str]) -> torch.device:
    """
    Resolve a device spec to torch.device with CUDA fallback to CPU if unavailable.
    """
    if isinstance(device, torch.device):
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn(f"Requested CUDA device {device} but CUDA is not available. Using CPU instead.",
                          RuntimeWarning)
            return torch.device("cpu")
        return device
    elif isinstance(device, str):
        dev_str = device.lower()
        if dev_str in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                warnings.warn("Requested CUDA device but CUDA is not available. Using CPU instead.", RuntimeWarning)
                return torch.device("cpu")
            return torch.device("cuda")
        elif dev_str.startswith("cuda:"):
            if not torch.cuda.is_available():
                warnings.warn(f"Requested CUDA device {dev_str} but CUDA is not available. Using CPU instead.",
                              RuntimeWarning)
                return torch.device("cpu")
            return torch.device(dev_str)
        elif dev_str == "cpu":
            return torch.device("cpu")
        else:
            try:
                return torch.device(dev_str)  # e.g., 'mps'
            except RuntimeError as e:
                raise ValueError(f"Invalid device string '{device}': {e}") from e
    else:
        raise TypeError(f"Device must be a string or torch.device, got {type(device)}.")
