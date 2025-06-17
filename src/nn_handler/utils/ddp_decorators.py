import functools
import warnings
from functools import wraps
from typing import Union, List, Callable, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import cloudpickle as _cp


def on_rank(rank: Union[int, List[int]], barrier: bool = False):
    """
    A decorator to execute a function only on a specific rank or ranks, with robust error handling.

    If the decorated function raises an exception on any of the target ranks, this decorator will:
    1. Catch the exception.
    2. Communicate the failure to all other ranks in the process group.
    3. Raise a RuntimeError on all ranks to ensure a clean, synchronized shutdown of the DDP job, preventing hangs.

    Args:
        rank (Union[int, List[int]]): The rank or list of ranks to execute on.
        barrier (bool): If True, a barrier is called after the function execution (only if no errors occurred).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not dist.is_available() or not dist.is_initialized():
                return func(*args, **kwargs)

            current_rank = dist.get_rank()
            ranks_to_run = rank if isinstance(rank, list) else [rank]

            # 1 for success, 0 for failure. Use a CPU tensor for simplicity.
            success_tensor = torch.tensor([1], dtype=torch.int)
            result = None

            if current_rank in ranks_to_run:
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    warnings.warn(f"Rank {current_rank} caught an exception in '{func.__name__}': {e}")
                    success_tensor[0] = 0

            # Communicate failure status across all ranks.
            # The all_reduce operation is itself a synchronization point.
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)

            # If success_tensor is 0, at least one rank failed.
            if success_tensor[0] == 0:
                # Raise an exception on all ranks to ensure a clean shutdown.
                raise RuntimeError(
                    f"A failure was detected on at least one rank during the execution of '{func.__name__}'. "
                    f"Terminating all ranks."
                )

            if barrier:
                dist.barrier()

            return result

        return wrapper

    return decorator


def parallel_on_all_devices(func):
    """
    A decorator to run a function in parallel on all available CUDA devices
    using torch.nn.DataParallel.

    This is suitable for single-node, multi-GPU data parallelism. It is simpler
    than DDP but often less performant due to factors like GIL contention and
    unbalanced workload on the primary GPU.

    The decorated function MUST be device-aware. It should:
    1.  Accept at least one tensor as input.
    2.  Infer the correct device from its input tensors (e.g., `device = my_tensor.device`).
    3.  Create any new tensors on that same device.

    Usage:
        @parallel_on_all_devices
        def my_model_forward(data_chunk):
            # data_chunk is a slice of the batch on a specific GPU
            device = data_chunk.device
            # ... perform operations on data_chunk ...
            new_tensor = torch.ones(1, device=device)
            # ... more operations ...
            return result_tensor

    How to call the decorated function:
        # 1. Move your *entire* input batch to the primary CUDA device.
        full_batch = torch.randn(128, 10).to('cuda:0')

        # 2. Call the function. DataParallel will automatically split the batch,
        #    move chunks to other GPUs, execute the function, and gather results.
        results = my_model_forward(full_batch)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available. Running on CPU without DataParallel.", RuntimeWarning)
            else:
                warnings.warn("Only one GPU available. Running without DataParallel.", RuntimeWarning)
            # Fallback to executing the function as-is on the current device.
            return func(*args, **kwargs)

        # A lightweight nn.Module to wrap the user's raw function.
        # This is necessary because DataParallel expects an nn.Module.
        class ParallelFunctionWrapper(nn.Module):
            def __init__(self, forward_func):
                super().__init__()
                self.forward_func = forward_func

            def forward(self, *args, **kwargs):
                # The user's function becomes the forward pass of our module.
                return self.forward_func(*args, **kwargs)

        # 1. Instantiate the module that wraps the user function.
        model = ParallelFunctionWrapper(func)

        # 2. Wrap the module with DataParallel.
        # This will replicate the module on all available GPUs. Input tensors
        # passed to parallel_model MUST be on the primary device (e.g., 'cuda:0').
        parallel_model = nn.DataParallel(model)

        # 3. Execute the function. DataParallel handles scattering the inputs,
        #    running the forward pass on each replica, and gathering the outputs.
        return parallel_model(*args, **kwargs)

    return wrapper


def _parallel_worker(rank, pickled_user_func, result_queue, pass_device):
    try:
        user_func = _cp.loads(pickled_user_func)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        # Ensure CUDA ops are complete before moving to CPU
        result = user_func(device=device) if pass_device else user_func()
        torch.cuda.synchronize(device)

        result_queue.put({'rank': rank, 'result': _as_cpu(result)})
    except Exception as e:
        import traceback
        error_str = f"Error in process for GPU {rank}:\n{repr(e)}\n{traceback.format_exc()}"
        result_queue.put({'rank': rank, 'error': error_str})
    finally:
        # A final synchronization can help ensure graceful exit.
        try:
            torch.cuda.synchronize(device)
        except (AttributeError, RuntimeError):
            pass


def _as_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, (list, tuple)):
        return type(obj)(_as_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _as_cpu(v) for k, v in obj.items()}
    return obj


class ParallelExecutor:
    def __init__(self, num_gpus: int = None, pass_device: bool = True):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for parallel execution.")
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        self.pass_device = pass_device
        if self.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive.")
        # We use torch.multiprocessing which defaults to and requires 'spawn'.
        # This check ensures we're in the right environment.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
        elif mp.get_start_method() != "spawn":
            warnings.warn(
                f"Multiprocessing start method is '{mp.get_start_method()}', but 'spawn' is required for CUDA.")

    def run(self, func, *args, **kwargs):
        if not callable(func):
            raise TypeError("The provided object must be a callable function.")

        task_with_args = functools.partial(func, *args, **kwargs)
        pickled_task = _cp.dumps(task_with_args)

        # Use torch.multiprocessing.Queue
        result_queue = mp.Queue()

        # Use torch.multiprocessing.Process
        processes = [mp.Process(target=_parallel_worker, args=(rank, pickled_task, result_queue, self.pass_device))
                     for rank in range(self.num_gpus)]
        for p in processes:
            p.start()

        results_map = {}
        errors = []
        for _ in range(self.num_gpus):
            output = result_queue.get()
            if 'error' in output:
                errors.append(output)
            else:
                results_map[output['rank']] = output['result']

        # The torch.multiprocessing process .join() is more robust.
        for p in processes:
            p.join()

        if errors:
            error_messages = "\n\n".join([e['error'] for e in sorted(errors, key=lambda x: x['rank'])])
            raise RuntimeError(f"One or more parallel processes failed:\n{error_messages}")

        return [results_map[i] for i in range(self.num_gpus)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def parallelize_on_gpus(num_gpus: int = None, pass_device: bool = True):
    """
    A decorator to run a function in parallel on multiple GPUs, designed to
    work robustly in interactive environments by using cloudpickle and
    torch.multiprocessing.

    Note:
        If encountering a EOFError, the torch multiprocessing kernel gets stopped prematurely and the results can't be
        shared between the processes. Try converting the results of your function to a list or a numpy array before
        returning.
    """

    def decorator(user_func: Callable) -> Callable:
        @functools.wraps(user_func)
        def wrapper(*args: Any, **kwargs: Any) -> list:
            with ParallelExecutor(num_gpus=num_gpus, pass_device=pass_device) as executor:
                return executor.run(user_func, *args, **kwargs)

        return wrapper

    return decorator
