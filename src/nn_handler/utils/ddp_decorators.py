import os
import warnings
from datetime import timedelta
from functools import wraps
from typing import Union, Optional, List

import torch
import torch.distributed as dist
from torch import nn
import torch.multiprocessing as mp


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


# This worker function must be a top-level function so it can be pickled
# and sent to the new processes.
def _parallel_worker(rank, user_func, result_queue):
    """
    Internal worker that runs in a separate process.
    - rank: The process index, which we use as the GPU index.
    - user_func: The user-provided function to execute.
    - result_queue: A queue to send the result back to the main process.
    """
    try:
        device = f'cuda:{rank}'
        torch.cuda.set_device(device)
        # Execute the user's function, passing the device name to it
        result = user_func(device=device)
        result_queue.put({'rank': rank, 'result': result})
    except Exception as e:
        result_queue.put({'rank': rank, 'error': e})


class ParallelExecutor:
    """
    A context manager to execute a function in parallel on multiple GPUs.

    This is for general-purpose parallel execution, not for model training. It's
    useful when you want to run the same code independently on several GPUs,
    for instance, for parallel data generation or simulations.

    Usage:
        def my_task(device):
            # This code will run on a specific GPU (e.g., 'cuda:0')
            data = torch.randn(5, device=device)
            return data.cpu() # Return data to the main process

        with ParallelExecutor() as executor:
            # .run() executes my_task on all available GPUs.
            results = executor.run(my_task)

        all_data = torch.cat(results)
    """

    def __init__(self, num_gpus: int = None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for parallel execution.")

        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        if self.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive.")

        # 'spawn' is the required start method for CUDA multiprocessing
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
        elif mp.get_start_method() != "spawn":
            warnings.warn(
                f"Warning: multiprocessing start method is '{mp.get_start_method()}', but 'spawn' is required for CUDA.")

    def run(self, func):
        """
        Executes the given function on each GPU in parallel.

        The function `func` must accept a keyword argument `device` (str),
        which will be the device name like 'cuda:0', 'cuda:1', etc.

        Args:
            func: A callable function to execute on each GPU.

        Returns:
            A list of results from each process, ordered by device index.
        """
        if not callable(func):
            raise TypeError("The provided object must be a callable function.")

        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        processes = [ctx.Process(target=_parallel_worker, args=(rank, func, result_queue)) for rank in
                     range(self.num_gpus)]
        for p in processes:
            p.start()

        # Collect results and potential errors
        results_map = {}
        errors = []
        for _ in range(self.num_gpus):
            output = result_queue.get()
            if 'error' in output:
                errors.append(output)
            else:
                results_map[output['rank']] = output['result']

        # Ensure all processes are cleaned up before proceeding
        for p in processes:
            p.join()

        # After cleanup, check if any errors were reported
        if errors:
            error_messages = "\n".join([f"  - GPU {e['rank']}: {e['error']}" for e in errors])
            raise RuntimeError(f"One or more parallel processes failed:\n{error_messages}")

        # Return results, sorted by the GPU index to ensure order
        return [results_map[i] for i in range(self.num_gpus)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Nothing to clean up


def parallelize_on_gpus(num_gpus: int = None):
    """
    A decorator to run a function in parallel on multiple GPUs.

    The decorated function must accept a `device` keyword argument. The decorator
    will automatically provide this argument when executing the function on each GPU.

    Args:
        num_gpus (int, optional): The number of GPUs to use. If None, uses all
                                  available GPUs.

    Returns:
        A list of the results from each parallel execution, ordered by device.
    """

    def decorator(user_func):
        @wraps(user_func)
        def wrapper(*args, **kwargs):
            # This is the function that will be executed inside each new process.
            # It accepts the `device` argument from the ParallelExecutor's worker.
            def task_on_gpu(device):
                # Call the original user function, passing along the original
                # arguments and injecting the new device argument.
                return user_func(*args, **kwargs, device=device)

            # Use the existing ParallelExecutor to run the task.
            with ParallelExecutor(num_gpus=num_gpus) as executor:
                results = executor.run(task_on_gpu)

            return results

        return wrapper

    return decorator
