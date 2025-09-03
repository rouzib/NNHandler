import functools
import warnings
from functools import wraps
from typing import Union, List, Callable, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import cloudpickle as _cp

from .dd_helpers import _collective_device_for_backend


def on_rank(rank: Union[int, List[int]], barrier: bool = False, check_exception: bool = False):
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
            # If not distributed, just run.
            if not dist.is_available() or not dist.is_initialized():
                return func(*args, **kwargs)

            current_rank = dist.get_rank()
            ranks_to_run = rank if isinstance(rank, list) else [rank]

            if check_exception:
                # Tensor must live on device compatible with the PG backend (e.g., CUDA for NCCL).
                device = _collective_device_for_backend()
                success_tensor = torch.ones(1, dtype=torch.int, device=device)

            result = None
            if current_rank in ranks_to_run:
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    warnings.warn(f"Rank {current_rank} caught an exception in '{func.__name__}': {e}")
                    if check_exception:
                        success_tensor.zero_()  # mark failure

            if check_exception:
                # Synchronize success/failure across ALL ranks.
                dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)

                if int(success_tensor.item()) == 0:
                    # Ensure a clean shutdown if any target rank failed.
                    dist.destroy_process_group()
                    raise RuntimeError(
                        f"A failure was detected on at least one rank during '{func.__name__}'. "
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
    """
    Internal worker function for parallel execution on a specific GPU.

    This function is executed in a separate process for each GPU. It unpickles the user function,
    sets up the appropriate CUDA device, executes the function, and puts the result in the queue.

    Args:
        rank (int): The GPU rank/index to use for this worker.
        pickled_user_func (bytes): The cloudpickle-serialized user function to execute.
        result_queue (multiprocessing.Queue): Queue to put results or errors into.
        pass_device (bool): If True, pass the device as an argument to the user function.

    Note:
        This function handles exceptions by capturing them and putting error information
        in the result queue instead of the result.
    """
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
    """
    Recursively moves PyTorch tensors to CPU, handling nested data structures.

    This utility function detaches tensors and moves them to CPU. It works recursively
    through lists, tuples, and dictionaries to handle nested data structures.

    Args:
        obj (Any): The object to process. Can be a tensor, list, tuple, dict, or any other type.
            - If a tensor: detaches it and moves it to CPU
            - If a list/tuple: processes each element recursively
            - If a dict: processes each value recursively
            - Otherwise: returns the object unchanged

    Returns:
        Any: The processed object with all tensors moved to CPU.

    Example:
        >>> tensor_on_gpu = torch.tensor([1, 2, 3]).cuda()
        >>> nested_structure = {'a': tensor_on_gpu, 'b': [tensor_on_gpu, 10]}
        >>> cpu_structure = _as_cpu(nested_structure)
        # All tensors in cpu_structure are now on CPU
    """
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, (list, tuple)):
        return type(obj)(_as_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _as_cpu(v) for k, v in obj.items()}
    return obj


class ParallelExecutor:
    """
    A class for executing functions in parallel across multiple GPUs.

    This class manages the creation of separate processes for each GPU, executes
    a function on each GPU, and collects the results. It handles process creation,
    error handling, and result aggregation.

    The class can be used as a context manager with the `with` statement.

    Attributes:
        num_gpus (int): Number of GPUs to use for parallel execution.
        pass_device (bool): Whether to pass the device as an argument to the function.
    """

    def __init__(self, num_gpus: int = None, pass_device: bool = True):
        """
        Initialize the ParallelExecutor.

        Args:
            num_gpus (int, optional): Number of GPUs to use. If None, uses all available GPUs.
            pass_device (bool, optional): If True, passes the device as an argument to the function.
                Defaults to True.

        Raises:
            RuntimeError: If CUDA is not available.
            ValueError: If the number of GPUs is not positive.
        """
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
        """
        Execute a function in parallel across multiple GPUs.

        This method creates a separate process for each GPU, executes the function
        on each GPU, and collects the results. It handles process creation, error
        handling, and result aggregation.

        Args:
            func (callable): The function to execute in parallel.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            list: A list of results from each GPU, in order of GPU index.

        Raises:
            TypeError: If the provided object is not callable.
            RuntimeError: If one or more parallel processes fail.

        Note:
            If `pass_device=True` was set during initialization, the function will
            receive a `device` argument with the appropriate CUDA device for each process.
        """
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
        """
        Enter the context manager.

        This method is called when entering a `with` statement.
        It returns the ParallelExecutor instance for use in the context.

        Returns:
            ParallelExecutor: The instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        This method is called when exiting a `with` statement.
        Currently, it performs no cleanup actions, but is included to support
        the context manager protocol.

        Args:
            exc_type: The exception type, if an exception was raised in the context.
            exc_val: The exception value, if an exception was raised in the context.
            exc_tb: The traceback, if an exception was raised in the context.
        """
        pass


def parallelize_on_gpus(num_gpus: int = None, pass_device: bool = True):
    """
    A decorator to run a function in parallel on multiple GPUs.

    This decorator distributes the execution of a function across multiple GPUs,
    creating a separate process for each GPU. It is designed to work robustly in
    interactive environments (like Jupyter notebooks) by using cloudpickle for
    serialization and torch.multiprocessing for process management.

    Args:
        num_gpus (int, optional): Number of GPUs to use. If None, uses all available GPUs.
        pass_device (bool, optional): If True, passes the device as an argument to the function.
            The device will be passed as a named argument 'device'. Defaults to True.

    Returns:
        callable: A wrapped function that, when called, executes the original function
            in parallel across multiple GPUs and returns a list of results.

    Example:
        >>> @parallelize_on_gpus()
        ... def process_on_gpu(device, data):
        ...     # This function will be executed on each GPU
        ...     return data.to(device) * 2
        ...
        >>> results = process_on_gpu(torch.tensor([1, 2, 3]))
        >>> # results is a list of tensors, one from each GPU

    Note:
        If encountering an EOFError, the torch multiprocessing kernel may have been stopped
        prematurely, preventing results from being shared between processes. Try converting
        the results of your function to a list or a numpy array before returning.
    """

    def decorator(user_func: Callable) -> Callable:
        """
        The actual decorator function that wraps the user's function.

        Args:
            user_func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function that will execute in parallel.
        """
        @functools.wraps(user_func)
        def wrapper(*args: Any, **kwargs: Any) -> list:
            """
            The wrapper function that executes the user's function in parallel.

            Args:
                *args: Positional arguments to pass to the user's function.
                **kwargs: Keyword arguments to pass to the user's function.

            Returns:
                list: A list of results from each GPU, in order of GPU index.
            """
            with ParallelExecutor(num_gpus=num_gpus, pass_device=pass_device) as executor:
                return executor.run(user_func, *args, **kwargs)

        return wrapper

    return decorator
