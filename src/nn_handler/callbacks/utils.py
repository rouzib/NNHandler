import time
import gc
from typing import Optional, Dict, Any

import torch

from .base import Callback


class SimpleGarbageCollector(Callback):
    """Periodically runs Python's garbage collector.

    Useful in memory-constrained environments or long training runs,
    though its impact might vary.

    Args:
        collect_freq_epoch (int): Run gc.collect() every N epochs. Defaults to 1.
        verbose (bool): Print a message when GC runs. Defaults to False.
    """

    def __init__(self, collect_freq_epoch: int = 1, verbose: bool = False):
        super().__init__()
        self.collect_freq_epoch = collect_freq_epoch
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        current_epoch_1_based = epoch + 1
        if current_epoch_1_based % self.collect_freq_epoch == 0:
            if self.verbose:
                print(f"\nRunning garbage collection at end of epoch {current_epoch_1_based}...")
            start_time = time.perf_counter()
            collected_count = gc.collect()  # Trigger garbage collection
            end_time = time.perf_counter()
            if self.verbose:
                print(
                    f"Garbage collection finished in {end_time - start_time:.3f}s. Collected {collected_count} objects.")


class CudaGarbageCollector(Callback):
    """
    Periodically runs Python's garbage collector and clears unused CUDA memory.

    Useful in memory-constrained environments or long training runs,
    particularly when GPU memory usage needs to be actively managed.

    Args:
        collect_freq_epoch (int): Run garbage collection every N epochs. Defaults to 1.
        verbose (bool): Print a message when GC runs. Defaults to False.
    """

    def __init__(self, collect_freq_epoch: int = 1, verbose: bool = False):
        super().__init__()
        self.collect_freq_epoch = collect_freq_epoch
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        current_epoch_1_based = epoch + 1
        # Check if it is time to run the garbage collector
        if current_epoch_1_based % self.collect_freq_epoch == 0:
            if self.verbose:
                print(f"\n[CUDA GC] Running garbage collection at end of epoch {current_epoch_1_based}...")

            # Run Python garbage collection
            python_gc_start = time.perf_counter()
            collected_count = gc.collect()  # Force Python's garbage collection
            python_gc_end = time.perf_counter()

            if self.verbose:
                print(f"Python GC collected {collected_count} objects in {python_gc_end - python_gc_start:.3f}s.")

            # Clear unused CUDA memory
            cuda_gc_start = time.perf_counter()
            torch.cuda.empty_cache()  # Release unused GPU memory
            cuda_gc_end = time.perf_counter()

            if self.verbose:
                print(f"CUDA GC cleared unused GPU memory in {cuda_gc_end - cuda_gc_start:.3f}s.")
                print(f"Memory stats after CUDA GC: {torch.cuda.memory_summary()}")
