from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable
from multiprocessing import Process
from multiprocessing import Queue as make_queue
from multiprocessing.queues import Queue
from typing import Any

from tqdm import tqdm

from .log_util import logger

__all__ = [
    "get_num_gpus",
    "run_gpu_multiprocessing",
]


def get_num_gpus() -> int:
    """
    Get the number of GPUs available with nvidia-smi.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len([line for line in result.stdout.strip().split("\n") if line.strip()])
    except Exception:
        return 0


def retry(
    func: Callable[[Any, ...], Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_retries: int = 3,
    retry_delay: int = 1,
) -> Any:
    """
    Retry a function with a delay between retries.
    """
    last_exception: Exception | None = None
    for _ in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            time.sleep(retry_delay)
    raise last_exception


def process_worker(
    rank: int,
    input_queue: Queue[tuple[tuple[Any, ...], dict[str, Any]] | None],
    output_queue: Queue[Any | None],
    func: Callable[[Any, ...], Any],
    kwargs: dict[str, Any],
) -> None:
    """
    A worker function for simple gpu multiprocessing.
    """
    import torch

    logger.info(f"Worker {rank} started with PID {os.getpid()}.")
    torch.cuda.set_device(torch.device(f"cuda:{rank}"))

    while True:
        params = input_queue.get()
        if params is None:
            break
        args, kwargs = params
        result = retry(func, args, kwargs)
        output_queue.put_nowait(result)


def run_gpu_multiprocessing(
    inputs: Iterable[Any],
    func: Callable[[Any, ...], Any],
    num_gpus: int | None = None,
    timeout: int | None = None,
    use_tqdm: bool = False,
    desc: str = "Processing",
    total: int | None = None,
    **kwargs: Any,
) -> list[Any]:
    """
    Run a function on multiple GPUs in parallel.
    """
    if num_gpus is None:
        num_gpus = get_num_gpus()

    input_queue = make_queue()
    output_queue = make_queue()

    processes = [
        Process(
            target=process_worker, args=(i, input_queue, output_queue, func, kwargs)
        )
        for i in range(num_gpus)
    ]

    for process in processes:
        process.start()

    num_expected_outputs = 0
    for input in inputs:
        if (
            isinstance(input, tuple)
            and len(input) == 2
            and isinstance(input[0], tuple)
            and isinstance(input[1], dict)
        ):
            args, kwargs = input
        elif isinstance(input, tuple):
            args = input
            kwargs = {}
        elif isinstance(input, dict):
            args = tuple()
            kwargs = input
        else:
            args = (input,)
            kwargs = {}

        input_queue.put((args, kwargs))
        num_expected_outputs += 1

    for _ in range(num_gpus):
        input_queue.put(None)

    results = []

    if use_tqdm:
        result_progress_bar = tqdm(total=num_expected_outputs, desc=desc)

    for _ in range(num_expected_outputs):
        results.append(output_queue.get(timeout=timeout))
        if use_tqdm:
            result_progress_bar.update(1)

    if use_tqdm:
        result_progress_bar.close()
    return results
