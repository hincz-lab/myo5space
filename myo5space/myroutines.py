#!/usr/bin/env python3
"""
File: myroutines.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains miscellaneous routines that need a home.
"""

import numpy as np
from pathlib import Path
from typing import Any, List, TypeVar, Callable
import multiprocessing as mp


def load_npz_to_dict(path: str | Path) -> dict[str, Any]:
    """
    Reads a numpy .npz file and returns its contents in a dictionary with
    keys equal to those used to save the original arrays.

    :param path: Path to npz file
    :return: A dictionary of the npz contents.
    """
    p = Path(path)
    contents = dict()
    with np.load(p) as npzfile:
        contents = {key: npzfile[key] for key in npzfile}
    return contents


def divide_equally(number: int, num_bins: int) -> List[int]:
    """
    Splits an integer number equally into some number of bins.
    The remainder is distributed over the first (number % num_bins) bins.

    :param number: The number to be split (must be positive).
    :param num_bins: The number of bins (must be positive).
    :return: A list of bin sizes.
    """
    if not isinstance(number, int) or number <= 0:
        raise ValueError("number must be a positive integer")

    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    base = number // num_bins
    remainder = number % num_bins
    res = [base + 1 if i < remainder else base for i in range(num_bins)]
    return res


R = TypeVar("R")
def mp_worker(args: tuple[int, Callable[..., R], tuple[Any, ...]]) -> R:
    """
    Worker function for multiprocessing execution.

    This function unpacks a single argument tuple and invokes the provided
    simulation function with the assigned number of trials and parameters.

    :param args: A 3-tuple containing:
        - trials (int): The number of trials for this worker to execute.
        - func (Callable): The simulation function to run. It must be callable
          as ``func(trials, *params)``.
        - params (tuple): A tuple of positional parameters passed to ``func``.
    :return: The result returned by ``func(trials, *params)``.
    """
    # Unpack args.
    trials, func, params = args
    # Run simulation and return results.
    res = func(trials, *params)
    return res


def mp_driver(
    trials: int,
    func: Callable[..., np.ndarray],
    out_shape: tuple[int, ...],
    processes: int,
    params: tuple[Any, ...],
) -> np.ndarray:
    """
    Driver function for running simulations in parallel using multiprocessing.

    This function divides the total number of trials across multiple worker
    processes, dispatches the work using ``mp_worker``, and assembles the
    returned partial results into a single output array.

    :param trials: The total number of simulation trials to execute. Must be a
        positive integer.
    :param func: The simulation function to run in each worker. The function
        must be callable as ``func(trials_for_job, *params)`` and return a
        NumPy array that can be assembled into the final output.
    :param out_shape: The expected shape of the final assembled output array.
        The second dimension must equal the total number of trials.
    :param processes: The number of worker processes to use. Must be a positive
        integer.
    :param params: A tuple of positional parameters passed to ``func`` in each
        worker.
    :return: A NumPy array of shape ``out_shape`` containing the assembled
        results from all workers.
    """
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive int")
    if not isinstance(processes, int) or processes <= 0:
        raise ValueError("processes must be a positive int")

    # Divide trials into per-process jobs.
    jobs = divide_equally(trials, processes)

    # Build worker inputs (only schedule nonzero jobs).
    worker_args: list[tuple[int, Callable[..., np.ndarray], tuple[Any, ...]]] = [
        (job_trials, func, params) for job_trials in jobs if job_trials > 0
    ]

    # Preallocate output.
    res = np.zeros(out_shape)
    count = 0

    # Run workers and assemble results as they arrive.
    with mp.Pool(processes) as pool:
        for r in pool.imap_unordered(mp_worker, worker_args, chunksize=1):
            size = r.shape[1]
            end = count + size
            res[:, count:end] = r
            count = end

    return res
