#!/usr/bin/env python3
"""
File: hpceval.py
Author: Joshua Holmes
Email: jbh92@case.edu

Functions to evaluate how well various motors describe the experimental results for
myosin-XI reported in Tominaga2003. Designed to be run on HPC resorces.
cluster.
"""

import numpy as np
from numpy.typing import NDArray
from myo5space.motor import Motor
import myo5space.fitting as fitting
import multiprocessing as mp
from itertools import product
from typing import Tuple, Callable


trials = 50000
method = "emc"
detach_force_cutoff = 0.25  # pN
dwell_force_cutoff = 0.5  # pN
vrun_force_cutoff = 0.29  # pN
time_cutoff = 0.02  # s
density_cutoff = 0.01


def make_kd1kd2_params() -> dict:
    """
    Make the parameter log-spaced grids for when kd1 and kd2 are the
    only free parameters. Min and max arguments for np.logspace are
    -1 and 4, respectively. Grid size is set to 200.

    :return: A dictionary with keys `kd1` and `kd2` and the grids as the values.
    """
    size = 200
    kd1s = np.logspace(-1, 4, size)
    kd2s = np.logspace(-1, 4, size)
    res = dict(kd1=kd1s, kd2=kd2s)
    return res


def make_kd1kd2nutheta_params() -> dict:
    """
    Make the parameter grids for when kd1, kd2, nu_c, and theata_c
    are the free parameters. kd1 and kd2 grids are log-spaced with
    min and max arguments -1 and 4, respectively. nu_c and theta_c
    grids are linspaced. Min and max arguments for nu_c are 50 and 1000.
    Min and max arguments for theta_c are pi/6 and 2pi/3. Grid size is set to 30.

    :return: A dictionary with keys `kd1`, `kd2`, `nu_c`, and `theta_c`. The grids
        are the values.
    """
    size = 30
    kd1s = np.logspace(-1, 4, size)
    kd2s = np.logspace(-1, 4, size)
    nu_cs = np.linspace(50, 1000, size)
    theta_cs = np.linspace(np.pi / 6, 2 * np.pi / 3, size)
    res = dict(kd1=kd1s, kd2=kd2s, nu_c=nu_cs, theta_c=theta_cs)
    return res


def make_jobs(param_grids: dict) -> list:
    """
    Generate jobs over the Cartesian product of parameter grids.

    :param param_grids: A dictionary of parameter grids. Keys are the parameter name.
        Values are the grids and should be np.arrays of floats.
    :return: A list of tuples with the following structure:
        tuple[0]: Index of the job within the CArtesian product of the grid.
        tuple[1]: Dictionary of parameter values. Keys are the same parameter 
            names as in `param_grids`. Values are the parameter values.
    """
    names = list(param_grids.keys())
    grids = list(param_grids.values())

    jobs = list()
    for ind, values in zip(product(*(range(len(g)) for g in grids)), product(*grids)):
        params = dict(zip(names, values))
        jobs.append((ind, params))

    return jobs


def dwell1um_worker(job: tuple) -> Tuple[tuple, float]:
    """
    A worker to fit a Motor paramaterized according to `job` to the experimental
    distribution of myosin-XI dwell times at 1 um ATP as in Tominaga2003. 
    Calculates the log-likelihood.

    :param job: A tuple with the following structure:
        job[0]: The index for the job within the Cartesian product of the 
            parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of 
            the parameter as in motor.Motor. Values are the parameter values.
    :return: A tuple with the following structure:
        tuple[0]: The index as passed by job[0]
        tuple[1]: Log-likelihood value.
    """
    ind = job[0]
    params = job[1]
    motor = Motor(kh=1000.0, **params)
    loglh = motor.loglh(
        "dwell 1um",
        trials,
        method=method,
        force_cutoff=dwell_force_cutoff,
        time_cutoff=time_cutoff,
        density_cutoff=density_cutoff,
    )
    return (ind, loglh)


def dwell100um_worker(job: tuple) -> Tuple[tuple, float]:
    """
    A worker to fit a Motor paramaterized according to `job` to the experimental
    distribution of myosin-XI dwell times at 100 um ATP as in Tominaga2003.
    Calculates the log-likelihood.

    :param job: A tuple with the following structure:
        job[0]: The index for the job within the Cartesian product of the parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of the 
            parameter as in motor.Motor. Values are the parameter values.
    :return: A tuple with the following structure:
        tuple[0]: The index as passed by job[0]
        tuple[1]: Log-likelihood value.
    """
    ind = job[0]
    params = job[1]
    motor = Motor(kh=1000.0, **params)
    loglh = motor.loglh(
        "dwell 100um",
        trials,
        method=method,
        force_cutoff=dwell_force_cutoff,
        time_cutoff=time_cutoff,
        density_cutoff=density_cutoff,
    )
    return (ind, loglh)


def vrun100um_worker(job: tuple) -> Tuple[tuple, float]:
    """
    A worker to find the average run velocity of a motor at after applying various
    inclusion and exclusion criteria to mimic those used in the experiments of
    Tominaga2003.

    :param job: A tuple with the following structure:
        job[0]: The index for the job within the Cartesian product of the parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of the
            parameter as in motor.Motor. Values are the parameter values.
    :return: A tuple with the following structure:
        tuple[0]: The index as passed by job[0]
        tuple[1]: The average run velocity after post-processing. np.nan if no values
            are left after post-processing.
    """
    ind = job[0]
    params = job[1]
    motor = Motor(kh=1000.0, **params)
    simres = motor.simulate("emc", "runstats", trials, k=4.1)
    vruns = fitting.proc_vrun(
        simres.vruns, simres.zruns, motor.Delta, 4.1, vrun_force_cutoff
    )
    avg = np.nan
    if vruns is None:
        avg = np.average(vruns)
    return (ind, avg)


def detach1um_worker(job: tuple) -> Tuple[tuple, float]:
    """
    A worker to fit a Motor paramaterized according to `job` to the experimental
    distribution of myosin-XI force-at-detachment at 1 um ATP as in Tominaga2003.
    Calculates the log-likelihood.

    :param job: A tuple with the following structure:
        job[0]: The index for the job within the Cartesian product of the parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of the 
            parameter as in motor.Motor. Values are the parameter values.
    :return: A tuple with the following structure:
        tuple[0]: The index as passed by job[0]
        tuple[1]: Log-likelihood value.
    """
    ind = job[0]
    params = job[1]
    motor = Motor(kh=1000.0, **params)
    loglh = motor.loglh(
        "detach 1um",
        trials,
        method=method,
        force_cutoff=detach_force_cutoff,
        time_cutoff=time_cutoff,
        density_cutoff=density_cutoff,
    )
    return (ind, loglh)


def detach100um_worker(job: tuple) -> Tuple[tuple, float]:
    """
    A worker to fit a Motor paramaterized according to `job` to the experimental
    distribution of myosin-XI force-at-detachment at 100 um ATP as in Tominaga2003.
    Calculates the log-likelihood.

    :param job: A tuple with the following structure:
        job[0]: The index for the job within the Cartesian product of the parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of the
            parameter as in motor.Motor. Values are the parameter values.
    :return: A tuple with the following structure:
        tuple[0]: The index as passed by job[0]
        tuple[1]: Log-likelihood value.
    """
    ind = job[0]
    params = job[1]
    motor = Motor(kh=1000.0, **params)
    loglh = motor.loglh(
        "detach 100um",
        trials,
        method=method,
        force_cutoff=detach_force_cutoff,
        time_cutoff=time_cutoff,
        density_cutoff=density_cutoff,
    )
    return (ind, loglh)


def run_jobs(jobs: list, worker: Callable, outshape: tuple) -> NDArray[np.float64]:
    """
    Run jobs using multiprocessing.

    :param jobs: A list of tuples with the following structure.
        job[0]: The index for the job within the Cartesian product of the parameter space.
        job[1]: A dictionary to paramaterize the motor. Keys are the name of the
    :param worker: A funciton to pass the job tuple to. Should return a tuple with the
        following structure.
        tuple[0]: The index as passed by job[0]
        tuple[1]: Number-like value.
    :param outshape: The shape of the parameter grid.
    :return: An NDArray with of the number-like values returned by the workers.
    """
    threads = mp.cpu_count()
    pool = mp.Pool(threads)
    pool_res = pool.imap_unordered(worker, jobs, chunksize=100)
    pool.close()

    res = np.zeros(outshape)
    count = 0
    for r in pool_res:
        ind = r[0]
        res[ind] = r[1]
        count += 1
        if count % 1000 == 0:
            print(100 * count / res.size)
    pool.join()

    return res
