#!/usr/bin/env python3
"""
File: simtesting.py
Author: Joshua Holmes
Email: jbh92@case.edu

Module for holding funcitons used to test and compare different simulation methods.
"""

from myo5space.motor import Motor
from myo5space import kmc

import pickle
from typing import Callable, Literal, Sequence, List
from numpy.typing import NDArray
import multiprocessing as mp
from pathlib import Path

import numpy as np
import numpy.random as nprand
import scipy.integrate as integ
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_motors(size: int, a: float, b: float) -> list[Motor]:
    """
    Make a list of motors with randomly generated values for kd1, kd2, and kh.
    Parameters are drawn from loguniform distributions. Other parameters are the
    defaults for motor.Motor.

    :param size: Number of motors to make.
    :param a: Min argument for loguniform distribution.
    :param b: Max argument for loguniform distribution.
    :return: A (size,) list of the generated motors.
    """
    kd1s = stats.loguniform.rvs(a, b, size=size)
    kd2s = stats.loguniform.rvs(a, b, size=size)
    khs = stats.loguniform.rvs(a, b, size=size)

    motors = [Motor(kd1=kd1, kd2=kd2, kh=kh) for kd1, kd2, kh in zip(kd1s, kd2s, khs)]
    return motors


def make_kfps(
    size: int, a: float, b: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Make arrays of randomly generated values for kfpp and kfpm drawn from a
    loguniform distribution. kfpp < kfpm for each pair of values.

    :param size: Number of motors to make.
    :param a: Min argument for loguniform distribution.
    :param b: Max argument for loguniform distribution.
    :return: A tuple of two (size,) arrays of kfpp and kfpm values.
    """
    kfpps = stats.loguniform.rvs(a, b, size=size)
    kfpms = np.array([stats.loguniform.rvs(a, kfpp) for kfpp in kfpps])
    return kfpps, kfpms


def sim_runstats(
    method: Literal["kmc", "emc"],
    motors: List[Motor],
    kfpps: NDArray[np.float64],
    kfpms: NDArray[np.float64],
    trials: int,
    processes: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate average zrun, trun, and vrun stats for a list of motors using either
    the kmc or emc method. kfpp and kfpm are explicity defined.

    :param method: 'kmc' or 'emc' to define the simulation method to use.
    :param motors: An (N,) list of motor.Motor objects to simulate.
    :param kfpps: An (N,) array of kfpps to simulate the motors with.
    :param kfpms: An (N,) array of kfpms to simulate the motors with.
    :param trials: The number to trials to simulate each motor for.
    :param processes: The number of processes to use during the simulation.
        Defaults to None, which will then use multiprocessing.cpu_count.
    :return: A tuple of three (N,) arrays with the resulting average zrun, trun, and vrun
        for each motor.
    """
    if not processes:
        processes = mp.cpu_count()

    zrunavg = np.zeros(len(motors), dtype=float)
    trunavg = np.zeros(len(motors), dtype=float)
    vrunavg = np.zeros(len(motors), dtype=float)

    for i, m in enumerate(motors):
        kfpp = kfpps[i]
        kfpm = kfpms[i]

        res = m.simulate(
            method,
            "runstats",
            trials,
            kfpp=kfpp,
            kfpm=kfpm,
            time_limit=100.0,
            processes=processes,
        )
        zrunavg[i] = res.zrun_mean()
        trunavg[i] = res.trun_mean()
        vrunavg[i] = res.vrun_mean()

        denom = 10 if len(motors) // 10 != 0 else len(motors)
        if i % (len(motors) // denom) == 0:
            print(f"{i}/{len(motors)}")

    return zrunavg, trunavg, vrunavg


def sim_tdwells(
    method: Literal["kmc", "emc"],
    motors: List[Motor],
    kfpps: NDArray[np.float64],
    kfpms: NDArray[np.float64],
    trials: int,
    processes: int | None = None,
) -> NDArray[np.float64]:
    """
    Generate average tdwell stats for a list of motors using either
    the kmc or emc method. kfpp and kfpm are explicity defined.

    :param method: 'kmc' or 'emc' to define the simulation method to use.
    :param motors: An (N,) list of motor.Motor objects to simulate.
    :param kfpps: An (N,) array of kfpps to simulate the motors with.
    :param kfpms: An (N,) array of kfpms to simulate the motors with.
    :param trials: The number to trials to simulate each motor for.
    :param processes: The number of processes to use during the simulation.
        Defaults to None, which will then use multiprocessing.cpu_count.
    :return: An (N,) arrays with the resulting average tdwell for each motor.
    """
    if not processes:
        processes = mp.cpu_count()

    tdwellavg = np.zeros(len(motors), dtype=float)

    for i, m in enumerate(motors):
        kfpp = kfpps[i]
        kfpm = kfpms[i]

        res = m.simulate(
            method,
            "dwell",
            trials,
            kfpp=kfpp,
            kfpm=kfpm,
            time_limit=100.0,
            processes=processes,
        )
        tdwellavg[i] = res.tdwell_mean()

        denom = 10 if len(motors) // 10 != 0 else len(motors)
        if i % (len(motors) // denom) == 0:
            print(f"{i}/{len(motors)}")

    return tdwellavg


def ana_avgs(
    motors: List[Motor], kfpps: NDArray[np.float64], kfpms: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Calculate values for the average zrun, trun, vrun, and tdwell for a list of
    motors using analytical methods. kfpp and kfpm are explicity defined.

    :param motors: An (N,) list of motor.Motor objects to calculate averages for.
    :param kfpps: An (N,) array of kfpps to simulate the motors with.
    :param kfpms: An (N,) array of kfpms to simulate the motors with.
    :return: A tuple of four (N,) arrays with the resulting average zrun, trun, vrun
        and tdwell for each motor.
    """
    zrunavg = np.zeros(len(motors), dtype=float)
    trunavg = np.zeros(len(motors), dtype=float)
    vrunavg = np.zeros(len(motors), dtype=float)
    tdwellavg = np.zeros(len(motors), dtype=float)

    for i, m in enumerate(motors):
        kfpp = kfpps[i]
        kfpm = kfpms[i]

        zrunavg[i] = m.zrun_mean(kfpp=kfpp, kfpm=kfpm)
        trunavg[i] = m.trun_mean2(kfpp=kfpp, kfpm=kfpm)
        vrunavg[i] = m.vrun_mean(kfpp=kfpp, kfpm=kfpm)
        tdwellavg[i] = m.tdwell_mean(kfpp=kfpp, kfpm=kfpm)

    return zrunavg, trunavg, vrunavg, tdwellavg


class SimTestingRes(object):
    def __init__(self, trials: int = 100000):
        self.trials = trials
        self.motors = None
        self.kfpps = None
        self.kfpms = None
        self.zrunavg_kmc = None
        self.trunavg_kmc = None
        self.vrunavg_kmc = None
        self.tdwellavg_kmc = None
        self.zrunavg_emc = None
        self.trunavg_emc = None
        self.vrunavg_emc = None
        self.tdwellavg_emc = None
        self.zrunavg_ana = None
        self.trunavg_ana = None
        self.vrunavg_ana = None
        self.tdwellavg_ana = None

    @classmethod
    def load(cls, path: str | Path) -> "SimTestingRes":
        """
        Load a pickled SimTestingResults object.

        :param path: Path to pickle file.
        :return: The loaded SimTestingResults object.
        """
        path = Path(path)
        with path.open("rb") as fh:
            obj = pickle.load(fh)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle does not contain a {cls.__name__}")

        return obj

    def save(self, path: str | Path) -> None:
        """
        Save this object to a Pickle file.

        :param path: The path to save this object under.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def check_ok(self) -> bool:
        non_vals = [val for key, val in self.__dict__.items() if val is None]
        if non_vals:
            return False

        n_motors = len(self.motors)
        if self.kfpms.size != n_motors or self.kfpms.size != n_motors:
            return False

        sims = [
            self.zrunavg_kmc,
            self.trunavg_kmc,
            self.vrunavg_kmc,
            self.tdwellavg_kmc,
            self.zrunavg_emc,
            self.trunavg_emc,
            self.vrunavg_emc,
            self.tdwellavg_emc,
            self.zrunavg_ana,
            self.trunavg_ana,
            self.vrunavg_ana,
            self.tdwellavg_ana,
        ]
        bad = [not isinstance(sim, np.ndarray) for sim in sims]
        if any(bad):
            return False
        bad_sizes = [sim.size != n_motors for sim in sims]
        if any(bad_sizes):
            return False
        return True

    def reset_parameters(self) -> None:
        """
        Reset all instance parameters except for self.trials.
        """
        self.motors = None
        self.kfpps = None
        self.kfpms = None
        self.zrunavg_kmc = None
        self.trunavg_kmc = None
        self.vrunavg_kmc = None
        self.tdwellavg_kmc = None
        self.zrunavg_emc = None
        self.trunavg_emc = None
        self.vrunavg_emc = None
        self.tdwellavg_emc = None
        self.zrunavg_ana = None
        self.trunavg_ana = None
        self.vrunavg_ana = None
        self.tdwellavg_ana = None

    def make_parameters(self, size: int = 1000) -> None:
        """
        Make a new set of slef.motors, self.kfpp, and self.kfpm parameters.
        All preexisting parameters and simulation results are cleared to avoid
        potential missmatches between saved parameters and results.

        :param size: The number of motors to generate.
        """
        # Reset all values
        self.reset_parameters()
        # Make new parameter set
        a = 1
        b = 1e6
        self.motors = make_motors(size, a, b)
        self.kfpps, self.kfpms = make_kfps(size, a, b)

    def run_kmc(self) -> None:
        """
        Generate simulation results using kmc methods. Sets the following
        instance parameters:
            self.zrunavg_kmc, self.trunavg_kmc, self.vrunavg_kmc, self.tdwellavg_kmc
        """
        print("Simulating KMC runstats.")
        self.zrunavg_kmc, self.trunavg_kmc, self.vrunavg_kmc = sim_runstats(
            "kmc", self.motors, self.kfpps, self.kfpms, self.trials
        )
        print("Simulating KMC tdwells.")
        self.tdwellavg_kmc = sim_tdwells(
            "kmc", self.motors, self.kfpps, self.kfpms, self.trials
        )

    def run_emc(self) -> None:
        """
        Generate simulation results using emc methods. Sets the following
        instance parameters:
            self.zrunavg_emc, self.trunavg_emc, self.vrunavg_emc, self.tdwellavg_emc
        """
        print("Simulating EMC runstats.")
        self.zrunavg_emc, self.trunavg_emc, self.vrunavg_emc = sim_runstats(
            "emc", self.motors, self.kfpps, self.kfpms, self.trials
        )
        print("Simulating EMC tdwells.")
        self.tdwellavg_emc = sim_tdwells(
            "emc", self.motors, self.kfpps, self.kfpms, self.trials
        )

    def run_ana(self) -> None:
        """
        Generate simulation results using analyitical methods. Sets the following
        instance parameters:
            self.zrunavg_ana, self.trunavg_ana, self.vrunavg_ana, self.tdwellavg_ana
        """
        self.zrunavg_ana, self.trunavg_ana, self.vrunavg_ana, self.tdwellavg_ana = (
            ana_avgs(self.motors, self.kfpps, self.kfpms)
        )

    def run_all(self, size=1000) -> None:
        """
        Make a new set of parameters and run all simulations and analytical
        calculations. Sets all instance parameters except for self.trials.

        :param size: The number of motors to generate.
        """
        self.make_parameters(size)
        self.run_kmc()
        self.run_emc()
        self.run_ana()


def calc_residual(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the elementwise residuals between two arrays relative to `a`.

    :param a: (N,) array of the first set of values.
    :param b: (N,) array of the second set of values.
    :return: (N,) array of the residuals relative to `a`.
    """
    residual = np.abs((a - b) / a)
    return residual


def find_nonzeros(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Find the indecies where both of two arrays are zero.

    :param a: (N,) array of the first set of values.
    :param b: (N,) array of the second set of values.
    :return: Array of the locations where `a` and `b` are zero.
    """
    a_nonzeros = np.where(a != 0)[0]
    b_nonzeros = np.where(b != 0)[0]
    nonzeros = np.intersect1d(a_nonzeros, b_nonzeros)
    return nonzeros


def calc_residuals(
    ana: NDArray[np.float64], kmc: NDArray[np.float64], emc: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the pairwise relative residuals for observables caclulated by either the
    kmc, emc, or analytical methods.

    :param ana: (N,) array of values calculated by analytical methods.
    :param kmc: (N,) array of values caclulted by kmc method.
    :param emc: (N,) array of values caclulated by emc method.
    :return: A three tuples of [(N,), (N,), (M,)] arrays of residuals with the
        following interpretations:
            result[0]: Residuals between analytical and kmc values.
            result[1]: Residuals between analytical and emc values.
            result[2]: Residuals between kmc and emc values. The result array
                can be of different length since it skips residual calculations
                where the values are equal to eachother.
    """
    ak = calc_residual(ana, kmc)
    ae = calc_residual(ana, emc)

    nonzeros = find_nonzeros(kmc, emc)
    ke = calc_residual(kmc[nonzeros], emc[nonzeros])

    return ak, ae, ke


def find_maxresidual(
    motors: list[Motor],
    kfpps: NDArray[np.float64],
    kfpms: NDArray[np.float64],
    resids: NDArray[np.float64],
) -> tuple[int, float, Motor, float, float]:
    """
    Find the parameter set that resulted in the maximum residual.

    :param motors: (N,) list of motor.Motor objects
    :param kfpps: (N,) array of kfpp values used in simulations with motors.
    :param kfpms: (N,) array of kfpm values used in simulations with motors.
    :param resids: (N,) array of relative residuals associated with the simulations run with
        motors, kfpps, and kfpms
    :return: A five tuple with the following interpretations:
        result[0]: The argument of the max value in resid array.
        result[1]: The maximum value in resdi array.
        result[2]: The motor producing the max residual.
        result[3]: The kfpp producing the max residual.
        result[4]: The kfpm producing the max residual.
    """
    arg = resids.argmax()
    val = resids.max()
    motor = motors[arg]
    kfpp = kfpps[arg]
    kfpm = kfpms[arg]
    return arg, val, motor, kfpp, kfpm


def check_maxresidual_zrun(
    motor: Motor,
    kfpp: float,
    kfpm: float,
    trials: int,
    method1: Literal["kmc", "emc", "ana"],
    method2: Literal["kmc", "emc", "ana"],
    processes: int = mp.cpu_count(),
) -> float:
    """
    Check that the maximum residual for the average run distance can be reduced
    by increasing the number of trials run.

    :param motor: The motor.Motor to simulate.
    :param kfpp: The kfpp value to use in simulations.
    :param kfpm: The kfpm value to use in simulations.
    :param trials: The number of trials to use.
    :param method1: The first methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param method2: The second methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param processes: The number of cpu processes to use. Defaults to mp.cpu_count()
    :return: The new residual for between the average run distance caclulated by
        the two methods.
    """
    avgs = list()
    for method in [method1, method2]:
        if "mc" in method:
            simres = motor.simulate(
                method,
                "runstats",
                trials,
                kfpp=kfpp,
                kfpm=kfpm,
                time_limit=100,
                processes=processes,
            )
            avg = simres.zrun_mean()
            avgs.append(avg)
        else:
            avg = motor.zrun_mean(kfpp=kfpp, kfpm=kfpm)
            avgs.append(avg)
    residual = calc_residual(*avgs)
    return residual


def check_maxresidual_trun(
    motor: Motor,
    kfpp: float,
    kfpm: float,
    trials: int,
    method1: Literal["kmc", "emc", "ana"],
    method2: Literal["kmc", "emc", "ana"],
    processes: int = mp.cpu_count(),
) -> float:
    """
    Check that the maximum residual for the average run time can be reduced
    by increasing the number of trials run.

    :param motor: The motor.Motor to simulate.
    :param kfpp: The kfpp value to use in simulations.
    :param kfpm: The kfpm value to use in simulations.
    :param trials: The number of trials to use.
    :param method1: The first methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param method2: The second methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param processes: The number of cpu processes to use. Defaults to mp.cpu_count()
    :return: The new residual for between the average run time caclulated by
        the two methods.
    """
    avgs = list()
    for method in [method1, method2]:
        if "mc" in method:
            simres = motor.simulate(
                method,
                "runstats",
                trials,
                kfpp=kfpp,
                kfpm=kfpm,
                time_limit=100000,
                processes=processes,
            )
            avg = simres.trun_mean()
            avgs.append(avg)
        else:
            avg = motor.trun_mean2(kfpp=kfpp, kfpm=kfpm)
            avgs.append(avg)
    residual = calc_residual(*avgs)
    return residual


def check_maxresidual_vrun(
    motor: Motor,
    kfpp: float,
    kfpm: float,
    trials: int,
    method1: Literal["kmc", "emc", "ana"],
    method2: Literal["kmc", "emc", "ana"],
    processes: int = mp.cpu_count(),
) -> float:
    """
    Check that the maximum residual for the average run velocity can be reduced
    by increasing the number of trials run.

    :param motor: The motor.Motor to simulate.
    :param kfpp: The kfpp value to use in simulations.
    :param kfpm: The kfpm value to use in simulations.
    :param trials: The number of trials to use.
    :param method1: The first methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param method2: The second methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param processes: The number of cpu processes to use. Defaults to mp.cpu_count()
    :return: The new residual for between the average run velocity caclulated by
        the two methods.
    """
    avgs = list()
    for method in [method1, method2]:
        if "mc" in method:
            simres = motor.simulate(
                method,
                "runstats",
                trials,
                kfpp=kfpp,
                kfpm=kfpm,
                time_limit=100,
                processes=processes,
            )
            avg = simres.vrun_mean()
            avgs.append(avg)
        else:
            avg = motor.vrun_mean(kfpp=kfpp, kfpm=kfpm)
            avgs.append(avg)
    residual = calc_residual(*avgs)
    return residual


def check_maxresidual_tdwell(
    motor: Motor,
    kfpp: float,
    kfpm: float,
    trials: int,
    method1: Literal["kmc", "emc", "ana"],
    method2: Literal["kmc", "emc", "ana"],
    processes: int = mp.cpu_count(),
) -> float:
    """
    Check that the maximum residual for the average dwell time can be reduced
    by increasing the number of trials run.

    :param motor: The motor.Motor to simulate.
    :param kfpp: The kfpp value to use in simulations.
    :param kfpm: The kfpm value to use in simulations.
    :param trials: The number of trials to use.
    :param method1: The first methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param method2: The second methods to use. Can be one of the following:
        "kmc": Use kmc method.
        "emc": Use emc method.
        "ana": Use analytical method.
    :param processes: The number of cpu processes to use. Defaults to mp.cpu_count()
    :return: The new residual for between the average dwell time caclulated by
        the two methods.
    """
    avgs = list()
    for method in [method1, method2]:
        if "mc" in method:
            simres = motor.simulate(
                method,
                "dwell",
                trials,
                kfpp=kfpp,
                kfpm=kfpm,
                time_limit=100,
                processes=processes,
            )
            avg = simres.tdwell_mean()
            avgs.append(avg)
        else:
            avg = motor.tdwell_mean(kfpp=kfpp, kfpm=kfpm)
            avgs.append(avg)
    residual = calc_residual(*avgs)
    return residual


def check_maxresiduals(
    param: Literal["zrun", "trun", "vrun", "tdwell"],
    trials: int,
    motors: list[Motor],
    kfpps: NDArray[np.float64],
    kfpms: NDArray[np.float64],
    ana: NDArray[np.float64],
    kmc: NDArray[np.float64],
    emc: NDArray[np.float64],
) -> str:
    """
    Check that the maximum residual for an observable can be reduced
    by increasing the number of trials run.

    :param param: The observable being tested. Can be one of the following:
        "zrun": Average run distance.
        "trun": Average run time.
        "vrun": Average run velocity.
        "tdwell": Average dwell time.
    :param trials: _de
    :param motor: The motor.Motor to simulate.
    :param kfpp: The kfpp value to use in simulations.
    :param kfpm: The kfpm value to use in simulations.
    :param trials: The number of trials to use.
    :param ana: (N,) array of parameter values from analytical method.
    :param kmc: (N,) array of parameter values from kmc method.
    :param emc: (N,) array of parameter values from emc method.
    :return: A string summary of the result comparing old and new residuals.
    """
    ak, ae, ke = calc_residuals(ana, kmc, emc)
    methods = [("ana", "kmc"), ("ana", "emc"), ("kmc", "emc")]
    func_dict = {
        "zrun": check_maxresidual_zrun,
        "trun": check_maxresidual_trun,
        "vrun": check_maxresidual_vrun,
        "tdwell": check_maxresidual_tdwell,
    }
    func = func_dict.get(param)

    ret_str = ""
    for residual, meth in zip([ak, ae, ke], methods):
        arg, val, motor, kfpp, kfpm = find_maxresidual(motors, kfpps, kfpms, residual)
        motor_str = f"Motor with the worst relative residual:\n     kd1: {motor.kd1} 1/s\n     kd2: {motor.kd2} 1/s\n     kh: {motor.kh} 1/s\n     kfpp: {kfpp} 1/s\n     kfpm: {kfpm} 1/s\n     {param}: {val}\n"
        
        new_val = func(motor, kfpp, kfpm, trials, meth[1], meth[0])
        string = f"Check {meth[0]} - {meth[1]} with {trials} trials:\n     Max Log10 residual: {np.log10(val)}\n     New Log10 resdiual {np.log10(new_val)}\n"
        ret_str += string
        ret_str += motor_str
        ret_str += "\n\n"
    return ret_str
