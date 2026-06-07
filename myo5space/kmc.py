#!/usr/bin/env python3
"""
File: kmc.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains implementations of a kinetic monte carlo simulations for the
myosin-v-like network.
"""

# Imports
import numpy as np
from numpy.typing import NDArray
from numba import njit
import myo5space.mymath as mm
import myo5space.myroutines as mr
import myo5space.motor_calculations as mc


@njit
def kmc_transmat(
    kd1: float, kd2: float, kh: float, kps: float, kfpp: float, kfpm: float, b: float
) -> float:
    """
    Creates the transition matrix for a single motor for use in kmc simulations.
    Uses convention that the columns add to zero. The columns refer to the
    following states:
        0: Resting state.
        1: Trailing lead detached, pre hydrolysis.
        2: Trailing head detached, post hydrolysis.
        3: Trailing head attached to forward binding position pre-powerstroke.
        4: Completion of forward step post-powerstroke (absorbing).
        5: Trailing head attached to original binding position.
        6: Leading head detached.
        7: Completion of backwards step (absorbing).
        8: Termination (absorbing).

    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param kfpp: The rate for a diffusing head to find the forward binding position (1/s).
    :param kfpm: The rate for a diffusing head to find the backward binding position (1/s)
    :param b: The binding penalty.

    :return: A (9x9) transition matrix.
    """
    matrix = np.zeros((9, 9))
    # Fill the matrix off diagonal elements with transition rate values such
    # that matrix[i,j] is the rate from j -> i.
    matrix[1, 0] = kd1
    matrix[6, 0] = kd2
    matrix[2, 1] = kh
    matrix[8, 1] = kd1
    matrix[3, 2] = kfpp
    matrix[5, 2] = b * kfpm
    matrix[8, 2] = kd1
    matrix[4, 3] = kps
    matrix[0, 5] = kps
    matrix[0, 6] = b * kfpp
    matrix[7, 6] = kfpm
    matrix[8, 6] = kd1

    # Fill the diagonal elements such that the columns add up to 0.
    for i in range(matrix.shape[1]):
        matrix[i, i] = -np.sum(matrix[:, i])

    return matrix


@njit
def update_transmat(
    matrix: NDArray[np.float64], newkfpp: float, newkfpm: float, b: float, kd1: float
) -> None:
    """
    Updates the values of newkfpp and newkfpm in the kmc transition matrix in place.

    :param matrix: The matrix from the "kmc_transmat" function.
    :param newkfpp: Value of kfpp to update the matrix with (1/s).
    :param newkfpm: Value of kfpm to update the matrix with (1/s)
    :param b: Original value of binding penalty (unitless).
    :param kd1: Original value trailing head detachment (1/s)

    :return None: Updates matrix in place.
    """
    matrix[3, 2] = newkfpp
    matrix[5, 2] = b * newkfpm
    matrix[0, 6] = b * newkfpp
    matrix[7, 6] = newkfpm
    matrix[2, 2] = -1 * (newkfpp + b * newkfpm + kd1)
    matrix[6, 6] = -1 * (b * newkfpp + newkfpm + kd1)

    return None


@njit
def init_transmat(
    const_force: float,
    kfpp: float,
    kfpm: float,
    kd1: float,
    kd2: float,
    kh: float,
    kps: float,
    b: float,
    Delta: float,
    L: float,
    l_p: float,
    beta: float,
    nu_c: float,
    theta_c: float,
    theta_F: float,
    a: float,
    D_h: float,
) -> NDArray[np.float64]:
    """
    Initialize the transition matrix for use in kmc simulations based on given
    parameters. The most important arguments are const_force, kfpp, and kfpm.
    If const_force >= 0, then kfpp and kfpm will be calculated using this force
    value and other parameters. If const_force is < 0 but both kfpp and kfpm > 0
    then the values of kfpp and kfpm will be used directly in the resulting
    transition matrix. If none of the above conditions are met, then the
    transition matrix is built using kfpp = kfpm = -1. This will indicate that
    the values of kfpp and kfpm need to be updated using the update_kmc_transmat
    function during the simulation.

    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.
    :param Delta: Length of a motor step (nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param theta_F: Angle of applied force (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)

    :return kmc_transmat: Transition matrix to be used in simulations.
        If the positions for kfpp and kfpm are < 0, then this indicates the need
        to calculate these values and update the transition matrix during
        the simulation.
    """
    transmat = kmc_transmat(kd1, kd2, kh, kps, -1.0, -1.0, b)
    if const_force >= 0:
        _kfpp, _kfpm = mc.kfp(
            const_force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
        )
        update_transmat(transmat, _kfpp, _kfpm, b, kd1)
    elif kfpp > 0 and kfpm > 0:
        update_transmat(transmat, kfpp, kfpm, b, kd1)
    return transmat


@njit
def is_transmat_const(transmat: NDArray[np.float64]) -> bool:
    """
    Test to see if a transition matrix constructed by the init_transmat function
    is meant to be constant or not by seeing if one of the kfpp elements is
    negative or positive.

    :param transmat: Matrix from init_transmat function to be tested.

    :return is_const: Boolean answer.
    """
    return transmat[3, 2] > 0


@njit
def draw_state(rates: NDArray[np.float64]) -> int:
    """
    Draws a new state based on the relative weight of kinetic rates.
    Only chooses from states with rates > 0.

    :param rates: A 1D array of transition rates to choose from.

    :return newstate: An index of an element in the rates argument corresponding
        to the chosen state according to the following:
        0: Resting state.
        1: Trailing lead detached, pre hydrolysis.
        2: Trailing head detached, post hydrolysis.
        3: Trailing head attached to forward binding position pre-powerstroke.
        4: Completion of forward step post-powerstroke (absorbing).
        5: Trailing head attached to original binding position.
        6: Leading head detached.
        7: Completion of backwards step (absorbing).
        8: Termination (absorbing).
    """
    connected_states = np.where(rates > 1e-15)[0]
    newpos = mm.sample_discrete(rates[connected_states])
    newstate = connected_states[newpos]
    return newstate


@njit
def kmc_trajectory(
    length: int,
    const_force: float,
    kfpp: float,
    kfpm: float,
    kd1: float,
    kd2: float,
    kh: float,
    kps: float,
    b: float,
    Delta: float,
    L: float,
    l_p: float,
    beta: float,
    nu_c: float,
    theta_c: float,
    theta_F: float,
    a: float,
    D_h: float,
    k: float,
) -> NDArray[np.float64]:
    """
    Generate a trajectory for a myosin-5-like motor traversing the kinetic
    network as described in Hinczewski2013.

    :param: length: The number of state transitions to record.
    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.
    :param Delta: Length of a motor step (nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param theta_F: Angle of applied force (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)
    :param k: Force constant of the trap in the case of non-const force (pN/nm)

    :return trajectory: A (3, length) np.ndarray with the following structure:
        trajectory[0]: The states visited by the motor (see kmc_transmat docstring).
        trajectory[1]: The current displacement of the motor (steps).
        trajectory[2]: The time since starting the trajectory (s).

    """
    # Initialize transition matrix.
    transmat = init_transmat(
        const_force,
        kfpp,
        kfpm,
        kd1,
        kd2,
        kh,
        kps,
        b,
        Delta,
        L,
        l_p,
        beta,
        nu_c,
        theta_c,
        theta_F,
        a,
        D_h,
    )
    update_tmat = not is_transmat_const(transmat)
    # Main kmc loop.
    trajectory = np.zeros((3, length))
    for i in range(length - 1):
        # Update transmat if not constant.
        if update_tmat:
            force = mc.force(trajectory[1, i] + 0.5, Delta, k)
            _kfpp, _kfpm = mc.kfp(
                force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
            )
            update_transmat(transmat, _kfpp, _kfpm, b, kd1)
        # Update the state.
        current_state = int(trajectory[0, i])
        connected_rates = transmat[:, current_state]
        newstate = draw_state(connected_rates)
        trajectory[0, i + 1] = newstate * (
            1 - (newstate == 4) - (newstate == 7) - (newstate == 8)
        )
        # Update the displacement
        trajectory[1, i + 1] = (
            trajectory[1, i]
            + (newstate == 4)
            - (newstate == 7)
            - trajectory[1, i] * (newstate == 8)
        )
        # update the time.
        totrate = -1 * connected_rates[current_state]
        trajectory[2, i + 1] = trajectory[2, i] + mm.sample_exponential(totrate)

    return trajectory


@njit
def kmc_runstats(
    trials: int,
    time_limit: float,
    const_force: float,
    kfpp: float,
    kfpm: float,
    kd1: float,
    kd2: float,
    kh: float,
    kps: float,
    b: float,
    Delta: float,
    L: float,
    l_p: float,
    beta: float,
    nu_c: float,
    theta_c: float,
    theta_F: float,
    a: float,
    D_h: float,
    k: float,
) -> NDArray[np.float64]:
    """
    Measure the run distance and run time for a myosin-V-like motor that behaves
    according to the theoretical model proposed by Hinczewski2013.

    :param: trials: The number of runs to record.
    :param time_limit: The upper limit of run times to prevent pseudo infinite loops.
    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.
    :param Delta: Length of a motor step (nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param theta_F: Angle of applied force (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)
    :param k: Force constant of the trap in the case of non-const force (pN/nm)

    :return runstats: A (2, length) np.ndarray with the following structure:
        runstats[0]: Run distances (steps).
        runstats[1]: Run times (s).
    """
    # Initialize transition matrix.
    transmat = init_transmat(
        const_force,
        kfpp,
        kfpm,
        kd1,
        kd2,
        kh,
        kps,
        b,
        Delta,
        L,
        l_p,
        beta,
        nu_c,
        theta_c,
        theta_F,
        a,
        D_h,
    )
    update_tmat = not is_transmat_const(transmat)
    # Main kmc loop.
    runstats = np.zeros((2, trials))
    for i in range(trials):
        state = 0
        n = 0
        time = 0.0
        end = False
        while not end:
            # Update transmat if not constant.
            if update_tmat:
                force = mc.force(n + 0.5, Delta, k)
                _kfpp, _kfpm = mc.kfp(
                    force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
                )
                update_transmat(transmat, _kfpp, _kfpm, b, kd1)
            # Update the state.
            connected_rates = transmat[:, state]
            totrate = -1 * connected_rates[state]
            time += mm.sample_exponential(totrate)
            state = draw_state(connected_rates)
            # Update the displacement
            n += (state == 4) - (state == 7)
            state = state * (1 - (state == 4) - (state == 7))

            if state == 8 or time > time_limit:
                end = True
        runstats[0, i] = n
        runstats[1, i] = time
    return runstats


@njit
def kmc_dwell(
    trials: int,
    time_limit: float,
    const_force: float,
    kfpp: float,
    kfpm: float,
    kd1: float,
    kd2: float,
    kh: float,
    kps: float,
    b: float,
    Delta: float,
    L: float,
    l_p: float,
    beta: float,
    nu_c: float,
    theta_c: float,
    theta_F: float,
    a: float,
    D_h: float,
    k: float,
) -> NDArray[np.float64]:
    """
    Simulation to record dwell times for a given myosin-V-like motor that follows
    the kinetic network as outlined in Hinczewski2013. A dwell time is defined as
    the time between the motor arriving at the resting state to it moving to another
    location on actin or terminating.

    :param: trials: The number of runs to record.
    :param time_limit: The upper limit of run times to prevent pseudo infinite loops.
    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.
    :param Delta: Length of a motor step (nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param theta_F: Angle of applied force (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)
    :param k: Force constant of the trap in the case of non-const force (pN/nm)

    :return tdwells: A (2, length) np.ndarray with the following structure:
        tdwells[0]: Dwell times (seconds).
        tdwells[1]: Displacement when dwell time is recorded (steps).
    """
    # Initialize transition matrix.
    transmat = init_transmat(
        const_force,
        kfpp,
        kfpm,
        kd1,
        kd2,
        kh,
        kps,
        b,
        Delta,
        L,
        l_p,
        beta,
        nu_c,
        theta_c,
        theta_F,
        a,
        D_h,
    )
    update_tmat = not is_transmat_const(transmat)
    # Main kmc loop.
    tdwells = np.zeros((2, trials))
    count = 0
    while count < trials:
        state = 0
        n = 0
        dwell_time = 0.0
        end = False
        while not end and count < trials:
            # Update transmat if not constant.
            if update_tmat:
                force = mc.force(n + 0.5, Delta, k)
                _kfpp, _kfpm = mc.kfp(
                    force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
                )
                update_transmat(transmat, _kfpp, _kfpm, b, kd1)
            # Update time.
            connected_rates = transmat[:, state]
            totrate = -1 * connected_rates[state]
            dwell_time += mm.sample_exponential(totrate)
            # Update the state.
            state = draw_state(connected_rates)
            dn = (state == 4) - (state == 7)
            state = state * (1 - (state == 4) - (state == 7))
            # Handel cases when dwell time finishes.
            if dn != 0:
                tdwells[0, count] = dwell_time
                tdwells[1, count] = n
                dwell_time = 0.0
                count += 1
                n += dn
            elif state == 8 or dwell_time > time_limit:
                end = True
                tdwells[0, count] = dwell_time
                tdwells[1, count] = n
                count += 1
    return tdwells


def kmc(
    params, trials, kind, const_force=-1, kfpp=-1, kfpm=-1, time_limit=None, processes=1
):
    """
    Driver function for the kmc based simulations.

    :param params: A list of parameters parameterizing a myosin motor. The list
        should be ordered as follows:
         kd1: Trailing head detachment rate (1/s).
         kd2: Leading head detachment rate (1/s)
         kh: ATP hydrolysis rate (1/s).
         kps: Power stroke rate (1/s).
         b: The binding penalty.
         Delta: Length of a motor step (nm).
         L: Length of the motor leg (nm).
         l_p: Persistence length of motor leg (nm).
         beta: 1 / (kB*T) (pN nm).
         nu_c: Strength of binding constraint (unitless).
         theta_c: Angle of constraint (rads).
         theta_F: Angle of applied force (rads).
         a: Debye radius (nm).
         D_h: Motor head diffusivity (nm^2/s)
         k: Force constant of the trap in the case of non-const force (pN/nm)
    :param trials: The number of runs to record.
    :param kind: The kind of simulation to run. Options include:
        'trajectory': Runs the kmc_trajectory function.
        'runstats': Runs the kmc_runstats function.
        'dwell': Runs the kmc_dwell function.
    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param time_limit: The upper limit of run times to prevent pseudo infinite loops.
        Default is None.
    :param processes: The number of processes to use during the simulation.
        If 'kind' == 'trajectory' then only single processes are allowed.

    :return result: An np.ndarray with the following characteristics depending on 'kind'.
        'trajectory': trajectory[0]: Index of the stats visited (see module docstring).
                      trajectory[1]: Current displacement (steps).
                      trajectory[2]: Current time (seconds).

        'runstats':   runstats[0]: Run distances (steps).
                      runstats[1]: Run times (seconds).

        'dwell':      tdwells[0]: Dwell times (seconds).
                      tdwells[1]: Displacement when dwell time is recorded (steps).
    """
    assert kind in ["trajectory", "dwell", "runstats"]
    if kind in ("dwell", "runstats"):
        assert time_limit is not None
    if kind == "trajectory":
        assert processes == 1
        time_limit = None

    kmc_funcs = dict(trajectory=kmc_trajectory, dwell=kmc_dwell, runstats=kmc_runstats)
    kmc_func = kmc_funcs.get(kind)

    out_shapes = dict(trajectory=(3, trials), dwell=(2, trials), runstats=(2, trials))
    out_shape = out_shapes.get(kind)

    _params = [time_limit, const_force, kfpp, kfpm]
    _params = [p for p in _params if p is not None]
    _params += params
    _params = tuple(_params)
    res = None
    if processes == 1:
        res = mr.mp_worker([trials, kmc_func, _params])
    else:
        res = mr.mp_driver(trials, kmc_func, out_shape, processes, _params)
    return res
