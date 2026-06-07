#!/usr/bin/env python3
"""
File: emc.py
Author: Joshua Holmes
Email: jbh92@case.edu

An event based implementation of the traditional kinetic monte carlo
method, dubbed event monte carlo (emc).

Contains code for simulating myosin motor dynamics. The method used in
these implementations differs slightly from a true kinetic monte carlo.
Here, the probabilities of the motor taking a forward/backward step,
trailing/leading stomp, and terminating one of three ways are used to
determine what event the motor does, then determines the time it takes
for that event to happen. This eliminates the need to explicitly traverse
a Markov state network.

To make the simulations as numerically efficient as possible,
events are assigned, then referred to, by the following indices:
    0: Forward step (F).
    1: Backward step (B).
    2: Trailing stomp (TS).
    3: Leading stomp (LS).
    4: Terminate before hydrolysis (T1).
    5: Terminate after hydrolysis (T2).
    6: Terminate after leading head detachment (T3).
"""


import numpy as np
from numpy.typing import NDArray
from numba import njit
from . import mymath as mm
from . import motor_calculations as mc
from . import myroutines as mr


@njit
def draw_event(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> int:
    """
    Draw event for the motor to execute.

    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s).
    :param kh: ATP hydrolysis rate (1/s).
    :param kfpp: Rate of free head binding to forward position (1/s).
    :param kfpm: Rate of free head binding to reverse position (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.

    :return event: The drawn event with the following keys.
        0: Forward step (F).
        1: Backward step (B).
        2: Trailing stomp (TS).
        3: Leading stomp (LS).
        4: Terminate before hydrolysis (T1).
        5: Terminate after hydrolysis (T2).
        6: Terminate after leading head detachment (T3).
    """
    # Calculate event probabilities.
    pF = mc.pF(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pB = mc.pB(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pTs = mc.pTs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pLs = mc.pLs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pT1 = mc.pT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pT2 = mc.pT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pT3 = mc.pT3(kd1, kd2, kh, kfpp, kfpm, kps, b)

    probs = np.array([pF, pB, pTs, pLs, pT1, pT2, pT3])

    # Draw event.
    event = mm.sample_discrete(probs)
    return event


@njit
def draw_event_time(
    event: int,
    kd1: float,
    kd2: float,
    kh: float,
    kfpp: float,
    kfpm: float,
    kps: float,
    b: float,
) -> float:
    """
    Determine the time it takes for the motor to complete a given event.

    :params event: Event index according to the following key:
        0: Forward step (F).
        1: Backward step (B).
        2: Trailing stomp (TS).
        3: Leading stomp (LS).
        4: Terminate before hydrolysis (T1).
        5: Terminate after hydrolysis (T2).
        6: Terminate after leading head detachment (T3).
    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s).
    :param kh: ATP hydrolysis rate (1/s).
    :param kfpp: Rate of free head binding to forward position (1/s).
    :param kfpm: Rate of free head binding to reverse position (1/s).
    :param kps: Power stroke rate (1/s).
    :param b: The binding penalty.

    :return event_time: The time it took to complete the event (seconds).
    """
    event_time = 0
    # This looks ugly, but its the most efficient numba implementaion.
    if event == 0 or event == 2:
        rates = np.array([kd1 + kd2, kd1 + kh, kfpp + b * kfpm + kd1, kps])
        event_time += mm.sample_hypoexponential(rates)
    elif event == 1 or event == 3:
        rates = np.array([kd1 + kd2, kd1 + kfpm + b * kfpp])
        event_time += mm.sample_hypoexponential(rates)
    elif event == 4:
        rates = np.array([kd1 + kd2, kd1 + kh])
        event_time += mm.sample_hypoexponential(rates)
    elif event == 5:
        rates = np.array([kd1 + kd2, kd1 + kh, kfpp + b * kfpm + kd1])
        event_time += mm.sample_hypoexponential(rates)
    elif event == 6:
        rates = np.array([kd1 + kd2, kd1 + kfpm + b * kfpp])
        event_time += mm.sample_hypoexponential(rates)
    return event_time


@njit
def event_to_disp(event: int, currdisp: float) -> float:
    """
    Convert events into the resulting positional displacements.
    The conversions are done as follows in terms of steps:
        F  ->  1
        B  -> -1
        TS ->  0
        LS ->  0
        T1 ->  -currdisp
        T2 ->  -currdisp
        T2 ->  -currdisp

    :param event: int of the event index according to the following:
        0: Forward step (F).
        1: Backward step (B).
        2: Trailing stomp (TS).
        3: Leading stomp (LS).
        4: Terminate before hydrolysis (T1).
        5: Terminate after hydrolysis (T2).
        6: Terminate after leading head detachment (T3).
    :param currdisp: The current displacement of the motor (steps).

    :return disp: The displacement caused by the event in terms of steps.
    """
    return (event == 0) - (event == 1) - currdisp * (event > 3)


@njit
def init_kfp(
    const_force: float,
    kfpp: float,
    kfpm: float,
    Delta: float,
    L: float,
    l_p: float,
    beta: float,
    nu_c: float,
    theta_c: float,
    theta_F: float,
    a: float,
    D_h: float,
) -> float:
    """
    Initialize the kfpp and kfpm values for use in emc simulations based on given
    parameters. The most important arguments are const_force, kfpp, and kfpm.
    If const_force >= 0, then kfpp and kfpm will be calculated using this force
    value and other parameters. If const_force is < 0 but both kfpp and kfpm > 0
    then the values of kfpp and kfpm will be used directly . If none of the above
    conditions are met, then the transition matrix is built using kfpp = kfpm = None.
    This will indicate that the values of kfpp and kfpm need to be updated using
    the update_kmc_transmat function during the simulation.

    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use (1/s).
    :param Delta: Length of a motor step (nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param theta_F: Angle of applied force (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)

    :return kfpp, kfpm: Values to be used in simulations (1/s) or None if the values
        are meant to be updated during the simulation.

    """
    _kfpp, _kfpm = None, None
    if const_force >= 0:
        _kfpp, _kfpm = mc.kfp(
            const_force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
        )
    elif kfpp >= 0 and kfpm >= 0:
        _kfpp, _kfpm = kfpp, kfpm
    return _kfpp, _kfpm


@njit
def emc_run(
    num_events: int,
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
    Generate a string of events that constitute a motor's run.

    :param num_events: The length of the run in events.
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

    :return run: (3,num_events) ndarray representing the motor's run.
        The information stored in run is as follows:
            run[0]: The events that were drawn.
            run[1]: The times at which the event finishes (seconds).
            run[2]: The position of the motor following the event (steps).
    """
    _kfpp, _kfpm = init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )
    kfp_is_const = _kfpp is not None and _kfpm is not None

    run = np.zeros((3, num_events + 1))
    for i in range(num_events):
        # Calculate diffusion rates.
        if not kfp_is_const:
            force = mc.force(run[1, i] + 0.5, Delta, k)
            _kfpp, _kfpm = mc.kfp(
                force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
            )

        # Determine this event's consequences.
        event = draw_event(kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        dt = draw_event_time(event, kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        dn = event_to_disp(event, run[1, i])

        # Update run.
        run[0, i + 1] = float(event)
        run[1, i + 1] = run[1, i] + dn
        run[2, i + 1] = run[2, i] + dt
    return run[:, 1:]


@njit
def emc_dwell(
    num_samples: int,
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
    Emc simulation to record dwell times for a given myosin-V-like motor that follows
    the kinetic network as outlined in Hinczewski2013. A dwell time in this case is
    defined as the time accumulated between displacing events (F, B, T1,2,3), which
    includes the final displacing event.

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
        tdwells[0]: Dwell times (s).
        tdwells[1]: Displacement when dwell time is recorded (steps).
    """
    _kfpp, _kfpm = init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )
    kfp_is_const = _kfpp is not None and _kfpm is not None

    tdwells = np.zeros((2, num_samples))
    i = 0
    n = 0
    t = 0.0
    while i < num_samples:
        # Calculate diffusion rates
        if not kfp_is_const:
            force = mc.force(n + 0.5, Delta, k)
            _kfpp, _kfpm = mc.kfp(
                force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
            )
        # Determine this event's consequences.
        event = draw_event(kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        dn = event_to_disp(event, n)
        dt = draw_event_time(event, kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        t += dt
        if (event != 2 and event != 3) or t > time_limit:
            tdwells[0, i] = t
            tdwells[1, i] = n
            t = 0
            i += 1
        n += dn
    return tdwells


@njit
def emc_runstats(
    num_samples: int,
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
    _kfpp, _kfpm = init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )
    kfp_is_const = _kfpp is not None and _kfpm is not None

    res = np.zeros((2, num_samples))
    i = 0
    n = 0
    t = 0.0
    while i < num_samples:
        # Calculate diffusion rates.
        if not kfp_is_const:
            force = mc.force(n + 0.5, Delta, k)
            _kfpp, _kfpm = mc.kfp(
                force, theta_F, Delta, beta, L, l_p, nu_c, theta_c, a, D_h
            )

        # Determine this event's consequences.
        event = draw_event(kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        dt = draw_event_time(event, kd1, kd2, kh, _kfpp, _kfpm, kps, b)
        dn = event_to_disp(event, n)
        t += dt

        # Record run position and time if motor terminates or excedes time limit.
        if event > 3 or t > time_limit:
            res[0, i] = float(n)
            res[1, i] = t
            t = 0.0
            n = 0
            i += 1
        else:
            n += dn
    return res


def emc(
    params,
    trials,
    kind,
    const_force=-1,
    kfpp=-1,
    kfpm=-1,
    time_limit=None,
    processes=1,
):
    """
    Driver function for the emc based simulations.

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
    :param trials: The number of values to record.
    :param kind: The kind of simulation to run. Options include:
        'run': Runs the emc_trajectory function.
        'dwell': Runs the emc_dwell function.
        'runstats': Runs the emc_runstats function.
    :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
    :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
    :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
    :param time_limit: The upper limit of run times to prevent pseudo infinite loops.
        Default is None.
    :param processes: The number of processes to use during the simulation.
        If 'kind' == 'run' then only single processes are allowed.

    :return result: An np.ndarray with the following characteristics depending on 'kind'.
        'run':        run[0]: The events that were drawn (see module docstring).
                      run[1]: The position of the motor following the event (steps).
                      run[2]: The times at which the event finishes (seconds).

        'runstats':   runstats[0]: Run distances (steps).
                      runstats[1]: Run times (seconds).

        'dwell':      dwells[0]: Dwell times (seconds).
                      dwells[1]: Displacement when dwell time is recorded (steps).
    """
    assert kind in ["run", "dwell", "runstats"]
    if kind in ("dwell", "runstats"):
        assert time_limit is not None
    if kind == "run":
        assert processes == 1
        time_limit = None

    emc_funcs = dict(run=emc_run, dwell=emc_dwell, runstats=emc_runstats)
    emc_func = emc_funcs.get(kind)

    out_shapes = dict(run=(3, trials), dwell=(2, trials), runstats=(2, trials))
    out_shape = out_shapes.get(kind)

    _params = [time_limit, const_force, kfpp, kfpm]
    _params += params
    _params = [p for p in _params if p is not None]
    _params = tuple(_params)
    res = None
    if processes == 1:
        res = mr.mp_worker([trials, emc_func, _params])
    else:
        res = mr.mp_driver(trials, emc_func, out_shape, processes, _params)
    return res
