#!/usr/bin/env python3
"""
File: test_emc.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains tests for the functions in the emc.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest
import myo5space.emc as emc


def test_draw_event_onlyF():
    """
    Test that the draw_event function only returns forward steps and
    terminating events when the kinetic parameters are such that only the trailing
    head can detach and stomps are impossible.
    """
    # Setup
    kd1 = 10.0
    kd2 = 0.0
    kh = 700.0
    kfpp = 3000.0
    kfpm = 0.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [0, 4, 5] for event in got_events])


def test_draw_event_onlyTs():
    """
    Test that the draw_event function only returns trailing stomps and
    terminating events when the kinetic parameters are such that only the trailing
    head can detach and stomps are impossible.
    """
    # Setup
    kd1 = 10.0
    kd2 = 0.0
    kh = 700.0
    kfpp = 0.0
    kfpm = 3000.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [2, 4, 5] for event in got_events])


def test_draw_event_onlyT1():
    """
    Test that the draw_event function only returns terminating events following
    hydrolysis when the kinetic parameters are such that only the trailing can
    detach, but can't hydrolyze ATP.
    """
    # Setup
    kd1 = 10.0
    kd2 = 0.0
    kh = 0.0
    kfpp = 0.0
    kfpm = 3000.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [4] for event in got_events])


def test_draw_event_onlyT1T2():
    """
    Test that the draw_event function only returns terminating events pre or post
    hydrolysis when the kinetic parameters are such that only the trailing head
    can't reattach to actin.
    """
    # Setup
    kd1 = 10.0
    kd2 = 0.0
    kh = 700.0
    kfpp = 0.0
    kfpm = 0.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [4, 5] for event in got_events])


def test_draw_event_onlyB():
    """
    Test that the draw_event function only returns backward steps when the kinetic
    parameters are such that only the leading head can detach then reattach to the
    previous actin binding site.
    """
    # Setup
    kd1 = 0.0
    kd2 = 10.0
    kh = 700.0
    kfpp = 0.0
    kfpm = 1000.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [1] for event in got_events])


def test_draw_event_onlyLs():
    """
    Test that the draw_event function only returns leading stomps when the kinetic
    parameters are such that only the leading head can detach then reattach to the
    original actin binding site.
    """
    # Setup
    kd1 = 0.0
    kd2 = 10.0
    kh = 700.0
    kfpp = 1000.0
    kfpm = 0.0
    kps = np.inf
    b = 0.065

    # Excercise
    got = np.array(
        [emc.draw_event(kd1, kd2, kh, kfpp, kfpm, kps, b) for _ in range(5000)]
    )
    got_events = np.unique(got)

    # Validate
    assert all([event in [3] for event in got_events])


@pytest.mark.parametrize(
    "event, exp",
    [(0, 2.33), (1, 0.833), (2, 2.33), (3, 0.833), (4, 1.00), (5, 1.33), (6, 0.833)],
)
def test_draw_event_time(event, exp):
    """
    Test that the draw_event_time function returns the expected average event
    time for each kind of event.
    """
    # Setup
    kd1 = 1.0
    kd2 = 1.0
    kh = 1.0
    kfpp = 1.0
    kfpm = 1.0
    kps = 1.0
    b = 1.0

    # Excercise
    got = np.average(
        [
            emc.draw_event_time(event, kd1, kd2, kh, kfpp, kfpm, kps, b)
            for _ in range(200000)
        ]
    )

    # Validate
    nptest.assert_allclose(exp, got, atol=0, rtol=0.01)


@pytest.mark.parametrize(
    "event, exp", [(0, 1), (1, -1), (2, 0), (3, 0), (4, -2), (5, -2), (6, -2)]
)
def test_event_to_disp(event, exp):
    """
    Test that the event_to_disp returns the correct change in displacement for
    each event type.
    """
    # Setup
    currdisp = 2

    # Excercise
    got = emc.event_to_disp(event, currdisp)

    # Validate
    nptest.assert_allclose(exp, got)


def test_init_kfp_myoV_constforce():
    """
    Test that the init_kfp function returns the correct values of kfpp and kfpm
    for a constant zero force.
    """
    # Setup
    const_force = 0.0
    kfpp = -1.0
    kfpm = -1.0

    Delta = 36.0
    beta = 0.24552442427253968
    L = 35.0
    l_p = 310.0
    nu_c = 184.0
    theta_c = np.pi / 3
    theta_F = 0.0
    a = 1.0
    D_h = 57000000.0

    # Excercise
    got_kfpp, got_kfpm = emc.init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )

    # Validate
    exp_kfpp = 2999.1290763466745
    exp_kfpm = 0.019268947349062707
    nptest.assert_allclose(exp_kfpp, got_kfpp)
    nptest.assert_allclose(exp_kfpm, got_kfpm)


def test_init_kfp_myoV_constkfp():
    """
    Test that the init_kfp function returns the correct values of kfpp and kfpm
    for given values of kfpp and kfpm.
    """
    # Setup
    const_force = -1.0
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707

    Delta = 36.0
    beta = 0.24552442427253968
    L = 35.0
    l_p = 310.0
    nu_c = 184.0
    theta_c = np.pi / 3
    theta_F = 0.0
    a = 1.0
    D_h = 57000000.0

    # Excercise
    got_kfpp, got_kfpm = emc.init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )

    # Validate
    nptest.assert_allclose(kfpp, got_kfpp)
    nptest.assert_allclose(kfpm, got_kfpm)


def test_init_kfp_myoV_varforce():
    """
    Test that the init_kfp function returns None for both kfpp and kfpm for
    variable force inputs.
    """
    # Setup
    const_force = -1.0
    kfpp = -1.0
    kfpm = -1.0

    Delta = 36.0
    beta = 0.24552442427253968
    L = 35.0
    l_p = 310.0
    nu_c = 184.0
    theta_c = np.pi / 3
    theta_F = 0.0
    a = 1.0
    D_h = 57000000.0

    # Excercise
    got_kfpp, got_kfpm = emc.init_kfp(
        const_force, kfpp, kfpm, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h
    )

    # Validate
    assert got_kfpp is None
    assert got_kfpm is None


@pytest.mark.parametrize(
    "const_force, kfpp, kfpm",
    [
        (0.0, -1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (-1, 2999.1290763466745, 0.019268947349062707),
    ],
)
def test_emc_myo5_run(const_force, kfpp, kfpm):
    """
    Test that the 'run' option of the ecm driver function returns valid results.
    Tests that the returned array is of the correct shape, that the times are
    all greater than zero, are monotonically increasing, and that the motor
    displacements are in accordance with the corresponding event.
    """
    # Setup
    num_events = 5000

    L = 35.0
    l_p = 310.0
    D_h = 5.7e7
    theta_c = np.pi / 3
    theta_F = 0.0
    nu_c = 184.0
    Delta = 36.0
    a = 1.0
    b = 0.065
    kh = 750.0
    kd1 = 12.0
    kd2 = 1.5
    kps = 1e9
    beta = 0.24552442427253968
    k = 4.2
    params = [
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
        k,
    ]
    # Excercise
    got = emc.emc(
        params, num_events, "run", const_force, kfpp, kfpm, time_limit=None, processes=1
    )
    got_events = got[0]
    got_uniqeevents = np.unique(got_events)
    got_disps = got[1]
    got_difdisps = np.diff(got_disps)
    got_times = got[2]
    got_diftimes = np.diff(got_times)

    # Validate
    assert got.shape == (3, num_events)
    assert np.all(got_times >= 0.0)
    assert np.all(got_diftimes > 0.0)
    assert all([event in [0, 1, 2, 3, 4, 5, 6] for event in got_uniqeevents])
    assert all(got_difdisps[got_events[1:] == 0] == 1)
    assert all(got_difdisps[got_events[1:] == 1] == -1)
    assert all(got_difdisps[got_events[1:] == 2] == 0)
    assert all(got_difdisps[got_events[1:] == 3] == 0)
    assert all(got_difdisps[got_events[1:] == 4] <= 0)
    assert all(got_difdisps[got_events[1:] == 5] <= 0)
    assert all(got_difdisps[got_events[1:] == 6] <= 0)


@pytest.mark.parametrize(
    "const_force, kfpp, kfpm, processes",
    [
        (0, -1, -1, 1),
        (0, -1, -1, 2),
        (-1, -1, -1, 1),
        (-1, -1, -1, 2),
        (-1, 2999.1290763466745, 0.019268947349062707, 1),
        (-1, 2999.1290763466745, 0.019268947349062707, 2),
    ],
)
def test_emc_myo5_dwell_zeroconstforce(const_force, kfpp, kfpm, processes):
    """
    Test that the 'dwell' option of the ecm driver function returns valid results.
    Tests that the returned array is of the correct shape, that the times are
    all greater than zero and that the motor and that the average dwell time is
    approximately equal to that expected for a myosin-V motor under a zero-force load
    according to Hinczewski2013. This is done three different ways: by setting the
    force to zero, by passing kfpp and kfpm argument equal to those calculated
    with zero force, and the last by letting the simulation calculate update
    kfp arguments each pass based on the current force, but with a zero force
    constant. Tests both single and multithreaded versions.
    """
    # Setup
    trials = 30000
    time_limit = 50.0

    L = 35.0
    l_p = 310.0
    D_h = 5.7e7
    theta_c = np.pi / 3
    theta_F = 0.0
    nu_c = 184.0
    Delta = 36.0
    a = 1.0
    b = 0.065
    kh = 750.0
    kd1 = 12.0
    kd2 = 1.5
    kps = 1e9
    beta = 0.24552442427253968
    k = 0.0
    params = [
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
        k,
    ]
    # Excercise
    got = emc.emc(
        params,
        trials,
        "dwell",
        const_force,
        kfpp,
        kfpm,
        time_limit=time_limit,
        processes=processes,
    )
    got_dwell = got[0]
    got_avgdwelll = np.average(got_dwell)

    # Validate
    exp_avgdwell = 0.0849
    assert got.shape == (2, trials)
    assert all(got_dwell > 0)
    nptest.assert_allclose(exp_avgdwell, got_avgdwelll, rtol=0.05, atol=0)


@pytest.mark.parametrize(
    "const_force, kfpp, kfpm, processes",
    [
        (0, -1, -1, 1),
        (0, -1, -1, 2),
        (-1, -1, -1, 1),
        (-1, -1, -1, 2),
        (-1, 2999.1290763466745, 0.019268947349062707, 1),
        (-1, 2999.1290763466745, 0.019268947349062707, 2),
    ],
)
def test_emc_myo5_runstats_zeroconstforce(const_force, kfpp, kfpm, processes):
    """
    Test that the 'runstats' option of the ecm driver function returns valid results
    when operating under zero force. This is done three different ways: by setting the
    force to zero, by passing kfpp and kfpm argument equal to those calculated
    with zero force, and the last by letting the simulation calculate update
    kfp arguments each pass based on the current force, but with a zero force
    constant. Tests both single and multithreaded versions. Tests that the
    returned array is of the correct shape, that the times are all greater
    than zero, and that the averages are approximately equal to that predicted by
    Hinczewski2013. Tests both single and multiprocesses versions.
    """
    # Setup
    trials = 30000
    time_limit = 50.0

    L = 35.0
    l_p = 310.0
    D_h = 5.7e7
    theta_c = np.pi / 3
    theta_F = 0.0
    nu_c = 184.0
    Delta = 36.0
    a = 1.0
    b = 0.065
    kh = 750.0
    kd1 = 12.0
    kd2 = 1.5
    kps = 1e9
    beta = 0.24552442427253968
    k = 0.0
    params = [
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
        k,
    ]
    # Excercise
    got = emc.emc(
        params,
        trials,
        "runstats",
        const_force,
        kfpp,
        kfpm,
        time_limit=time_limit,
        processes=processes,
    )
    got_disp = got[0]
    got_avgdisp = np.average(got_disp)
    got_time = got[1]
    got_avgtime = np.average(got_time)
    # Validate
    exp_avgtime = 3.1791  # s
    exp_avgdisp = 36.418  # Steps
    assert got.shape == (2, trials)
    assert all(got_time > 0)
    nptest.assert_allclose(exp_avgtime, got_avgtime, rtol=0.05, atol=0)
    nptest.assert_allclose(exp_avgdisp, got_avgdisp, rtol=0.05, atol=0)


def test_emc_myo5_runstats_compforceconst():
    """
    Tests that the average run distance decreases when the myoV motor works against a
    variable load with increased force constant.
    """
    # Setup
    trials = 30000
    time_limit = 50.0
    const_force = -1.0
    kfpp = -1.0
    kfpm = -1.0
    processes = 2

    L = 35.0
    l_p = 310.0
    D_h = 5.7e7
    theta_c = np.pi / 3
    theta_F = 0.0
    nu_c = 184.0
    Delta = 36.0
    a = 1.0
    b = 0.065
    kh = 750.0
    kd1 = 12.0
    kd2 = 1.5
    kps = 1e9
    beta = 0.24552442427253968
    params = [kd1, kd2, kh, kps, b, Delta, L, l_p, beta, nu_c, theta_c, theta_F, a, D_h]
    soft_params = params.copy() + [0.0]
    strong_params = params.copy() + [4.2]
    # Excercise
    got_soft = emc.emc(
        soft_params,
        trials,
        "runstats",
        const_force,
        kfpp,
        kfpm,
        time_limit=time_limit,
        processes=processes,
    )
    got_strong = emc.emc(
        strong_params,
        trials,
        "runstats",
        const_force,
        kfpp,
        kfpm,
        time_limit=time_limit,
        processes=processes,
    )

    got_soft_avgdisp = np.average(got_soft[0])
    got_strong_avgdips = np.average(got_strong[0])
    # Validate
    assert got_soft_avgdisp > got_strong_avgdips
