#!/usr/bin/env python3
"""
File: test_motor_calculation.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains test of the funcion in the myo5space/motor_calculations.py module.
"""

import pytest
import myo5space.motor_calculations as mc
import numpy as np
import numpy.random as nprand
import numpy.testing as nptest


@pytest.mark.parametrize(
    "n, d, k, exp", [(1, 2, 3, 6e-3), (1.0, 2, 3, 6e-3), (1.0, 2, 3, 6e-3)]
)
def test_force(n, d, k, exp):
    """
    Test the force function. Mainly looking for that it always returns a float
    and calculates the expected value.
    """
    # Setup-none
    # Exercise
    got = mc.force(n, d, k)
    # Verify
    assert type(got) is float
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize("argtype", [int, float])
def test_kfp_myo5(argtype):
    """
    Test that the kfp function returns a tuple of two floats, and that they
    are the expected values for kfpp and kfpm for myosin-V at zero force and
    at a temperature of 295K as calculated in Hinczewski2013.
    Also tests when some of the inputs are ints or floats.
    """
    # Setup
    f = argtype(0.0)
    theta_f = argtype(0.0)
    Delta = argtype(36.0)
    beta = 0.24552442427253968
    L = argtype(35.0)
    l_p = argtype(310.0)
    nu_c = argtype(184.0)
    theta_c = np.pi / 3
    a = argtype(1.0)
    D_h = 5.7e7
    # Excercise
    got = mc.kfp(f, theta_f, Delta, beta, L, l_p, nu_c, theta_c, a, D_h)
    # Verify
    assert type(got) is tuple
    assert len(got) == 2
    assert all([type(k) is float for k in got])
    exp_kfpp = 2999.1290763466745
    exp_kfpm = 0.019268947349062707
    nptest.assert_allclose(exp_kfpp, got[0])
    nptest.assert_allclose(exp_kfpm, got[1])


@pytest.mark.parametrize("argtype", [int, float])
def test_mu_z_myo5(argtype):
    """
    Test that the mu_z function returns a float, whether arguments are float or
    ints, and that it is the expected value for myosin-V as calculated in
    Hinczewski2013.
    """
    # Setup
    L = argtype(35.0)
    l_p = argtype(310.0)
    nu_c = argtype(184.0)
    theta_c = np.pi / 3
    # Exercise
    got = mc.mu_z(l_p, L, nu_c, theta_c)
    # Verify
    assert isinstance(got, float)
    exp = 16.45831363376521
    nptest.assert_allclose(exp, got)


def test_pF_myo5():
    """
    Test the pF function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pF(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.8714036479717147
    nptest.assert_allclose(exp, got)


def test_pF_leq_one():
    """
    Test that the pF function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pF(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pB_myo5():
    """
    Test the pB function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pB(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 1.0344833030136622e-5
    nptest.assert_allclose(exp, got)


def test_pB_leq_one():
    """
    Test that the pB function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pB(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pTs_myo5():
    """
    Test the pTs function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pTs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 3.639113182634726e-7
    nptest.assert_allclose(exp, got)


def test_pTs_leq_one():
    """
    Test that the pTs function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pTs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pLs_myo5():
    """
    Test the pLs function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pLs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.10465838029233701
    nptest.assert_allclose(exp, got)


def test_pLs_leq_one():
    """
    Test that the pLs function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pLs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pT1_myo5():
    """
    Test the pT1 function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.013998250218722658
    nptest.assert_allclose(exp, got)


def test_pT1_leq_one():
    """
    Test that the pT1 function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pT2_myo5():
    """
    Test the pT2 function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.003486626787133269
    nptest.assert_allclose(exp, got)


def test_pT2_leq_one():
    """
    Test that the pT2 function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pT3_myo5():
    """
    Test the pT3 function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pT3(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.0064423859857439415
    nptest.assert_allclose(exp, got)


def test_pT3_leq_one():
    """
    Test that the pT3 function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pT3(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_pT_myo5():
    """
    Test the pT function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.pT(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.02392726299159987
    nptest.assert_allclose(exp, got)


def test_pT_leq_one():
    """
    Test that the pT function always returns a value less than one given normal
    inputs.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)
    # Excercise
    got = mc.pT(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    assert got.size == numsamples
    assert np.all(got < 1.0)
    assert np.all(got > 0.0)


def test_psum_eq_one():
    """
    Test that all the event probabilities add up to one.
    """
    # Setup
    nprand.seed(33)
    numsamples = 10000
    kd1 = nprand.uniform(0, 1e6, numsamples)
    kd2 = nprand.uniform(0, 1e6, numsamples)
    kh = nprand.uniform(0, 1e6, numsamples)
    kfpp = nprand.uniform(0, 1e6, numsamples)
    kfpm = nprand.uniform(0, 1e6, numsamples)
    kps = nprand.uniform(0, 1e9, numsamples)
    b = nprand.uniform(0, 1, numsamples)

    pf = mc.pF(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pb = mc.pB(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pts = mc.pTs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pls = mc.pLs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pt1 = mc.pT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pt2 = mc.pT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
    pt3 = mc.pT3(kd1, kd2, kh, kfpp, kfpm, kps, b)

    # Excercise
    got = pf + pb + pts + pls + pt1 + pt2 + pt3

    # Verify
    exp = np.ones(numsamples)
    nptest.assert_allclose(exp, got)


def test_tF_myo5():
    """
    Test the tF function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tF(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07571851223355378
    nptest.assert_allclose(exp, got)


def test_tB_myo5():
    """
    Test the tB function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tB(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


def test_tTs_myo5():
    """
    Test the tTs function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tTs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07571851223355378
    nptest.assert_allclose(exp, got)


def test_tLs_myo5():
    """
    Test the tLs function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tLs(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


def test_tT1_myo5():
    """
    Test the tT1 function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07538641003207933
    nptest.assert_allclose(exp, got)


def test_tT2_myo5():
    """
    Test the tT2 function with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07571851123355378
    nptest.assert_allclose(exp, got)


def test_tT3_myo5():
    """
    Test the tT3 with the parameters of myosin-V as defined in
    Hinczewski2013 at zero force.
    """
    # Setup
    kd1 = 12
    kd2 = 1.5
    kh = 750
    kfpp = 2999.1290763466745
    kfpm = 0.019268947349062707
    kps = 1e9
    b = 0.065
    # Excercise
    got = mc.tT3(kd1, kd2, kh, kfpp, kfpm, kps, b)
    # Validate
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize("argtype", [int, float])
def test_fstall_myo5(argtype):
    """
    Test that the f_stall function returns the correct value with parameters
    for myosin-V as defined in Hinkzewski2013 with both int and float inputs.
    """
    # Setup
    Delta = argtype(36.0)
    beta = 0.24552442427253968
    L = argtype(35.0)
    l_p = argtype(310.0)
    nu_c = argtype(184.0)
    theta_c = np.pi / 3
    theta_F = 0.0
    b = 0.065
    g = argtype(8.0)
    # Excercise
    got = mc.f_stall(beta, theta_F, L, l_p, nu_c, theta_c, Delta, g, b)
    # Verify
    exp = 1.8820604251027886
    nptest.assert_allclose(exp, got)


def test_trun_transmat():
    """
    Test that the trum_trunsmat function returns the correct 2D matrix of
    shape (7,7).
    """
    # Setup
    kd1 = 1.0
    kd2 = 2.0
    kh = 3.0
    kps = 4.0
    kfpp = 5.0
    kfpm = 6.0
    b = 0.5
    # Excercise
    got = mc.trun_transmat(kd1, kd2, kh, kps, kfpp, kfpm, b)
    # Verify
    exp = np.array(
        [
            [-3.0, 0.0, 0.0, 4.0, 4.0, 8.5, 0.0],
            [1.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, -9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, -4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, -9.5, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -0.0],
        ]
    )

    assert got.shape == (7, 7)
    nptest.assert_allclose(exp, got)


def test_tdwell_transmat():
    """
    Test that the tdwell_transmat function returns the correct 2D matrix of
    shape (7,7).
    """
    # Setup
    kd1 = 1.0
    kd2 = 2.0
    kh = 3.0
    kps = 4.0
    kfpp = 5.0
    kfpm = 6.0
    b = 0.5
    # Excercise
    got = mc.tdwell_transmat(kd1, kd2, kh, kps, kfpp, kfpm, b)
    # Verify
    exp = np.array(
        [
            [-3.0, 0.0, 0.0, 0.0, 4.0, 2.5, 0.0],
            [1.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, -9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, -4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, -9.5, 0.0],
            [0.0, 1.0, 1.0, 4.0, 0.0, 7.0, -0.0],
        ]
    )

    assert got.shape == (7, 7)
    nptest.assert_allclose(exp, got)
