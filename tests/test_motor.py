#!/usr/bin/env python3
"""
File: test_motor.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains tests for the classes and functions in the motor.py module.
"""

import pytest
import numpy as np
import numpy.testing as nptest
import scipy.integrate as integ
import myo5space.motor as motor


def test_motor_default_params():
    """
    Test that the default parameters for a Motor are those for myosin-V as described
    in Hinczewski2013.
    """
    m = motor.Motor()
    nptest.assert_allclose(35, m.L)
    nptest.assert_allclose(310, m.l_p)
    nptest.assert_allclose(5.7e7, m.D_h)
    nptest.assert_allclose(np.pi / 3, m.theta_c)
    nptest.assert_allclose(0, m.theta_F)
    nptest.assert_allclose(184, m.nu_c)
    nptest.assert_allclose(5e-6, m.t_r)
    nptest.assert_allclose(36, m.Delta)
    nptest.assert_allclose(1, m.a)
    nptest.assert_allclose(0.065, m.b)
    nptest.assert_allclose(750, m.kh)
    nptest.assert_allclose(12, m.kd1)
    nptest.assert_allclose(1.5, m.kd2)
    nptest.assert_allclose(np.inf, m.kps)


@pytest.mark.parametrize("val", [-10, "-10"])
def test_motor_bad_params(val):
    """
    Test that a ValueError is raised when the Motor constructor is passed
    a bad value.
    """
    with pytest.raises(ValueError):
        motor.Motor(L=val)


def test_myo5_g():
    """
    Test that the gating parameter is correct for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.g
    exp = 8.0
    nptest.assert_allclose(exp, got)


def test_myo5_kapp():
    """
    Test that the kappa parameter is correct for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.kappa
    exp = 0.11290322580645161
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize("temp, exp", [(295, 0.24552442427253968), (0, np.inf)])
def test_myo5_beta(temp, exp):
    """
    Test that the beta method calculates the correct value given a zero and non-zero
    temperature.
    """
    m = motor.Motor()
    got = m.beta(temp)
    nptest.assert_allclose(exp, got)


def test_myo5_force():
    """
    Test that the force method calculates the correct value for myosin-V parameters.
    """
    m = motor.Motor()
    n = 1
    k = 2
    got = m.force(n, k)
    exp = 0.072
    nptest.assert_allclose(exp, got)


def test_myo5_kfp():
    """
    Test that the kfp method calculates the correct values for myosin-V parameters.
    """
    m = motor.Motor()
    gotkfpp, gotkfpm = m.kfp()
    expkfpp, expkfpm = 2999.1290763466745, 0.019268947349062707
    nptest.assert_allclose(expkfpp, gotkfpp)
    nptest.assert_allclose(expkfpm, gotkfpm)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pF(kfpp, kfpm):
    """
    Test that the pF method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pF(kfpp=kfpp, kfpm=kfpm)
    exp = 0.8714036479717147
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pB(kfpp, kfpm):
    """
    Test that the pB method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pB(kfpp=kfpp, kfpm=kfpm)
    exp = 1.0344833030136622e-5
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pTs(kfpp, kfpm):
    """
    Test that the pTs method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pTs(kfpp=kfpp, kfpm=kfpm)
    exp = 3.639113182634726e-7
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pLs(kfpp, kfpm):
    """
    Test that the pLs method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pLs(kfpp=kfpp, kfpm=kfpm)
    exp = 0.10465838029233701
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pT1(kfpp, kfpm):
    """
    Test that the pT1 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pT1(kfpp=kfpp, kfpm=kfpm)
    exp = 0.013998250218722658
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pT2(kfpp, kfpm):
    """
    Test that the pT2 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pT2(kfpp=kfpp, kfpm=kfpm)
    exp = 0.003486626787133269
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pT3(kfpp, kfpm):
    """
    Test that the pT3 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pT3(kfpp=kfpp, kfpm=kfpm)
    exp = 0.0064423859857439415
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_pT(kfpp, kfpm):
    """
    Test that the pT method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.pT(kfpp=kfpp, kfpm=kfpm)
    exp = 0.02392726299159987
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tF(kfpp, kfpm):
    """
    Test that the tF method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tF(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07571851223355378
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tB(kfpp, kfpm):
    """
    Test that the tB method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tB(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tTs(kfpp, kfpm):
    """
    Test that the tTs method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tTs(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07571851223355378
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tLs(kfpp, kfpm):
    """
    Test that the tLs method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tLs(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tT1(kfpp, kfpm):
    """
    Test that the tT1 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tT1(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07538641003207933
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tT2(kfpp, kfpm):
    """
    Test that the tT2 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tT2(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07571851123355378
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tT3(kfpp, kfpm):
    """
    Test that the tT3 method returns the expected value for myosin-V under
    zero force. Test cases when kfpp and kfpm are either implicitly or explicitly
    defined.
    """
    m = motor.Motor()
    got = m.tT3(kfpp=kfpp, kfpm=kfpm)
    exp = 0.07890586356338203
    nptest.assert_allclose(exp, got)


def test_myo5_fstall():
    """
    Test that the f_stall method returns the expected value for myosin-V at
    295K.
    """
    m = motor.Motor()
    got = m.f_stall()
    exp = 1.8820604251027886
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_truntm(kfpp, kfpm):
    """
    Test that the trun_tm method reutrns the expected transition matrix for
    myosin-V under zero force. Test cases when kfpp and kfpm are either implicitly
    or explicitly defined.
    """
    m = motor.Motor()
    got = m.trun_tm(kfpp=kfpp, kfpm=kfpm)
    exp = np.array(
        [
            [
                -1.35000000e01,
                0.00000000e00,
                0.00000000e00,
                np.inf,
                np.inf,
                1.94962659e02,
                0.00000000e00,
            ],
            [
                1.20000000e01,
                -7.62000000e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                7.50000000e02,
                -3.01113033e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                2.99912908e03,
                -np.inf,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                1.25248158e-03,
                0.00000000e00,
                -np.inf,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                1.50000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -2.06962659e02,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                1.20000000e01,
                1.20000000e01,
                0.00000000e00,
                0.00000000e00,
                1.20000000e01,
                -0.00000000e00,
            ],
        ]
    )
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tdwelltm(kfpp, kfpm):
    """
    Test that the tdwell_tm method reutrns the expected transition matrix for
    myosin-V under zero force. Test cases when kfpp and kfpm are either implicitly
    or explicitly defined.
    """
    m = motor.Motor()
    got = m.tdwell_tm(kfpp=kfpp, kfpm=kfpm)
    exp = np.array(
        [
            [
                -1.35000000e01,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                np.inf,
                1.94943390e02,
                0.00000000e00,
            ],
            [
                1.20000000e01,
                -7.62000000e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                7.50000000e02,
                -3.01113033e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                2.99912908e03,
                -np.inf,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                1.25248158e-03,
                0.00000000e00,
                -np.inf,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                1.50000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -2.06962659e02,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                1.20000000e01,
                1.20000000e01,
                np.inf,
                0.00000000e00,
                1.20192689e01,
                -0.00000000e00,
            ],
        ]
    )
    nptest.assert_allclose(exp, got)


def test_myo5_trun_p0():
    """
    Test that the trun_p0 returns the expected initial probability vector.
    """
    m = motor.Motor()
    got = m.trun_p0()
    exp = np.array([1, 0, 0, 0, 0, 0])
    nptest.assert_allclose(exp, got)


def test_myo5_tdwell_p0():
    """
    Test that the tdwell_p0 returns the expected initial probability vector.
    """
    m = motor.Motor()
    got = m.tdwell_p0()
    exp = np.array([1, 0, 0, 0, 0, 0])
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_trunsubtm(kfpp, kfpm):
    """
    Test that the trun_subtm method reutrns the expected transition matrix for
    myosin-V under zero force. Test cases when kfpp and kfpm are either implicitly
    or explicitly defined.
    """
    m = motor.Motor()
    got = m.trun_subtm(kfpp=kfpp, kfpm=kfpm, infval=1e20)
    exp = np.array(
        [
            [
                -1.35000000e01,
                0.00000000e00,
                0.00000000e00,
                1e20,
                1e20,
                1.94962659e02,
                0.00000000e00,
            ],
            [
                1.20000000e01,
                -7.62000000e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                7.50000000e02,
                -3.01113033e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                2.99912908e03,
                -1e20,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                1.25248158e-03,
                0.00000000e00,
                -1e20,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                1.50000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -2.06962659e02,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                1.20000000e01,
                1.20000000e01,
                0.00000000e00,
                0.00000000e00,
                1.20000000e01,
                -0.00000000e00,
            ],
        ]
    )
    nptest.assert_allclose(exp[:-1, :-1], got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tdwellsubtm(kfpp, kfpm):
    """
    Test that the tdwell_subtm method reutrns the expected transition matrix for
    myosin-V under zero force. Test cases when kfpp and kfpm are either implicitly
    or explicitly defined.
    """
    m = motor.Motor()
    got = m.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=1e20)
    exp = np.array(
        [
            [
                -1.35000000e01,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                1e20,
                1.94943390e02,
                0.00000000e00,
            ],
            [
                1.20000000e01,
                -7.62000000e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                7.50000000e02,
                -3.01113033e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                2.99912908e03,
                -1e20,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                1.25248158e-03,
                0.00000000e00,
                -1e20,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                1.50000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -2.06962659e02,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                1.20000000e01,
                1.20000000e01,
                1e20,
                0.00000000e00,
                1.20192689e01,
                -0.00000000e00,
            ],
        ]
    )
    nptest.assert_allclose(exp[:-1, :-1], got)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000.0, 1000.0)],
)
def test_zrunpmf_sumtoone(kfpp, kfpm):
    """
    Test that the zrun_pmf method sums to one.
    """
    m = motor.Motor()
    z = np.arange(-10, 3000)
    got = m.zrun_pmf(z, kfpp=kfpp, kfpm=kfpm)
    nptest.assert_allclose(1.0, got.sum())


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_zrunmean(kfpp, kfpm):
    """
    Test that the zrun_mean method returns the expected value for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.zrun_mean(kfpp=kfpp, kfpm=kfpm)
    exp = 36.418427943246336
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000, 1000)],
)
def test_zrunmean_geom(kfpp, kfpm):
    """
    Test that the zrun_mean method returns the mean for a geometric distribution
    the special case when the motor cannot take backward steps.
    """
    m = motor.Motor(kd2=0)
    got = m.zrun_mean(kfpp=kfpp, kfpm=kfpm)
    pT = m.pT(kfpp=kfpp, kfpm=kfpm) / (
        1 - m.pTs(kfpp=kfpp, kfpm=kfpm) - m.pLs(kfpp=kfpp, kfpm=kfpm)
    )
    exp = 1 / pT - 1
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000, 1000)],
)
def test_trunpdf_sumtoone(kfpp, kfpm):
    """
    Test that the trun_pdf method inegrates to one.
    """
    m = motor.Motor()
    got = integ.quad(
        lambda x: m.trun_pdf(x, kfpp=kfpp, kfpm=kfpm),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    nptest.assert_allclose(1.0, got)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_trunmean(kfpp, kfpm):
    """
    Test that the trun_mean method returns the expected value for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.trun_mean(kfpp=kfpp, kfpm=kfpm)
    exp = 3.17913
    nptest.assert_allclose(exp, got, atol=1e-4)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_trunmean2(kfpp, kfpm):
    """
    Test that the trun_mean2 method returns the expected value for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.trun_mean2(kfpp=kfpp, kfpm=kfpm)
    exp = 3.17913
    nptest.assert_allclose(exp, got, atol=1e-4)


@pytest.mark.parametrize("kd1", [12, 1200])
def test_trunmean_agreement(kd1):
    """
    Test that the two methods for calculating mean run times agree with eachother
    for slow and fast motors.
    """
    m = motor.Motor(kd1=kd1, kps=1e12)
    gotmean1 = m.trun_mean()
    gotmean2 = m.trun_mean2()

    nptest.assert_allclose(gotmean1, gotmean2, rtol=1e-5)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000, 1000)],
)
def test_tdwellpdf_sumtoone(kfpp, kfpm):
    """
    Test that the tdwell_pdf method inegrates to one.
    """
    m = motor.Motor()
    got = integ.quad(
        lambda x: m.tdwell_pdf(x, kfpp=kfpp, kfpm=kfpm),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    nptest.assert_allclose(1.0, got, rtol=1e-6)


@pytest.mark.parametrize(
    "kfpp, kfpm", [(None, None), (2999.1290763466745, 0.019268947349062707)]
)
def test_myo5_tdwellmean(kfpp, kfpm):
    """
    Test that the tdwell_mean method returns the expected value for myosin-V parameters.
    """
    m = motor.Motor()
    got = m.tdwell_mean(kfpp=kfpp, kfpm=kfpm)
    exp = 0.0849597877301841
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000, 1000)],
)
def test_vrunpdf_sumstoone(kfpp, kfpm):
    """
    Test that the vrun_pdf method roughly integrates to one. "Roughly" becuase
    these are very, very slow tests.
    """
    m = motor.Motor()
    got = integ.quad(
        lambda x: m.vrun_pdf([x], kfpp=kfpp, kfpm=kfpm),
        1e-20,
        1e3,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    got += integ.quad(
        lambda x: m.vrun_pdf([x], kfpp=kfpp, kfpm=kfpm),
        -10.0,
        -1e-20,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    got += m.vrun_pdf([0], kfpp=kfpp, kfpm=kfpm)
    nptest.assert_allclose(1.0, got, rtol=1e-3)


@pytest.mark.parametrize(
    "kfpp, kfpm",
    [(None, None), (2999.1290763466745, 0.019268947349062707), (2000, 1000)],
)
def test_vrunpdf2_sumstoone(kfpp, kfpm):
    """
    Test that the vrun_pdf method roughly integrates to one. "Roughly" becuase
    these are very, very slow tests.
    """
    m = motor.Motor(kd2=0, kd1=12000)
    got = integ.quad(
        lambda x: m.vrun_pdf2(x, kfpp=kfpp, kfpm=kfpm, zmax=1000),
        0.0,
        1e6,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    gotzero = m.zrun_pmf(0)
    nptest.assert_allclose(1.0, got + gotzero, rtol=1e-2)


def test_myo5_vrun_mean():
    """
    Test that the vrun_mean method returns the expected result for myo5 params.
    """
    m = motor.Motor()
    got = m.vrun_mean()
    exp = 11.413153239411669
    nptest.assert_allclose(exp, got)


def test_myo5_vrun_mean_force_comparison():
    """
    Test that the run velocity decreases uniformly as the force increases.
    Uses the defualt myo5 params.
    """
    m = motor.Motor()
    forces = np.linspace(0, 2, 50)
    got_vruns = [m.vrun_mean(f=f) for f in forces]
    got_diffs = np.diff(got_vruns)
    assert np.all(got_diffs < 0)


def test_vrun_mean2():
    """
    Teset that the vrun_mean2 method returns the same value as the vrun_mean
    method for motors with kd2=0 and a variety of kd1 values.
    """
    kd1s = np.logspace(1, 4, 10)
    ms = [motor.Motor(kd1=kd1, kd2=0.0) for kd1 in kd1s]
    got_means = [m.vrun_mean() for m in ms]
    got_means2 = [m.vrun_mean2() for m in ms]
    nptest.assert_allclose(got_means, got_means2, rtol=1e-4)


def test_vrun_mean3():
    """
    Test that the vrun_mean3 method returns approximatly the same value as the
    vrun_mean2 and vrun_mean3 methods for parameters that give a shape value of
    approximatly 2.0. The tolerances are loose since the shape value differs from
    2.0 by about 3%.
    """
    m = motor.Motor(kd1=1000.0, kd2=0.0)
    got_mean = m.vrun_mean()
    got_mean2 = m.vrun_mean2()
    got_mean3 = m.vrun_mean3()

    # Loose rtol becuase shape param is not exactly 2.
    nptest.assert_allclose(got_mean, got_mean3, rtol=0.04)
    nptest.assert_allclose(got_mean2, got_mean3, rtol=0.04)


@pytest.mark.parametrize("method", ["kmc", "emc"])
def test_simulate_runstats(method):
    """
    Test that the 'runstats' option for the Motor.simulate method returns the
    expected RunStatsRes object with expected values, array shapes, and mean
    values for myo5.
    """
    m = motor.Motor()
    trials = 50000
    got = m.simulate(
        method, "runstats", trials, const_force=0.0, time_limit=1000, processes=3
    )
    assert got.motor is m
    nptest.assert_allclose(got.f, 0.0)
    nptest.assert_allclose(got.kfpp, 2999.1290763466745)
    nptest.assert_allclose(got.kfpm, 0.019268947349062707)
    assert got.zruns.shape == (trials,)
    assert got.truns.shape == (trials,)
    assert got.vruns.shape == (trials,)
    nptest.assert_allclose(got.zrun_mean(), 36.418427943246336, rtol=0.02)
    nptest.assert_allclose(got.trun_mean(), 3.1791396394070235, rtol=0.02)
    nptest.assert_allclose(got.vrun_mean(), 11.413153239411669, rtol=0.02)


@pytest.mark.parametrize("method", ["kmc", "emc"])
def test_simulate_dwell(method):
    """
    Test that the 'dwell' option for the Motor.simulate method returns the
    expected DwellRes object with expected values, array shapes, and mean
    values for myo5.
    """
    m = motor.Motor()
    trials = 20000
    got = m.simulate(
        method, "dwell", trials, const_force=0.0, time_limit=1000, processes=3
    )
    assert got.motor is m
    nptest.assert_allclose(got.f, 0.0)
    nptest.assert_allclose(got.kfpp, 2999.1290763466745)
    nptest.assert_allclose(got.kfpm, 0.019268947349062707)
    assert got.tdwells.shape == (trials,)
    assert got.zlocs.shape == (trials,)
    nptest.assert_allclose(got.tdwell_mean(), 0.08495978773018413, rtol=0.02)


def test_simulate_trajectory():
    """
    Test that the 'trajectory' option for the Motor.simulate method returns the
    expected TrajRes object with expected values and array shapes.
    """
    m = motor.Motor()
    trials = 1000
    got = m.simulate("kmc", "trajectory", trials, const_force=0.0)
    assert got.motor is m
    nptest.assert_allclose(got.f, 0.0)
    nptest.assert_allclose(got.kfpp, 2999.1290763466745)
    nptest.assert_allclose(got.kfpm, 0.019268947349062707)
    assert got.states.shape == (trials,)
    assert got.disps.shape == (trials,)
    assert got.times.shape == (trials,)


def test_simulate_run():
    """
    Test that the 'run' option for the Motor.simulate method returns the
    expected RunRes object with expected values and array shapes.
    """
    m = motor.Motor()
    trials = 1000
    got = m.simulate("emc", "run", trials, const_force=0.0)
    assert got.motor is m
    nptest.assert_allclose(got.f, 0.0)
    nptest.assert_allclose(got.kfpp, 2999.1290763466745)
    nptest.assert_allclose(got.kfpm, 0.019268947349062707)
    assert got.events.shape == (trials,)
    assert got.disps.shape == (trials,)
    assert got.times.shape == (trials,)


@pytest.mark.parametrize(
    "dataset, method",
    [
        ("detach 1um", "kmc"),
        ("detach 100um", "kmc"),
        ("detach 1um", "emc"),
        ("detach 100um", "emc"),
    ],
)
def test_fitting_detach(dataset, method):
    """
    Test that the fitting method returns valid results for detachment datasets.
    """
    m = motor.Motor()
    trials = 10000
    force_cutoff = 0.0
    density_cutoff = 0.0
    got = m.loglh(
        dataset,
        trials,
        method=method,
        force_cutoff=force_cutoff,
        density_cutoff=density_cutoff,
    )
    assert isinstance(got, float)
    assert got < 0


@pytest.mark.parametrize(
    "dataset, method",
    [
        ("dwell 1um", "kmc"),
        ("dwell 100um", "kmc"),
        ("dwell 1um", "emc"),
        ("dwell 100um", "emc"),
    ],
)
def test_fitting_dwell(dataset, method):
    """
    Test that the fitting method returns valid results fro the dwelltime datsets.
    """
    m = motor.Motor()
    trials = 10000
    force_cutoff = 0
    time_cutoff = 0
    density_cutoff = 0.01
    got = m.loglh(
        dataset,
        trials,
        method=method,
        force_cutoff=force_cutoff,
        time_cutoff=time_cutoff,
        density_cutoff=density_cutoff
    )
    assert isinstance(got, float)
    assert got < 0
