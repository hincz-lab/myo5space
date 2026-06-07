#!/usr/bin/env python3
"""
File : motor_calculations.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains calculations that are performed on a set of motor parameters.
These are contained in a separate module so that high performance functions
using numba can call them while being decoupled from a container class.
"""

# Imports
import numpy as np
from numba import njit
from . import mymath


@njit
def force(n: float, Delta: float, k: float) -> float:
    """
    Calculates the force experienced by a myosin motor working against a
    compliant optical trap with a linear force constant, k. It's Hooks law.

    :param n: The number of steps the motor is away from the spring's
        equilibrium point.
    :param Delta: Distance per step (nm).
    :param k: Spring constant (pN/nm)

    :return f: The force exerted by the trap (pN).
    """
    # The 1e-3 factor converts Delta from nm to pm.
    return Delta * n * k * 1e-3


@njit
def kfp(
    f: float,
    theta_F: float,
    Delta: float,
    beta: float,
    L: float,
    l_p: float,
    nu_c: float,
    theta_c: float,
    a: float,
    D_h: float,
) -> float:
    """
    Calculate the force dependent head diffusion rates.

    :param f: Magnitude of force that the motor works against (pN).
    :param theta_F: Angle of applied force (rads).
    :param Delta: Length of a motor step (nm).
    :param beta: 1 / (kB*T) (pN nm).
    :param L: Length of the motor leg (nm).
    :param l_p: Persistence length of motor leg (nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).
    :param a: Debye radius (nm).
    :param D_h: Motor head diffusivity (nm^2/s)

    :return kfpp, kfpm: The diffusion rates to the forward and reverse
        binding positions.
    """

    # Precalculations.
    kappa = L / l_p
    T = 1.0 + 20.0 * nu_c / (20.0 + 7.0 * kappa * nu_c)
    Tx = T * np.sin(theta_c) + beta * f * L * np.sin(theta_F)
    Tz = T * np.cos(theta_c) - beta * f * L * np.cos(theta_F)
    Tprime = np.sqrt(Tx**2 + Tz**2)

    # End probability calculations.
    fact1 = (
        (3.0 * kappa * (7.0 * kappa + 20.0) + 200.0)
        * Tprime
        / (1600.0 * np.pi * L**2 * Delta * np.sinh(Tprime))
    )
    fact2 = mymath.i0(Tx * np.sqrt(1.0 - (Delta / (2.0 * L)) ** 2))
    exparg = Tz * Delta / (2.0 * L)
    fact3_p = np.exp(exparg)
    fact3_m = np.exp(-1.0 * exparg)
    p_plus = fact1 * fact2 * fact3_p
    p_minus = fact1 * fact2 * fact3_m

    # Rate calculations.
    kfpp = 4 * np.pi * a * D_h * p_plus
    kfpm = 4 * np.pi * a * D_h * p_minus

    return kfpp, kfpm


@njit
def mu_z(l_p: float, L: float, nu_c: float, theta_c: float) -> float:
    """
    Calculate the average z-axis of the free motor leg when diffusively
    searching for its next binding position.

    :param l_p: Persistence length of motor leg (nm).
    :param L: Length of the motor leg (nm).
    :param nu_c: Strength of binding constraint (unitless).
    :param theta_c: Angle of constraint (rads).

    :return mu_z: Motor's average z-axis.
    """
    return (
        l_p
        * (1.0 - np.exp(-L / l_p))
        * (np.tanh(nu_c) ** (-1) - 1.0 / nu_c)
        * np.cos(theta_c)
    )


@njit
def pF(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Calculate the probability that the motor takes a forwards step.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pF: Probability of taking a forward step.
    """
    return kd1 / (kd1 + kd2) * kh / (kd1 + kh) * kfpp / (kd1 + kfpp + b * kfpm)


@njit
def pB(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability  that the motor takes a backwards step.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pB: Probability of taking a forward step.

    """
    return kd2 / (kd1 + kd2) * kfpm / (kd1 + b * kfpp + kfpm)


@njit
def pTs(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability that the motor does a trailing stomp.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pTs: Probability of doing a trailing stomp.

    """
    return kd1 / (kd1 + kd2) * kh / (kd1 + kh) * b * kfpm / (kd1 + kfpp + b * kfpm)


@njit
def pLs(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probabiltiy that the motor does a leading stomp.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pLs: Probability of doing a leading stomp.
    """
    return kd2 / (kd1 + kd2) * b * kfpp / (kd1 + b * kfpp + kfpm)


@njit
def pT1(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability that the motor detaches before hydrolysis.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pT1: Probability of termination before hydrolysis.
    """
    return kd1 / (kd1 + kd2) * kd1 / (kd1 + kh)


@njit
def pT2(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability that the motor detaches after hydrolysis.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pT2: Probability of termination after hydrolysis.
    """
    return kd1 / (kd1 + kd2) * kh / (kd1 + kh) * kd1 / (kd1 + kfpp + b * kfpm)


@njit
def pT3(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability that the motor detaches before completing a backwards step.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pT3: Probability of termination beffore a backwards step.
    """
    return kd2 / (kd1 + kd2) * kd1 / (kd1 + b * kfpp + kfpm)


def pT(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Probability that the motor detaches.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return pT: Total probability of termination.
    """
    return (
        pT1(kd1, kd2, kh, kfpp, kfpm, kps, b)
        + pT2(kd1, kd2, kh, kfpp, kfpm, kps, b)
        + pT3(kd1, kd2, kh, kfpp, kfpm, kps, b)
    )


@njit
def tF(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Calculates the average time it takes to complete a forward step.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tF: Average time for a forward step.
    """
    t0 = 1 / (kd1 + kd2)
    t1 = 1 / (kd1 + kh)
    t2 = 1 / (kd1 + kfpp + b * kfpm)
    t3 = 1 / kps

    return t0 + t1 + t2 + t3


@njit
def tB(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Calculates the average time it takes to complete a backwards step.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tB: Average time for a backwards step.
    """
    t0 = 1 / (kd1 + kd2)
    t5 = 1 / (kd1 + b * kfpp + kfpm)

    return t0 + t5


@njit
def tTs(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Calculates the average time it takes to complete a trailing stomp.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tTs: Average time for a trailing stomp.
    """
    return tF(kd1, kd2, kh, kfpp, kfpm, kps, b)


@njit
def tLs(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
     Calculates the average time it takes to complete a leading stomp.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tLs: Average time for leading stomp.
    """
    return tB(kd1, kd2, kh, kfpp, kfpm, kps, b)


@njit
def tT1(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
     Calculates the average time it takes to terminate following trailing
     head detachment.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tT1: Average time to terminate following trailing head detachment.
    """
    t0 = 1 / (kd1 + kd2)
    t1 = 1 / (kd1 + kh)

    return t0 + t1


@njit
def tT2(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
     Calculates the average time it takes to terminate following hydrolysis.

    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tT2: Average time to terminate following hydrolysis.
    """
    t0 = 1 / (kd1 + kd2)
    t1 = 1 / (kd1 + kh)
    t2 = 1 / (kd1 + kfpp + b * kfpm)

    return t0 + t1 + t2


@njit
def tT3(
    kd1: float, kd2: float, kh: float, kfpp: float, kfpm: float, kps: float, b: float
) -> float:
    """
    Calculates the average time it takes to terminate following leading
    head detachment.


    :param kd1: Rate of trailing head detachment (1/s).
    :param kd2: Rate of leading head detachment (1/s).
    :param kh: Rate of ATP hydrolysis (1/s).
    :param kfpp: Average rate a free head binding to the forward position (1/s).
    :param kfpm: Average rate of free head binding to reverse position (1/s).
    :param kps: Rate of the power stroke (1/s).
    :param b: Binding penalty (unitless).

    :return tT3: Average time to terminate following leading head detachment.
    """
    t0 = 1 / (kd1 + kd2)
    t5 = 1 / (kd1 + b * kfpp + kfpm)

    return t0 + t5


@njit
def f_stall(
    beta: float,
    theta_F: float,
    L: float,
    l_p: float,
    nu_c: float,
    theta_c: float,
    Delta: float,
    g: float,
    b: float,
) -> float:
    """
    Note: All parameters can be used with numpy nd arrays
        without problems.

    :param beta: 1/kbT (pN nm)
    :param theta_F: Angle at which the force on the motor
        is being applied (rad).
    :param L: Contour length of motor arm (nm).
    :param l_p: Persistence length of motor arm (nm).
    :param nu_c: Angle constraint strength of attached head (unitless).
    :param theta_c: Angle constraint of attached head with actin (rad).
    :param Delta: Step length of motor (nm).
    :param g: Ratio of kd1/kd2 (unitless)
    :param b: Binding penalty (unitless).

    :return f_stall: The motor's stall force (pN).
    """
    # Intermediate calculations.
    kappa = L / l_p
    T = 1.0 + 20.0 * nu_c / (20.0 + 7.0 * kappa * nu_c)
    term1 = T / L * np.cos(theta_c)
    term2 = np.log((g - 1 + np.sqrt((g - 1) ** 2 + 4 * g * b**2)) / (2 * b)) / Delta

    return 1 / (np.cos(theta_F) * beta) * (term1 + term2)


@njit
def trun_transmat(
    kd1: float, kd2: float, kh: float, kps: float, kfpp: float, kfpm: float, b: float
) -> float:
    """
    Calculates the transition matrix for a motor's run where termination results
    in absorption.
    Used to calculate the distribution of motor run times in at a given position.
    Uses convention that the columns add to zero. The columns refer to the
    following states:
    0: Resting state.
    1: Trailing lead detached, pre hydrolysis.
    2: Trailing head detached, post hydrolysis.
    3: Trailing head attached to forward binding position.
    4: Trailing head attached to original binding position.
    5: Leading head detached.
    6: Termination (absorbing).

    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param kfpp: The rate for a diffusing head to find the forward binding position (1/s).
    :param kfpm: The rate for a diffusing head to find the backward binding position (1/s)
    :param b: The binding penalty.

    :return omega: (7,7) ndarray. The transition matrix.
    """
    matrix = np.zeros((7, 7))

    # Fill the matrix off diagonal elements with transition rate values such that matrix[i,j] is the rate from j -> i.
    matrix[1, 0] = kd1
    matrix[5, 0] = kd2
    matrix[2, 1] = kh
    matrix[6, 1] = kd1
    matrix[3, 2] = kfpp
    matrix[4, 2] = b * kfpm
    matrix[6, 2] = kd1
    matrix[0, 3] = kps
    matrix[0, 4] = kps
    matrix[0, 5] = b * kfpp + kfpm
    matrix[6, 5] = kd1

    # Fill the diagonal elements such that the columns add up to 0.
    for i in range(matrix.shape[1]):
        matrix[i, i] = -np.sum(matrix[:, i])

    return matrix


@njit
def tdwell_transmat(
    kd1: float, kd2: float, kh: float, kps: float, kfpp: float, kfpm: float, b: float
) -> float:
    """
    Calculates the transition matrix for a single motor event where the
    completion of a forwards step, backwards step
    leading stomp, trailing stomp, or termination results in absorption.
    Used to calculate the distribution of motor dwell times in at a given position.
    Uses convention that the columns add to zero. The columns refer to the
    following states:
    0: Resting state.
    1: Trailing lead detached, pre hydrolysis.
    2: Trailing head detached, post hydrolysis.
    3: Trailing head attached to forward binding position.
    4: Trailing head attached to original binding position.
    5: Leading head detached.
    6: Completion of forward step, backward step, termination (absorbing).

    :param kd1: Trailing head detachment rate (1/s).
    :param kd2: Leading head detachment rate (1/s)
    :param kh: ATP hydrolysis rate (1/s).
    :param kps: Power stroke rate (1/s).
    :param kfpp: The rate for a diffusing head to find the forward binding position (1/s).
    :param kfpm: The rate for a diffusing head to find the backward binding position (1/s)
    :param b: The binding penalty.

    :return omega: (7,7) ndarray. The transition matrix.
    """
    matrix = np.zeros((7, 7))

    # Fill the matrix off diagonal elements with transition rate values such that 
    # matrix[i,j] is the rate from j -> i.
    matrix[1, 0] = kd1
    matrix[5, 0] = kd2
    matrix[2, 1] = kh
    matrix[6, 1] = kd1
    matrix[3, 2] = kfpp
    matrix[4, 2] = b * kfpm
    matrix[6, 2] = kd1
    matrix[6, 3] = kps
    matrix[0, 4] = kps
    matrix[0, 5] = b * kfpp
    matrix[6, 5] = kfpm + kd1

    # Fill the diagonal elements such that the columns add up to 0.
    for i in range(matrix.shape[1]):
        matrix[i, i] = -np.sum(matrix[:, i])

    return matrix
