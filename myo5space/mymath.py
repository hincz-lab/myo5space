#!/usr/bin/env python3
"""
File: mymath.py
Author: Joshua Holmes
Email: jbh92@case.edu

This module contains miscelaneous math functions.
Functions include those related to special functions and probability distributions.
"""

# Imports
import ctypes
import math
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as linalg
import scipy.optimize as opt
import scipy.integrate as integ
from numba import vectorize, njit
from numba.extending import get_cython_function_address

# Zeroth order modified Bessel function of the first kind.
_i0_addr = get_cython_function_address("scipy.special.cython_special", "i0")
_i0_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_i0_fn = _i0_functype(_i0_addr)


@vectorize
def _vec_i0(x):
    return _i0_fn(x)


@njit
def i0(x: ArrayLike) -> ArrayLike:
    """
    Numba compatible version of the zeroth order modified Bessel function of
    the first kind.

    :param x: Argument of the function.
    :return: Calculated values.
    """
    return _vec_i0(x)

# Binomial coefficients.
_binom_addr = get_cython_function_address("scipy.special.cython_special", "binom")
_binom_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
_binom_fn = _binom_functype(_binom_addr)


@vectorize
def _vec_binom(x, y):
    """
    Vectorized version of binom function.
    """
    return _binom_fn(x, y)


# This is the function that you actually use.
@njit
def binom(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Numba compatible version of the binom coefficients.

    :param x: The top number in binomial calculation.
    :param y: The bottom nubmer in binomial calculation.
    :return: Binomial coefficient.
    """
    return _vec_binom(x, y)


def is_invertible(a: NDArray[np.floating | np.integer]) -> bool:
    """
    Check if a matrix is invertible based on the conditions that:
    1) It is square.
    2) The rank of the matrix is equal to its shape.

    :param a: The matrix.
    :return: True if the matrix can be inverted, False otherwise.
    """
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def expm_eig(a):
    """
    Computes the matrix exponential using eigenvalue decomposition.
    This function was borrowed from:
        https://github.com/scipy/scipy/blob/v0.12.0/scipy/linalg/matfuncs.py#L57

    :param a: (N,N) array_like. Matrix to be exponentiated.
    :except Singular eignensystem (ValueError): Throws a ValueError if the
        eigensystem of a is singular.
    :return expm: (N,N) ndarray. Matrix exponential of A.
    """
    a = np.asarray(a)
    t = a.dtype.char
    if t not in ["f", "F", "d", "D"]:
        a = a.astype("d")
        t = "d"
    s, vr = linalg.eig(a)
    if not is_invertible(vr):
        raise ValueError(
            "Eigensystem is singular. \
                    Consider using other matrix exponentiating method."
        )
    vri = linalg.inv(vr)
    r = vr @ np.diag(np.exp(s)) @ vri
    if t in ["f", "d"]:
        return r.real.astype(t)
    else:
        return r.astype(t)
    

def expm(a: ArrayLike) -> np.ndarray:
    """
    Computes the matrix exponential of a, preferring eigen-decomposition when possible.
    Falls back to scipy.linalg.expm when eigen-decomposition is not suitable.
    """
    a = np.asarray(a)
    try:
        return expm_eig(a)
    except (ValueError, linalg.LinAlgError):
        return linalg.expm(a)


@njit
def sample_discrete(probs: NDArray[np.floating]) -> int:
    """
    Numba compatible function to make a selection from choices with
    arbitrary probabilities.

    :param probs: (N,) np.ndarray with positive elements representing
        the weight of that choice. The array does not have to be normalized.
        Negative values are made positive.
    :return choice: The index of the choice that was made.
    """
    _probs = np.abs(probs)
    _probs /= _probs.sum()
    u = np.random.rand()
    R = _probs.cumsum()
    choice = np.searchsorted(R, u)
    return choice


@njit
def sample_exponential(rate: float) -> float:
    """
    Numba compatible function for sampling an exponetial distribution
    of a given rate.

    :param rate: The rate of the exponential distribution to sample from.
        Rate must be greater than 0.
    :return: An exponential sample.
    """
    u = np.random.rand()
    res = -np.log(u) / rate
    return res


@njit
def sample_hypoexponential(rates: NDArray[np.floating]) -> float:
    """
    Numba compatibe function for drawing samples from a hypoexponential
    distribution (a string of exponential dwell times with arbitrary rates).

    :param rates: A one dimensional ndarray of rates, all of which are 
        greater than 0.
    :return sample: A hypoexponential sample.
    """
    res = 0.0
    for r in rates:
        res += sample_exponential(r)
    return res


@njit
def lngamma_pdf(x: float, a: float) -> float:
    """
    Numba compatible calculation of the natural log of the probability density
    function (pdf) of the gamma distribution for x >= 0 (up to an additive
    constant -log(Gamma(a)), depending on parameterization).

    :param x: Argument of the pdf. Should satisfy x >= 0.
    :param a: Shape parameter (typically a > 0).
    :return: Calculated natural log of gamma density.
    """
    atol = 1e-12
    if np.abs(a - 1.0) < atol:
        return -x
    else:
        return (a - 1.0) * np.log(x) - x - math.lgamma(a)


@njit
def gamma_pdf(x, a, b):
    """
    Numba compatible calculation of the pro  bability density function (pdf) of
    the gamma distribution. For computational purposes, this function is only
    valid for x>=0.

    :param x: Argument of the pdf.
    :param a: Shape parameter.
    :param b: Scale parameter.
    :return: Calculated gamma density.
    """
    _x = x / b
    res = np.exp(lngamma_pdf(_x, a)) / b
    return res


def _ph_pdf(t: float, p0: np.ndarray, subtm: np.ndarray) -> float:
    """
    Backend for the ph_pdf function.

    :param t: Time argument (t >= 0).
    :param p0: (N,) ndarray. Initial probability vector.
    :param subtm: (N,N) ndarray. Sub-transition matrix.
    :return: Phase-type density at time t.
    """
    if t < 0:
        return 0.0
    ones = np.ones(subtm.shape[0])
    s0 = -1 * ones @ subtm
    return s0 @ expm(t * subtm) @ p0


# Vecotrized calc_ph_den function.
_vph_pdf = np.vectorize(_ph_pdf, otypes=[np.float64])
for i in range(1, 3):
    _vph_pdf.excluded.add(i)


def ph_pdf(t: ArrayLike, p0: ArrayLike, subtm: ArrayLike) -> ArrayLike:
    """
    Calculates the density of a phase-type distribution.

    :param t: Scalar or array-like of time points.
    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix parameter.
    :return: Density value(s). Returns float if t is scalar, else np.ndarray.
    """
    p0_arr = np.asarray(p0, dtype=float)
    subtm_arr = np.asarray(subtm, dtype=float)

    if np.isscalar(t):
        return _ph_pdf(float(t), p0_arr, subtm_arr)
    else:
        t_arr = np.asarray(t, dtype=float)
        return _vph_pdf(t_arr, p0_arr, subtm_arr)


def _ph_cdf(t: float, p0: np.ndarray, subtm: np.ndarray) -> float:
    """
    Backend for the ph_cdf function.

    :param t: Time argument (t >= 0).
    :param p0: (N,) ndarray. Initial probability vector.
    :param subtm: (N,N) ndarray. Sub-transition matrix.
    :return: Phase-type CDF at time t.
    """
    if t < 0.0:
        return 0.0
    ones = np.ones(subtm.shape[0], dtype=float)
    return float(1.0 - ones @ expm(t * subtm) @ p0)


# Vecotrized calc_ph_cdf function.
_vph_cdf = np.vectorize(_ph_cdf, otypes=[np.float64])
for i in range(1, 3):
    _vph_cdf.excluded.add(i)


def ph_cdf(t: ArrayLike, p0: ArrayLike, subtm: ArrayLike) -> ArrayLike:
    """
    Calculate the cumulative distribution function (cdf) of a phase-type distribution.

    :param t: Scalar or array-like of time points.
    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix.
    :return: CDF value(s). Returns float if t is scalar, else np.ndarray.
    """
    p0_arr = np.asarray(p0, dtype=float)
    subtm_arr = np.asarray(subtm, dtype=float)

    if np.isscalar(t):
        return _ph_cdf(float(t), p0_arr, subtm_arr)
    t_arr = np.asarray(t, dtype=float)
    return _vph_cdf(t_arr, p0_arr, subtm_arr)


def ph_mean(p0: ArrayLike, subtm: ArrayLike) -> float:
    """
    Calculates the mean of a phase-type distribution using an analytic expression.
    WARNING: This method often runs into issues with finite precision issues in
    the summation steps during the matrix multiplication steps.
    In these cases, use the numeric version.

    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix parameter.
    :return: Mean of the phase-type distribution.
    """
    p0_arr = np.asarray(p0, dtype=float)
    subtm_arr = np.asarray(subtm, dtype=float)

    ones = np.ones(subtm_arr.shape[0], dtype=float)
    s_inv = linalg.pinv(subtm_arr)
    mean = -1.0 * (ones @ s_inv @ p0_arr)
    return float(mean)


def ph_var(p0: ArrayLike, subtm: ArrayLike) -> float:
    """
    Calculates the variance of a phase-type distribution using an analytic expression.
    WARNING: This method often runs into issues with finite precision issues in
    the summation steps during the matrix multiplication steps.
    In these cases, use the numeric version.

    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix parameter.
    :return: Variance of the phase-type distribution.
    """
    p0_arr = np.asarray(p0, dtype=float)
    subtm_arr = np.asarray(subtm, dtype=float)

    ones = np.ones(subtm_arr.shape[0], dtype=float)
    s_inv = linalg.pinv(subtm_arr)

    term1 = 2.0 * (ones @ s_inv @ s_inv @ p0_arr)
    term2 = (ones @ s_inv @ p0_arr)
    var = term1 - term2**2
    return float(var)


def ph_mean_var_num(p0: ArrayLike, subtm: ArrayLike, level: float = 0.90) -> Tuple[float, float]:
    """
    Calculate the mean and variance of a phase type distribution using numerical methods.

    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix.
    :param level: float. A value between 0 and 1 non inclusive (default=0.90).
        If out of bounds, it is reset to 0.90.
    :return: (mean, var) tuple.
    """
    p0_arr = np.asarray(p0, dtype=float)
    subtm_arr = np.asarray(subtm, dtype=float)

    # Check the level
    if level >= 1.0 or level <= 0.0:
        level = 0.90

    # Root finding function to find where the phasetype distribution's cdf == level.
    def f(x: float) -> float:
        return _ph_cdf(x, p0_arr, subtm_arr) - level

    # Root find to level.
    x1 = ph_mean(p0_arr, subtm_arr)
    while f(x1) < 0.0:
        x1 *= 2.0

    locend = opt.brentq(f, 0.0, x1)
    locend *= 20.0

    # Numerically calculate the first and second moments.
    mu1 = integ.quad(lambda x: x * ph_pdf(x, p0_arr, subtm_arr), 0.0, locend, limit=100)[0]
    mu2 = integ.quad(lambda x: x * x * ph_pdf(x, p0_arr, subtm_arr), 0.0, locend, limit=100)[0]

    var = mu2 - mu1**2
    return float(mu1), float(var)


def approx_gamma_shape(mean: float, var: float) -> float:
    """
    Calculates the approximate shape parameter of the gamma distribution by
    the moment matching method.

    :param mean: Mean of the distribution (must be > 0).
    :param var: Variance of the distribution (must be > 0).
    :return: Approximate shape parameter.
    """
    temp = 2 * np.log(mean) - np.log(var)
    return np.exp(temp)


def approx_gamma_scale(mean: float, var: float):
    """
    Calculates the approximate scale parameter of the gamma distribution by
    the moment matching method.

    :param mean: The mean of some distribution.
    :param var: The variance of some distribution.
    :return scale: The approximate scale parameter.
    """
    temp = np.log(var) - np.log(mean)
    return np.exp(temp)


def approx_ph_gamma(
    p0: ArrayLike,
    subtm: ArrayLike,
    method: str = "analytic",
    level: float = 0.90,
) -> Tuple[float, float, float, float]:
    """
    Approximate a phase-type distribution with a gamma distribution using
    moment matching (analytic or numeric).

    :param p0: (N,) array-like. Initial probability vector.
    :param subtm: (N,N) array-like. Sub-transition matrix.
    :param method: 'analytic' or 'numeric'. Defaults to 'analytic'.
        If not recognized, defaults to 'numeric'.
    :param level: Passed to ph_mean_var_num if method == 'numeric'.
    :return: (mean, var, shape, scale) of the approximate gamma distribution.
    """

    if method not in ("analytic", "numeric"):
        method = "numeric"

    if method == "analytic":
        mean = ph_mean(p0, subtm)
        var = ph_var(p0, subtm)
    else:
        mean, var = ph_mean_var_num(p0, subtm, level=level)

    shape = approx_gamma_shape(mean, var)
    scale = approx_gamma_scale(mean, var)
    return float(mean), float(var), float(shape), float(scale)


def gaussian_pdf(x: ArrayLike, mean: float, std: float) -> ArrayLike:
    """
    Gaussian (normal) probability density function.

    :param x: Scalar or array-like of evaluation points.
    :param mean: Mean of the distribution.
    :param std: Standard deviation (must be > 0).
    :return: Gaussian pdf evaluated at x.
    """
    x_arr = np.asarray(x, dtype=float)
    return np.exp(-((x_arr - mean) ** 2) / (2.0 * std**2)) / (np.sqrt(2.0 * np.pi) * std)


def log_gaussian_pdf(x: ArrayLike, mean: float, std: float) -> ArrayLike:
    """
    Natural log of the Gaussian (normal) probability density function.

    :param x: Scalar or array-like of evaluation points.
    :param mean: Mean of the distribution.
    :param std: Standard deviation (must be > 0).
    :return: Log Gaussian pdf evaluated at x.
    """
    x_arr = np.asarray(x, dtype=float)
    term1 = -np.log(std * np.sqrt(2.0 * np.pi))
    term2 = -((x_arr - mean) ** 2) / (2.0 * std**2)
    return term1 + term2
