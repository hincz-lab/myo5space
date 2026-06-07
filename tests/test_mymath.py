"""
File: test_mymath.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains test for the mymath module using the pytest framework.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("numba")
pytest.importorskip("numba")

from scipy import special
from scipy import linalg
from numba import njit
import math
import myo5space.mymath as mymath


# Tests for mymath.i0
def test_ctypes_i0_callable_returns_finite_float() -> None:
    # Setup
    x = 0.5
    # Exercise
    y = mymath._i0_fn(x)
    # Validate
    assert isinstance(y, float)
    assert np.isfinite(y)
    # Cleanup - None


@pytest.mark.parametrize("x", [0.0, 0.1, 1.0, 5.0, -2.0, 20.0])
def test_vec_i0_matches_scipy_for_scalar_inputs(x: float) -> None:
    # Setup
    expected = float(special.i0(x))
    # Exercise
    out = mymath._vec_i0(x)
    # Validate
    # _vec_i0 may return a numpy scalar; cast to float for comparison.
    assert np.isclose(float(out), expected, rtol=1e-13, atol=0.0)
    # Cleanup - None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vec_i0_matches_scipy_for_array_inputs(dtype) -> None:
    # Setup
    x = np.linspace(-10, 10, 2048, dtype=dtype)
    expected = np.asarray(special.i0(x.astype(np.float64)), dtype=np.float64)
    # Exercise
    out = mymath._vec_i0(x)
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    # Cleanup - None


def test_i0_njit_compiles_and_matches_py_func_for_scalar() -> None:
    """
    Ensures:
      - i0 compiles in nopython mode
      - compiled result matches the undecorated python implementation (py_func)
    """
    # Setup
    x = 3.25
    # Exercise
    out_py = mymath.i0.py_func(x)  # no JIT compile
    out_jit = mymath.i0(x)  # triggers compilation
    # Validate
    assert np.isclose(float(out_jit), float(out_py), rtol=1e-13, atol=0.0)
    assert getattr(mymath.i0, "nopython_signatures", ())
    assert len(mymath.i0.signatures) >= 1
    # Cleanup - None


def test_i0_njit_matches_scipy_for_array_inputs() -> None:
    # Setup
    x = np.linspace(-5, 5, 512, dtype=np.float64)
    expected = np.asarray(special.i0(x), dtype=np.float64)
    # Exercise
    out = mymath.i0(x)
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert out.shape == x.shape
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    assert getattr(mymath.i0, "nopython_signatures", ())
    # Cleanup - None


@pytest.mark.parametrize(
    "x",
    [
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    ],
)
def test_i0_handles_common_float_dtypes(x: np.ndarray) -> None:
    # Setup
    expected = np.asarray(special.i0(x.astype(np.float64)), dtype=np.float64)
    # Exercise
    out = mymath.i0(x)
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    # Cleanup - None


def test_i0_rejects_non_numeric_inputs() -> None:
    """
    In nopython mode, Numba should raise for unsupported input types.
    Exact exception varies by Numba version, so keep broad.
    """
    # Setup
    bad = "not-a-number"
    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.i0(bad)  # type: ignore[arg-type]
    # Cleanup - None


# Tests for mymath.binon
def test_ctypes_binom_callable_returns_finite_float() -> None:
    # Setup
    x = 5.0
    y = 2.0
    # Exercise
    out = mymath._binom_fn(x, y)
    # Validate
    assert isinstance(out, float)
    assert np.isfinite(out)
    # Cleanup - None


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 1.0),
        (5.0, 2.0),
        (10.0, 5.0),
        (20.0, 10.0),
    ],
)
def test_vec_binom_matches_scipy_for_integer_like_scalars(x: float, y: float) -> None:
    # Setup
    expected = float(special.binom(x, y))
    # Exercise
    out = mymath._vec_binom(x, y)
    # Validate
    # These small integer-like cases should match exactly as float in SciPy.
    assert float(out) == expected
    # Cleanup - None


@pytest.mark.parametrize(
    "x,y",
    [
        (0.5, 0.0),
        (5.5, 2.0),
        (10.2, 3.7),
        (-0.5, 2.0),
        (20.25, 7.5),
    ],
)
def test_vec_binom_matches_scipy_for_real_scalars(x: float, y: float) -> None:
    # Setup
    expected = float(special.binom(x, y))
    # Exercise
    out = mymath._vec_binom(float(x), float(y))
    # Validate
    # Use a tolerance for gamma-based computations.
    assert np.isclose(float(out), expected, rtol=1e-12, atol=0.0)
    # Cleanup - None


def test_vec_binom_matches_scipy_for_array_inputs() -> None:
    # Setup
    x = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float64)
    y = np.array([0.0, 2.0, 5.0, 10.0], dtype=np.float64)
    expected = np.asarray(special.binom(x, y), dtype=np.float64)
    # Exercise
    out = mymath._vec_binom(x, y)
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert out.shape == expected.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    # Cleanup - None


def test_vec_binom_broadcasting_matches_scipy() -> None:
    # Setup
    x = np.array([5.0, 10.0, 20.0], dtype=np.float64)  # shape (3,)
    y = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)  # shape (3,1)
    expected = np.asarray(special.binom(x, y), dtype=np.float64)
    # Exercise
    out = mymath._vec_binom(x, y)
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    # Cleanup - None


def test_binom_njit_compiles_and_matches_py_func_for_scalars() -> None:
    """
    Ensures:
      - binom compiles in nopython mode
      - compiled result matches the undecorated python implementation (py_func)
    """
    # Setup
    x = 10.0
    y = 3.0
    # Exercise
    out_py = mymath.binom.py_func(x, y)  # no JIT compile
    out_jit = mymath.binom(x, y)  # triggers compilation
    # Validate
    assert np.isclose(float(out_jit), float(out_py), rtol=1e-12, atol=0.0)
    assert getattr(mymath.binom, "nopython_signatures", ())
    assert len(mymath.binom.signatures) >= 1
    # Cleanup - None


def test_binom_njit_matches_scipy_for_arrays() -> None:
    # Setup
    x = np.array([5.0, 10.0, 20.0], dtype=np.float64)
    y = np.array([2.0, 5.0, 10.0], dtype=np.float64)
    expected = np.asarray(special.binom(x, y), dtype=np.float64)
    # Exercise
    out = mymath.binom(x, y)  # triggers compilation for array signature if needed
    # Validate
    out = np.asarray(out, dtype=np.float64)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)
    assert getattr(mymath.binom, "nopython_signatures", ())
    # Cleanup - None


def test_binom_rejects_non_numeric_inputs() -> None:
    """
    In nopython mode, Numba should raise for unsupported input types.
    Exact exception varies by Numba version, so keep broad.
    """
    # Setup
    bad_x = "10"
    bad_y = 3.0
    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.binom(bad_x, bad_y)  # type: ignore[arg-type]
    # Cleanup - None


# Tests for mymath.is_invertible
@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_property_identity_is_always_invertible(n: int) -> None:
    # Setup
    a = np.eye(n)
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert out
    # Cleanup - None


def test_invertible_random_full_rank_true() -> None:
    # Setup
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert out
    # Cleanup - None


def test_non_square_matrix_false() -> None:
    # Setup
    a = np.ones((3, 4))
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert not out
    # Cleanup - None


def test_square_but_singular_false_duplicate_rows() -> None:
    # Setup
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],  # duplicate row -> rank deficient
            [4.0, 5.0, 6.0],
        ]
    )
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert not out
    # Cleanup - None


def test_square_but_singular_false_zero_row() -> None:
    # Setup
    a = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # zero row -> rank deficient
            [1.0, 0.0, 3.0],
        ]
    )
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert not out
    # Cleanup - None


def test_integer_matrix_full_rank_true() -> None:
    # Setup
    a = np.array([[1, 2], [3, 5]], dtype=int)  # det = -1, invertible
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert out
    # Cleanup - None


def test_integer_matrix_singular_false() -> None:
    # Setup
    a = np.array([[1, 2], [2, 4]], dtype=int)  # rank 1
    # Exercise
    out = mymath.is_invertible(a)
    # Validate
    assert not out
    # Cleanup - None


# Tests for mymath.expm_eig
def test_expm_eig_matches_scipy_for_diagonal_matrix() -> None:
    # Setup
    a = np.diag([0.0, 1.0, -2.0])
    expected = linalg.expm(a)

    # Exercise
    out = mymath.expm_eig(a)

    # Validate
    assert out.shape == a.shape
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_expm_eig_matches_scipy_for_symmetric_matrix() -> None:
    # Setup
    a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=float)  # symmetric -> diagonalizable
    expected = linalg.expm(a)

    # Exercise
    out = mymath.expm_eig(a)

    # Validate
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_expm_eig_casts_non_float_input_and_matches_scipy() -> None:
    # Setup
    a = np.array(
        [[1, 2], [3, 4]], dtype=int
    )  # integer dtype -> cast to float internally
    expected = linalg.expm(a.astype(float))

    # Exercise
    out = mymath.expm_eig(a)

    # Validate
    # For int input, expm_eig returns float (per its casting logic)
    assert out.dtype.kind in ("f", "c")
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_expm_eig_preserves_real_dtype_for_real_input() -> None:
    # Setup
    a = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)  # rotation generator
    expected = linalg.expm(a)

    # Exercise
    out = mymath.expm_eig(a)

    # Validate
    assert out.dtype.kind == "f"
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_expm_eig_handles_complex_input_and_matches_scipy() -> None:
    # Setup
    a = np.array([[1.0 + 1.0j, 0.2], [0.0, -0.3 + 0.5j]], dtype=np.complex128)
    expected = linalg.expm(a)

    # Exercise
    out = mymath.expm_eig(a)

    # Validate
    assert out.dtype.kind == "c"
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_expm_eig_raises_value_error_for_defective_matrix() -> None:
    """
    A Jordan block has a defective eigen-decomposition: eigenvector matrix is singular.
    expm_eig should raise ValueError via is_invertible(vr) check.
    """
    # Setup
    a = np.array(
        [[1.0, 1.0], [0.0, 1.0]], dtype=float
    )  # defective (non-diagonalizable)

    # Exercise / Validate
    with pytest.raises(ValueError, match="Eigensystem is singular"):
        mymath.expm_eig(a)

    # Cleanup
    # nothing


def test_expm_uses_expm_eig_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    a = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)

    orig_expm_eig = mymath.expm_eig
    orig_scipy_expm = mymath.linalg.expm

    called = {"eig": 0, "scipy": 0}

    def wrapped_expm_eig(x):
        called["eig"] += 1
        return orig_expm_eig(x)  # call original, not patched

    def wrapped_scipy_expm(x):
        called["scipy"] += 1
        return orig_scipy_expm(x)  # call original, not patched

    monkeypatch.setattr(mymath, "expm_eig", wrapped_expm_eig)
    monkeypatch.setattr(mymath.linalg, "expm", wrapped_scipy_expm)

    expected = orig_expm_eig(a)

    # Exercise
    out = mymath.expm(a)

    # Validate
    assert called["eig"] == 1
    assert called["scipy"] == 0
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # monkeypatch restores originals


def test_expm_falls_back_to_scipy_when_expm_eig_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setup
    a = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)

    orig_scipy_expm = mymath.linalg.expm

    called = {"eig": 0, "scipy": 0}

    def failing_expm_eig(_x):
        called["eig"] += 1
        raise ValueError("Eigensystem is singular.")

    def wrapped_scipy_expm(x):
        called["scipy"] += 1
        return orig_scipy_expm(x)  # call original

    monkeypatch.setattr(mymath, "expm_eig", failing_expm_eig)
    monkeypatch.setattr(mymath.linalg, "expm", wrapped_scipy_expm)

    expected = orig_scipy_expm(a)

    # Exercise
    out = mymath.expm(a)

    # Validate
    assert called["eig"] == 1
    assert called["scipy"] == 1
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # monkeypatch restores originals


# Tests for mymath.sample_discrete
@njit
def _draw_many_discrete(probs: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Deterministic helper for tests: seeds Numba RNG and draws n samples.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = mymath.sample_discrete(probs)
    return out


def test_single_choice_always_zero_pyfunc() -> None:
    # Setup
    probs = np.array([123.0], dtype=np.float64)

    # Exercise
    out = mymath.sample_discrete.py_func(probs)

    # Validate
    assert out == 0

    # Cleanup
    # nothing


def test_returns_index_in_bounds_pyfunc() -> None:
    # Setup
    probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    # Exercise
    out = mymath.sample_discrete.py_func(probs)

    # Validate
    assert isinstance(out, (int, np.integer))
    assert 0 <= int(out) < probs.size

    # Cleanup
    # nothing


def test_negative_probs_are_treated_as_positive_pyfunc() -> None:
    # Setup
    probs_pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    probs_neg = np.array([-1.0, 2.0, -3.0], dtype=np.float64)

    # Exercise
    np.random.seed(0)
    draws_pos = np.array(
        [mymath.sample_discrete.py_func(probs_pos) for _ in range(2000)]
    )

    np.random.seed(0)
    draws_neg = np.array(
        [mymath.sample_discrete.py_func(probs_neg) for _ in range(2000)]
    )

    # Validate
    # With identical seeds and identical abs(probs), sequences should match for py_func.
    assert np.array_equal(draws_pos, draws_neg)

    # Cleanup
    # nothing


def test_compiles_nopython_and_returns_int() -> None:
    # Setup
    probs = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # Exercise
    out = mymath.sample_discrete(probs)  # triggers compilation

    # Validate
    assert 0 <= int(out) < probs.size
    assert getattr(mymath.sample_discrete, "nopython_signatures", ())
    assert len(mymath.sample_discrete.signatures) >= 1

    # Cleanup
    # nothing


def test_distribution_matches_weights_reasonably() -> None:
    """
    Statistical sanity check (not too strict to avoid flakiness).

    We seed Numba RNG inside _draw_many so this is deterministic.
    """
    # Setup
    probs = np.array([1.0, 2.0, 7.0], dtype=np.float64)
    p = probs / probs.sum()
    n = 50_000
    seed = 123

    # Exercise
    draws = _draw_many_discrete(probs, n, seed)
    counts = np.bincount(draws, minlength=probs.size)
    freqs = counts / n

    # Validate
    # Allow a few standard errors; 6*sigma is very safe and deterministic here.
    # sigma ~ sqrt(p(1-p)/n) for multinomeal distribution
    tol = 6.0 * np.sqrt(p * (1.0 - p) / n)
    assert np.all(np.abs(freqs - p) <= tol)

    # Cleanup
    # nothing


def test_works_with_unnormalized_inputs() -> None:
    # Setup
    probs1 = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    probs2 = np.array([10.0, 10.0, 20.0], dtype=np.float64)  # same proportions
    n = 20_000
    seed = 999

    # Exercise
    draws1 = _draw_many_discrete(probs1, n, seed)
    draws2 = _draw_many_discrete(probs2, n, seed)
    freq1 = np.bincount(draws1, minlength=3) / n
    freq2 = np.bincount(draws2, minlength=3) / n

    # Validate
    assert np.allclose(freq1, freq2, rtol=0.0, atol=2e-3)

    # Cleanup
    # nothing


# Tests for mymath.sample_exponential
@njit
def _draw_many_exponential(rate: float, n: int, seed: int) -> np.ndarray:
    """
    Deterministic helper for tests: seeds Numba RNG and draws n samples.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = mymath.sample_exponential(rate)
    return out


def test_sample_exponential_returns_positive_pyfunc() -> None:
    # Setup
    rate = 2.5

    # Exercise
    out = mymath.sample_exponential.py_func(rate)

    # Validate
    assert isinstance(out, float)
    assert out > 0.0

    # Cleanup
    # nothing


def test_sample_exponential_compiles_nopython() -> None:
    # Setup
    rate = 1.0

    # Exercise
    out = mymath.sample_exponential(rate)  # triggers compilation

    # Validate
    assert out > 0.0
    assert getattr(mymath.sample_exponential, "nopython_signatures", ())
    assert len(mymath.sample_exponential.signatures) >= 1

    # Cleanup
    # nothing


def test_deterministic_with_seed() -> None:
    # Setup
    rate = 1.7

    # Exercise
    np.random.seed(0)
    a = mymath.sample_exponential.py_func(rate)

    np.random.seed(0)
    b = mymath.sample_exponential.py_func(rate)

    # Validate
    assert a == b

    # Cleanup
    # nothing


def test_mean_matches_theory_reasonably() -> None:
    """
    Statistical sanity check:
    E[X] = 1 / rate for exponential distribution.
    """
    # Setup
    rate = 2.0
    expected_mean = 1.0 / rate
    n = 50_000
    seed = 123

    # Exercise
    samples = _draw_many_exponential(rate, n, seed)
    sample_mean = samples.mean()

    # Validate
    # Standard error of the mean for exponential:
    # Var = 1 / rate^2, so SE = 1 / (rate * sqrt(n))
    se = 1.0 / (rate * np.sqrt(n))
    assert abs(sample_mean - expected_mean) <= 6.0 * se

    # Cleanup
    # nothing


def test_distribution_scales_with_rate() -> None:
    # Setup
    n = 20_000
    seed = 999
    rate1 = 1.0
    rate2 = 4.0  # mean should be 4x smaller

    # Exercise
    samples1 = _draw_many_exponential(rate1, n, seed)
    samples2 = _draw_many_exponential(rate2, n, seed)

    # Validate
    ratio = samples1.mean() / samples2.mean()
    assert np.isclose(ratio, rate2 / rate1, rtol=0.05)

    # Cleanup
    # nothing


# Tests for mymath.sample_hypoexponential
@njit
def _draw_many_hypoexponential(rates: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Deterministic helper: seed Numba RNG and draw n hypoexponential samples.
    """
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = mymath.sample_hypoexponential(rates)
    return out


def test_returns_positive_pyfunc() -> None:
    # Setup
    rates = np.array([2.0, 3.0, 5.0], dtype=np.float64)

    # Exercise
    out = mymath.sample_hypoexponential.py_func(rates)

    # Validate
    assert isinstance(out, float)
    assert out > 0.0

    # Cleanup
    # nothing


def test_compiles_nopython() -> None:
    # Setup
    rates = np.array([1.0, 2.0], dtype=np.float64)

    # Exercise
    out = mymath.sample_hypoexponential(rates)  # triggers compilation

    # Validate
    assert out > 0.0
    assert getattr(mymath.sample_hypoexponential, "nopython_signatures", ())
    assert len(mymath.sample_hypoexponential.signatures) >= 1

    # Cleanup
    # nothing


def test_single_rate_matches_exponential_distribution_mean() -> None:
    """
    If rates has length 1, sample_hypoexponential should behave like exponential.
    We compare sample means from deterministic draws.
    """
    # Setup
    rate = 2.5
    rates = np.array([rate], dtype=np.float64)
    n = 30_000
    seed = 123

    # Exercise
    h = _draw_many_hypoexponential(rates, n, seed)
    e = _draw_many_exponential(rate, n, seed)

    # Validate
    # Means should be extremely close since RNG sequence is identical and both call
    # one exponential draw per sample.
    assert np.isclose(h.mean(), e.mean(), rtol=0.0, atol=1e-12)

    # Cleanup
    # nothing


def test_mean_matches_theory_reasonably2() -> None:
    """
    For independent exponentials, the sum has:
      E[sum_i X_i] = sum_i 1/rate_i
      Var[sum_i X_i] = sum_i 1/(rate_i^2)
    We test the sample mean against theory with a non-flaky tolerance.
    """
    # Setup
    rates = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    expected_mean = float(np.sum(1.0 / rates))
    expected_var = float(np.sum(1.0 / (rates * rates)))

    n = 60_000
    seed = 999

    # Exercise
    samples = _draw_many_hypoexponential(rates, n, seed)
    sample_mean = float(samples.mean())

    # Validate
    # SE(mean) = sqrt(Var / n)
    se = np.sqrt(expected_var / n)
    assert abs(sample_mean - expected_mean) <= 6.0 * se

    # Cleanup
    # nothing


def test_distribution_scales_when_all_rates_scaled() -> None:
    """
    If all rates are multiplied by c, each exponential mean scales by 1/c,
    so the hypoexponential mean scales by 1/c.
    """
    # Setup
    rates1 = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    c = 4.0
    rates2 = rates1 * c

    n = 40_000
    seed = 42

    # Exercise
    s1 = _draw_many_hypoexponential(rates1, n, seed)
    s2 = _draw_many_hypoexponential(rates2, n, seed)

    # Validate
    ratio = s1.mean() / s2.mean()
    assert np.isclose(ratio, c, rtol=0.05)

    # Cleanup
    # nothing


# Tests for mymath.lngamma_pdf and mymath.gamma_pdf
def test_lngamma_pdf_compiles_nopython() -> None:
    # Setup
    x = 1.25
    a = 2.5

    # Exercise
    out = mymath.lngamma_pdf(x, a)  # triggers compilation

    # Validate
    assert math.isfinite(out)
    assert getattr(mymath.lngamma_pdf, "nopython_signatures", ())
    assert len(mymath.lngamma_pdf.signatures) >= 1

    # Cleanup
    # nothing


def test_gamma_pdf_compiles_nopython() -> None:
    # Setup
    x = 1.25
    a = 2.5
    b = 3.0

    # Exercise
    out = mymath.gamma_pdf(x, a, b)  # triggers compilation

    # Validate
    assert out > 0.0
    assert getattr(mymath.gamma_pdf, "nopython_signatures", ())
    assert len(mymath.gamma_pdf.signatures) >= 1

    # Cleanup
    # nothing


def test_lngamma_pdf_special_case_a_near_one() -> None:
    # Setup
    x = 3.0
    a = 1.0 + 5e-13  # within atol=1e-12 in implementation

    # Exercise
    out = mymath.lngamma_pdf.py_func(x, a)

    # Validate
    assert out == -x

    # Cleanup
    # nothing


def test_lngamma_pdf_general_case_differs_from_special_case_outside_tolerance() -> None:
    """
    Checks that once we move outside the atol window, the result is no
    longer forced to -x.
    """
    # Setup
    x = 3.0
    a = 1.0 + 2e-12  # outside atol

    # Exercise
    out = mymath.lngamma_pdf.py_func(x, a)

    # Validate
    assert out != -x

    # Cleanup
    # nothing


@pytest.mark.parametrize(
    "x,a",
    [
        (0.1, 2.0),
        (1.0, 2.0),
        (5.0, 2.0),
        (0.5, 5.5),
        (2.2, 0.8),
    ],
)
def test_lngamma_pdf_compiled_matches_pyfunc(x: float, a: float) -> None:
    # Setup
    out_py = mymath.lngamma_pdf.py_func(x, a)

    # Exercise
    out_jit = mymath.lngamma_pdf(x, a)

    # Validate
    assert np.isclose(out_jit, out_py, rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_lngamma_pdf_x_zero_behavior_a_eq_one_is_zero() -> None:
    # Setup
    x = 0.0
    a = 1.0

    # Exercise
    out = mymath.lngamma_pdf.py_func(x, a)

    # Validate
    assert out == -0.0  # equals 0.0

    # Cleanup
    # nothing


def test_lngamma_pdf_x_zero_behavior_a_gt_one_is_neg_inf() -> None:
    # Setup
    x = 0.0
    a = 2.0

    # Exercise
    out = mymath.lngamma_pdf.py_func(x, a)

    # Validate
    assert out == -np.inf

    # Cleanup
    # nothing


def test_gamma_pdf_reduces_to_exponential_when_a_is_one() -> None:
    """
    For a=1, Gamma(a=1, scale=b) is Exponential(scale=b):
      f(x) = (1/b) * exp(-x/b) for x >= 0
    """
    # Setup
    x = 1.7
    a = 1.0
    b = 2.5
    expected = (1.0 / b) * math.exp(-x / b)

    # Exercise
    out = mymath.gamma_pdf.py_func(x, a, b)

    # Validate
    assert np.isclose(out, expected, rtol=1e-15, atol=0.0)

    # Cleanup
    # nothing


@pytest.mark.parametrize(
    "x,a,b",
    [
        (0.1, 2.0, 1.0),
        (1.0, 2.0, 1.0),
        (5.0, 2.0, 3.0),
        (0.5, 5.5, 2.0),
        (2.2, 0.8, 4.0),
    ],
)
def test_gamma_pdf_compiled_matches_pyfunc(x: float, a: float, b: float) -> None:
    # Setup
    out_py = mymath.gamma_pdf.py_func(x, a, b)

    # Exercise
    out_jit = mymath.gamma_pdf(x, a, b)

    # Validate
    assert np.isclose(out_jit, out_py, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_gamma_pdf_integrates_to_one_approximately() -> None:
    """
    Numerical sanity check: integrate pdf over [0, xmax] where xmax is large
    relative to scale. For well-behaved shapes, most mass is captured.
    """
    # Setup
    a = 2.5
    b = 1.3
    xmax = 40.0 * b  # should capture ~all mass
    xs = np.linspace(0.0, xmax, 200_001, dtype=np.float64)

    # Exercise
    ys = np.array(
        [mymath.gamma_pdf.py_func(float(x), a, b) for x in xs], dtype=np.float64
    )
    area = np.trapz(ys, xs)

    # Validate
    assert np.isfinite(area)
    assert 0.995 <= area <= 1.005

    # Cleanup
    # nothing


# Tests for mymath.ph_pdf
def test_ph_pdf_negative_t_returns_zero_scalar() -> None:
    # Setup
    t = -0.1
    p0 = np.array([1.0])
    subtm = np.array([[-2.0]])  # valid sub-generator

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    assert out == 0.0

    # Cleanup
    # nothing


def test_ph_pdf_negative_t_returns_zero_vectorized() -> None:
    # Setup
    t = np.array([-1.0, -0.5, -0.1], dtype=float)
    p0 = np.array([1.0])
    subtm = np.array([[-2.0]])

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert out.shape == t.shape
    assert np.all(out == 0.0)

    # Cleanup
    # nothing


def test_ph_pdf_scalar_vs_array_consistency() -> None:
    # Setup
    p0 = np.array([1.0, 0.0])
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    t_scalar = 0.7
    t_array = np.array([t_scalar], dtype=float)

    # Exercise
    out_scalar = mymath.ph_pdf(t_scalar, p0, subtm)
    out_array = mymath.ph_pdf(t_array, p0, subtm)

    # Validate
    assert np.isclose(float(out_scalar), float(out_array[0]), rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_ph_pdf_output_nonnegative_for_valid_subgenerator() -> None:
    # Setup
    # A simple valid sub-generator: negative diagonals, nonnegative off-diagonals, row sums <= 0
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)
    t = np.linspace(0.0, 10.0, 200, dtype=float)

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-14)  # tiny numerical noise allowance

    # Cleanup
    # nothing


def test_ph_pdf_one_state_reduces_to_exponential_pdf() -> None:
    """
    For N=1 with subtm = [[-lambda]] and p0=[1], PH is Exp(rate=lambda):
      f(t) = lambda * exp(-lambda t), t>=0
    """
    # Setup
    lam = 2.5
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-lam]], dtype=float)
    t = np.array([0.0, 0.1, 1.0, 3.0], dtype=float)
    expected = lam * np.exp(-lam * t)

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_pdf_two_independent_absorbing_phases_equals_mixture_of_exponentials() -> (
    None
):
    """
    If subtm is diagonal with rates lam1, lam2 and p0 is a mixture weight,
    this is a mixture of exponentials:
      f(t) = w1*lam1*exp(-lam1 t) + w2*lam2*exp(-lam2 t)
    """
    # Setup
    lam1, lam2 = 1.0, 4.0
    w1, w2 = 0.3, 0.7
    p0 = np.array([w1, w2], dtype=float)
    subtm = np.array([[-lam1, 0.0], [0.0, -lam2]], dtype=float)
    t = np.linspace(0.0, 5.0, 200, dtype=float)
    expected = w1 * lam1 * np.exp(-lam1 * t) + w2 * lam2 * np.exp(-lam2 * t)

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_pdf_integrates_to_one_for_valid_subgenerator() -> None:
    """
    PH distribution density should integrate to ~1 over [0, large].
    Use a simple 2-state sub-generator and integrate numerically.
    """
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    tmax = 40.0
    ts = np.linspace(0.0, tmax, 20001, dtype=float)

    # Exercise
    ys = mymath.ph_pdf(ts, p0, subtm)
    ys = np.asarray(ys, dtype=float)
    area = np.trapz(ys, ts)

    # Validate
    assert np.isfinite(area)
    assert 0.995 <= area <= 1.005

    # Cleanup
    # nothing


def test_ph_pdf_accepts_list_inputs_and_returns_numpy_array_for_vector_t() -> None:
    # Setup
    t = [0.0, 0.5, 1.0]
    p0 = [1.0]
    subtm = [[-2.0]]

    # Exercise
    out = mymath.ph_pdf(t, p0, subtm)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(out >= 0.0)

    # Cleanup
    # nothing


def test_ph_pdf_shapes_mismatch_raises() -> None:
    # Setup
    t = 1.0
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.ph_pdf(t, p0, subtm)

    # Cleanup
    # nothing


# Tests for mymath.ph_cdf
def test_ph_cdf_negative_t_returns_zero_scalar() -> None:
    # Setup
    t = -0.1
    p0 = np.array([1.0])
    subtm = np.array([[-2.0]])

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    assert out == 0.0

    # Cleanup
    # nothing


def test_ph_cdf_negative_t_returns_zero_vectorized() -> None:
    # Setup
    t = np.array([-1.0, -0.5, -0.1], dtype=float)
    p0 = np.array([1.0])
    subtm = np.array([[-2.0]])

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert out.shape == t.shape
    assert np.all(out == 0.0)

    # Cleanup
    # nothing


def test_ph_cdf_scalar_vs_array_consistency() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    t_scalar = 0.7
    t_array = np.array([t_scalar], dtype=float)

    # Exercise
    out_scalar = mymath.ph_cdf(t_scalar, p0, subtm)
    out_array = mymath.ph_cdf(t_array, p0, subtm)

    # Validate
    assert np.isclose(float(out_scalar), float(out_array[0]), rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_ph_cdf_in_range_for_valid_subgenerator() -> None:
    # Setup
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)
    t = np.linspace(0.0, 10.0, 200, dtype=float)

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-14)
    assert np.all(out <= 1.0 + 1e-14)

    # Cleanup
    # nothing


def test_ph_cdf_is_monotone_nondecreasing_for_valid_subgenerator() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    t = np.linspace(0.0, 20.0, 2000, dtype=float)

    # Exercise
    out = np.asarray(mymath.ph_cdf(t, p0, subtm), dtype=float)
    diffs = np.diff(out)

    # Validate
    assert np.all(diffs >= -1e-10)  # allow small numerical noise

    # Cleanup
    # nothing


def test_ph_cdf_at_zero_matches_expected_formula() -> None:
    """
    For t=0: expm(0)=I, so CDF(0)=1 - 1^T I p0 = 1 - sum(p0).
    For a proper PH, sum(p0)=1 -> CDF(0)=0.
    """
    # Setup
    p0 = np.array([0.2, 0.8], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    out = float(mymath.ph_cdf(0.0, p0, subtm))

    # Validate
    assert np.isclose(out, 1.0 - float(p0.sum()), rtol=0.0, atol=0.0)

    # Cleanup
    # nothing


def test_ph_cdf_one_state_reduces_to_exponential_cdf() -> None:
    """
    For N=1 with subtm=[[-lambda]] and p0=[1], PH is Exp(rate=lambda):
      F(t) = 1 - exp(-lambda t), t>=0
    """
    # Setup
    lam = 2.5
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-lam]], dtype=float)
    t = np.array([0.0, 0.1, 1.0, 3.0], dtype=float)
    expected = 1.0 - np.exp(-lam * t)

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_cdf_diagonal_subtm_gives_mixture_of_exponential_cdfs() -> None:
    """
    If subtm is diagonal with rates lam1, lam2 and p0=[w1,w2],
    then survival is w1*exp(-lam1 t) + w2*exp(-lam2 t),
    so CDF is 1 - that.
    """
    # Setup
    lam1, lam2 = 1.0, 4.0
    w1, w2 = 0.3, 0.7
    p0 = np.array([w1, w2], dtype=float)
    subtm = np.array([[-lam1, 0.0], [0.0, -lam2]], dtype=float)
    t = np.linspace(0.0, 5.0, 200, dtype=float)
    expected = 1.0 - (w1 * np.exp(-lam1 * t) + w2 * np.exp(-lam2 * t))

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    out = np.asarray(out, dtype=float)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_cdf_tends_to_one_for_large_t() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    out = float(mymath.ph_cdf(1e3, p0, subtm))

    # Validate
    assert 0.999999 <= out <= 1.0 + 1e-12

    # Cleanup
    # nothing


def test_ph_cdf_accepts_list_inputs_and_returns_numpy_array_for_vector_t() -> None:
    # Setup
    t = [0.0, 0.5, 1.0]
    p0 = [1.0]
    subtm = [[-2.0]]

    # Exercise
    out = mymath.ph_cdf(t, p0, subtm)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(out >= 0.0)

    # Cleanup
    # nothing


def test_ph_cdf_shapes_mismatch_raises() -> None:
    # Setup
    t = 1.0
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.ph_cdf(t, p0, subtm)

    # Cleanup
    # nothing


# Tests for mymath.ph_mean
def test_ph_mean_one_state_exponential_mean() -> None:
    """
    For N=1 with subtm=[[-lambda]] and p0=[1], PH is Exp(rate=lambda),
    so mean = 1/lambda.
    """
    # Setup
    lam = 2.5
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-lam]], dtype=float)
    expected = 1.0 / lam

    # Exercise
    out = mymath.ph_mean(p0, subtm)

    # Validate
    assert np.isclose(out, expected, rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_ph_mean_diagonal_subtm_mixture_of_exponentials_mean() -> None:
    """
    Diagonal subtm with p0 mixture weights corresponds to a mixture of independent
    absorbing phases, and mean is sum_i p0_i * (1/lambda_i).
    """
    # Setup
    lam1, lam2 = 1.0, 4.0
    w1, w2 = 0.3, 0.7
    p0 = np.array([w1, w2], dtype=float)
    subtm = np.array([[-lam1, 0.0], [0.0, -lam2]], dtype=float)
    expected = w1 * (1.0 / lam1) + w2 * (1.0 / lam2)

    # Exercise
    out = mymath.ph_mean(p0, subtm)

    # Validate
    assert np.isclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_mean_linearity_in_p0() -> None:
    """
    ph_mean is linear in p0: mean(alpha p + (1-alpha) q) = alpha mean(p) + (1-alpha) mean(q)
    for fixed subtm.
    """
    # Setup
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    p = np.array([1.0, 0.0], dtype=float)
    q = np.array([0.0, 1.0], dtype=float)
    alpha = 0.25
    mix = alpha * p + (1.0 - alpha) * q

    # Exercise
    m_p = mymath.ph_mean(p, subtm)
    m_q = mymath.ph_mean(q, subtm)
    m_mix = mymath.ph_mean(mix, subtm)

    # Validate
    assert np.isclose(m_mix, alpha * m_p + (1.0 - alpha) * m_q, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_mean_positive_mean_for_valid_subgenerator() -> None:
    # Setup
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)

    # Exercise
    out = mymath.ph_mean(p0, subtm)

    # Validate
    assert np.isfinite(out)
    assert out > 0.0

    # Cleanup
    # nothing


def test_ph_mean_accepts_list_inputs() -> None:
    # Setup
    p0 = [1.0, 0.0]
    subtm = [[-2.0, 2.0], [0.0, -3.0]]

    # Exercise
    out = mymath.ph_mean(p0, subtm)

    # Validate
    assert isinstance(out, float)
    assert out > 0.0

    # Cleanup
    # nothing


def test_ph_mean_shapes_mismatch_raises() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.ph_mean(p0, subtm)

    # Cleanup
    # nothing


def test_ph_mean_matches_numeric_integration_of_survival_function() -> None:
    """
    Mean of a nonnegative random variable is integral of survival:
      E[T] = ∫_0^∞ S(t) dt, where S(t)=1 - CDF(t).
    For PH: S(t) = 1^T expm(t S) p0.

    This test uses SciPy expm as an oracle and numerical integration on a large interval.
    """
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    ones = np.ones(subtm.shape[0], dtype=float)

    tmax = 60.0
    ts = np.linspace(0.0, tmax, 20001, dtype=float)
    surv = np.array(
        [float(ones @ linalg.expm(t * subtm) @ p0) for t in ts], dtype=float
    )
    numeric_mean = np.trapz(surv, ts)

    # Exercise
    analytic_mean = mymath.ph_mean(p0, subtm)

    # Validate
    assert np.isfinite(numeric_mean)
    assert np.isclose(analytic_mean, numeric_mean, rtol=5e-3, atol=0.0)

    # Cleanup
    # nothing


# Tests for mymath.ph_var
def test_ph_var_one_state_exponential_variance() -> None:
    """
    For N=1 with subtm=[[-lambda]] and p0=[1], PH is Exp(rate=lambda),
    so Var = 1/lambda^2.
    """
    # Setup
    lam = 2.5
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-lam]], dtype=float)
    expected = 1.0 / (lam * lam)

    # Exercise
    out = mymath.ph_var(p0, subtm)

    # Validate
    assert np.isclose(out, expected, rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_ph_var_diagonal_subtm_mixture_of_exponentials_variance() -> None:
    """
    Diagonal subtm with p0 mixture weights corresponds to a mixture of exponentials.
    For a mixture, Var(T) = E[Var(T|Z)] + Var(E[T|Z])
      = sum w_i * 1/l_i^2 + sum w_i * (1/l_i)^2 - (sum w_i * 1/l_i)^2
      = sum w_i * 1/l_i^2 + (sum w_i * 1/l_i^2?) no; use the standard formula:
      Var = sum w_i*(v_i + m_i^2) - (sum w_i*m_i)^2,
      where m_i=1/l_i, v_i=1/l_i^2.
      => Var = sum w_i*(2/l_i^2) - (sum w_i*(1/l_i))^2
    """
    # Setup
    lam1, lam2 = 1.0, 4.0
    w1, w2 = 0.3, 0.7
    p0 = np.array([w1, w2], dtype=float)
    subtm = np.array([[-lam1, 0.0], [0.0, -lam2]], dtype=float)

    expected = (2.0 * (w1 / (lam1 * lam1) + w2 / (lam2 * lam2))) - (
        w1 / lam1 + w2 / lam2
    ) ** 2

    # Exercise
    out = mymath.ph_var(p0, subtm)

    # Validate
    assert np.isclose(out, expected, rtol=1e-12, atol=0.0)

    # Cleanup
    # nothing


def test_ph_var_linearity_in_p0_does_not_hold() -> None:
    """
    Variance is not linear in p0. This test guards against accidental linearization bugs.
    We assert the mixture variance differs from the weighted sum of variances
    for a simple case.
    """
    # Setup
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    p = np.array([1.0, 0.0], dtype=float)
    q = np.array([0.0, 1.0], dtype=float)
    alpha = 0.25
    mix = alpha * p + (1.0 - alpha) * q

    # Exercise
    v_p = mymath.ph_var(p, subtm)
    v_q = mymath.ph_var(q, subtm)
    v_mix = mymath.ph_var(mix, subtm)

    # Validate
    # Not strictly guaranteed for all generators, but should hold for this case.
    assert not np.isclose(
        v_mix, alpha * v_p + (1.0 - alpha) * v_q, rtol=1e-12, atol=1e-12
    )

    # Cleanup
    # nothing


def test_ph_var_nonnegative_for_valid_subgenerator() -> None:
    # Setup
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)

    # Exercise
    out = mymath.ph_var(p0, subtm)

    # Validate
    assert np.isfinite(out)
    assert out >= -1e-12  # allow tiny numerical noise

    # Cleanup
    # nothing


def test_ph_var_accepts_list_inputs() -> None:
    # Setup
    p0 = [1.0, 0.0]
    subtm = [[-2.0, 2.0], [0.0, -3.0]]

    # Exercise
    out = mymath.ph_var(p0, subtm)

    # Validate
    assert isinstance(out, float)
    assert out >= -1e-12

    # Cleanup
    # nothing


def test_ph_var_shapes_mismatch_raises() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.ph_var(p0, subtm)

    # Cleanup
    # nothing


def test_ph_var_matches_numeric_second_moment_relation_when_available() -> None:
    """
    For nonnegative T: Var(T) = E[T^2] - (E[T])^2.
    For PH, moments can be computed from expm numerically:
      E[T]   = ∫_0^∞ S(t) dt
      E[T^2] = 2 ∫_0^∞ t S(t) dt
    where S(t)=1^T expm(t S) p0.

    We numerically integrate on a large interval and compare to ph_var.
    """
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)
    ones = np.ones(subtm.shape[0], dtype=float)

    tmax = 80.0
    ts = np.linspace(0.0, tmax, 40001, dtype=float)
    surv = np.array(
        [float(ones @ linalg.expm(t * subtm) @ p0) for t in ts], dtype=float
    )

    e1 = np.trapz(surv, ts)
    e2 = 2.0 * np.trapz(ts * surv, ts)
    numeric_var = e2 - e1 * e1

    # Exercise
    analytic_var = mymath.ph_var(p0, subtm)

    # Validate
    assert np.isfinite(numeric_var)
    assert np.isclose(analytic_var, numeric_var, rtol=1e-2, atol=0.0)

    # Cleanup
    # nothing


# Test mymath.ph_mean_var_num
def test_ph_mean_var_num_returns_finite_positive_for_valid_subgenerator() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    mu, var = mymath.ph_mean_var_num(p0, subtm)

    # Validate
    assert np.isfinite(mu) and mu > 0.0
    assert np.isfinite(var) and var >= -1e-10  # allow tiny numerical noise

    # Cleanup
    # nothing


def test_ph_mean_var_num_level_out_of_bounds_defaults_to_point_nine() -> None:
    # Setup
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)

    # Exercise
    mu_bad_low, var_bad_low = mymath.ph_mean_var_num(p0, subtm, level=-1.0)
    mu_bad_high, var_bad_high = mymath.ph_mean_var_num(p0, subtm, level=2.0)
    mu_default, var_default = mymath.ph_mean_var_num(p0, subtm, level=0.90)

    # Validate
    assert np.isclose(mu_bad_low, mu_default, rtol=0.0, atol=1e-10)
    assert np.isclose(var_bad_low, var_default, rtol=0.0, atol=1e-10)

    assert np.isclose(mu_bad_high, mu_default, rtol=0.0, atol=1e-10)
    assert np.isclose(var_bad_high, var_default, rtol=0.0, atol=1e-10)

    # Cleanup
    # nothing


def test_ph_mean_var_num_matches_closed_form_for_exponential_case() -> None:
    """
    For N=1 with subtm=[[-lambda]] and p0=[1], PH is Exp(rate=lambda).
    Mean = 1/lambda, Var = 1/lambda^2.
    """
    # Setup
    lam = 2.0
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-lam]], dtype=float)
    expected_mu = 1.0 / lam
    expected_var = 1.0 / (lam * lam)

    # Exercise
    mu, var = mymath.ph_mean_var_num(p0, subtm, level=0.90)

    # Validate
    # Numerical integration + truncation -> allow modest tolerance
    assert np.isclose(mu, expected_mu, rtol=2e-3, atol=0.0)
    assert np.isclose(var, expected_var, rtol=5e-3, atol=0.0)

    # Cleanup
    # nothing


def test_ph_mean_var_num_close_to_analytic_ph_mean_and_ph_var() -> None:
    """
    For a well-conditioned PH, numeric mean/var should match analytic ph_mean/ph_var.
    """
    # Setup
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)

    mu_analytic = mymath.ph_mean(p0, subtm)
    var_analytic = mymath.ph_var(p0, subtm)

    # Exercise
    mu_num, var_num = mymath.ph_mean_var_num(p0, subtm, level=0.90)

    # Validate
    assert np.isclose(mu_num, mu_analytic, rtol=1e-2, atol=0.0)
    assert np.isclose(var_num, var_analytic, rtol=3e-2, atol=0.0)

    # Cleanup
    # nothing


def test_ph_mean_var_num_increases_locend_when_level_high() -> None:
    """
    Higher level should push the brentq root (quantile) higher and thus typically
    increase the truncation bound locend, so the returned mean should not shrink
    drastically. We compare two levels and just sanity-check ordering.
    """
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    mu_low, var_low = mymath.ph_mean_var_num(p0, subtm, level=0.50)
    mu_high, var_high = mymath.ph_mean_var_num(p0, subtm, level=0.99)

    # Validate
    assert np.isfinite(mu_low) and np.isfinite(mu_high)
    assert mu_high > 0.0 and mu_low > 0.0
    # Both should be in the same ballpark; high level truncation should not reduce mean.
    assert mu_high >= mu_low * 0.95
    assert var_high >= -1e-10 and var_low >= -1e-10

    # Cleanup
    # nothing


def test_ph_mean_var_num_accepts_list_inputs() -> None:
    # Setup
    p0 = [1.0, 0.0]
    subtm = [[-2.0, 2.0], [0.0, -3.0]]

    # Exercise
    mu, var = mymath.ph_mean_var_num(p0, subtm)

    # Validate
    assert isinstance(mu, float)
    assert isinstance(var, float)
    assert mu > 0.0
    assert var >= -1e-10

    # Cleanup
    # nothing


def test_ph_mean_var_num_shapes_mismatch_raises() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.ph_mean_var_num(p0, subtm)

    # Cleanup
    # nothing


# Tests for mymath.approx_gamma_shape and mymath.approx_gamma_scale
def test_approx_gamma_shape_and_scale_positive() -> None:
    # Setup
    mean = 2.5
    var = 1.7

    # Exercise
    shape = mymath.approx_gamma_shape(mean, var)
    scale = mymath.approx_gamma_scale(mean, var)

    # Validate
    assert shape > 0.0
    assert scale > 0.0
    assert np.isfinite(shape)
    assert np.isfinite(scale)

    # Cleanup
    # nothing


def test_approx_gamma_shape_scale_reproduce_mean_and_variance() -> None:
    """
    For Gamma(shape=k, scale=θ):
      mean = k θ
      var  = k θ^2

    Moment-matching should reproduce these exactly.
    """
    # Setup
    mean = 3.2
    var = 5.8

    # Exercise
    k = mymath.approx_gamma_shape(mean, var)
    theta = mymath.approx_gamma_scale(mean, var)

    # Validate
    assert np.isclose(k * theta, mean, rtol=1e-14, atol=0.0)
    assert np.isclose(k * theta * theta, var, rtol=1e-14, atol=0.0)

    # Cleanup
    # nothing


def test_approx_gamma_shape_scale_identity_relation() -> None:
    """
    Algebraic identity from definitions:
      shape = mean^2 / var
      scale = var / mean
    """
    # Setup
    mean = 4.0
    var = 2.0

    # Exercise
    shape = mymath.approx_gamma_shape(mean, var)
    scale = mymath.approx_gamma_scale(mean, var)

    # Validate
    assert np.isclose(shape, mean * mean / var, rtol=1e-12, atol=1e-12)
    assert np.isclose(scale, var / mean, rtol=1e-12, atol=1e-12)

    # Cleanup
    # nothing


def test_approx_gamma_shape_scale_scale_invariance() -> None:
    """
    Scaling a distribution by c:
      mean -> c mean
      var  -> c^2 var

    Shape should be invariant; scale should scale by c.
    """
    # Setup
    mean = 1.5
    var = 0.7
    c = 3.4

    # Exercise
    k1 = mymath.approx_gamma_shape(mean, var)
    theta1 = mymath.approx_gamma_scale(mean, var)

    k2 = mymath.approx_gamma_shape(c * mean, c * c * var)
    theta2 = mymath.approx_gamma_scale(c * mean, c * c * var)

    # Validate
    assert np.isclose(k1, k2, rtol=1e-14, atol=0.0)
    assert np.isclose(theta2, c * theta1, rtol=1e-14, atol=0.0)

    # Cleanup
    # nothing


def test_approx_gamma_shape_monotone_in_variance_for_fixed_mean() -> None:
    """
    For fixed mean:
      larger variance -> smaller shape parameter
    """
    # Setup
    mean = 2.0
    var_small = 0.5
    var_large = 3.0

    # Exercise
    k_small = mymath.approx_gamma_shape(mean, var_small)
    k_large = mymath.approx_gamma_shape(mean, var_large)

    # Validate
    assert k_small > k_large

    # Cleanup
    # nothing


def test_approx_gamma_scale_monotone_in_variance_for_fixed_mean() -> None:
    """
    For fixed mean:
      larger variance -> larger scale parameter
    """
    # Setup
    mean = 2.0
    var_small = 0.5
    var_large = 3.0

    # Exercise
    theta_small = mymath.approx_gamma_scale(mean, var_small)
    theta_large = mymath.approx_gamma_scale(mean, var_large)

    # Validate
    assert theta_large > theta_small

    # Cleanup
    # nothing


def test_approx_gamma_shape_scale_equal_mean_variance_gives_shape_one() -> None:
    """
    If mean == variance, shape == mean and scale == 1.
    (Exponential distribution)
    """
    # Setup
    mean = 2.0
    var = 2.0

    # Exercise
    shape = mymath.approx_gamma_shape(mean, var)
    scale = mymath.approx_gamma_scale(mean, var)

    # Validate
    assert np.isclose(shape, mean, rtol=1e-12, atol=1e-12)
    assert np.isclose(scale, 1.0, rtol=1e-12, atol=1e-12)

    # Cleanup
    # nothing


# Tests for mymath.approx_ph_gamma
def test_approx_ph_gamma_analytic_returns_finite_positive_params() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    mean, var, shape, scale = mymath.approx_ph_gamma(p0, subtm, method="analytic")

    # Validate
    assert np.isfinite(mean) and mean > 0.0
    assert np.isfinite(var) and var >= -1e-10
    assert np.isfinite(shape) and shape > 0.0
    assert np.isfinite(scale) and scale > 0.0

    # Gamma identities should hold for the returned parameters.
    assert np.isclose(shape * scale, mean, rtol=1e-10, atol=0.0)
    assert np.isclose(shape * scale * scale, var, rtol=1e-10, atol=0.0)

    # Cleanup
    # nothing


def test_approx_ph_gamma_numeric_returns_finite_positive_params() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0, 2.0], [0.0, -3.0]], dtype=float)

    # Exercise
    mean, var, shape, scale = mymath.approx_ph_gamma(
        p0, subtm, method="numeric", level=0.90
    )

    # Validate
    assert np.isfinite(mean) and mean > 0.0
    assert np.isfinite(var) and var >= -1e-10
    assert np.isfinite(shape) and shape > 0.0
    assert np.isfinite(scale) and scale > 0.0
    assert np.isclose(shape * scale, mean, rtol=1e-8, atol=0.0)
    assert np.isclose(shape * scale * scale, var, rtol=1e-8, atol=0.0)

    # Cleanup
    # nothing


def test_approx_ph_gamma_invalid_method_defaults_to_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setup
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)

    called = {"num": 0}

    orig_num = mymath.ph_mean_var_num

    def wrapped_num(p0_in, subtm_in, level=0.90):
        called["num"] += 1
        return orig_num(p0_in, subtm_in, level=level)

    monkeypatch.setattr(mymath, "ph_mean_var_num", wrapped_num)

    # Exercise
    _ = mymath.approx_ph_gamma(p0, subtm, method="banana")  # invalid -> numeric

    # Validate
    assert called["num"] == 1

    # Cleanup
    # monkeypatch restores originals


def test_approx_ph_gamma_analytic_matches_ph_mean_and_ph_var() -> None:
    # Setup
    p0 = np.array([0.6, 0.4], dtype=float)
    subtm = np.array([[-2.0, 0.5], [0.0, -1.0]], dtype=float)

    expected_mean = mymath.ph_mean(p0, subtm)
    expected_var = mymath.ph_var(p0, subtm)

    # Exercise
    mean, var, shape, scale = mymath.approx_ph_gamma(p0, subtm, method="analytic")

    # Validate
    assert np.isclose(mean, expected_mean, rtol=1e-12, atol=0.0)
    assert np.isclose(var, expected_var, rtol=1e-12, atol=0.0)
    assert np.isclose(
        shape, expected_mean * expected_mean / expected_var, rtol=1e-10, atol=0.0
    )
    assert np.isclose(scale, expected_var / expected_mean, rtol=1e-10, atol=0.0)

    # Cleanup
    # nothing


def test_approx_ph_gamma_numeric_level_out_of_bounds_still_runs() -> None:
    # Setup
    p0 = np.array([1.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)

    # Exercise
    mean1, var1, shape1, scale1 = mymath.approx_ph_gamma(
        p0, subtm, method="numeric", level=-1.0
    )
    mean2, var2, shape2, scale2 = mymath.approx_ph_gamma(
        p0, subtm, method="numeric", level=2.0
    )

    # Validate
    for v in (mean1, var1, shape1, scale1, mean2, var2, shape2, scale2):
        assert np.isfinite(v)
    assert mean1 > 0.0 and mean2 > 0.0
    assert shape1 > 0.0 and scale1 > 0.0
    assert shape2 > 0.0 and scale2 > 0.0

    # Cleanup
    # nothing


def test_approx_ph_gamma_accepts_list_inputs() -> None:
    # Setup
    p0 = [1.0, 0.0]
    subtm = [[-2.0, 2.0], [0.0, -3.0]]

    # Exercise
    mean, var, shape, scale = mymath.approx_ph_gamma(p0, subtm, method="analytic")

    # Validate
    assert isinstance(mean, float)
    assert isinstance(var, float)
    assert isinstance(shape, float)
    assert isinstance(scale, float)
    assert mean > 0.0
    assert shape > 0.0
    assert scale > 0.0

    # Cleanup
    # nothing


def test_approx_ph_gamma_shapes_mismatch_raises() -> None:
    # Setup
    p0 = np.array([1.0, 0.0], dtype=float)
    subtm = np.array([[-2.0]], dtype=float)  # wrong shape

    # Exercise / Validate
    with pytest.raises(Exception):
        mymath.approx_ph_gamma(p0, subtm, method="analytic")

    # Cleanup
    # nothing


# Tests for mymath.gaussian_pdf
def test_gaussian_pdf_returns_finite_positive() -> None:
    # Setup
    x = 0.0
    mean = 1.0
    std = 2.0

    # Exercise
    out = mymath.gaussian_pdf(x, mean, std)

    # Validate
    assert np.isfinite(out)
    assert out > 0.0

    # Cleanup
    # nothing


def test_gaussian_pdf_peak_at_mean() -> None:
    """
    PDF should attain its maximum at x = mean.
    """
    # Setup
    mean = 1.5
    std = 0.7
    x = np.array([mean - 0.1, mean, mean + 0.1], dtype=float)

    # Exercise
    out = mymath.gaussian_pdf(x, mean, std)

    # Validate
    assert out[1] >= out[0]
    assert out[1] >= out[2]

    # Cleanup
    # nothing


def test_gaussian_pdf_symmetry_about_mean() -> None:
    """
    Gaussian is symmetric: f(mean + d) == f(mean - d)
    """
    # Setup
    mean = 2.0
    std = 1.3
    d = np.array([0.1, 0.5, 1.0], dtype=float)

    # Exercise
    left = mymath.gaussian_pdf(mean - d, mean, std)
    right = mymath.gaussian_pdf(mean + d, mean, std)

    # Validate
    assert np.allclose(left, right, rtol=1e-14, atol=0.0)

    # Cleanup
    # nothing


def test_gaussian_pdf_value_at_mean_matches_known_constant() -> None:
    """
    f(mean) = 1 / (sqrt(2*pi) * std)
    """
    # Setup
    mean = 0.0
    std = 2.5
    expected = 1.0 / (np.sqrt(2.0 * np.pi) * std)

    # Exercise
    out = mymath.gaussian_pdf(mean, mean, std)

    # Validate
    assert np.isclose(out, expected, rtol=0.0, atol=0.0)

    # Cleanup
    # nothing


def test_gaussian_pdf_integrates_to_one() -> None:
    """
    Numerical sanity check: integral over a wide range should be ~1.
    """
    # Setup
    mean = 1.0
    std = 0.8
    xmin = mean - 8.0 * std
    xmax = mean + 8.0 * std
    xs = np.linspace(xmin, xmax, 20001, dtype=float)

    # Exercise
    ys = mymath.gaussian_pdf(xs, mean, std)
    area = np.trapz(ys, xs)

    # Validate
    assert np.isfinite(area)
    assert 0.999 <= area <= 1.001

    # Cleanup
    # nothing


def test_gaussian_pdf_scaling_with_std() -> None:
    """
    Larger std -> lower peak height at the mean.
    """
    # Setup
    mean = 0.0
    std_small = 0.5
    std_large = 2.0

    # Exercise
    peak_small = mymath.gaussian_pdf(mean, mean, std_small)
    peak_large = mymath.gaussian_pdf(mean, mean, std_large)

    # Validate
    assert peak_small > peak_large

    # Cleanup
    # nothing


def test_gaussian_pdf_vector_input_shape_preserved() -> None:
    # Setup
    x = np.linspace(-1.0, 1.0, 11)
    mean = 0.0
    std = 1.0

    # Exercise
    out = mymath.gaussian_pdf(x, mean, std)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape

    # Cleanup
    # nothing


def test_gaussian_pdf_accepts_list_input() -> None:
    # Setup
    x = [-1.0, 0.0, 1.0]
    mean = 0.0
    std = 1.0

    # Exercise
    out = mymath.gaussian_pdf(x, mean, std)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(out > 0.0)

    # Cleanup
    # nothing


# Tests for mymath.log_gaussian_pdf
def test_log_gaussian_pdf_returns_finite() -> None:
    # Setup
    x = 0.0
    mean = 1.0
    std = 2.0

    # Exercise
    out = mymath.log_gaussian_pdf(x, mean, std)

    # Validate
    assert np.isfinite(out)

    # Cleanup
    # nothing


def test_log_gaussian_pdf_peak_at_mean() -> None:
    """
    Log-PDF should attain its maximum at x = mean.
    """
    # Setup
    mean = 1.5
    std = 0.7
    x = np.array([mean - 0.1, mean, mean + 0.1], dtype=float)

    # Exercise
    out = mymath.log_gaussian_pdf(x, mean, std)

    # Validate
    assert out[1] >= out[0]
    assert out[1] >= out[2]

    # Cleanup
    # nothing


def test_log_gaussian_pdf_symmetry_about_mean() -> None:
    """
    log f(mean + d) == log f(mean - d)
    """
    # Setup
    mean = 2.0
    std = 1.3
    d = np.array([0.1, 0.5, 1.0], dtype=float)

    # Exercise
    left = mymath.log_gaussian_pdf(mean - d, mean, std)
    right = mymath.log_gaussian_pdf(mean + d, mean, std)

    # Validate
    assert np.allclose(left, right, rtol=1e-14, atol=0.0)

    # Cleanup
    # nothing


def test_log_gaussian_pdf_value_at_mean_matches_known_constant() -> None:
    """
    log f(mean) = -log(std * sqrt(2*pi))
    """
    # Setup
    mean = 0.0
    std = 2.5
    expected = -np.log(std * np.sqrt(2.0 * np.pi))

    # Exercise
    out = mymath.log_gaussian_pdf(mean, mean, std)

    # Validate
    assert np.isclose(out, expected, rtol=0.0, atol=0.0)

    # Cleanup
    # nothing


def test_log_gaussian_pdf_consistent_with_gaussian_pdf() -> None:
    """
    log_gaussian_pdf(x) == log(gaussian_pdf(x))
    """
    # Setup
    x = np.linspace(-2.0, 2.0, 21)
    mean = 0.5
    std = 1.2

    # Exercise
    log_pdf = mymath.log_gaussian_pdf(x, mean, std)
    pdf = mymath.gaussian_pdf(x, mean, std)

    # Validate
    assert np.allclose(log_pdf, np.log(pdf), rtol=1e-13, atol=0.0)

    # Cleanup
    # nothing


def test_log_gaussian_pdf_vector_input_shape_preserved() -> None:
    # Setup
    x = np.linspace(-1.0, 1.0, 11)
    mean = 0.0
    std = 1.0

    # Exercise
    out = mymath.log_gaussian_pdf(x, mean, std)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape

    # Cleanup
    # nothing


def test_log_gaussian_pdf_accepts_list_input() -> None:
    # Setup
    x = [-1.0, 0.0, 1.0]
    mean = 0.0
    std = 1.0

    # Exercise
    out = mymath.log_gaussian_pdf(x, mean, std)

    # Validate
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))

    # Cleanup
    # nothing
