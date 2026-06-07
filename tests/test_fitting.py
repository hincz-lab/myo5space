#!/usr/bin/env python3
"""
File: test_fitting .py
Author: Joshua Holmes
Email: jbh92@case.edu

Containes tests for the functions in the fitting.py module.
"""


from pathlib import Path
import pytest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import myo5space.fitting as fit
import myo5space.mymath as mm
import scipy.integrate as integ
import scipy.optimize as opt


def test_datadict_keys():
    """
    Test that the datadict dictionary has the expected keys.
    """
    exp_keys = ["dwell 1um", "dwell 100um", "detach 1um", "detach 100um", "vrun 100um"]
    got_keys = fit.datadict.keys()
    assert all([key in exp_keys for key in got_keys])


def test_binsdict_keys():
    """
    Test that the binsdict dictionary has the expected keys.
    """
    exp_keys = ["dwell 1um", "dwell 100um", "detach 1um", "detach 100um"]
    got_keys = fit.binsdict.keys()
    assert all([key in exp_keys for key in got_keys])


def test_make_histdata():
    """
    Test that the make_histdata function returns the appropriate array
    when passed simple arguments.
    """
    centers = [0, 1, 2, 3]
    counts = [1, 2, 3, 4]
    got = fit.make_histdata(centers, counts)
    exp = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    nptest.assert_allclose(exp, got)


@pytest.mark.parametrize(
    "dataset, expsize",
    [("dwell 1um", 30), ("dwell 100um", 20), ("detach 1um", 26), ("detach 100um", 26)],
)
def test_get_expdata(dataset, expsize):
    """
    Test that the get_expdata function returns arrays of counts and bins that
    have the expected sizes.
    """
    got_counts, got_bins = fit.get_expdata(dataset)
    assert got_counts.size == expsize
    assert got_bins.size == expsize + 1


@pytest.mark.mpl_image_compare(hash_library="images/baseline.json")
def test_get_expdata_detach_1um():
    """
    Test that the data returned by the get_expdata function generates the correct
    histogram for 'detach 1um' as compared with hash library.
    """
    counts, bins = fit.get_expdata("detach 1um")
    fig, ax = plt.subplots(1, 1)
    ax.hist(bins[:-1], bins, weights=counts)
    return fig


@pytest.mark.mpl_image_compare(hash_library="images/baseline.json")
def test_get_expdata_detach_100um():
    """
    Test that the data returned by the get_expdata function generates the correct
    histogram for 'detach 100um' as compared with hash library.
    """
    counts, bins = fit.get_expdata("detach 100um")
    fig, ax = plt.subplots(1, 1)
    ax.hist(bins[:-1], bins, weights=counts)
    return fig


@pytest.mark.mpl_image_compare(hash_library="images/baseline.json")
def test_get_expdata_dwell_1um():
    """
    Test that the data returned by the get_expdata function generates the correct
    histogram for 'dwell 1um' as compared with hash library.
    """
    counts, bins = fit.get_expdata("dwell 1um")
    fig, ax = plt.subplots(1, 1)
    ax.hist(bins[:-1], bins, weights=counts)
    return fig


@pytest.mark.mpl_image_compare(hash_library="images/baseline.json")
def test_get_expdata_dwell_100um():
    """
    Test that the data returned by the get_expdata function generates the correct
    histogram for dwell 100um' as compared with hash library.
    """
    counts, bins = fit.get_expdata("dwell 100um")
    fig, ax = plt.subplots(1, 1)
    ax.hist(bins[:-1], bins, weights=counts)
    return fig


def test_vrun_data():
    """
    Test that the experimental vrun mean and SEM are as expected.
    """
    got = fit.datadict.get("vrun 100um")
    exp = (4.27, 0.35)
    assert exp == got


def test_proc_dwell_returns_none_when_no_samples_left():
    dwells = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    pdf = fit.proc_dwell(
        dwells=dwells,
        forces=forces,
        force_cutoff=1.0,  # removes all (forces not > 1)
        time_cutoff=0.0,
        density_cutoff=0.1,
    )
    assert pdf is None


def test_proc_dwell_returns_none_when_density_cutoff_fails():
    dwells = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    forces = np.array(
        [0.0, 2.0, 0.0, 0.0], dtype=np.float64
    )  # only one survives force cutoff

    pdf = fit.proc_dwell(
        dwells=dwells,
        forces=forces,
        force_cutoff=1.0,  # keeps only index 1
        time_cutoff=0.0,
        density_cutoff=0.5,  # requires at least 50% remain; only 25% remain
    )
    assert pdf is None


def test_proc_dwell_returns_callable_when_enough_samples():
    rng = np.random.default_rng(0)
    dwells = rng.lognormal(mean=-2.0, sigma=0.3, size=200).astype(np.float64)
    forces = rng.normal(loc=2.0, scale=0.2, size=200).astype(np.float64)

    pdf = fit.proc_dwell(
        dwells=dwells,
        forces=forces,
        force_cutoff=1.0,
        time_cutoff=0.0,
        density_cutoff=0.1,
    )

    assert callable(pdf)


def test_proc_dwell_pdf_zero_below_time_cutoff():
    rng = np.random.default_rng(1)
    dwells = rng.lognormal(mean=-2.0, sigma=0.3, size=200).astype(np.float64)
    forces = np.full(dwells.shape, 2.0, dtype=np.float64)

    time_cutoff = 0.02
    pdf = fit.proc_dwell(
        dwells, forces, force_cutoff=0.0, time_cutoff=time_cutoff, density_cutoff=0.1
    )
    assert pdf is not None

    assert pdf(time_cutoff - 1e-6) == 0.0


def test_proc_dwell_pdf_vectorized():
    rng = np.random.default_rng(2)
    dwells = rng.lognormal(mean=-2.0, sigma=0.3, size=200).astype(np.float64)
    forces = np.full(dwells.shape, 2.0, dtype=np.float64)

    time_cutoff = 0.01
    pdf = fit.proc_dwell(
        dwells, forces, force_cutoff=0.0, time_cutoff=time_cutoff, density_cutoff=0.1
    )
    assert pdf is not None

    t = np.array(
        [time_cutoff - 1e-6, time_cutoff, time_cutoff + 0.01], dtype=np.float64
    )
    out = pdf(t)

    assert isinstance(out, np.ndarray)
    assert out.shape == t.shape
    assert out[0] == 0.0
    assert out[1] >= 0.0
    assert out[2] >= 0.0


def test_proc_dwell_pdf_normalizes_over_truncated_domain():
    rng = np.random.default_rng(3)
    dwells = rng.lognormal(mean=-2.0, sigma=0.35, size=400).astype(np.float64)
    forces = np.full(dwells.shape, 2.0, dtype=np.float64)

    time_cutoff = 0.01
    pdf = fit.proc_dwell(
        dwells, forces, force_cutoff=0.0, time_cutoff=time_cutoff, density_cutoff=0.1
    )
    assert pdf is not None

    # Integrate from time_cutoff to a large upper bound
    # (lognormal-ish dwells -> mass beyond, say, 2s is negligible here)
    area = integ.quad(lambda x: pdf(x), time_cutoff, 2.0)[0]
    assert area == pytest.approx(1.0, rel=5e-2, abs=5e-2)


def test_proc_dwell_shape_mismatch_raises():
    dwells = np.array([0.1, 0.2], dtype=np.float64)
    forces = np.array([1.0], dtype=np.float64)

    with pytest.raises(AssertionError):
        fit.proc_dwell(
            dwells, forces, force_cutoff=0.0, time_cutoff=0.0, density_cutoff=0.1
        )


def test_make_convolved_pdf_stdnormal():
    """
    Test that the make_convolved_pdf function returns the same as a standard
    normal distribution when passed the appropriate arguments.
    """
    x = 0.0
    pmf = 1.0
    sigma = 1.0
    values = np.linspace(-1, 1, 10)
    got = fit.make_convolved_pdf(x, pmf, sigma)
    got_val = got(values)

    exp_val = (
        1 / (np.sqrt(2 * np.pi) * sigma**2) * np.exp(-1 * values**2 / (2 * sigma**2))
    )
    nptest.assert_allclose(exp_val, got_val)


@pytest.mark.parametrize("force_cutoff", [2.6, 4.6])
def test_proc_vrun(force_cutoff):
    """
    Test that the proc_vrun function returns values above the specified
    cutoff value.
    """
    vruns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    delta = 1.0
    k = 1000.0

    got = fit.proc_vrun(vruns, steps, delta, k, force_cutoff)
    exp = vruns[int(np.round(force_cutoff)) :]
    nptest.assert_allclose(got, exp)


def test_proc_vrun_bad():
    """
    Test that the proc_vrun function returns values above the specified
    cutoff value.
    """
    vruns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    steps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    delta = 1.0
    k = 1000.0
    force_cutoff = 1

    got = fit.proc_vrun(vruns, steps, delta, k, force_cutoff)
    assert got is None


def test_proc_detach_gaussian():
    """
    Test that the proc_detachemnt funciton gives a single Gaussian with the expected
    mean and variance given inputs and cutoffs that result in only a single force
    bin being left.
    """
    steps = np.arange(11)
    delta = 1000.0
    k = 1.0
    beta = 1e-3
    force_cutoff = 10
    density_cutoff = 0

    got = fit.proc_detach(steps, delta, k, beta, force_cutoff, density_cutoff)
    got_area = integ.quad(got, -np.inf, np.inf)[0]
    got_mean = integ.quad(lambda x: x * got(x), -np.inf, np.inf)[0]
    got_var = integ.quad(lambda x: (x - 10.5) ** 2 * got(x), -np.inf, np.inf)[0]

    nptest.assert_allclose(1, got_area)
    nptest.assert_allclose(10.5, got_mean)
    nptest.assert_allclose(1.0, got_var)


def test_proc_detach_none_density():
    """
    Test that the proc_detachment function returns None when it fails the density
    test due to density_cutoff being too high.
    """
    steps = np.arange(11)
    delta = 1000.0
    k = 1.0
    beta = 1e-3
    force_cutoff = 10
    density_cutoff = 1.0

    got = fit.proc_detach(steps, delta, k, beta, force_cutoff, density_cutoff)
    assert got is None


def test_proc_detach_none_force():
    """
    Test that the proc_detachment function returns None when it fails the density
    test due to no samples being left after force_cutoff has been enforced.
    """
    steps = np.arange(11)
    delta = 1000.0
    k = 1.0
    beta = 1e-3
    force_cutoff = 11
    density_cutoff = 0.0

    got = fit.proc_detach(steps, delta, k, beta, force_cutoff, density_cutoff)
    assert got is None


def test_loglh_good():
    """
    Test that the loglh function returns the correct result given simple input
    data. This also tests whether the function ignores bins with zero counts.
    """
    counts = np.array([0, 10])
    probs = np.array([0.9, 0.1])

    got = fit.loglh(counts, probs)
    exp = -23.025850929940454
    nptest.assert_allclose(exp, got)


def test_loglh_bad():
    """
    Test that the loglh function returns -np.inf if one one of the probabilities is
    leq zero.
    """
    counts = np.array([0, 10])
    probs = np.array([1, 0])

    got = fit.loglh(counts, probs)
    exp = -1 * np.inf
    nptest.assert_allclose(exp, got)


def test_vrun_logprior_zerosize():
    """
    Test that the vrun_logprior function returns minus infinity when the input
    array has a size of zero.
    """
    vruns = None
    got = fit.vrun_logprior(vruns)
    exp = -np.inf
    nptest.assert_allclose(exp, got)


def teset_vrun_logprior_max():
    """
    Test that the maximum logprior occurs at the experimental mean vrun.
    """
    res = opt.minimize_scalar(lambda x: -1 * fit.vrun_logprior(x), bracket=(1, 6))
    got = res.x0
    exp = 4.2
    nptest.assert_allclose(exp, got)


REQUIRED_FILES = ("dwell1um.npy", "vrun100um.npy", "detach1um.npy", "detach100um.npy")
REQUIRED_KEYS = ("dwell1um", "vrun100um", "detach1um", "detach100um")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for name in REQUIRED_FILES:
        (tmp_path / name).touch()
    return tmp_path


def test_make_tominaga_paths_returns_dict(data_dir: Path) -> None:
    result = fit.make_tominaga_paths(data_dir)
    assert isinstance(result, dict)


def test_make_tominaga_paths_keys_are_correct(data_dir: Path) -> None:
    result = fit.make_tominaga_paths(data_dir)
    assert set(result.keys()) == set(REQUIRED_KEYS)


def test_make_tominaga_paths_values_are_paths_and_exist(data_dir: Path) -> None:
    result = fit.make_tominaga_paths(data_dir)
    for p in result.values():
        assert isinstance(p, Path)
        assert p.exists()


def test_make_tominaga_paths_points_to_expected_files(data_dir: Path) -> None:
    result = fit.make_tominaga_paths(data_dir)
    assert result["dwell1um"] == data_dir / "dwell1um.npy"
    assert result["vrun100um"] == data_dir / "vrun100um.npy"
    assert result["detach1um"] == data_dir / "detach1um.npy"
    assert result["detach100um"] == data_dir / "detach100um.npy"


@pytest.mark.parametrize("missing_name", REQUIRED_FILES)
def test_make_tominaga_paths_missing_file_raises_assertion(
    tmp_path: Path, missing_name: str
) -> None:
    # Create all but one file
    for name in REQUIRED_FILES:
        if name != missing_name:
            (tmp_path / name).touch()

    with pytest.raises(AssertionError) as excinfo:
        fit.make_tominaga_paths(tmp_path)

    # Make sure the error message names the missing file (regression + usability)
    assert missing_name in str(excinfo.value)


def test_calc_vrun_logprior_shape_and_type():
    vruns = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mean = 2.0
    std = 0.5

    out = fit.calc_vrun_logprior(vruns, mean, std)

    assert isinstance(out, np.ndarray)
    assert out.shape == vruns.shape
    assert out.dtype == np.float64


def test_calc_vrun_logprior_matches_log_gaussian():
    vruns = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    mean = 2.0
    std = 0.3

    expected = mm.log_gaussian_pdf(vruns, mean, std)
    result = fit.calc_vrun_logprior(vruns, mean, std)

    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_calc_vrun_logprior_nan_replaced_with_minus_inf():
    vruns = np.array([np.nan, 1.0], dtype=np.float64)
    mean = 0.0
    std = 1.0

    result = fit.calc_vrun_logprior(vruns, mean, std)

    assert result[0] == -np.inf
    assert np.isfinite(result[1])


def test_calc_vrun_logprior_inf_replaced_with_minus_inf():
    vruns = np.array([np.inf, -np.inf], dtype=np.float64)
    mean = 0.0
    std = 1.0

    result = fit.calc_vrun_logprior(vruns, mean, std)

    assert np.all(result == -np.inf)


def test_calc_vrun_logprior_zero_std_behaviour():
    """
    A zero std should produce NaNs or infs from the pdf,
    which must be converted to -inf.
    """
    vruns = np.array([0.0, 1.0], dtype=np.float64)
    mean = 0.0
    std = 0.0

    result = fit.calc_vrun_logprior(vruns, mean, std)

    assert np.all(result == -np.inf)


def test_load_vrun_vals_shape_and_dtype(tmp_path: Path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_vals(file, delta=36.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == data.shape
    assert out.dtype == np.float64


def test_load_vrun_vals_unit_conversion(tmp_path: Path):
    data = np.array([10.0, 20.0], dtype=np.float64)  # steps/s
    delta = 8.0  # nm
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_vals(file, delta)

    expected = data * (delta / 1000.0)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_load_vrun_vals_does_not_modify_file(tmp_path: Path):
    data = np.array([5.0, 6.0], dtype=np.float64)
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    _ = fit.load_vrun_vals(file, delta=10.0)

    reloaded = np.load(file)
    np.testing.assert_array_equal(reloaded, data)


def test_load_vrun_vals_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError):
        fit.load_vrun_vals(missing, delta=36.0)


def test_load_vrun_vals_empty_array(tmp_path: Path):
    data = np.array([], dtype=np.float64)
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_vals(file, delta=36.0)

    assert out.size == 0
    assert out.dtype == np.float64


def test_load_vrun_logprior_shape_and_dtype(tmp_path: Path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # steps/s
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_logprior(file, delta=36.0, mean=0.0, std=1.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == data.shape
    assert out.dtype == np.float64


def test_load_vrun_logprior_matches_composed_functions(tmp_path: Path):
    data = np.array([10.0, 20.0, 30.0], dtype=np.float64)  # steps/s
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    delta = 8.0
    mean = 1.2
    std = 0.7

    expected_vals = fit.load_vrun_vals(file, delta)
    expected = fit.calc_vrun_logprior(expected_vals, mean, std)

    result = fit.load_vrun_logprior(file, delta, mean, std)

    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_load_vrun_logprior_has_no_nan_or_posinf(tmp_path: Path):
    data = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float64)
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_logprior(file, delta=36.0, mean=0.0, std=1.0)

    assert not np.any(np.isnan(out))
    assert not np.any(np.isposinf(out))
    assert np.any(np.isneginf(out))  # -inf is allowed


def test_load_vrun_logprior_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError):
        fit.load_vrun_logprior(missing, delta=36.0, mean=0.0, std=1.0)


def test_load_vrun_logprior_empty_array(tmp_path: Path):
    data = np.array([], dtype=np.float64)
    file = tmp_path / "vrun.npy"
    np.save(file, data)

    out = fit.load_vrun_logprior(file, delta=36.0, mean=0.0, std=1.0)

    assert out.size == 0
    assert out.dtype == np.float64


def _save(tmp_path: Path, name: str, arr: np.ndarray) -> Path:
    p = tmp_path / name
    np.save(p, arr)
    return p


def test_load_loglhs_loads_and_preserves_keys(tmp_path: Path):
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([0.0, -0.1, -3.0], dtype=np.float64)

    pa = _save(tmp_path, "a.npy", a)
    pb = _save(tmp_path, "b.npy", b)

    paths = {"condA": pa, "condB": pb}
    out = fit.load_loglhs(paths)

    assert set(out.keys()) == set(paths.keys())
    np.testing.assert_array_equal(out["condA"], a)
    np.testing.assert_array_equal(out["condB"], b)


def test_load_loglhs_empty_dict_ok():
    out = fit.load_loglhs({})
    assert out == {}


@pytest.fixture
def tominaga_dir(tmp_path: Path) -> Path:
    # Valid log-likelihood arrays (<=0 and no NaNs)
    _save(tmp_path, "dwell1um.npy", np.array([-1.0, -2.0, 0.0], dtype=np.float64))
    _save(tmp_path, "detach1um.npy", np.array([-0.5, -3.0], dtype=np.float64))
    _save(tmp_path, "detach100um.npy", np.array([-10.0, -1.0], dtype=np.float64))

    # vrun values in steps/s (can include odd values; downstream should sanitize logprior if needed)
    _save(tmp_path, "vrun100um.npy", np.array([10.0, 20.0, 30.0], dtype=np.float64))

    return tmp_path


def test_load_scores_keys_and_types(tominaga_dir: Path):
    out = fit.load_scores(tominaga_dir, delta=8.0, mean=1.2, std=0.7)

    # Should include the three loglh datasets plus vrun outputs
    expected_keys = {
        "dwell1um",
        "detach1um",
        "detach100um",
        "vrun100um_vals",
        "vrun100um",
    }
    assert set(out.keys()) == expected_keys

    for k, v in out.items():
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float64


def test_load_scores_vrun_vals_unit_conversion(tominaga_dir: Path):
    delta = 8.0  # nm
    out = fit.load_scores(tominaga_dir, delta=delta, mean=0.0, std=1.0)

    raw = np.load(tominaga_dir / "vrun100um.npy").astype(np.float64)
    expected_vals = raw * (delta / 1000.0)

    np.testing.assert_allclose(out["vrun100um_vals"], expected_vals, rtol=1e-12)


def test_load_scores_vrun_logprior_matches_composed(tominaga_dir: Path):
    delta = 8.0
    mean = 1.2
    std = 0.7

    out = fit.load_scores(tominaga_dir, delta=delta, mean=mean, std=std)

    vrun_path = tominaga_dir / "vrun100um.npy"
    expected_vals = fit.load_vrun_vals(vrun_path, delta)
    expected_prior = fit.calc_vrun_logprior(expected_vals, mean, std)

    np.testing.assert_allclose(out["vrun100um"], expected_prior, rtol=1e-12)


def test_combine_same_atp_basic_addition():
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    b = np.array([-0.5, -1.5, -2.5], dtype=np.float64)

    out = fit.combine_same_atp(a, b)

    expected = a + b
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_same_atp_does_not_modify_inputs():
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([-3.0, -4.0], dtype=np.float64)

    a_copy = a.copy()
    b_copy = b.copy()

    _ = fit.combine_same_atp(a, b)

    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(b, b_copy)


def test_combine_same_atp_returns_float64():
    a = np.array([-1, -2], dtype=np.int64)
    b = np.array([-3, -4], dtype=np.int64)

    out = fit.combine_same_atp(a, b)

    assert out.dtype == np.float64


def test_combine_same_atp_allows_neg_inf():
    a = np.array([-np.inf, -1.0], dtype=np.float64)
    b = np.array([-2.0, -3.0], dtype=np.float64)

    out = fit.combine_same_atp(a, b)

    assert np.isneginf(out[0])
    assert out[1] == -4.0


def test_combine_same_atp_shape_mismatch_raises():
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([-3.0], dtype=np.float64)

    with pytest.raises(AssertionError):
        fit.combine_same_atp(a, b)


def test_combine_same_atp_empty_arrays():
    a = np.array([], dtype=np.float64)
    b = np.array([], dtype=np.float64)

    out = fit.combine_same_atp(a, b)

    assert out.size == 0
    assert out.dtype == np.float64


def test_combine_2diff_atp_basic_outer_sum():
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([-0.5, -1.5, -2.5], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    expected = np.array(
        [
            [-1.5, -2.5, -3.5],
            [-2.5, -3.5, -4.5],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_2diff_atp_shape():
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    b = np.array([-4.0], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    assert out.shape == (3, 1)


def test_combine_2diff_atp_does_not_modify_inputs():
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([-3.0, -4.0], dtype=np.float64)

    a_copy = a.copy()
    b_copy = b.copy()

    _ = fit.combine_2diff_atp(a, b)

    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(b, b_copy)


def test_combine_2diff_atp_returns_float64():
    a = np.array([-1, -2], dtype=np.int32)
    b = np.array([-3, -4], dtype=np.int32)

    out = fit.combine_2diff_atp(a, b)

    assert out.dtype == np.float64


def test_combine_2diff_atp_allows_neg_inf():
    a = np.array([-np.inf, -1.0], dtype=np.float64)
    b = np.array([-2.0], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    assert np.isneginf(out[0, 0])
    assert out[1, 0] == -3.0


def test_combine_2diff_atp_empty_first():
    a = np.array([], dtype=np.float64)
    b = np.array([-1.0, -2.0], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    assert out.shape == (0, 2)


def test_combine_2diff_atp_empty_second():
    a = np.array([-1.0, -2.0], dtype=np.float64)
    b = np.array([], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    assert out.shape == (2, 0)


def test_combine_2diff_atp_both_empty():
    a = np.array([], dtype=np.float64)
    b = np.array([], dtype=np.float64)

    out = fit.combine_2diff_atp(a, b)

    assert out.shape == (0, 0)


def test_combine_all_scores_basic():
    # 1uM components (same length)
    dwell1um = np.array([-1.0, -2.0], dtype=np.float64)
    detach1um = np.array([-0.1, -0.2], dtype=np.float64)

    # 100uM components (same length)
    vrun100um = np.array([-0.5, -1.5, -2.5], dtype=np.float64)
    detach100um = np.array([-10.0, -11.0, -12.0], dtype=np.float64)

    scores = {
        "dwell1um": dwell1um,
        "detach1um": detach1um,
        "vrun100um": vrun100um,
        "detach100um": detach100um,
    }

    out = fit.combine_all_scores(scores)

    # Expected:
    # score_1um = dwell1um + detach1um -> length 2
    # score_100um = vrun100um + detach100um -> length 3
    # outer sum -> shape (2, 3)
    expected_1um = dwell1um + detach1um
    expected_100um = vrun100um + detach100um
    expected = expected_1um[:, None] + expected_100um[None, :]

    assert out.shape == (2, 3)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_combine_all_scores_does_not_modify_inputs():
    dwell1um = np.array([-1.0, -2.0], dtype=np.float64)
    detach1um = np.array([-0.1, -0.2], dtype=np.float64)
    vrun100um = np.array([-0.5, -1.5], dtype=np.float64)
    detach100um = np.array([-10.0, -11.0], dtype=np.float64)

    scores = {
        "dwell1um": dwell1um,
        "detach1um": detach1um,
        "vrun100um": vrun100um,
        "detach100um": detach100um,
    }

    snapshots = {k: v.copy() for k, v in scores.items()}

    _ = fit.combine_all_scores(scores)

    for k, v in scores.items():
        np.testing.assert_array_equal(v, snapshots[k])


def test_combine_all_scores_shape_mismatch_propagates():
    # dwell1um and detach1um must match shape for combine_same_atp
    scores = {
        "dwell1um": np.array([-1.0, -2.0], dtype=np.float64),
        "detach1um": np.array([-0.1], dtype=np.float64),  # mismatch
        "vrun100um": np.array([-0.5], dtype=np.float64),
        "detach100um": np.array([-10.0], dtype=np.float64),
    }

    with pytest.raises(AssertionError):
        fit.combine_all_scores(scores)
