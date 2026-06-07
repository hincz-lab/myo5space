import numpy as np
import pytest

import myo5space.hpceval as hpceval


def test_make_kd1kd2_params_returns_dict():
    res = hpceval.make_kd1kd2_params()

    assert isinstance(res, dict)
    assert set(res.keys()) == {"kd1", "kd2"}


def test_make_kd1kd2_params_values_are_numpy_arrays():
    res = hpceval.make_kd1kd2_params()

    assert isinstance(res["kd1"], np.ndarray)
    assert isinstance(res["kd2"], np.ndarray)


def test_make_kd1kd2_params_array_sizes():
    res = hpceval.make_kd1kd2_params()

    assert res["kd1"].shape == (200,)
    assert res["kd2"].shape == (200,)


def test_make_kd1kd2_params_logspace_bounds():
    res = hpceval.make_kd1kd2_params()

    assert np.isclose(res["kd1"][0], 1e-1)
    assert np.isclose(res["kd1"][-1], 1e4)

    assert np.isclose(res["kd2"][0], 1e-1)
    assert np.isclose(res["kd2"][-1], 1e4)


def test_make_kd1kd2_params_kd1_kd2_identical():
    res = hpceval.make_kd1kd2_params()

    np.testing.assert_allclose(res["kd1"], res["kd2"])


def test_make_kd1kd2_params_monotonic():
    res = hpceval.make_kd1kd2_params()

    assert np.all(np.diff(res["kd1"]) > 0)
    assert np.all(np.diff(res["kd2"]) > 0)


def test_make_kd1kd2nutheta_params_returns_dict():
    res = hpceval.make_kd1kd2nutheta_params()

    assert isinstance(res, dict)
    assert set(res.keys()) == {"kd1", "kd2", "nu_c", "theta_c"}


def test_make_kd1kd2nutheta_params_values_are_numpy_arrays():
    res = hpceval.make_kd1kd2nutheta_params()

    assert isinstance(res["kd1"], np.ndarray)
    assert isinstance(res["kd2"], np.ndarray)
    assert isinstance(res["nu_c"], np.ndarray)
    assert isinstance(res["theta_c"], np.ndarray)


def test_make_kd1kd2nutheta_params_array_sizes():
    res = hpceval.make_kd1kd2nutheta_params()

    assert res["kd1"].shape == (30,)
    assert res["kd2"].shape == (30,)
    assert res["nu_c"].shape == (30,)
    assert res["theta_c"].shape == (30,)


def test_make_kd1kd2nutheta_params_kd_logspace_bounds():
    res = hpceval.make_kd1kd2nutheta_params()

    assert np.isclose(res["kd1"][0], 1e-1)
    assert np.isclose(res["kd1"][-1], 1e4)

    assert np.isclose(res["kd2"][0], 1e-1)
    assert np.isclose(res["kd2"][-1], 1e4)


def test_make_kd1kd2nutheta_params_kd1_kd2_identical():
    res = hpceval.make_kd1kd2nutheta_params()

    np.testing.assert_allclose(res["kd1"], res["kd2"])


def test_make_kd1kd2nutheta_params_nu_c_bounds():
    res = hpceval.make_kd1kd2nutheta_params()

    assert np.isclose(res["nu_c"][0], 50.0)
    assert np.isclose(res["nu_c"][-1], 1000.0)


def test_make_kd1kd2nutheta_params_theta_c_bounds():
    res = hpceval.make_kd1kd2nutheta_params()

    assert np.isclose(res["theta_c"][0], np.pi / 6)
    assert np.isclose(res["theta_c"][-1], 2 * np.pi / 3)


def test_make_kd1kd2nutheta_params_monotonic():
    res = hpceval.make_kd1kd2nutheta_params()

    assert np.all(np.diff(res["kd1"]) > 0)
    assert np.all(np.diff(res["kd2"]) > 0)
    assert np.all(np.diff(res["nu_c"]) > 0)
    assert np.all(np.diff(res["theta_c"]) > 0)


def test_make_jobs_returns_list():
    param_grids = {"a": [1, 2], "b": [10, 20, 30]}
    jobs = hpceval.make_jobs(param_grids)

    assert isinstance(jobs, list)


def test_make_jobs_length_matches_cartesian_product():
    param_grids = {"a": [1, 2], "b": [10, 20, 30]}
    jobs = hpceval.make_jobs(param_grids)

    assert len(jobs) == 6


def test_make_jobs_elements_are_index_and_params_dict():
    param_grids = {"a": [1, 2], "b": [10, 20, 30]}
    jobs = hpceval.make_jobs(param_grids)

    assert jobs, "Expected non-empty jobs list"

    ind, params = jobs[0]
    assert isinstance(ind, tuple)
    assert all(isinstance(x, int) for x in ind)
    assert isinstance(params, dict)
    assert set(params.keys()) == {"a", "b"}


def test_make_jobs_index_shapes_match_number_of_grids():
    param_grids = {"a": [1, 2], "b": [10, 20, 30], "c": [7]}
    jobs = hpceval.make_jobs(param_grids)

    for ind, _params in jobs:
        assert len(ind) == 3


def test_make_jobs_index_ranges_are_valid():
    param_grids = {"a": [1, 2], "b": [10, 20, 30], "c": [7, 8]}
    jobs = hpceval.make_jobs(param_grids)

    lengths = {k: len(v) for k, v in param_grids.items()}

    for ind, _params in jobs:
        assert 0 <= ind[0] < lengths["a"]
        assert 0 <= ind[1] < lengths["b"]
        assert 0 <= ind[2] < lengths["c"]


def test_make_jobs_params_values_come_from_grids():
    param_grids = {"a": [1, 2], "b": [10, 20, 30], "c": [7, 8]}
    jobs = hpceval.make_jobs(param_grids)

    for _ind, params in jobs:
        assert params["a"] in param_grids["a"]
        assert params["b"] in param_grids["b"]
        assert params["c"] in param_grids["c"]


def test_make_jobs_first_and_last_job_match_expected_ordering():
    # This test checks the ordering implied by product(*grids) and
    # the corresponding index product(*(range(len(g)) ...)).
    # With grids in insertion order: a, b.
    param_grids = {"a": [1, 2], "b": [10, 20, 30]}
    jobs = hpceval.make_jobs(param_grids)

    assert jobs[0] == ((0, 0), {"a": 1, "b": 10})
    assert jobs[-1] == ((1, 2), {"a": 2, "b": 30})


def test_make_jobs_with_numpy_arrays():
    param_grids = {
        "kd1": np.array([0.1, 1.0, 10.0]),
        "kd2": np.array([5.0, 50.0]),
    }
    jobs = hpceval.make_jobs(param_grids)

    assert len(jobs) == 3 * 2
    # Spot-check a couple of entries
    assert jobs[0] == ((0, 0), {"kd1": 0.1, "kd2": 5.0})
    assert jobs[-1] == ((2, 1), {"kd1": 10.0, "kd2": 50.0})


def test_make_jobs_empty_grids_returns_empty_list():
    param_grids = {"a": [], "b": [1, 2]}
    jobs = hpceval.make_jobs(param_grids)

    assert jobs == []


class DummyPool:
    """
    Stand-in for multiprocessing.Pool that runs everything in-process.
    Captures calls to close/join for assertions.
    """
    def __init__(self, threads):
        self.threads = threads
        self.closed = False
        self.joined = False
        self.imap_calls = []

    def imap_unordered(self, worker, jobs, chunksize=1):
        self.imap_calls.append({"worker": worker, "jobs": jobs, "chunksize": chunksize})
        for job in jobs:
            yield worker(job)

    def close(self):
        self.closed = True

    def join(self):
        self.joined = True


def test_run_jobs_returns_numpy_array_with_shape(monkeypatch):
    # Patch mp.cpu_count and mp.Pool inside hpceval
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 4)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: DummyPool(threads))

    jobs = [((0,), 2.0), ((1,), 4.0), ((2,), 6.0)]

    def worker(job):
        # Identity: job already is (ind, value)
        return job

    out = hpceval.run_jobs(jobs=jobs, worker=worker, outshape=(3,))

    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_run_jobs_populates_result_array_correctly(monkeypatch):
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 2)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: DummyPool(threads))

    jobs = [((0, 0), 1.5), ((1, 2), -3.0), ((2, 1), 9.0)]

    def worker(job):
        return job  # (ind, value)

    out = hpceval.run_jobs(jobs=jobs, worker=worker, outshape=(3, 3))

    expected = np.zeros((3, 3))
    expected[(0, 0)] = 1.5
    expected[(1, 2)] = -3.0
    expected[(2, 1)] = 9.0

    np.testing.assert_allclose(out, expected)


def test_run_jobs_uses_cpu_count_and_chunksize_100(monkeypatch):
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 8)

    pool = DummyPool(threads=8)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: pool)

    jobs = [((0,), 1.0), ((1,), 2.0)]

    def worker(job):
        return job

    _out = hpceval.run_jobs(jobs=jobs, worker=worker, outshape=(2,))

    assert pool.threads == 8
    assert pool.imap_calls, "Expected imap_unordered to be called"
    assert pool.imap_calls[0]["chunksize"] == 100


def test_run_jobs_closes_and_joins_pool(monkeypatch):
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 1)

    pool = DummyPool(threads=1)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: pool)

    jobs = [((0,), 7.0)]

    def worker(job):
        return job

    _out = hpceval.run_jobs(jobs=jobs, worker=worker, outshape=(1,))

    assert pool.closed is True
    assert pool.joined is True


def test_run_jobs_progress_prints_every_1000(monkeypatch, capsys):
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 1)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: DummyPool(threads))

    # Need res.size >= 1000 for the printed percentage to be meaningful.
    # outshape=(1000,) makes res.size == 1000, so at count==1000 we expect one print.
    jobs = [((i,), float(i)) for i in range(1000)]

    def worker(job):
        return job

    out = hpceval.run_jobs(jobs=jobs, worker=worker, outshape=(1000,))
    captured = capsys.readouterr()

    # Should print exactly once at count == 1000
    # It prints: 100 * count / res.size == 100.0
    # print() adds newline.
    assert captured.out.strip() == "100.0"
    # Sanity: output array is filled
    assert out[0] == 0.0
    assert out[-1] == 999.0


def test_run_jobs_empty_jobs_returns_zeros(monkeypatch):
    monkeypatch.setattr(hpceval.mp, "cpu_count", lambda: 1)
    monkeypatch.setattr(hpceval.mp, "Pool", lambda threads: DummyPool(threads))

    def worker(job):
        return job

    out = hpceval.run_jobs(jobs=[], worker=worker, outshape=(2, 3))

    np.testing.assert_allclose(out, np.zeros((2, 3)))