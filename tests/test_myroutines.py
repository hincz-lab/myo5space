#!/usr/bin/env python3
"""
File: test_myroutines.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains tests for the functions in the myroutines.py file.
"""
import myo5space.myroutines as myroutines
import pytest
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable
import pickle
import numpy as np
import numpy.testing as nptest


# Tests for myroutines.load_npz_to_dict()
def _make_npz(tmp_path: Path, name: str, **arrays: Any) -> Path:
    """
    Setup helper: write an npz file and return its path.
    """
    p = tmp_path / name
    np.savez(p, **arrays)
    return p


def test_returns_dict_with_same_keys_and_values(tmp_path: Path) -> None:
    # Setup
    p = _make_npz(
        tmp_path,
        "example.npz",
        a=np.array([1, 2, 3]),
        b=np.arange(6).reshape(2, 3),
        c=np.array("hello"),
    )
    # Exercise
    out = myroutines.load_npz_to_dict(p)
    # Validate
    assert isinstance(out, dict)
    assert set(out.keys()) == {"a", "b", "c"}
    np.testing.assert_array_equal(out["a"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(out["b"], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(out["c"], np.array("hello"))
    # Cleanup - None


def test_accepts_str_path(tmp_path: Path) -> None:
    # Setup
    p = _make_npz(tmp_path, "as_str.npz", x=np.array([10, 20]))
    # Exercise
    out = myroutines.load_npz_to_dict(str(p))
    # Validate
    assert set(out.keys()) == {"x"}
    np.testing.assert_array_equal(out["x"], np.array([10, 20]))
    # Cleanup - None


def test_accepts_pathlib_path(tmp_path: Path) -> None:
    # Setup
    p = _make_npz(tmp_path, "as_path.npz", x=np.array([1.5, 2.5]))

    # Exercise
    out = myroutines.load_npz_to_dict(p)

    # Validate
    assert set(out.keys()) == {"x"}
    np.testing.assert_array_equal(out["x"], np.array([1.5, 2.5]))
    # Cleanup - None


def test_zero_length_array_round_trips(tmp_path: Path) -> None:
    # Setup
    p = _make_npz(tmp_path, "emptyish.npz", empty=np.array([]))
    # Exercise
    out = myroutines.load_npz_to_dict(p)
    # Validate
    assert set(out.keys()) == {"empty"}
    assert isinstance(out["empty"], np.ndarray)
    assert out["empty"].size == 0
    # Cleanup - None


def test_raises_file_not_found(tmp_path: Path) -> None:
    # Setup
    missing = tmp_path / "does_not_exist.npz"
    # Exercise / Validate
    with pytest.raises(FileNotFoundError):
        myroutines.load_npz_to_dict(missing)
    # Cleanup - None


def test_propagates_bad_npz_error(tmp_path: Path) -> None:
    # Setup
    p = tmp_path / "not_really.npz"
    p.write_bytes(b"this is not a zip file")
    # Exercise / Validate
    # Exception type varies across NumPy/Python versions (zipfile-related), so keep broad.
    with pytest.raises(Exception):
        myroutines.load_npz_to_dict(p)
    # Cleanup - None


def test_uses_numpy_load_as_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    class FakeNPZ:
        def __init__(self) -> None:
            self._data = {"k1": np.array([1]), "k2": np.array([2, 3])}

        def __iter__(self):
            return iter(self._data.keys())

        def __getitem__(self, k: str):
            return self._data[k]

    class FakeContext:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False
            self.npz = FakeNPZ()

        def __enter__(self):
            self.entered = True
            return self.npz

        def __exit__(self, exc_type, exc, tb):
            self.exited = True
            return False

    ctx = FakeContext()
    called = {"path": None}

    def fake_load(path):
        called["path"] = path
        return ctx

    monkeypatch.setattr(np, "load", fake_load)

    # Exercise
    out = myroutines.load_npz_to_dict("whatever.npz")

    # Validate
    assert called["path"] == Path("whatever.npz")
    assert ctx.entered is True
    assert ctx.exited is True
    assert set(out.keys()) == {"k1", "k2"}
    np.testing.assert_array_equal(out["k1"], np.array([1]))
    np.testing.assert_array_equal(out["k2"], np.array([2, 3]))

    # Cleanup


# Tests for myroutines.divide_equally
@pytest.mark.parametrize(
    "number,num_bins,expected",
    [
        (10, 5, [2, 2, 2, 2, 2]),
        (10, 3, [4, 3, 3]),
        (7, 3, [3, 2, 2]),
        (5, 10, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        (1, 1, [1]),
        (1, 5, [1, 0, 0, 0, 0]),
    ],
)
def test_divide_equally_examples(number: int, num_bins: int, expected: list[int]) -> None:
    # Setup - None
    # Exercise
    result = myroutines.divide_equally(number, num_bins)
    # Validate
    assert result == expected
    # Cleanup - None


def test_sum_is_preserved() -> None:
    # Setup
    number = 37
    num_bins = 8
    # Exercise
    result = myroutines.divide_equally(number, num_bins)
    # Validate
    assert len(result) == num_bins
    assert sum(result) == number
    # Cleanup - None


def test_bins_differ_by_at_most_one() -> None:
    # Setup
    number = 101
    num_bins = 17
    # Exercise
    result = myroutines.divide_equally(number, num_bins)
    # Validate
    assert max(result) - min(result) <= 1
    # Cleanup - None


def test_remainder_distributed_to_first_bins() -> None:
    # Setup
    number = 10
    num_bins = 4
    # Exercise
    result = myroutines.divide_equally(number, num_bins)
    # Validate
    assert result == [3, 3, 2, 2]  # base=2, remainder=2
    # Cleanup - None


@pytest.mark.parametrize(
    "number,num_bins",
    [
        (0, 5),
        (-1, 5),
        (5, 0),
        (5, -3),
        (0, 0),
    ],
)
def test_invalid_values_raise_value_error(number: int, num_bins: int) -> None:
    # Setup - None
    # Exercise / Validate
    with pytest.raises(ValueError):
        myroutines.divide_equally(number, num_bins)
    # Cleanup - None


@pytest.mark.parametrize(
    "number,num_bins",
    [
        (10.5, 3),
        ("10", 3),
        (10, 3.5),
        (10, "3"),
        (None, 3),
    ],
)
def test_non_integer_inputs_raise_value_error(number, num_bins) -> None:
    # Setup - None
    # Exercise / Validate
    with pytest.raises(ValueError):
        myroutines.divide_equally(number, num_bins)
    # Cleanup - None


# Tests for myroutines.mp_worker
def _sim_add(trials: int, a: int, b: int) -> int:
    """
    Top-level function so it's pickleable across processes.
    """
    return trials + a + b


def _sim_pack(trials: int, *params: object) -> tuple[int, tuple[object, ...]]:
    """
    Returns exactly what it received (for validating argument passing).
    """
    return trials, params


def _sim_raises(trials: int, msg: str) -> None:
    raise RuntimeError(f"{trials}:{msg}")


def test_calls_func_with_trials_and_params_and_returns_result() -> None:
    # Setup
    args = (5, _sim_add, (2, 3))
    # Exercise
    out = myroutines.mp_worker(args)
    # Validate
    assert out == 10
    # Cleanup - None


def test_unpacks_params_correctly() -> None:
    # Setup
    args = (7, _sim_pack, ("alpha", 2, 3.0))
    # Exercise
    out = myroutines.mp_worker(args)
    # Validate
    assert out == (7, ("alpha", 2, 3.0))
    # Cleanup


def test_empty_params_supported() -> None:
    # Setup
    args = (9, _sim_pack, ())
    # Exercise
    out = myroutines.mp_worker(args)
    # Validate
    assert out == (9, ())
    # Cleanup - None


def test_propagates_exceptions_from_func() -> None:
    # Setup
    args = (3, _sim_raises, ("boom",))
    # Exercise / Validate
    with pytest.raises(RuntimeError, match=r"^3:boom$"):
        myroutines.mp_worker(args)
    # Cleanup - None


def test_args_tuple_shape_errors_propagate() -> None:
    # Setup
    bad_args = (1, _sim_add)  # missing params element
    # Exercise / Validate
    with pytest.raises(ValueError):
        # Unpacking will raise ValueError: not enough values to unpack
        myroutines.mp_worker(bad_args)  # type: ignore[arg-type]
    # Cleanup - None


def test_non_callable_func_errors_propagate() -> None:
    # Setup
    args = (1, 123, ())  # func is not callable
    # Exercise / Validate
    with pytest.raises(TypeError):
        myroutines.mp_worker(args)  # type: ignore[arg-type]
    # Cleanup - None


def test_pickle_roundtrip_of_args_tuple() -> None:
    """
    This is a lightweight proxy for 'works under multiprocessing spawn':
    if the args tuple (including func) pickles, it's suitable for ProcessPool.
    """
    # Setup
    args = (4, _sim_add, (10, 20))
    # Exercise
    blob = pickle.dumps(args)
    args2 = pickle.loads(blob)
    out = myroutines.mp_worker(args2)
    # Validate
    assert out == 34
    # Cleanup - None


# Tests for myroutines.mp_driver
class _FakePool:
    """
    A minimal stand-in for multiprocessing.Pool that:
      - supports context manager protocol
      - implements imap_unordered by calling the worker in-process
    """

    def __init__(self, processes: int) -> None:
        self.processes = processes
        self.closed = False

    def __enter__(self) -> "_FakePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.closed = True
        return False  # do not suppress exceptions

    def imap_unordered(
        self,
        worker: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int = 1,
    ):
        # simulate unordered behavior by just yielding in input order
        # (order doesn't matter for these tests because we validate packed output)
        for item in iterable:
            yield worker(item)


def _make_fake_pool(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """
    Monkeypatches module.mp.Pool to use _FakePool, keeping tests fast and deterministic.
    """
    monkeypatch.setattr(module.mp, "Pool", lambda processes: _FakePool(processes))


def test_invalid_trials_raise_value_error() -> None:
    # Setup
    func = lambda t, *_: np.zeros((1, t))
    out_shape = (1, 1)

    # Exercise / Validate
    with pytest.raises(ValueError, match="trials must be a positive int"):
        myroutines.mp_driver(trials=0, func=func, out_shape=out_shape, processes=1, params=())

    # Cleanup - None


def test_invalid_processes_raise_value_error() -> None:
    # Setup
    func = lambda t, *_: np.zeros((1, t))
    out_shape = (1, 1)

    # Exercise / Validate
    with pytest.raises(ValueError, match="processes must be a positive int"):
        myroutines.mp_driver(trials=1, func=func, out_shape=out_shape, processes=0, params=())

    # Cleanup - None


def test_basic_assembly_with_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    # Import module to patch its mp.Pool reference (not multiprocessing globally).
    import myo5space.myroutines as m

    _make_fake_pool(monkeypatch, m)

    def sim(trials_for_job: int, offset: float) -> np.ndarray:
        # rows=2, cols=trials_for_job
        return np.vstack(
            [
                np.arange(trials_for_job, dtype=float) + offset,
                np.arange(trials_for_job, dtype=float) + offset + 100.0,
            ]
        )

    trials = 5
    processes = 1
    params = (10.0,)
    out_shape = (2, trials)

    # Exercise
    out = m.mp_driver(trials=trials, func=sim, out_shape=out_shape, processes=processes, params=params)

    # Validate
    expected = sim(trials, *params)
    np.testing.assert_array_equal(out, expected)

    # Cleanup - None


def test_assembly_multiple_processes_preserves_total_width(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    import myo5space.myroutines as m

    _make_fake_pool(monkeypatch, m)

    def sim(trials_for_job: int, tag: int) -> np.ndarray:
        # Make each job's output content depend on trials_for_job so we can validate packing.
        # 1 row, cols=trials_for_job
        return np.full((1, trials_for_job), fill_value=float(tag + trials_for_job))

    trials = 10
    processes = 3  # divide_equally should produce [4,3,3]
    params = (7,)
    out_shape = (1, trials)

    # Exercise
    out = m.mp_driver(trials=trials, func=sim, out_shape=out_shape, processes=processes, params=params)

    # Validate
    assert out.shape == out_shape
    # Expect three contiguous blocks sized 4,3,3 (in some order of completion).
    # Our FakePool yields in input order, so it will be [4,3,3] specifically.
    expected = np.concatenate(
        [
            np.full((1, 4), 7 + 4, dtype=float),
            np.full((1, 3), 7 + 3, dtype=float),
            np.full((1, 3), 7 + 3, dtype=float),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(out, expected)

    # Cleanup - None


def test_skips_zero_trial_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    import myo5space.myroutines as m

    _make_fake_pool(monkeypatch, m)

    calls: list[int] = []

    def sim(trials_for_job: int) -> np.ndarray:
        calls.append(trials_for_job)
        return np.zeros((1, trials_for_job))

    trials = 2
    processes = 5  # divide_equally should include zeros, e.g. [1,1,0,0,0]
    out_shape = (1, trials)

    # Exercise
    out = m.mp_driver(trials=trials, func=sim, out_shape=out_shape, processes=processes, params=())

    # Validate
    assert out.shape == out_shape
    assert calls == [1, 1]  # zero-trial jobs should not have been scheduled

    # Cleanup - None


def test_propagates_worker_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup
    import myo5space.myroutines as m

    _make_fake_pool(monkeypatch, m)

    def sim(trials_for_job: int) -> np.ndarray:
        raise RuntimeError(f"boom:{trials_for_job}")

    trials = 5
    processes = 2
    out_shape = (1, trials)

    # Exercise / Validate
    with pytest.raises(RuntimeError, match=r"boom:"):
        m.mp_driver(trials=trials, func=sim, out_shape=out_shape, processes=processes, params=())

    # Cleanup - None


def test_out_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If out_shape doesn't have enough columns to hold all concatenated results,
    NumPy should raise during slice assignment.
    """
    # Setup
    import myo5space.myroutines as m

    _make_fake_pool(monkeypatch, m)

    def sim(trials_for_job: int) -> np.ndarray:
        return np.ones((2, trials_for_job))

    trials = 6
    processes = 2
    params: tuple[Any, ...] = ()
    out_shape = (2, trials - 1)  # too small by 1 column

    # Exercise / Validate
    with pytest.raises(ValueError):
        m.mp_driver(trials=trials, func=sim, out_shape=out_shape, processes=processes, params=params)

    # Cleanup - None
