#!/usr/bin/env python3
"""
File: fitting.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains code for fitting a parameterized myosin motor to experimental data from
Tominaga2003.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional
from pathlib import Path
import scipy.integrate as integ
import scipy.stats as stats
import myo5space.mymath as mm
import myo5space.hpceval as hpceval


_detach_1um_data = np.array(
    [
        [0.198275862068965, 9.67180883242577],
        [0.295977011494252, 63.6834543254688],
        [0.399425287356321, 69.0146702964306],
        [0.499999999999999, 52.3045220810647],
        [0.600574712643678, 38.2259528130671],
        [0.701149425287356, 35.9894888082274],
        [0.795977011494253, 16.9729280096792],
        [0.902298850574713, 21.6481397459163],
        [1.000000000000000, 13.8176799758013],
        [1.10057471264368, 10.9233212341196],
        [1.19540229885058, 8.35412885662413],
        [1.29885057471264, 13.3563974591649],
        [1.39367816091954, 4.20825771324849],
        [1.59770114942529, 6.64511494252861],
        [1.89942528735632, 3.88309134906217],
    ]
)

_detach_100um_data = np.array(
    [
        [0.298850574712643, 79.4748185117966],
        [0.39655172413793, 118.683832425892],
        [0.50287356321839, 104.280096793708],
        [0.594827586206895, 85.590592861464],
        [0.701149425287356, 45.5289624924378],
        [0.801724137931034, 48.8846037507561],
        [0.899425287356322, 23.9488808227464],
        [1.000000000000000, 8.88346944948557],
        [1.10057471264368, 8.94963702359325],
        [1.19540229885058, 4.40676043557153],
        [1.50287356321839, 5.26693889897149],
    ]
)

_dwell_1um_data = np.array(
    [
        [0.183447749809306, 18.9956775997966],
        [0.337528604118995, 29.1482329010933],
        [0.511314518179508, 22.1090770404272],
        [0.706458174421561, 8.95245359776253],
        [0.927790490719554, 12.8184591914569],
        [1.12954487668447, 13.9588100686499],
        [1.27243834223239, 6.83956267480295],
        [1.49834731756929, 3.68039664378335],
        [1.6675565725909, 3.66641240783116],
        [1.85214848715993, 3.65115687770152],
        [2.07983727434528, 1.64886854818205],
        [2.30917874396135, 0.720823798627052],
        [2.52453597762522, 0.703025680142403],
        [2.70912789219425, 0.687770150012739],
        [2.89677091278922, 2.65573353674048],
        [3.07831172133232, 0.657259089753381],
        [3.30778032036613, -0.188151538265927],
        [3.47698957538774, -0.202135774218092],
        [3.70874650394101, 0.439867785405582],
        [3.86168319349097, -0.151284007119244],
        [4.09674548690567, 2.63920671243332],
        [4.3232901093313, -0.106788710907679],
        [4.58479532163743, -0.128400711924641],
        [4.89346554792779, 0.507246376811665],
        [5.06140350877193, 0.333079074497746],
        [5.27790490719553, 1.39282990083915],
        [5.47787948131198, 1.37630307653201],
    ]
)

_dwell_100um_data = np.array(
    [
        [0.024499258160238, 63.5163204747775],
        [0.031179525222552, 23.2477744807122],
        [0.040274480712166, 14.2091988130564],
        [0.050597181008902, 14.3827893175075],
        [0.060758902077151, 6.20771513353122],
        [0.070905786350149, 4.5074183976262],
        [0.080577151335312, 2.52373887240361],
        [0.090272997032641, 1.25667655786359],
        [0.100902448071217, 0.709198813056446],
        [0.110985163204748, 2.4673590504452],
        [0.12021884272997, 1.48961424332356],
        [0.130401335311573, 0.649851632047614],
        [0.140841988130564, 1.54896142433248],
    ]
)

_vrun_100um_data = (4.27, 0.35)  # (Mean, SEM), um/s

datadict = {
    "dwell 1um": _dwell_1um_data,
    "dwell 100um": _dwell_100um_data,
    "detach 1um": _detach_1um_data,
    "detach 100um": _detach_100um_data,
    "vrun 100um": _vrun_100um_data,
}

binsdict = {
    "detach 1um": np.linspace(-0.05, 2.55, 27),
    "dwell 1um": np.linspace(0, 6, 31),
    "dwell 100um": np.linspace(0, 0.2, 21),
    "detach 100um": np.linspace(-0.05, 2.55, 27),
}


def make_histdata(
    centers: NDArray[np.float64], counts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Generate an array of values used to generate a histogram.

    :param centers: An (N,) shaped array of values to to generate.
    :param counts: An (N,) shaped array where the corresponding element in centers
        is represented counts number of times.

    :return hist_data: An 1D array of data that can be passed to np.histogram.
    """
    hist_data = np.array([])
    for ce, co in zip(centers, counts):
        width_array = ce * np.ones(co)
        hist_data = np.concatenate((hist_data, width_array))
    return hist_data


def get_expdata(dataset: str) -> NDArray[np.float64]:
    """
    Get a histogram for the dwell-time and force-at-detachment data reported in
    Tominaga2003. Specific datasets available include those shown in figure 5B
    (force-at-detachment at 1uM and 100uM ATP) and figure 8B (dwell-times at 1uM
    and 100uM ATP).

    :param dataset: A key for the dataset to retrieve. Options include:
        'detach 1uM':   Figure 5B, 1uM ATP.
        'detach 100uM': Figure 5B, 100uM ATP
        'dwell 1uM':    Figure 8B, 1uM ATP
        'dwell 100uM':  Figure 8B, 100uM ATP

    :return counts, bins: The counts and bin edges for the histogram.
    """
    data = datadict.get(dataset)
    centers = data[:, 0]
    counts = np.round(data[:, 1]).astype(int)
    histdata = make_histdata(centers, counts)
    bins = binsdict.get(dataset)
    return np.histogram(histdata, bins=bins)


def proc_dwell(
    dwells: NDArray[np.float64],
    forces: NDArray[np.float64],
    force_cutoff: float,
    time_cutoff: float,
    density_cutoff: float,
) -> Optional[Callable[[NDArray[np.float64] | float], NDArray[np.float64] | float]]:
    """
    Process dwell times to simulate the processing done by Tominaga2003.

    Keeps dwell times that occur above `force_cutoff` (thermal-noise filtering)
    and longer than `time_cutoff` (instrument rise-time filtering).

    Returns a callable PDF from the remaining samples using a Gaussian kernel KDE.
    Returns None if the remaining samples fail the density cutoff.

    :param dwells: (N,) array of dwell times.
    :param forces: (N,) array of forces at which dwell times were measured.
    :param force_cutoff: Only keep dwells where force > force_cutoff (pN).
    :param time_cutoff: Only keep dwells where dwell > time_cutoff (s).
    :param density_cutoff: Minimum remaining fraction required to return a PDF.
    :return: Callable PDF or None.
    """
    dwells = np.asarray(dwells, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)

    assert dwells.shape == forces.shape, "dwells and forces must have the same shape."
    if dwells.size == 0:
        return None

    newdwells = dwells[forces > force_cutoff]
    newdwells = newdwells[newdwells > time_cutoff]

    if newdwells.size / dwells.size < density_cutoff:
        return None

    _pdf = stats.gaussian_kde(newdwells)

    # Mass below time_cutoff (normalization for truncation)
    lt_timecutoff = float(integ.quad(lambda x: float(_pdf(x)), -np.inf, time_cutoff)[0])
    norm = max(1.0 - lt_timecutoff, np.finfo(np.float64).tiny)

    def pdf(t: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        t_arr = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t_arr)

        mask = t_arr >= time_cutoff
        if np.any(mask):
            out[mask] = _pdf(t_arr[mask]) / norm

        # Return scalar if scalar was passed
        if np.isscalar(t):
            return float(out)
        return out

    return pdf


def make_convolved_pdf(
    x: NDArray[np.float64], pmf: NDArray[np.float64], sigma: float
) -> Callable:
    """
    Make a continuous pdf from a discrete pmf by summing together convolved Gaussians
    with some standard deviation, weighted by the pmf values.

    :param x: (N,) array of values of the pmf.
    :param pmf: (N,) array of the probabilities of the pmf.
    :param sigma: Scalar standard deviation of the convolved Gaussians.

    :return pdf: A callable, vectorized function representing the
        convolved distribution.
    """

    def _pdf(_x):
        numer = pmf * np.exp(-((_x - x) ** 2) / (2 * sigma**2))
        numer = np.sum(numer)
        denom = np.sqrt(2 * np.pi * sigma**2)

        return numer / denom

    pdf = np.vectorize(_pdf)
    return pdf


def proc_vrun(
    vruns: NDArray[np.float64],
    steps: NDArray[np.float64],
    delta: float,
    k: float,
    force_cutoff: float,
) -> NDArray[np.float64]:
    """
    Enforce a cutoff for vrun values based on the force at which they were obtained at.

    :param vruns: (N,) array of vrun values to be processed.
    :param steps: (N,) Array of steps of displacement when the vrun data were taken.
    :param delta: Step size of the motor (nm).
    :param k: Force constant of the optical trap (pN/nm)
    :param force_cutoff: Cut off to enforce (pN).

    :return proc_vruns: (M,) array of vrun values above force_cutoff.
    """
    f = steps * delta * k / 1e3
    pos = np.where(f > force_cutoff)[0]
    new_vruns = vruns[pos]

    if new_vruns.size == 0:
        return None

    return new_vruns


def proc_detach(
    steps: NDArray[np.float64],
    delta: float,
    k: float,
    beta: float,
    force_cutoff: float,
    density_cutoff: float,
) -> NDArray[np.float64]:
    """
    Process detachment data to simulate the processing done by Tominaga2003.
    In particular, only keep the detachments that occur above a given force
    (simulates the need to avoid thermal noise). If the percentage of remaining
    events is not above a certain threshold, return None as a means to enforce
    a certain amount of confidence in the distribution.

    :param seteps: The displacement at detachment in units of motor steps.
    :param Delta: Length of a motor step (nm).
    :param k: Force constant of the trap in the case of non-const force (pN/nm)
    :param beta: 1 / (kB*T) (pN nm).
    :param force_cutoff: Scalar. Only keep dwell times that occur at forces larger
        than the cutoff (pN).
    :param density_cutoff: Scalar. If the proportion of remaining samples after
        enforcing the force cutoff is less than this value, None is returned.

    :return pdf: Callable. The convolved detachment pdf. If the samples failed the
        density_cutoff, then None.
    """
    n, counts = np.unique(steps, return_counts=True)
    n = n.astype(float)
    n += 0.5
    f = n * delta * k / 1e3

    pos = np.where(f > force_cutoff)[0]
    f = f[pos]
    counts = counts[pos]

    density = counts.sum() / steps.size
    if density_cutoff < 0 or density <= 0 or density < density_cutoff:
        return None

    pmf = counts / counts.sum()
    sigma = np.sqrt(k / beta / 1e3)
    pdf = make_convolved_pdf(f, pmf, sigma)
    return pdf


def loglh(datacounts: NDArray[np.float64], modelprobs: NDArray[np.float64]) -> float:
    """
    Calculate the log-likely-hood given counts from data and the probabilities
    of observing those data in a model.

    :param datacounts: (N,) array of data counts.
    :param modelprobs: (N,) array of the probability of observing one of the associated
        data points as determined by some parameterized model.

    :return loglh: The log-likely-hood of observing the data with the given mode.
    """
    res = 0.0
    for c, p in zip(datacounts, modelprobs):
        if c > 0:
            res += c * np.log(p) if p > 0 else -np.inf
    return res


def loglh_frompdf(
    pdf: Callable, datacounts: NDArray[np.float64], databins: NDArray[np.float64]
) -> float:
    """
    Calculate the log-likelihood from a continuous distribution.

    :param pdf: A callable function representing the distribution from the model
    :param datacounts: (M,) array representing the number of experimental data points
        in each bin.
    :param databins: (M+1,) array of the edges used to bin the experimental data.
        These will be used to bin the pdf into a discrete distribution.
    :return loglh: The log-likelihood.
    """
    if pdf is None:
        return -1 * np.inf
    probs = np.zeros(datacounts.size)
    for i in range(probs.size - 1):
        probs[i] = integ.quad(
            pdf, databins[i], databins[i + 1], epsabs=1e-12, epsrel=1e-12
        )[0]
    return loglh(datacounts, probs)


def vrun_logprior(vrun: float | None) -> float:
    """
    Calculate a Gaussian log prior for the run velocity.

    :param vrun: Run velocity (nm/s).
    :return logprior: Is minus np.inf if None.
    """
    if vrun is None:
        return -np.inf

    mean, sem = datadict.get("vrun 100um")
    return np.log(mm.gaussian_pdf(vrun, mean, sem))


def make_tominaga_paths(path: Path) -> dict:
    """
    Helper function to make paths to files with fitting data from Tominaga2003.

    :param path: (Path) Path to directory that holds the fitting data
    :return: (dict) Dictionary of paths to fitting data with stem for keys
        and paths for values. Keys are as follows.
            'dwell1um': path/'dwell1um.npy'
            'vrun100um': path/'vrun100um.npy'
            'detach1um': path/'detach1um.npy'
            'detach100um': path/'detach100um.npy'
    """
    stems = ["dwell1um.npy", "vrun100um.npy", "detach1um.npy", "detach100um.npy"]
    paths = [path.joinpath(stem) for stem in stems]
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"Missing files: {missing}"
    path_dict = {path.stem: path for path in paths}
    return path_dict


def calc_vrun_logprior(
    vruns: NDArray[np.float64], mean: float, std: float
) -> NDArray[np.float64]:
    """
    Calculate the logprior for a vrun value based on a log-Gaussian.
    Parameters for the log-Gaussian distribution are taken from  the
    Tominaga2003 paper.

    :param vruns: (N,) NDArray of the vruns to calculate logpriors for.
    :param mean: (float) mean of the prior distribution.
    :param std: (float) standard deviation of the prior distribution.
    :return: (N,) NDArray of the logpriors.
    """
    logprior = mm.log_gaussian_pdf(vruns, mean, std)
    logprior[np.isnan(logprior)] = -np.inf
    return logprior


def load_vrun_vals(path: Path, delta: float) -> NDArray[np.float64]:
    """
    Loads vrun values from "path", converts vrun values from steps/s to um/s.

    :param path: (Path) of .npy file containing (N,) vrun array in steps/s.
    :param delta: (float) Step size of motor in nm.
    :return: (N,) NDArray of vrun values in um/s.
    """
    vrun_vals = np.load(path).astype(np.float64, copy=True)
    vrun_vals *= delta / 1000  # Convert from steps/s to um/s
    return vrun_vals


def load_vrun_logprior(
    path: Path, delta: float, mean: float, std: float
) -> NDArray[np.float64]:
    """
    Main function for loading and calculating the vrun logprior based on a
    Log-Gaussian distribution. Parameters for the log-Gaussian distribution
    are taken from the Tominaga2003 paper.

    :param path: (Path) of .npy file containing (N,) vrun array in steps/s.
    :param delta: (float) Step size of motor in nm.
    :param mean: (float) mean of the prior distribution.
    :param std: (float) standard deviation of the prior distribution.
    :return: (N,) NDArray of the logpriors.
    """
    vrun_vals = load_vrun_vals(path, delta)
    vrun_logprior = calc_vrun_logprior(vrun_vals, mean, std)
    assert not np.any(np.isnan(vrun_logprior))
    return vrun_logprior


def check_loglh(loglh: NDArray[np.float64]) -> bool:
    """
    Check that all the log-likelihood values are not nan and less than zero.

    :param loglh: Array of log-likelihood values to check.
    :return: (bool) of if all values are okay.
    """
    ok = True
    if np.any(np.isnan(loglh)):
        ok = False
    if np.any(loglh > 0):
        ok = False
    return ok


def load_loglhs(paths: dict) -> dict:
    """
    Load arrays of loglh values from .npy files contained in a dictionary.
    Keys are arbitrary. Values are the paths to the .npy files.

    :param paths: (dict) Dictionary where the values are paths to .npy files
        containing loglh values
    :return: (dict) Dictionary with the same keys as in paths and values are
        the NDArray[np.float64] loglhs.
    """
    loglhs_dict = {key: np.load(path) for key, path in paths.items()}
    test = {name: check_loglh(loglh) for name, loglh in loglhs_dict.items()}
    assert all(test.values())
    return loglhs_dict


def load_scores(data_path: Path, delta: float, mean: float, std: float) -> dict:
    """
    Load the fitting scores into a dictionary.

    :param data_path: (Path) The directory with .npy files for the following.
        1) Log-likelihood for fit to dwell1um distribution.
        2) Log-likelihood for fit to detach1um distribution.
        3) Log-likelihood for fit to detach100um distribution.
        4) vrun values at 100um.
    :param delta: (float) Step size of motor in nm.
    :param mean: (float) mean of the prior distribution.
    :param std: (float) standard deviation of the prior distribution.
    :return: Dictionary with keys equal to that of "make_tominaga_paths" and values
         are the loaded NDArrays.
    """
    paths = make_tominaga_paths(data_path)
    vrun_path = paths.pop("vrun100um")
    scores_dict = load_loglhs(paths)
    vrun_vals = load_vrun_vals(vrun_path, delta)
    scores_dict["vrun100um_vals"] = vrun_vals
    vrun_logprior = load_vrun_logprior(vrun_path, delta, mean, std)
    scores_dict["vrun100um"] = vrun_logprior
    return scores_dict


def combine_same_atp(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Combine the fitting scores for two data sets at the same ATP.

    :param first: First set of scores.
    :param second: Second set of scores. Must have the same shape as "first".
    :return: Combined scores.
    """
    assert first.shape == second.shape, "Score arrays must have the same shape."
    res = np.array(first, dtype=np.float64, copy=True)
    res += second
    return res


def combine_2diff_atp(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Combine the fitting scores for two datasets at different ATP concentrations.

    Produces an outer-sum matrix where:
        combined[i, j] = first[i] + second[j]

    :param first: (N,) array of scores at ATP condition 1.
    :param second: (M,) array of scores at ATP condition 2.
    :return: (N, M) array of combined scores.
    """
    first = first.astype(float)
    second = second.astype(float)
    combined = first[..., :, None] + second[..., None, :]
    return combined


def combine_all_scores(scores: dict) -> NDArray[np.float64]:
    """
    Combine all scores from the fitting to the Tominaga2003 data.

    Expects the following keys in `scores`:
        - 'dwell1um':   Log-likelihood
        - 'detach1um':  Log-likelihood
        - 'vrun100um'   Average vrun in steps/s.
        - 'detach100um' Log-likelyhood

    :param scores: Dictionary containing all the fitting scores to the Tominaga data.
    :return: (N, M) NDArray of the total combined fitting scores.
    """
    score_1um = combine_same_atp(scores.get("dwell1um"), scores.get("detach1um"))
    score_100um = combine_same_atp(scores.get("vrun100um"), scores.get("detach100um"))
    score = combine_2diff_atp(score_1um, score_100um)
    return score


def get_expvel_mean() -> float:
    return 4.27  # um/s


def get_expvel_std() -> float:
    return 0.35  # um/s


def get_myoXI_stepsize() -> float:
    return 35.0  # nm


def get_kd1kd2_params() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    params = hpceval.make_kd1kd2_params()
    kd1s = params["kd1"]
    kd2s = params["kd2"]
    return kd1s, kd2s


def get_kd1kd2nucthetac_params() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    params = hpceval.make_kd1kd2nutheta_params()
    kd1s = params["kd1"]
    kd2s = params["kd2"]
    nucs = params["nu_c"]
    thetacs = params["theta_c"]
    return kd1s, kd2s, nucs, thetacs


def find_bestfit_args(score, axes, axlabels=None, axunits=None, output=False):
    inds = np.unravel_index(score.argmax(), score.shape)
    args = np.array([a[i] for a, i in zip(axes, inds)])

    if output:
        to_print = "Best fit parameters\n"
        for label, a, unit in zip(axlabels, args, axunits):
            to_print += f"     {label}: {a:.3e} {unit}\n"
        print(to_print)
    args_dict = {label: arg for label, arg in zip(axlabels, args)}
    return inds, args_dict


def find_bestfit_vrun(vruns, inds):
    best = vruns[inds]
    return best


def find_bestfit_args_2free(score):
    params = hpceval.make_kd1kd2_params()
    axes = [params.get("kd2"), params.get("kd1"), params.get("kd1")]
    axlabels = ["kd2", "kd1 1uM", "kd1 100uM"]
    axunits = ["1/s", "1/s", "1/s"]
    inds, args_dict = find_bestfit_args(
        score, axes, axlabels=axlabels, axunits=axunits, output=True
    )
    return inds, args_dict


def find_bestfit_args_4free(score):
    params = hpceval.make_kd1kd2nutheta_params()
    axes = [
        params.get("theta_c"),
        params.get("nu_c"),
        params.get("kd2"),
        params.get("kd1"),
        params.get("kd1"),
    ]
    axlabels = ["ThetaC", "nuC", "kd2", "kd1 1uM", "kd1 100uM"]
    axunits = ["rad", "", "1/s", "1/s", "1/s"]
    inds, args_dict = find_bestfit_args(
        score, axes, axlabels=axlabels, axunits=axunits, output=True
    )
    return inds, args_dict
