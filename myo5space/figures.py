#!/usr/bin/env python3
"""
File: figures.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains code used in the `Figures.ipynb` notebook.
"""

import numpy as np
import multiprocessing as mp
from pathlib import Path
import myo5space.fitting as fitting
from myo5space.motor import Motor
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integ
import scipy.optimize as opt


def make_bestfit_motors(
    args_2free: dict, args_4free: dict
) -> tuple[Motor, Motor, Motor, Motor]:
    """
    Make Motor objects using the best fit parameters.

    :param args_2free: Dictionary of best fit parameters when only kd1 and kd2
        are free. Expected keys are `kd1 1um`, `kd1 100um`, and `kd2`.
    :param args_4free: Dictionary of best fit parameters when kd1, kd2, theat_c,
        and nu_c are free.Expected keys are `kd1 1um`, `kd1 100um`, `kd2`, `ThetaC`, and `nuC`.
    :return: A four tuple of Motor objects made using the the best fits.
        motor_2free1um: Besyt fit motor at 1 uM ATP with two free params.
        motor_2free100um: Best fit motor at 100 uM ATP with two free params.
        motor_4free1um: Best fit motor at 1 uM ATP with four free params.
        motor_4free100um: Best fit motor at 100 uM ATP with four free params.
    """
    kh = 1000  # 1/s
    motor_2free1um = Motor(
        kd1=args_2free.get("kd1 1uM"), kd2=args_2free.get("kd2"), kh=kh
    )
    motor_2free100um = Motor(
        kd1=args_2free.get("kd1 100uM"), kd2=args_2free.get("kd2"), kh=kh
    )

    motor_4free1um = Motor(
        kd1=args_4free.get("kd1 1uM"),
        kd2=args_4free.get("kd2"),
        theta_c=args_4free.get("ThetaC"),
        nu_c=args_4free.get("nuC"),
        kh=kh,
    )
    motor_4free100um = Motor(
        kd1=args_4free.get("kd1 100uM"),
        kd2=args_4free.get("kd2"),
        theta_c=args_4free.get("ThetaC"),
        nu_c=args_4free.get("nuC"),
        kh=kh,
    )
    return motor_2free1um, motor_2free100um, motor_4free1um, motor_4free100um


def make_dwell_simdata(
    motor: Motor, trials: int, time_limit: float, k: float, processes: int | None
):
    """
    See motor.simulate docstring for documentation on emc dwell.
    """
    simdata = motor.simulate(
        "emc", "dwell", trials, time_limit=time_limit, k=k, processes=processes
    )
    return simdata


def make_detach_simdata(
    motor: Motor, trials: int, time_limit: float, k: float, processes: int | None
):
    """
    See motor.simulate docstring for documentation on emc runstats.
    """
    simdata = motor.simulate(
        "emc", "runstats", trials, time_limit=time_limit, k=k, processes=processes
    )
    return simdata


def make_dwell_pdf(
    simdata, k: float, force_cutoff: float, time_cutoff: float, density_cutoff: float
):
    """
    See fitting.proc_dwell docstring for documentation.
    """
    motor = simdata.motor
    pdf = fitting.proc_dwell(
        simdata.tdwells,
        motor.force(simdata.zlocs + 0.5, k),
        force_cutoff,
        time_cutoff,
        density_cutoff,
    )
    return pdf


def make_detach_pdf(simdata, k: float, force_cutoff: float, density_cutoff: float):
    """
    See fitting.proc_detach docstring for documentation.
    """
    motor = simdata.motor
    pdf = fitting.proc_detach(
        simdata.zruns, motor.Delta, k, motor.beta(295), force_cutoff, density_cutoff
    )
    return pdf


def make_dwell_xdata(bins, n=300):
    xdata = np.linspace(0, 1.3 * bins[-1], n)
    return xdata


def make_detach_xdata(bins, n=300):
    xdata = np.linspace(0.1 * bins[0], 1.3 * bins[-1], n)
    return xdata


def make_bestfitplot_data(
    path: Path, motor_2free1um, motor_2free100um, motor_4free1um, motor_4free100um
) -> None:
    # Simulation parameters
    trials = 100000
    processes = mp.cpu_count()
    time_limit = 50.0
    k = 4.1

    # Run simulations
    simdata_dwell_2free1um = make_dwell_simdata(
        motor_2free1um, trials, time_limit, k, processes
    )
    simdata_detach_2free1um = make_detach_simdata(
        motor_2free1um, trials, time_limit, k, processes
    )
    simdata_detach_2free100um = make_detach_simdata(
        motor_2free100um, trials, time_limit, k, processes
    )
    simdata_dwell_4free1um = make_dwell_simdata(
        motor_4free1um, trials, time_limit, k, processes
    )
    simdata_detach_4free1um = make_detach_simdata(
        motor_4free1um, trials, time_limit, k, processes
    )
    simdata_detach_4free100um = make_detach_simdata(
        motor_4free100um, trials, time_limit, k, processes
    )

    # Pdf making parameters
    detach_force_cutoff = 0.25
    dwell_force_cutoff = 0.5
    time_cutoff = 0.02
    density_cutoff = 0.01

    # Make pdfs
    pdf_dwell_2free1um = make_dwell_pdf(
        simdata_dwell_2free1um, k, dwell_force_cutoff, time_cutoff, density_cutoff
    )
    pdf_detach_2free1um = make_detach_pdf(
        simdata_detach_2free1um, k, detach_force_cutoff, density_cutoff
    )
    pdf_detach_2free100um = make_detach_pdf(
        simdata_detach_2free100um, k, detach_force_cutoff, density_cutoff
    )
    pdf_dwell_4free1um = make_dwell_pdf(
        simdata_dwell_4free1um, k, dwell_force_cutoff, time_cutoff, density_cutoff
    )
    pdf_detach_4free1um = make_detach_pdf(
        simdata_detach_4free1um, k, detach_force_cutoff, density_cutoff
    )
    pdf_detach_4free100um = make_detach_pdf(
        simdata_detach_4free100um, k, detach_force_cutoff, density_cutoff
    )

    # Histogram data from experiments
    counts_dwell1um, bins_dwell1um = fitting.get_expdata("dwell 1um")
    counts_detach1um, bins_detach1um = fitting.get_expdata("detach 1um")
    counts_detach100um, bins_detach100um = fitting.get_expdata("detach 100um")

    hist_dict = dict(
        bins_dwell1um=bins_dwell1um,
        counts_dwell1um=counts_dwell1um,
        bins_detach1um=bins_detach1um,
        counts_detach1um=counts_detach1um,
        bins_detach100um=bins_detach100um,
        counts_detach100um=counts_detach100um,
    )
    hist_dict = {k: pd.Series(v) for k, v in hist_dict.items()}
    hist_df = pd.DataFrame(hist_dict)
    hist_df.to_csv(path.joinpath("bestfitplot_histdata.csv"))

    # X-data for pdf
    xdata_dwell1um = make_dwell_xdata(bins_dwell1um)
    xdata_detach1um = make_detach_xdata(bins_detach1um)
    xdata_detach100um = make_detach_xdata(bins_detach100um)

    # Y-data for pdf.
    ydata_dwell_2free1um = pdf_dwell_2free1um(xdata_dwell1um)
    ydata_detach_2free1um = pdf_detach_2free1um(xdata_detach1um)
    ydata_detach_2free100um = pdf_detach_2free100um(xdata_detach100um)
    ydata_dwell_4free1um = pdf_dwell_4free1um(xdata_dwell1um)
    ydata_detach_4free1um = pdf_detach_4free1um(xdata_detach1um)
    ydata_detach_4free100um = pdf_detach_4free100um(xdata_detach100um)

    pdf_dict = dict(
        x_dwell_2free1um=xdata_dwell1um,
        y_dwell_2free1um=ydata_dwell_2free1um,
        x_detach_2free1um=xdata_detach1um,
        y_detach_2free1um=ydata_detach_2free1um,
        x_detach_2free100um=xdata_detach100um,
        y_detach_2free100um=ydata_detach_2free100um,
        x_dwell_4free1um=xdata_dwell1um,
        y_dwell_4free1um=ydata_dwell_4free1um,
        x_detach_4free1um=xdata_detach1um,
        y_detach_4free1um=ydata_detach_4free1um,
        x_detach_4free100um=xdata_detach100um,
        y_detach_4free100um=ydata_detach_4free100um,
    )
    pdf_df = pd.DataFrame(pdf_dict)
    pdf_df.to_csv(path.joinpath("bestfitplot_pdfdata.csv"))


def check_bestfitplot_data(path: Path) -> bool:
    path_histdata = path.joinpath("bestfitplot_histdata.csv")
    path_pdfdata = path.joinpath("bestfitplot_pdfdata.csv")

    if not path_histdata.exists() or not path_pdfdata.exists():
        return False

    return True


def calc_zrunforce(forces, motor):
    zruns = np.array([motor.zrun_mean(f=f) for f in forces])
    return motor.Delta * zruns / 1000


def calc_tdwellforce(forces, motor):
    tdwells = np.array([motor.tdwell_mean(f=f) for f in forces])
    return tdwells


def calc_vrunforce(forces, motor):
    vruns = np.array([motor.vrun_mean(f=f) for f in forces])
    vruns *= motor.Delta / 1000
    return vruns


def calc_kfpforce(forces, motor):
    kfps = np.array([motor.kfp(f) for f in forces])
    return kfps


def make_bestfitforce_data(path: Path, motor_2free100um, motor_4free100um):
    forces = np.linspace(0, 2.6, 50)

    zruns_2free100um = calc_zrunforce(forces, motor_2free100um)
    tdwells_2free100um = calc_tdwellforce(forces, motor_2free100um)
    vruns_2free100um = calc_vrunforce(forces, motor_2free100um)
    kfps_2free100um = calc_kfpforce(forces, motor_2free100um)

    zruns_4free100um = calc_zrunforce(forces, motor_4free100um)
    tdwells_4free100um = calc_tdwellforce(forces, motor_4free100um)
    vruns_4free100um = calc_vrunforce(forces, motor_4free100um)
    kfps_4free100um = calc_kfpforce(forces, motor_4free100um)

    data_dict = dict(
        forces=forces,
        zruns_2free100um=zruns_2free100um,
        tdwells_2free100um=tdwells_2free100um,
        vruns_2free100um=vruns_2free100um,
        kfpps_2free100um=kfps_2free100um[:, 0],
        kfpms_2free100um=kfps_2free100um[:, 1],
        zruns_4free100um=zruns_4free100um,
        tdwells_4free100um=tdwells_4free100um,
        vruns_4free100um=vruns_4free100um,
        kfpps_4free100um=kfps_4free100um[:, 0],
        kfpms_4free100um=kfps_4free100um[:, 1],
    )
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(path.joinpath("bestfitforce_data.csv"))


def check_bestfitforce_data(path: Path) -> bool:
    path_data = path.joinpath("bestfitforce_data.csv")

    if not path_data.exists():
        return False
    return True


def make_simtraces_data(path: Path, myoV_motor: Motor, myoXI_motor: Motor):
    myoV_traj = myoV_motor.simulate("kmc", "trajectory", trials=4000, const_force=0.0)
    myoXI_traj = myoV_motor.simulate("kmc", "trajectory", trials=4000, const_force=1.33)

    res = np.zeros((4000, 4))
    res[: myoV_traj.times.size, 0] = myoV_traj.times
    res[: myoV_traj.times.size, 1] = myoV_traj.disps
    res[:, 2] = myoXI_traj.times
    res[:, 3] = myoXI_traj.disps

    data_df = pd.DataFrame(
        res,
        columns=[
            "MyoV Time (s)",
            "MyoV Disp (um)",
            "MyoXI Time (s)",
            "MyoXI Disp (um)",
        ],
    )
    data_df.to_csv(path.joinpath("simtraces_data.csv"))


def check_simtraces_data(path: Path) -> bool:
    path_data = path.joinpath("simtraces_data.csv")

    if not path_data.exists():
        return False
    return True


def make_gammaapprox_data(path: Path):
    # Make random motors
    a = 1
    b = 1e6
    size = 5000
    kd1s = stats.loguniform.rvs(a, b, size=size)
    kd2s = np.array([stats.loguniform.rvs(a, kd1) for kd1 in kd1s])
    khs = stats.loguniform.rvs(a, b, size=size)
    kfpps = stats.uniform.rvs(1, 5000, size=size)
    kfpms = [stats.uniform.rvs(1, kfpp) for kfpp in kfpps]
    motors = [
        Motor(kd1=kd1, kd2=kd2, kh=kh, kps=1e9) for kd1, kd2, kh in zip(kd1s, kd2s, khs)
    ]

    # Caclulate approximating shapes and KLD between true and approx tdwell distributions.
    shapes = np.zeros(size)
    klds = np.zeros(size)
    count = 0

    for m, kfpp, kfpm in zip(motors, kfpps, kfpms):
        mean, _, shape, scale = m.tdwell_gammaapprox(kfpp=kfpp, kfpm=kfpm)

        def f(x):
            p = m.tdwell_pdf(x, kfpp=kfpp, kfpm=kfpm)
            q = m.tdwell_gamma_pdf(x, kfpp=kfpp, kfpm=kfpm)
            return p * (np.log(p) - np.log(q))

        kld = integ.quad(f, 0, 20 * mean, limit=200)[0]
        shapes[count] = shape
        klds[count] = kld

        if count % (size // 10) == 0:
            print(f"{count}/{size}")
        count += 1

    # Save shape-kld data.
    shape_kld_dict = dict(shapes=shapes, klds=klds)
    shape_kld_df = pd.DataFrame(shape_kld_dict)
    shape_kld_df.to_csv(path.joinpath("gammaapprox_shapekld_data.csv"))

    # Find the motor with the highest KLD.
    arg = klds.argmax()
    m = motors[arg]
    kfpp = kfpps[arg]
    kfpm = kfpms[arg]
    tdwell_mean = m.tdwell_mean(kfpp=kfpp, kfpm=kfpm)

    # Evaluate the ture and approx distributions in the worst case.
    t = np.linspace(0, 10 * tdwell_mean, 300)
    true_tdwell = m.tdwell_pdf(t, kfpp=kfpp, kfpm=kfpm)
    approx_tdwell = m.tdwell_gamma_pdf(t, kfpp=kfpp, kfpm=kfpm)

    # Save true and approx distribution data.
    worst_dict = dict(tdwells=t, true_pdf=true_tdwell, approx_pdf=approx_tdwell)
    worst_df = pd.DataFrame(worst_dict)
    worst_df.to_csv(path.joinpath("gammaapprox_worst_data.csv"))


def check_gammaapprox_data(path: Path):
    path_shapekld = path.joinpath("gammaapprox_shapekld_data.csv")
    path_worst = path.joinpath("gammaapprox_worst_data.csv")

    if not path_shapekld.exists() or not path_worst.exists():
        return False
    return True


def calc_vmax_kh(khs, kfpp, kfpm):
    res = np.zeros((2, khs.size))
    for i, kh in enumerate(khs):

        def f(kd1, kh):
            m = Motor(kd1=kd1, kd2=0, kh=kh)
            vrun_mean = m.vrun_mean2(kfpp=kfpp, kfpm=kfpm)
            return vrun_mean

        minres = opt.minimize_scalar(
            lambda kd1: -1 * f(kd1, kh), bounds=(1e-1, kh * 10**3), method="Bounded"
        )
        res[0, i] = minres.x
        res[1, i] = -1 * minres.fun
    return res


def make_kd1khline_data(path: Path):
    a = 1
    b = 1e6
    size = 500
    kfpps = stats.loguniform.rvs(a, b, size=size)
    kfpms = np.array([stats.loguniform.rvs(a, kfpp) for kfpp in kfpps])
    khs = np.logspace(1, 12, 100)
    v_kh_res = [calc_vmax_kh(khs, kfpp, kfpm) for kfpp, kfpm in zip(kfpps, kfpms)]

    kd1_kh_shapes = list()
    for kfpp, kfpm, res in zip(kfpps, kfpms, v_kh_res):
        kd1s = res[0]
        shapes = list()
        for kd1, kh in zip(kd1s, khs):
            motor = Motor(kd1=kd1, kh=kh, kd2=0)
            _, _, shape, _ = motor.tdwell_gammaapprox(kfpp=kfpp, kfpm=kfpm)
            shapes.append(shape)
        kd1_kh_shapes.append(np.array(shapes))
    kd1_kh_shapes = np.array(kd1_kh_shapes)

    v_kh_res_myoV = calc_vmax_kh(khs, *Motor().kfp(f=0))
    v_kh_res.append(v_kh_res_myoV)

    kd1_kh_lines = [
        stats.linregress(np.log10(khs), np.log10(res[0])) for res in v_kh_res
    ]
    kd1_kh_slopes = np.array([line.slope for line in kd1_kh_lines])
    kd1_kh_intercepts = np.array([line.intercept for line in kd1_kh_lines])
    kd1_kh_rvals = np.array([line.rvalue for line in kd1_kh_lines])

    myoV_dict = dict(
        khs=khs,
        myoV_vkh=v_kh_res_myoV[0],
    )
    myoV_df = pd.DataFrame(myoV_dict)
    myoV_df.to_csv(path.joinpath("kd1khline_myoV_data.csv"))

    lines_dict = dict(
        slopes=kd1_kh_slopes,
        intercepts=kd1_kh_intercepts,
        rvals=kd1_kh_rvals,
    )
    lines_df = pd.DataFrame(lines_dict)
    lines_df.to_csv(path.joinpath("kd1khline_lines_data.csv"))

    pd.DataFrame(dict(shapes=kd1_kh_shapes.flatten())).to_csv(
        path.joinpath("kd1khline_shapes_data.csv")
    )


def check_kd1khline_data(path: Path):
    path_myoV = path.joinpath("kd1khline_myoV_data.csv")
    path_lines = path.joinpath("kd1khline_lines_data.csv")
    path_shapes = path.joinpath("kd1khline_shapes_data.csv")
    if not path_myoV.exists() or not path_lines.exists() or not path_shapes.exists():
        return False
    return True


def make_speedlimit_data(path: Path):
    a = 1
    b = 1e6
    size = 500

    kd1s = stats.loguniform.rvs(a, b, size=size)
    khs = stats.loguniform.rvs(a, b, size=size)
    kfpps = stats.loguniform.rvs(a, b, size=size)
    kfpms = np.array([stats.loguniform.rvs(a, kfpp) for kfpp in kfpps])

    vruns = list()
    count = 0
    for kd1, kh, kfpp, kfpm in zip(kd1s, khs, kfpps, kfpms):
        m = Motor(kd1=kd1, kd2=0, kh=kh)
        simres = m.simulate(
            "kmc",
            "runstats",
            100000,
            kfpp=kfpp,
            kfpm=kfpm,
            time_limit=100,
            processes=mp.cpu_count(),
        )
        vrun_avg = simres.vrun_mean()
        productive_avg = simres.vrun_productive_mean()

        _, _, shape, _ = m.tdwell_gammaapprox(kfpp=kfpp, kfpm=kfpm)
        vruns.append([kfpp, vrun_avg, productive_avg, shape])

        if count % (size // 10) == 0:
            print(f"{count}/{size}")

        count += 1

    vruns = np.array(vruns)
    df = pd.DataFrame(vruns, columns=["kfpp", "vrun", "vrun_productive", "shape"])
    df.to_csv(path.joinpath("speedlimit_data.csv"))


def check_speedlimit_data(path: Path):
    path_data = path.joinpath("speedlimit_data.csv")
    if not path_data.exists():
        return False
    return True


def vz_worker(args):
    kd1 = args[0]
    kd2 = args[1]
    kh = args[2]
    m = Motor(kd1=kd1, kh=kh, kd2=-0)
    m_kd2 = Motor(kd1=kd1, kd2=kd2, kh=kh)
    zrun = m.zrun_mean()
    vrun = m.vrun_mean2()
    zrun2 = zrun / (1 - m.pT())
    vrun2 = vrun / (1 - m.pT())
    zrun_kd2 = m_kd2.zrun_mean()
    vrun_kd2 = m_kd2.vrun_mean()
    zrun2_kd2 = zrun_kd2 / (1 - m_kd2.pT())
    vrun2_kd2 = vrun_kd2 / (1 - m_kd2.pT())
    return [zrun, vrun, zrun2, vrun2, zrun_kd2, vrun_kd2, zrun2_kd2, vrun2_kd2]


def make_pareto_data(path: Path):
    a = 1e-2
    b = 1e6
    size = 5000

    kd1s_vz = stats.loguniform.rvs(a, b, size=size)
    khs_vz = stats.loguniform.rvs(a, b, size=size)
    kd2s_vz = np.array([stats.loguniform.rvs(a, kd1, size=1)[0] for kd1 in kd1s_vz])

    vz = None
    with mp.Pool(12) as p:
        vz = np.array(p.map(vz_worker, zip(kd1s_vz, kd2s_vz, khs_vz), chunksize=12)).T

    vz_shapes = np.zeros((2, size))
    i = 0
    for kd1, kh, kd2 in zip(kd1s_vz, khs_vz, kd2s_vz):
        m = Motor(kd1=kd1, kd2=0, kh=kh)
        m2 = Motor(kd1=kd1, kd2=kd2, kh=kh)
        _, _, shape, _ = m.tdwell_gammaapprox()
        _, _, shape2, _ = m2.tdwell_gammaapprox()
        vz_shapes[0, i] = shape
        vz_shapes[1, i] = shape2
        i += 1

    kfpp, _ = Motor().kfp()
    x = np.logspace(-1, 6, 100)
    y = (khs_vz.max() * kfpp) / (khs_vz.max() + kfpp) / x
    line_dict = dict(xdata=x, ydata=y, kfpp=kfpp * np.ones(100))
    line_df = pd.DataFrame(line_dict)
    line_df.to_csv(path.joinpath("pareto_line_data.csv"))

    vz_dict = dict(
        zrun=vz[0],
        vrun=vz[1],
        zrun2=vz[2],
        vrun2=vz[3],
        zrun_kd2=vz[4],
        vrun_kd2=vz[5],
        zrun2_kd2=vz[6],
        vrun2_kd2=vz[7],
        shapes=vz_shapes[0],
        shapes_kd2=vz_shapes[1],
    )
    vz_df = pd.DataFrame(vz_dict)
    vz_df.to_csv(path.joinpath("pareto_vz_data.csv"))


def check_pareto_data(path: Path):
    path_line = path.joinpath("pareto_line_data.csv")
    path_vz = path.joinpath("pareto_vz_data.csv")

    if not path_line.exists() or not path_vz.exists():
        return False
    return True


def make_trapmode_data(path: Path):
    a = 1
    b = 1e6
    size = 1000

    kd1s = stats.loguniform.rvs(a, b, size=size)
    kd2s = np.array([stats.loguniform.rvs(a, kd1) for kd1 in kd1s])
    khs = stats.loguniform.rvs(a, b, size=size)

    zruns = np.zeros((2, size))
    vruns = np.zeros((2, size))

    i = 0
    for kd1, kd2, kh in zip(kd1s, kd2s, khs):
        m = Motor(kd1=kd1, kd2=kd2, kh=kh)

        varforce = m.simulate(
            "emc", "runstats", 50000, time_limit=100, processes=mp.cpu_count()
        )
        zruns[0, i] = varforce.zrun_mean()
        vruns[0, i] = varforce.vrun_mean()

        zeroforce = m.simulate(
            "emc",
            "runstats",
            50000,
            time_limit=100,
            const_force=0.0,
            processes=mp.cpu_count(),
        )
        zruns[1, i] = zeroforce.zrun_mean()
        vruns[1, i] = zeroforce.vrun_mean()

        if i % (size // 10) == 0:
            print(f"{i}/{size}")

        i += 1

    def exp(x, k):
        return np.exp(x * k)

    zrun_exp = opt.curve_fit(exp, zruns[0], zruns[1], p0=[1.0])
    vrun_line = stats.linregress(vruns[0], vruns[1])

    zrun_xdata = np.linspace(0, 10, 200)
    vrun_xdata = xdata = np.linspace(0, 950, 200)
    zrun_ydata = exp(zrun_xdata, *zrun_exp[0])
    zrun_exponent = np.ones_like(zrun_xdata) * zrun_exp[0][0]
    vrun_ydata = vrun_line.slope * vrun_xdata + vrun_line.intercept
    vrun_slope = np.ones_like(vrun_xdata) * vrun_line.slope
    vrun_intercept = np.ones_like(xdata) * vrun_line.intercept
    vrun_rvalue = np.ones_like(vrun_xdata) * vrun_line.rvalue

    trapmode_fits_dict = dict(
        zrun_xdata=zrun_xdata,
        zrun_ydata=zrun_ydata,
        vrun_ydata=vrun_ydata,
        zrun_exp=zrun_exponent,
        vrun_xdata=vrun_xdata,
        vrun_slope=vrun_slope,
        vrun_intercept=vrun_intercept,
        vrun_rvalue=vrun_rvalue,
    )

    trapmode_data_dict = dict(
        zrun_comp=zruns[0], zrun_const=zruns[1], vrun_comp=vruns[0], vrun_const=vruns[1]
    )
    trapmode_fits_df = pd.DataFrame(trapmode_fits_dict)
    trapmode_data_df = pd.DataFrame(trapmode_data_dict)

    trapmode_data_df.to_csv(path.joinpath("trapmode_vz_data.csv"))
    trapmode_fits_df.to_csv(path.joinpath("trapmode_fits_data.csv"))


def check_trapmode_data(path: Path):
    path_fits = path.joinpath("trapmode_fits_data.csv")
    path_vz = path.joinpath("trapmode_vz_data.csv")

    if not path_fits.exists() or not path_vz.exists():
        return False
    return True


if __name__ == "__main__":
    path = Path(
        "/home/joshuaholmes/Documents/Code/_myo5space/JupyterNotebooks/Figures/FigureData"
    )
    make_trapmode_data(path)
