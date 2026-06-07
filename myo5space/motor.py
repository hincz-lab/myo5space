#!/usr/bin/env pythonc
"""
File: motor.py
Author: Joshua Holmes
Email: jbh92@case.edu

Contains code to generate myosin motors with desired
kinetic and structural parameters.
"""

# Imports
from numbers import Number
from functools import lru_cache
from numba import njit
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.constants as const
import scipy.special as special
import myo5space.motor_calculations as mc
import myo5space.mymath as mm
import myo5space.kmc as kmc
import myo5space.emc as emc
import myo5space.fitting as fit


class Motor(object):
    """
    Class for holding the structural and kinetic parameters
    to describe a MyosinV-like molecular motor and interfacing with the simulation
    and fitting functionality of the program.

    Default parameters are for myosin-V as described in Hinkzewski2013.

    Instance variables:
        L (float): Leg contour length (nm).
        l_p (float): Leg persistence length (nm).
        D_h (float): Head diffusivity (nm^2/s).
        theta_c (float): Constraint angle (rads).
        theta_F (float): Load force angle (rads).
        nu_c (float): Constraint strength (unitless).
        t_r (float): Relaxation time (s).
        Delta (float): Binding site separation (nm).
        a (float): Capture radius (nm).
        b (float): Binding penalty (unitless).
        kh (float): ATP hydrolysis rate (1/s).
        kd1 (float): Trailing head detachment rate (1/s).
        kd2 (float): Leading head detachment rate (1/s).
        kps (float): Power stroke rate (1/s).
    """

    # Structural parameters.
    def __init__(
        self,
        L: float = 35,
        l_p: float = 310,
        D_h: float = 5.7e7,
        theta_c: float = np.pi / 3,
        theta_F: float = 0.0,
        nu_c: float = 184,
        t_r: float = 5e-6,
        Delta: float = 36,
        a: float = 1,
        b: float = 0.065,
        kh: float = 750,
        kd1: float = 12,
        kd2: float = 1.5,
        kps: float = np.inf,
    ) -> None:
        """
        Default parameter values are for Myosin V as in Hinczewski2013
        """
        self.L = L
        self.l_p = l_p
        self.D_h = D_h
        self.theta_c = theta_c
        self.theta_F = theta_F
        self.nu_c = nu_c
        self.t_r = t_r
        # Binding parameters.
        self.Delta = Delta
        self.a = a
        self.b = b
        # Chemical rates.
        self.kh = kh
        self.kd1 = kd1
        self.kd2 = kd2
        self.kps = kps

        self.check_params()

    def check_params(self) -> None:
        key_vals = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(key)
        }
        for key, value in key_vals.items():
            if not isinstance(value, Number):
                raise ValueError(f"{key} = {value} is not a number.")
            if value < 0:
                raise ValueError(f"{key} = {value} is less than zero.")

    @property
    def g(self) -> float:
        """
        :return g: The motor's gating parameter.
        """
        if self.kd1 == 0:
            return np.inf
        else:
            return self.kd1 / self.kd2

    @property
    def kappa(self) -> float:
        """
        :return kappa: Ratio of contour length to persistence length.
        """
        if self.l_p == 0:
            return np.inf
        else:
            return self.L / self.l_p

    @staticmethod
    def beta(temperature: float) -> float:
        """
        :param temperature: Temperature to calculate beta for.
        :return beta: 1/(kB*T), where kB is Boltzmann's constant and
        T is self.temperature (pN nm).
        """
        if temperature == 0:
            return np.inf
        else:
            kbt = const.Boltzmann * temperature * 10**12 * 10**9
            return 1 / kbt

    def force(self, n: float, k: float) -> float:
        """
        Calculate the force experienced by a motor in a compliant optical trap.

        :param n: The number of steps the motor is displaced from the trap's zero point.
        :param k: The stiffness of the trap (pN/nm).

        :return force: The force on the motor (pN).
        """
        return mc.force(n, self.Delta, k)

    def kfp(self, f: float = 0.0, temperature: float = 295.0) -> tuple[float, float]:
        """
        Calculate the force dependent rates of first passage for a free diffusing
        head to find either the forward (kfpp) or reverse (kfpm) position on an
        actin track.

        :param f: The force experienced by the motor (pN).
        :param temperature: Temperature of the experiment (K).

        :return kfpp, kfpm: The rates of first passage (1/s).
        """
        return mc.kfp(
            f,
            self.theta_F,
            self.Delta,
            self.beta(temperature),
            self.L,
            self.l_p,
            self.nu_c,
            self.theta_c,
            self.a,
            self.D_h,
        )

    def pF(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will take a forward step.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pF: Probability of taking a forward step.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pF(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pB(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will take a backwards step.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pB: Probability of taking a reverse step.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pB(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pTs(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will complete a trailing stomp.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pTs: Probability of doing completing a trailing stomp.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pTs(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pLs(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will complete a leading stomp.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pLs: Probability of doing completing a leading stomp.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pLs(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pT1(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will terminate before hydrolysis.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pT1: Probability of terminating before hydrolysis.
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pT1(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pT2(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will terminate after hydrolysis.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pT2: Probability of terminating after hydrolysis.
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pT2(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pT3(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability that the motor will terminate after leading
        head detachment. The first passage rates can either be explicitly specified
        or calculated from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pT3: Probability of terminating after leading head detachment.
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pT3(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def pT(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the overall probability of termination.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pT3: Probability of terminating after leading head detachment.
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.pT(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tF(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to complete a forward step.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tF: Average time to complete a forward step (s).
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tF(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tB(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to complete a backwards step.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tB: Average time to complete a backwards step (s).
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tB(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tTs(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to complete a trailing stomp.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tTs: Average time to complete a trailing stomp (s).
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tTs(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tLs(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to complete a leading stomp.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tLs: Average time to complete a leading stomp (s).
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tLs(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tT1(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to terminate before hydrolysis.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tT1: Average time to terminate before hydrolysis (s).
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tT1(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tT2(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to terminate after hydrolysis.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tT2: Average time to terminate after hydrolysis (s).
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tT2(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def tT3(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the average time needed to terminate after leading head detachment.
        The first passage rates can either be explicitly specified or calculated
        from a force. Both rates must be given for the former.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tT3: Average time to terminate after leading head detachment.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        return mc.tT3(self.kd1, self.kd2, self.kh, kfpp, kfpm, self.kps, self.b)

    def f_stall(self, temperature: float = 295.0) -> float:
        """
        Calculate the force at which the probability of taking a forward step
        equals the probability of taking a backward step.

        :param temperature: The temperature of the experiment (K).

        :return fstall: The stall force (pN).
        """
        return mc.f_stall(
            self.beta(temperature),
            self.theta_F,
            self.L,
            self.l_p,
            self.nu_c,
            self.theta_c,
            self.Delta,
            self.g,
            self.b,
        )

    def trun_tm(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> NDArray[np.float64]:
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
            6: Termination in any way (absorbing).

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return truntm: (7,7) transition matrix.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        return mc.trun_transmat(
            self.kd1, self.kd2, self.kh, self.kps, kfpp, kfpm, self.b
        )

    def tdwell_tm(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> NDArray[np.float64]:
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

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return tdwelltm: (7,7) transition matrix.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        return mc.tdwell_transmat(
            self.kd1, self.kd2, self.kh, self.kps, kfpp, kfpm, self.b
        )

    def trun_p0(self) -> NDArray[np.float64]:
        """
        Generate an initial probability vector associated with the trun transition
        matrix where the motor starts in the resting state.

        :return p0: (6,) vector of zeros except with p0[0] == 1.
        """
        transmat = self.trun_tm()
        p0 = np.zeros(transmat.shape[0] - 1, dtype=float)
        p0[0] = 1.0
        return p0

    def tdwell_p0(self) -> NDArray[np.float64]:
        """
        Generate an initial probability vector associated with the trun transition
        matrix where the motor starts in the resting state.

        :return p0: (6,) vector of zeros except with p0[0] == 1.
        """
        transmat = self.tdwell_tm()
        p0 = np.zeros(transmat.shape[0] - 1, dtype=float)
        p0[0] = 1.0
        return p0

    def trun_subtm(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> NDArray[np.float64]:
        """
        Calculates the transition matrix for a motor's run where termination results
        in absorption.
        Used to calculate the distribution of motor dwell times in at a given position.
        Uses convention that the columns add to zero. The columns refer to the
        following states:
            0: Resting state.
            1: Trailing lead detached, pre hydrolysis.
            2: Trailing head detached, post hydrolysis.
            3: Trailing head attached to forward binding position.
            4: Trailing head attached to original binding position.
            5: Leading head detached.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinite values with.

        :return trunsubtm: (6,6) subtransition matrix.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        transmat = self.trun_tm(kfpp=kfpp, kfpm=kfpm)
        transmat[transmat == np.inf] = infval
        transmat[transmat == -np.inf] = -infval
        return transmat[:-1, :-1]

    def tdwell_subtm(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> NDArray[np.float64]:
        """
        Calculates the subtransition matrix for a single motor event where the
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

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinite values with.

        :return tdwelltm: (6,6) subtransition matrix.
        """

        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        transmat = self.tdwell_tm(kfpp=kfpp, kfpm=kfpm)
        transmat[transmat == np.inf] = infval
        transmat[transmat == -np.inf] = -infval
        return transmat[:-1, :-1]

    def zrun_pmf(
        self,
        z: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the probability of the motor obtaining a certain run distance.

        :param z: The run distance (steps). May be positive or negative.
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return pmf: The probability of obtaining the run distance.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        pF = self.pF(kfpp=kfpp, kfpm=kfpm)
        pB = self.pB(kfpp=kfpp, kfpm=kfpm)
        pTs = self.pTs(kfpp=kfpp, kfpm=kfpm)
        pLs = self.pLs(kfpp=kfpp, kfpm=kfpm)
        pT = self.pT(kfpp=kfpp, kfpm=kfpm)

        denom = 1 - pLs - pTs
        x = 4 * pF * pB / (denom**2)
        fact1 = pT / denom / np.sqrt(1 - x)

        pFB = np.ones_like(z, dtype=float)
        pFB[z >= 0] = pF
        pFB[z < 0] = pB
        fact2 = (2 * pFB / denom / (1 + np.sqrt(1 - x))) ** np.abs(z)

        return fact1 * fact2

    def zrun_mean(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
    ) -> float:
        """
        Calculate the mean run distance.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).

        :return mean: The mean run distance (steps).
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        pF = self.pF(kfpp=kfpp, kfpm=kfpm)
        pB = self.pB(kfpp=kfpp, kfpm=kfpm)
        pT = self.pT(kfpp=kfpp, kfpm=kfpm)

        return (pF - pB) / pT

    def trun_pdf(
        self,
        t: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the probability distribution of run times.

        :param t: The time to calculate the probability at (seconds).
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinites with.

        :return trun pdf: The probability density at t.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.trun_p0()
        subtm = self.trun_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        return mm.ph_pdf(t, p0, subtm)

    def trun_mean(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the mean run time using the definition of the mean of a phase-type
        distribution. Note that this can sometime be numerically unstable with rate
        values > 1e15.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinities with.

        :return mean trun: The mean run time.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.trun_p0()
        subtm = self.trun_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        return mm.ph_mean(p0, subtm)

    def trun_mean2(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the mean run time using an event-based approach.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinities with.

        :return mean trun: The mean run time.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p = np.asarray(
            [
                self.pF(kfpp=kfpp, kfpm=kfpm),
                self.pB(kfpp=kfpp, kfpm=kfpm),
                self.pTs(kfpp=kfpp, kfpm=kfpm),
                self.pLs(kfpp=kfpp, kfpm=kfpm),
                self.pT1(kfpp=kfpp, kfpm=kfpm),
                self.pT2(kfpp=kfpp, kfpm=kfpm),
                self.pT3(kfpp=kfpp, kfpm=kfpm),
            ]
        )
        t = np.asarray(
            [
                self.tF(kfpp=kfpp, kfpm=kfpm),
                self.tB(kfpp=kfpp, kfpm=kfpm),
                self.tTs(kfpp=kfpp, kfpm=kfpm),
                self.tLs(kfpp=kfpp, kfpm=kfpm),
                self.tT1(kfpp=kfpp, kfpm=kfpm),
                self.tT2(kfpp=kfpp, kfpm=kfpm),
                self.tT3(kfpp=kfpp, kfpm=kfpm),
            ]
        )
        return np.sum(t * p) / (self.pT(kfpp=kfpp, kfpm=kfpm))

    def tdwell_pdf(
        self,
        t: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the probability distribution of dwell times.

        :param t: The time to calculate the probability at (seconds).
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinites with.

        :return tdwell pdf: The probability density at t.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        return mm.ph_pdf(t, p0, subtm)

    @lru_cache(maxsize=1)
    def tdwell_gammaapprox(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Find an approximate gamma distribution to describe the dwell time
        phase-type distribution using a moment-matching method.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinites with.

        :return mean: The mean of the distribution.
        :return var: The variance of the distribution.
        :return shape: The shape parameter of the approximating gamma distribution.
        :return scale: The scale parameter of the approximating gamma distribution.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        return mm.approx_ph_gamma(p0, subtm)

    def tdwell_gamma_pdf(
        self,
        t: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the probability distribution of dwell times using an approximate
        gamma distribution.

        :param t: The time to calculate the probability at (seconds).
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinites with.

        :return approx tdwell pdf: The approximate probability density at t.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        _, _, shape, scale = self.tdwell_gammaapprox(
            kfpp=kfpp, kfpm=kfpm, infval=infval
        )
        return mm.gamma_pdf(t, shape, scale)

    def tdwell_mean(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
    ) -> float:
        """
        Calculate the mean dwell time using the definition of the mean of a phase-type
        distribution. Note that this can sometime be numerically unstable with rate
        values > 1e15.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Value to replace infinites with.

        :return mean tdwell: The mean dwell time.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        return mm.ph_mean(p0, subtm)

    @staticmethod
    @njit
    def _vrun_pdf(
        v: float,
        pF: float,
        pB: float,
        pT: float,
        a: float,
        b: float,
        zmax: int,
        nmax: int,
    ) -> float:
        """
        A numba compatible helper function to the vrun_pdf method.
        This function is vectorized in the first argument.

        :param v: The velocities to calculate the density at.
        :param pF: Probability of the motor taking a forward step conditioned on
            not staying put.
        :param pB: Probability of the motor taking a backwards step conditioned on
            not staying put.
        :param pT: Probability of the motor terminating conditioned on not staying put.
        :param a: The shape parameter of the gamma approximation to the phase-type
            distribution of dwell times.
        :param b: The scale parameter of the gamma approximation to the phase-type
            distribution of dwell times.
        :param z_max: The max z to loop to.
        :param n_max: The max n to loop to.

        :return density: The calculated run velocity density.
        """
        res = np.zeros_like(v)
        # Sum over the run distances.
        for z in range(1, zmax):
            t = z / v
            _res = np.zeros_like(v)
            # Sum over the number of reverse steps.
            for n in range(0, nmax):
                binom = mm.binom(z + 2 * n, z + n)
                fact1 = binom * pF ** (z + n) * pB**n
                gamma = mm.gamma_pdf(t, (z + 2 * n + 1) * a, b)
                _res += fact1 * gamma
            res += z * _res
        res *= pT / v**2
        return res

    def vrun_pdf(
        self,
        v: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
        zmax: int = 400,
        nmax: int = 350,
    ) -> float:
        """
        Calculate the run velocity density for a given motor using the most general scheme.
        This function is vectorized in the first argument.

        :param v: Velocities (steps/s) to calculate the density at.
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param z_max: The max z to loop to. Default is 400.
        :param n_max: The max n to loop to. Default is 350.
        :return density: The calculated run velocity density.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)
        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        _, _, shape, scale = mm.approx_ph_gamma(p0, subtm)

        pN = 1.0 - self.pLs(kfpp=kfpp, kfpm=kfpm) - self.pTs(kfpp=kfpp, kfpm=kfpm)
        pF = self.pF(kfpp=kfpp, kfpm=kfpm) / pN
        pB = self.pB(kfpp=kfpp, kfpm=kfpm) / pN
        pT = self.pT(kfpp=kfpp, kfpm=kfpm) / pN

        v = np.asarray(v).astype(float)
        vgtr0 = np.where(v > 0.0)[0]
        vles0 = np.where(v < 0.0)[0]
        veq0 = np.where(v == 0.0)[0]
        # Calculate result for each domain.
        res = np.zeros_like(v)
        v_temp = v[vgtr0]
        if v_temp.size != 0:
            res[vgtr0] = self._vrun_pdf(v_temp, pF, pB, pT, shape, scale, zmax, nmax)
        v_temp = v[vles0]
        if v_temp.size != 0:
            v_temp *= -1
        res[vles0] = self._vrun_pdf(v_temp, pB, pF, pT, shape, scale, zmax, nmax)
        res[veq0] = self.zrun_pmf(0)
        if np.any(np.isnan(res)):
            warnings.warn("Some values are nan. Try reducing n_max first, then z_max.")
        return res

    def vrun_pdf2(
        self,
        v: float,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
        zmax: int = 500,
    ) -> float:
        """
        Calculate the run velocity density for a given motor assuming that the probability of the motor
        taking a backwards step is approx. zero.
        This function is vectorized in the first argument.

        :param v: Velocities (steps/s) to calculate the density at.
        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Replace infinite rates with this value.
        :param z_max: The max z to loop to. Default is 500.

        :return density: The calculated run velocity density.
        """
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        # Generate approx. gamma dist. params.
        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        _, _, shape, scale = mm.approx_ph_gamma(p0, subtm)
        # Calculate conditioned probabilities.
        pN = self.pF(kfpp=kfpp, kfpm=kfpm) + self.pT(kfpp=kfpp, kfpm=kfpm)
        pT = self.pT(kfpp=kfpp, kfpm=kfpm) / pN
        res = np.zeros_like(v)
        # Sum over run distances.
        for z in range(1, zmax):
            fact1 = z * (1 - pT) ** z
            fact2 = mm.gamma_pdf(z / v, (z + 1) * shape, scale)
            res += fact1 * fact2
        res *= pT / v**2
        return res

    @staticmethod
    @njit
    def _vrun_mean(
        pF: float,
        pB: float,
        pT: float,
        shape: float,
        scale: float,
        zmax: int,
        nmax: int,
    ) -> float:
        """
        A numba compatible helper function to the vrun_mean method when case when pF>pB.

        :param pF: Probability of the motor taking a forward step conditioned on
            not staying put.
        :param pB: Probability of the motor taking a backward step conditioned on
            not staying put.
        :param pT: Probability of the motor terminating conditioned on not staying put.
        :param shape: The shape parameter that approximates the dwell time PH
            distribution with a gamma distribution.
        :param scale: The shape parameter that approximates the dwell time PH
            distribution with a gamma distribution.
        :param z_max: The max z to loop to.
        :param n_max: The max n to loot to.

        :return density: The calculated run velocity density.
        """
        res = np.zeros_like(pF)
        # Sum over the run distances.
        for z in range(1, zmax):
            # Sum over the number of backwards steps.
            for n in range(0, nmax):
                fact1 = z / ((z + 2 * n + 1) * shape - 1)
                fact2 = mm.binom(z + 2 * n, z + n)
                fact3 = (pF ** (z + n) * pB**n) - (pB ** (z + n) * pF**n)
                res += fact1 * fact2 * fact3
        res *= pT / scale
        return res[()]

    def vrun_mean(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        zmax: int = 700,
        nmax: int = 300,
        infval: float = 1e12,
        **kwargs,
    ) -> float:
        """
        Calculate the average run velocity for a given motor using the most
        general scheme.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param z_max: The max z to loop to. Note that the value of z_max should several times
            larger than the average run distance to ensure proper coverage of the vrun
            distribution. Numerical instability issues begin for values greater than 700.
        :param n_max: The max n to loop to. Numerical instability issues begin for values
            greater than about 300.
        :param infval: Replace infinite rates with this value.

        :return average: The average run velocity (steps/s).
        """
        # Generate approx. gamma dist. params.
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        _, _, shape, scale = mm.approx_ph_gamma(p0, subtm, **kwargs)
        # Calculate conditioned probabilities.
        pN = 1 - self.pLs(kfpp=kfpp, kfpm=kfpm) - self.pTs(kfpp=kfpp, kfpm=kfpm)
        pF = self.pF(kfpp=kfpp, kfpm=kfpm) / pN
        pB = self.pB(kfpp=kfpp, kfpm=kfpm) / pN
        pT = self.pT(kfpp=kfpp, kfpm=kfpm) / pN

        res = self._vrun_mean(pF, pB, pT, shape, scale, zmax, nmax)
        return res

    def vrun_mean2(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
        **kwargs,
    ) -> float:
        """
        Calculate the average run velocity for a given motor assuming that the
        probability of taking a reverse step is approx. zero.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Replace infinite rates with this value.

        :return average: The average run velocity (steps/s).
        """
        # Generate approx. gamma dist. params.
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        _, _, shape, scale = mm.approx_ph_gamma(p0, subtm, **kwargs)
        # Calculate conditioned probabilities.
        pN = 1 - self.pLs(kfpp=kfpp, kfpm=kfpm) - self.pTs(kfpp=kfpp, kfpm=kfpm)
        pF = self.pF(kfpp=kfpp, kfpm=kfpm) / pN
        fact1 = pF / (2 * shape * scale - scale)
        fact2 = special.hyp2f1(1, 1 - 1 / shape, 3 - 1 / shape, pF)

        return fact1 * fact2

    def vrun_mean3(
        self,
        f: float = 0.0,
        kfpp: float | None = None,
        kfpm: float | None = None,
        infval: float = 1e12,
        **kwargs,
    ) -> float:
        """
        Calculate the average run velocity assuming that the probability of taking a reverse step is approx zero
        and that the shape parameter from the gamma approximation to the phase-type distribution of the dwell times is
        equal to 2.

        :param f: The force experienced by the motor (pN).
        :param kfpp: Rate of first passage to the forward actin binding spot (1/s).
        :param kfpm: Rate of first passage to the reverse actin binding spot (1/s).
        :param infval: Replace infinite rates with this value.

        :return average: The average run velocity (steps/s).
        """
        # Generate approx. gamma dist. params.
        if kfpp is None or kfpm is None:
            kfpp, kfpm = self.kfp(f)

        p0 = self.tdwell_p0()
        subtm = self.tdwell_subtm(kfpp=kfpp, kfpm=kfpm, infval=infval)
        _, _, _, scale = mm.approx_ph_gamma(p0, subtm, **kwargs)

        # Calculate conditioned probabilities.
        pN = 1 - self.pLs(kfpp=kfpp, kfpm=kfpm) - self.pTs(kfpp=kfpp, kfpm=kfpm)
        pF = self.pF(kfpp=kfpp, kfpm=kfpm) / pN

        # Calculate results.
        x = np.sqrt(pF)
        numer = x + (pF - 1) * np.arctanh(x)
        denom = 2 * scale * x
        res = numer / denom
        return res

    def params(self, temperature: float, k: float = 4.2) -> list:
        params = [
            self.kd1,
            self.kd2,
            self.kh,
            self.kps,
            self.b,
            self.Delta,
            self.L,
            self.l_p,
            self.beta(temperature),
            self.nu_c,
            self.theta_c,
            self.theta_F,
            self.a,
            self.D_h,
            k,
        ]
        return params

    def simulate(
        self,
        method: str,
        kind: str,
        trials: int,
        const_force: float | None = None,
        kfpp: float | None = None,
        kfpm: float | None = None,
        time_limit: float | None = None,
        temperature: float = 295,
        k: float = 4.1,
        processes: int = 1,
    ):
        """
        Simulate the behavior of this motor according to the model proposed in
        Hinczewski2013. The available simulation options report observable quantities
        such as run distances, run times, and dwell times. Additional, options to
        to obtain trajectories motors take along the underlying kinetic network
        and the sequence of events executed are available.

        :param method: The simulation backend to use. They differ in implementation only.
            'kmc': A kinetic monte carlo based backend.
            'emc': An event monte carlo based backend.
        :param kind: The type of simulation to run:
            'trajectory': A trajectory of the states visited by the motor.
                Only available with the 'kmc' method.
            'run': The events executed by the motor during its run.
                Only available with the the 'emc' method.
            'runstats': Determine run distances and run times.
                 Available with both methods.
            'dwell': Record dwell times.
                 Availible with both methods.
        :param trials: Depends of the value of kind:
            kind='trajectory': The number of states to record in the trajectory.
            kind='run': The number of events to record from the run.
            kind='runstats': The number of runs to record the run time and distance of.
            kind='dwell': The number of dwell times to record.
        :param const_force: If >=0, force value to use when calculating kfpp and kfpm (pN).
        :param kfpp: If > 0 with kfpm, kfpp value to use in final matrix (1/s).
        :param kfpm: If > 0 with kfpp, kfpm value to use in final matrix (1/s).
        :param time_limit: The upper limit of run or dwell times to prevent sudo
            infinite loops. Defualt is None but must be set to a positive value for
            'kind' = 'dwell' and 'runstats'.
        :param temperature: The temperature of the experiment (K).
        :param k: The force constant of the compliant trap (pN/nm).
        :param processes: The number of processes to use during the simulation.
            If 'kind' = 'trajectory' or 'run' then only a single process is allowed.

        :return result: An np.ndarray with the following characteristics depending on 'kind'.
            'trajectory': trajectory[0]: Index of the stats visited (see module docstring).
                          trajectory[1]: Current displacement (steps).
                          trajectory[2]: Current time (seconds).

            'run':        run[0]: The events that were drawn (see module docstring).
                          run[1]: The position of the motor following the event (steps).
                          run[2]: The times at which the event finishes (seconds).

            'runstats':   runstats[0]: Run distances (steps).
                          runstats[1]: Run times (seconds).

            'dwell':      tdwells[0]: Dwell times (seconds).
                          tdwells[1]: Displacement when dwell time is recorded (steps).
        """

        assert method in ["kmc", "emc"]
        assert kind in ["trajectory", "run", "runstats", "dwell"]
        if method == "kmc":
            assert kind != "run"
        elif method == "emc":
            assert kind != "trajectory"

        const_force = const_force if const_force is not None else -1
        kfpp = kfpp if kfpp is not None else -1
        kfpm = kfpm if kfpm is not None else -1

        params = self.params(temperature, k)
        sim_func = kmc.kmc if method == "kmc" else emc.emc
        simres = sim_func(
            params, trials, kind, const_force, kfpp, kfpm, time_limit, processes
        )
        classdict = {
            "trajectory": TrajRes,
            "run": RunRes,
            "runstats": RunStatsRes,
            "dwell": DwellRes,
        }
        resclass = classdict.get(kind)
        return resclass.from_sim(self, const_force, kfpp, kfpm, simres)

    def loglh(
        self,
        dataset: str,
        trials: int,
        time_limit: float = 50.0,
        method: str = "emc",
        temperature: float = 295,
        k: float = 4.2,
        processes: int = 1,
        force_cutoff: float | None = None,
        time_cutoff: float | None = None,
        density_cutoff: float | None = None,
    ) -> float:
        """
        Calculate the log likelihood that this motor describes the experimental
        force-at-detachment (figure 5B) or dwell time (figure 8B)
        distributions for Nt. Myosin-XI obtained under a compliant optical trap
        published in Tominaga2003.

        The logLH value is calculated by first running simulations, processing
        the results to mimic the data analysis of Tominaga2003, then binned
        according to those used in Tominaga2003. The resulting probabilities
        associated with each bin are then used to calculate the logLH in conjunction
        with the experimentally observed counts for each bin.

        :param dataset: A key for the dataset to retrieve. Options include:
            'detach 1uM':   Figure 5B, 1uM ATP.
            'detach 100uM': Figure 5B, 100uM ATP
            'dwell 1uM':    Figure 8B, 1uM ATP
            'dwell 100uM':  Figure 8B, 100uM ATP
        :param trials: Number of trials that will be used to determine the model's
            distribution.
        :param method: The simulation backend to use. They differ in implementation only.
            'kmc': A kinetic monte carlo based backend.
            'emc': An event monte carlo based backend.
        :param temperature: The temperature of the experiment (K).
        :param k: The force constant of the compliant trap (pN/nm).
        :param processes: The number of processes to use during the simulation.
        :param force_cutoff: Scalar. Only keep dwell times that occur at forces larger
            than the cutoff. This mimics how thermal noise obscures low force values (pN).
            Needed for both kinds of datasets.
        :param time_cutoff: Scalar. Only keep dwell times longer than the cutoff.
            This simulates the non-infinitesimal rise times of the optical trap (seconds).
            Only needed for dwell time datasets.
        :param density_cutoff: Scalar. If the proportion of remaining samples after
            enforcing the force cutoff is too small, then the logLH is -infinity. This is used to
            ensure that enough samples remain following the force cutoff. Only
            needed for detachment datasets.

        :returns logLH: The log likelihood that this motor describes the dataset.
        """

        kind_dict = {
            "dwell 1um": "dwell",
            "dwell 100um": "dwell",
            "detach 1um": "runstats",
            "detach 100um": "runstats",
            "vrun 100um": "runstats",
        }

        assert dataset in kind_dict.keys()
        assert method in ["kmc", "emc"]

        kind = kind_dict.get(dataset)
        simres = self.simulate(
            method, kind, trials, time_limit=time_limit, processes=processes
        )
        if "vrun" in dataset:
            vruns = fit.proc_vrun(
                simres.vruns, simres.zruns, self.Delta, k, force_cutoff
            )
            if vruns is None:
                return -np.inf

            vrun_mean = np.average(vruns)
            return fit.vrun_logprior(vrun_mean * self.Delta / 1000)

        pdf = None
        if "dwell" in dataset:
            pdf = fit.proc_dwell(
                simres.tdwells,
                self.force(simres.zlocs + 0.5, k),
                force_cutoff,
                time_cutoff,
                density_cutoff,
            )
        elif "detach" in dataset:
            pdf = fit.proc_detach(
                simres.zruns,
                self.Delta,
                k,
                self.beta(temperature),
                force_cutoff,
                density_cutoff,
            )

        expdata = fit.get_expdata(dataset)
        return fit.loglh_frompdf(pdf, *expdata)


class SimRes(object):
    """
    Class to hold simulation results.
    """

    def __init__(self, motor: Motor, f: float, kfpp: float | None, kfpm: float | None):
        self.motor = motor
        self.f = f
        self.kfpp = kfpp
        self.kfpm = kfpm

        if self.f >= 0.0 and (self.kfpp < 0 or self.kfpm < 0):
            self.kfpp, self.kfpm = self.motor.kfp(f=self.f)


class RunStatsRes(SimRes):
    """
    Class to hold results from RunStat stimulations.
    """

    def __init__(
        self,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        zruns: NDArray[np.float64],
        truns: NDArray[np.float64],
    ):
        self.zruns = zruns
        self.truns = truns
        self.vruns = self.zruns / self.truns
        SimRes.__init__(self, motor, f, kfpp, kfpm)

    @classmethod
    def from_sim(
        cls,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        simdata: NDArray[np.float64],
    ):
        return cls(motor, f, kfpp, kfpm, simdata[0], simdata[1])

    def zrun_mean(self) -> float:
        """
        Calculate the mean run distance from the simulation data.

        :return: Mean run distance (steps).
        """
        return np.average(self.zruns)

    def trun_mean(self) -> float:
        """
        Calculate the mean run time from the simulation data.

        :return: Mean run time (s).
        """
        return np.average(self.truns)

    def vrun_mean(self) -> float:
        """
        Calculate the mean run velocity from the simulation data.

        :return: Mean run velocity (steps/s).
        """
        return np.average(self.vruns)

    def vrun_productive_mean(self) -> float:
        """
        Calculate the mean run velocity from teh simulation data only from
        entries with non zero run velocity.

        :return: Mean productive run velocity (steps/s).
        """
        productive_runs = self.vruns > 0
        productive_vruns = self.vruns[productive_runs]
        if productive_vruns.size == 0:
            return 0.0
        else:
            return np.average(productive_vruns)


class DwellRes(SimRes):
    """
    Class to hold results from tdwell simulations.
    """

    def __init__(
        self,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        tdwells: NDArray[np.float64],
        zlocs: NDArray[np.float64],
    ):
        self.tdwells = tdwells
        self.zlocs = zlocs
        SimRes.__init__(self, motor, f, kfpp, kfpm)

    @classmethod
    def from_sim(
        cls,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        simdata: NDArray[np.float64],
    ):
        return cls(motor, f, kfpp, kfpm, simdata[0], simdata[1])

    def tdwell_mean(self) -> float:
        """
        Calculate the mean dwell time from the simulation data.

        :return: Mean dwell time (s).
        """
        return np.average(self.tdwells)


class TrajRes(SimRes):
    """
    Class to hold results from trajectory simulations.
    """

    def __init__(
        self,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        states: NDArray[np.float64],
        disp: NDArray[np.float64],
        time: NDArray[np.float64],
    ):
        self.states = states
        self.disps = disp
        self.times = time
        SimRes.__init__(self, motor, f, kfpp, kfpm)

    @classmethod
    def from_sim(
        cls,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        simdata: NDArray[np.float64],
    ):
        return cls(motor, f, kfpp, kfpm, simdata[0], simdata[1], simdata[2])


class RunRes(SimRes):
    """
    Class to hold results from run simulations.
    """

    def __init__(self, motor, f, kfpp, kfpm, events, disp, times):
        self.events = events
        self.disps = disp
        self.times = times
        SimRes.__init__(self, motor, f, kfpp, kfpm)

    @classmethod
    def from_sim(
        cls,
        motor: Motor,
        f: float,
        kfpp: float | None,
        kfpm: float | None,
        simdata: NDArray[np.float64],
    ):
        return cls(motor, f, kfpp, kfpm, simdata[0], simdata[1], simdata[2])
