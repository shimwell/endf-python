# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/{angle_energy,uncorrelated,correlated,kalbach_mann,
# nbody}.py, restricted to reading. The ACE laws are dispatched by
# AngleEnergy.from_ace.

"""Joint distributions of secondary particle angle and energy.

A reaction product's angle and energy may be described independently
(:class:`UncorrelatedAngleEnergy`), as an angular distribution conditional on
the outgoing energy (:class:`CorrelatedAngleEnergy`), through the Kalbach-Mann
systematics (:class:`KalbachMann`), or by N-body phase space kinematics
(:class:`NBodyPhaseSpace`).
"""

from __future__ import annotations

from abc import ABC
from warnings import warn

import numpy as np

from .data import EV_PER_MEV
from .function import Tabulated1D, INTERPOLATION_SCHEME
from .mf5 import (
    ContinuousTabular, DiscretePhoton, Evaporation, GeneralEvaporation,
    LevelInelastic, MaxwellEnergy, WattEnergy, _ace_incident_grid,
    _ace_outgoing_energy,
)
from .univariate import Tabular, Uniform


__all__ = [
    "AngleEnergy", "UncorrelatedAngleEnergy", "CorrelatedAngleEnergy",
    "KalbachMann", "NBodyPhaseSpace",
]


class AngleEnergy(ABC):
    """Distribution in secondary angle and energy."""

    @staticmethod
    def from_ace(table, location_dist: int, location_start: int,
                 rx=None) -> AngleEnergy:
        """Generate an angle-energy distribution from ACE data.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        location_dist
            Index in the XSS array corresponding to the start of a block,
            e.g. JXS(11) for the DLW block
        location_start
            Index in the XSS array corresponding to the start of an energy
            distribution array
        rx : endf.Reaction, optional
            Reaction this distribution belongs to. Only needed for law 66, which
            takes the reaction Q value.

        Returns
        -------
        Secondary angle-energy distribution

        """
        idx = location_dist + location_start - 1

        law = int(table.xss[idx + 1])
        location_data = int(table.xss[idx + 2])

        # Position index for reading law data
        idx = location_dist + location_data - 1

        if law == 2:
            return UncorrelatedAngleEnergy(
                energy=DiscretePhoton.from_ace(table, idx))
        if law in (3, 33):
            return UncorrelatedAngleEnergy(
                energy=LevelInelastic.from_ace(table, idx))
        if law == 4:
            return UncorrelatedAngleEnergy(
                energy=ContinuousTabular.from_ace(table, idx, location_dist))
        if law == 5:
            return UncorrelatedAngleEnergy(
                energy=GeneralEvaporation.from_ace(table, idx))
        if law == 7:
            return UncorrelatedAngleEnergy(
                energy=MaxwellEnergy.from_ace(table, idx))
        if law == 9:
            return UncorrelatedAngleEnergy(
                energy=Evaporation.from_ace(table, idx))
        if law == 11:
            return UncorrelatedAngleEnergy(
                energy=WattEnergy.from_ace(table, idx))
        if law == 44:
            return KalbachMann.from_ace(table, idx, location_dist)
        if law == 61:
            return CorrelatedAngleEnergy.from_ace(table, idx, location_dist)
        if law == 66:
            return NBodyPhaseSpace.from_ace(table, idx, rx.q_reaction)
        raise ValueError(
            f"Unsupported ACE secondary energy distribution law {law}")


class UncorrelatedAngleEnergy(AngleEnergy):
    """Uncorrelated angle-energy distribution.

    The angle and energy of the secondary particle are sampled independently.

    Parameters
    ----------
    angle : endf.AngleDistribution, optional
        Distribution of outgoing angles represented as scattering cosines
    energy : optional
        Distribution of outgoing energies

    Attributes
    ----------
    angle : endf.AngleDistribution or None
        Distribution of outgoing angles represented as scattering cosines
    energy : None or an energy distribution
        Distribution of outgoing energies

    """

    def __init__(self, angle=None, energy=None):
        self.angle = angle
        self.energy = energy

    def __repr__(self) -> str:
        energy = type(self.energy).__name__ if self.energy is not None else None
        return f"<UncorrelatedAngleEnergy: energy={energy}>"


class KalbachMann(AngleEnergy):
    """Kalbach-Mann distribution.

    The outgoing energy is tabulated, and the angular distribution at each
    outgoing energy is described by the Kalbach-Mann systematics through a
    precompound fraction and a slope value.

    Parameters
    ----------
    breakpoints
        Breakpoints defining interpolation regions
    interpolation
        Interpolation codes
    energy
        Incident energies in eV at which distributions exist
    energy_out
        Distribution of outgoing energies corresponding to each incident energy
    precompound
        Precompound factor 'r' as a function of outgoing energy for each
        incident energy
    slope
        Kalbach-Chadwick angular distribution slope value of 'a' as a function
        of outgoing energy for each incident energy

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incident energies in eV at which distributions exist
    energy_out : Iterable of Univariate
        Distribution of outgoing energies corresponding to each incident energy
    precompound : Iterable of Tabulated1D
        Precompound factor 'r' as a function of outgoing energy
    slope : Iterable of Tabulated1D
        Kalbach-Chadwick angular distribution slope value of 'a'

    """

    def __init__(self, breakpoints, interpolation, energy, energy_out,
                 precompound, slope):
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out
        self.precompound = precompound
        self.slope = slope

    def __repr__(self) -> str:
        return f"<KalbachMann: {len(self.energy)} incident energies>"

    @classmethod
    def from_ace(cls, table, idx: int, ldis: int) -> KalbachMann:
        """Generate a Kalbach-Mann distribution from ACE data.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Index in the XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis
            Index in the XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        """
        breakpoints, interpolation, energy, loc_dist = _ace_incident_grid(
            table, idx)

        energy_out = []
        km_r = []
        km_a = []
        for i in range(len(energy)):
            idx = ldis + loc_dist[i] - 1
            eout_i, data = _ace_outgoing_energy(table, idx, n_cols=5)
            energy_out.append(eout_i)
            km_r.append(Tabulated1D(data[0], data[3]))
            km_a.append(Tabulated1D(data[0], data[4]))

        return cls(breakpoints, interpolation, energy, energy_out, km_r, km_a)


class CorrelatedAngleEnergy(AngleEnergy):
    """Correlated angle-energy distribution.

    The outgoing energy is tabulated, and each outgoing energy carries its own
    tabulated angular distribution.

    Parameters
    ----------
    breakpoints
        Breakpoints defining interpolation regions
    interpolation
        Interpolation codes
    energy
        Incident energies in eV at which distributions exist
    energy_out
        Distribution of outgoing energies corresponding to each incident energy
    mu
        Distribution of scattering cosines for each pair of incident and
        outgoing energies

    Attributes
    ----------
    breakpoints : Iterable of int
        Breakpoints defining interpolation regions
    interpolation : Iterable of int
        Interpolation codes
    energy : Iterable of float
        Incident energies in eV at which distributions exist
    energy_out : Iterable of Univariate
        Distribution of outgoing energies corresponding to each incident energy
    mu : Iterable of Iterable of Univariate
        Distribution of scattering cosines for each pair of incident and
        outgoing energies

    """

    _name = 'correlated'

    def __init__(self, breakpoints, interpolation, energy, energy_out, mu):
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out
        self.mu = mu

    def __repr__(self) -> str:
        return f"<CorrelatedAngleEnergy: {len(self.energy)} incident energies>"

    @classmethod
    def from_ace(cls, table, idx: int, ldis: int) -> CorrelatedAngleEnergy:
        """Generate a correlated angle-energy distribution from ACE data.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Index in the XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        ldis
            Index in the XSS array of the start of the energy distribution block
            (e.g. JXS[11])

        """
        breakpoints, interpolation, energy, loc_dist = _ace_incident_grid(
            table, idx)

        energy_out = []
        mu = []
        for i in range(len(energy)):
            idx = ldis + loc_dist[i] - 1
            eout_i, data = _ace_outgoing_energy(table, idx, n_cols=4)
            energy_out.append(eout_i)

            # The fourth column locates each outgoing energy's angular
            # distribution; zero means isotropic.
            lc = data[3].astype(int)
            n_energy_out = data.shape[1]
            mu_i = []
            for j in range(n_energy_out):
                if lc[j] > 0:
                    idx = ldis + abs(lc[j]) - 1
                    intt = int(table.xss[idx])
                    n_cosine = int(table.xss[idx + 1])
                    mu_data = table.xss[idx + 2:idx + 2 + 3 * n_cosine]
                    mu_data.shape = (3, n_cosine)

                    mu_ij = Tabular(mu_data[0], mu_data[1],
                                    INTERPOLATION_SCHEME[intt])
                    mu_ij.c = mu_data[2]
                else:
                    mu_ij = Uniform(-1., 1.)
                mu_i.append(mu_ij)

            mu.append(mu_i)

        return cls(breakpoints, interpolation, energy, energy_out, mu)


class NBodyPhaseSpace(AngleEnergy):
    """N-body phase space distribution.

    Parameters
    ----------
    total_mass
        Total mass of product particles
    n_particles
        Number of product particles
    atomic_weight_ratio
        Atomic weight ratio of the target nuclide
    q_value
        Q value for the reaction in eV

    Attributes
    ----------
    total_mass : float
        Total mass of product particles
    n_particles : int
        Number of product particles
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide
    q_value : float
        Q value for the reaction in eV

    """

    def __init__(self, total_mass, n_particles, atomic_weight_ratio, q_value):
        self.total_mass = total_mass
        self.n_particles = n_particles
        self.atomic_weight_ratio = atomic_weight_ratio
        self.q_value = q_value

    def __repr__(self) -> str:
        return f"<NBodyPhaseSpace: {self.n_particles} particles>"

    @classmethod
    def from_ace(cls, table, idx: int, q_value: float) -> NBodyPhaseSpace:
        """Generate an N-body phase space distribution from ACE data.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Index in the XSS array of the start of the energy distribution data
            (LDIS + LOCC - 1)
        q_value
            Q value for the reaction in eV

        """
        n_particles = int(table.xss[idx])
        total_mass = table.xss[idx + 1]
        return cls(total_mass, n_particles, table.atomic_weight_ratio, q_value)
