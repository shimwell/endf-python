# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

from abc import ABC
from typing import TextIO
from warnings import warn

import numpy as np

from .data import EV_PER_MEV
from .function import Tabulated1D, INTERPOLATION_SCHEME
from .records import get_tab1_record, get_tab2_record, get_head_record
from .univariate import Discrete, Mixture, Tabular


def parse_mf5(file_obj: TextIO) -> dict:
    ZA, AWR, _, _, NK, _ = get_head_record(file_obj)

    data = {'ZA': ZA, 'AWR': AWR, 'NK': NK}
    data['subsections'] = []
    for _ in range(NK):
        subsection = {}
        params, applicability = get_tab1_record(file_obj)
        subsection['LF'] = LF = params[3]
        subsection['p'] = applicability
        if LF == 1:
            dist = ArbitraryTabulated.dict_from_endf(file_obj, params)
        elif LF == 5:
            dist = GeneralEvaporation.dict_from_endf(file_obj, params)
        elif LF == 7:
            dist = MaxwellEnergy.dict_from_endf(file_obj, params)
        elif LF == 9:
            dist = Evaporation.dict_from_endf(file_obj, params)
        elif LF == 11:
            dist = WattEnergy.dict_from_endf(file_obj, params)
        elif LF == 12:
            dist = MadlandNix.dict_from_endf(file_obj, params)

        subsection['distribution'] = dist
        data['subsections'].append(subsection)

    return data



class EnergyDistribution(ABC):
    """Abstract superclass for all energy distributions."""
    def __init__(self):
        pass

    @staticmethod
    def from_endf(file_obj: TextIO, params: list):
        """Generate energy distribution from MF=5 data

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        A sub-class of :class:`EnergyDistribution`

        """
        LF = params[3]
        if LF == 1:
            return ArbitraryTabulated.from_endf(file_obj, params)
        elif LF == 5:
            return GeneralEvaporation.from_endf(file_obj, params)
        elif LF == 7:
            return MaxwellEnergy.from_endf(file_obj, params)
        elif LF == 9:
            return Evaporation.from_endf(file_obj, params)
        elif LF == 11:
            return WattEnergy.from_endf(file_obj, params)
        elif LF == 12:
            return MadlandNix.from_endf(file_obj, params)

    @staticmethod
    def from_dict(subsection: dict):
        LF = subsection['LF']
        data = subsection['distribution']
        if LF == 1:
            return ArbitraryTabulated.from_dict(data)
        elif LF == 5:
            return GeneralEvaporation.from_dict(data)
        elif LF == 7:
            return MaxwellEnergy.from_dict(data)
        elif LF == 9:
            return Evaporation.from_dict(data)
        elif LF == 11:
            return WattEnergy.from_dict(data)
        elif LF == 12:
            return MadlandNix.from_dict(data)


class ArbitraryTabulated(EnergyDistribution):
    r"""Arbitrary tabulated function given in ENDF MF=5, LF=1 represented as

    .. math::
         f(E \rightarrow E') = g(E \rightarrow E')

    Parameters
    ----------
    energy : numpy.ndarray
        Array of incident neutron energies
    pdf : list of openmc.data.Tabulated1D
        Tabulated outgoing energy distribution probability density functions

    Attributes
    ----------
    energy : numpy.ndarray
        Array of incident neutron energies
    pdf : list of openmc.data.Tabulated1D
        Tabulated outgoing energy distribution probability density functions

    """

    def __init__(self, energy, pdf):
        super().__init__()
        self.energy = energy
        self.pdf = pdf

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse arbitrary tabulated distribution (LF=1)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.

        Returns
        -------
        dict
            Arbitrary tabulated distribution data

        """
        data = {}
        params, data['E_int'] = get_tab2_record(file_obj)
        n_energies = params[5]

        energy = np.zeros(n_energies)
        pdf = []
        for j in range(n_energies):
            params, func = get_tab1_record(file_obj)
            energy[j] = params[1]
            pdf.append(func)
        data['E'] = energy
        data['g'] = pdf
        return data

    @classmethod
    def from_endf(cls, file_obj: TextIO, params: list):
        data = cls.dict_from_endf(file_obj, params)
        return cls(data['E'], data['g'])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['E'], data['g'])



class GeneralEvaporation(EnergyDistribution):
    r"""General evaporation spectrum given in ENDF MF=5, LF=5 represented as

    .. math::
        f(E \rightarrow E') = g(E'/\theta(E))

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy :math:`E`
    g : openmc.data.Tabulated1D
        Tabulated function of :math:`x = E'/\theta(E)`
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy :math:`E`
    g : openmc.data.Tabulated1D
        Tabulated function of :math:`x = E'/\theta(E)`
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, g, u):
        super().__init__()
        self.theta = theta
        self.g = g
        self.u = u

    @classmethod
    def from_ace(cls, ace, idx=0):
        """Not implemented: ACE law 5 is not read.

        `AngleEnergy.from_ace` dispatches here for law 5, so without this the
        call fails with `AttributeError` and reads like an internal error
        rather than an unsupported format. OpenMC carries the same stub. See
        issue #19.
        """
        raise NotImplementedError(
            "ACE law 5 (general evaporation) is not implemented"
        )

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse general evaporation spectrum (MF=5)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        openmc.data.GeneralEvaporation
            General evaporation spectrum

        """
        _, theta = get_tab1_record(file_obj)
        _, g = get_tab1_record(file_obj)
        return {'U': params[0], 'theta': theta, 'g': g}

    @classmethod
    def from_endf(cls, file_obj: TextIO, params: list):
        data = cls.dict_from_endf(file_obj, params)
        return cls(data['theta'], data['g'], data['U'])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['theta'], data['g'], data['U'])


class MaxwellEnergy(EnergyDistribution):
    r"""Simple Maxwellian fission spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{\sqrt{E'}}{I} e^{-E'/\theta(E)}

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    @classmethod
    def from_ace(cls, table, idx=0):
        """Create a Maxwell fission spectrum from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Offset to read from in the XSS array

        """
        # Nuclear temperature, stored in MeV
        theta = Tabulated1D.from_ace(table, idx)
        theta.y = theta.y * EV_PER_MEV

        # Restriction energy
        nr = int(table.xss[idx])
        ne = int(table.xss[idx + 1 + 2 * nr])
        u = table.xss[idx + 2 + 2 * nr + 2 * ne] * EV_PER_MEV
        return cls(theta, u)

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse Maxwellian fission spectrum (LF=7)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        dict
            Maxwellian distribution data

        """
        _, theta = get_tab1_record(file_obj)
        return {'U': params[0], 'theta': theta}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['theta'], data['U'])


class Evaporation(EnergyDistribution):
    r"""Evaporation spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{E'}{I} e^{-E'/\theta(E)}

    Parameters
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    theta : openmc.data.Tabulated1D
        Tabulated function of incident neutron energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, theta, u):
        super().__init__()
        self.theta = theta
        self.u = u

    @classmethod
    def from_ace(cls, table, idx=0):
        """Create a evaporation spectrum from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Offset to read from in the XSS array

        """
        # Nuclear temperature, stored in MeV
        theta = Tabulated1D.from_ace(table, idx)
        theta.y = theta.y * EV_PER_MEV

        # Restriction energy
        nr = int(table.xss[idx])
        ne = int(table.xss[idx + 1 + 2 * nr])
        u = table.xss[idx + 2 + 2 * nr + 2 * ne] * EV_PER_MEV
        return cls(theta, u)

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse evaporation spectrum (LF=9)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        data
            Evaporation spectrum data

        """
        _, theta = get_tab1_record(file_obj)
        return {'U': params[0], 'theta': theta}

    @classmethod
    def from_endf(cls, file_obj: TextIO, params: list):
        data = cls.dict_from_endf(file_obj, params)
        return cls(data['theta'], data['U'])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['theta'], data['U'])


class WattEnergy(EnergyDistribution):
    r"""Energy-dependent Watt spectrum represented as

    .. math::
        f(E \rightarrow E') = \frac{e^{-E'/a}}{I} \sinh \left ( \sqrt{bE'}
        \right )

    Parameters
    ----------
    a, b : openmc.data.Tabulated1D
        Energy-dependent parameters tabulated as function of incident neutron
        energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    Attributes
    ----------
    a, b : openmc.data.Tabulated1D
        Energy-dependent parameters tabulated as function of incident neutron
        energy
    u : float
        Constant introduced to define the proper upper limit for the final
        particle energy such that :math:`0 \le E' \le E - U`

    """

    def __init__(self, a, b, u):
        super().__init__()
        self.a = a
        self.b = b
        self.u = u

    @classmethod
    def from_ace(cls, table, idx=0):
        """Create a Watt fission spectrum from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Offset to read from in the XSS array

        """
        # Energy-dependent a parameter, stored in MeV
        a = Tabulated1D.from_ace(table, idx)
        a.y = a.y * EV_PER_MEV

        nr = int(table.xss[idx])
        ne = int(table.xss[idx + 1 + 2 * nr])
        idx += 2 + 2 * nr + 2 * ne

        # Energy-dependent b parameter, stored in MeV^-1
        b = Tabulated1D.from_ace(table, idx)
        b.y = b.y / EV_PER_MEV

        nr = int(table.xss[idx])
        ne = int(table.xss[idx + 1 + 2 * nr])
        idx += 2 + 2 * nr + 2 * ne

        # Restriction energy
        u = table.xss[idx] * EV_PER_MEV
        return cls(a, b, u)

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse energy-dependent Watt spectrum (MF=11)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        data
            Watt fission spectrum data

        """
        _, a = get_tab1_record(file_obj)
        _, b = get_tab1_record(file_obj)
        return {'U': params[0], 'a': a, 'b': b}

    @classmethod
    def from_endf(cls, file_obj: TextIO, params: list):
        data = cls.dict_from_endf(file_obj, params)
        return cls(data['a'], data['b'], data['U'])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['a'], data['b'], data['U'])

class MadlandNix(EnergyDistribution):
    r"""Energy-dependent fission neutron spectrum (Madland and Nix) given in
    ENDF MF=5, LF=12 represented as

    .. math::
        f(E \rightarrow E') = \frac{1}{2} [ g(E', E_F(L)) + g(E', E_F(H))]

    where

    .. math::
        g(E',E_F) = \frac{1}{3\sqrt{E_F T_M}} \left [ u_2^{3/2} E_1 (u_2) -
        u_1^{3/2} E_1 (u_1) + \gamma \left ( \frac{3}{2}, u_2 \right ) - \gamma
        \left ( \frac{3}{2}, u_1 \right ) \right ] \\ u_1 = \left ( \sqrt{E'} -
        \sqrt{E_F} \right )^2 / T_M \\ u_2 = \left ( \sqrt{E'} + \sqrt{E_F}
        \right )^2 / T_M.

    Parameters
    ----------
    efl, efh : float
        Constants which represent the average kinetic energy per nucleon of the
        fission fragment (efl = light, efh = heavy)
    tm : openmc.data.Tabulated1D
        Parameter tabulated as a function of incident neutron energy

    Attributes
    ----------
    efl, efh : float
        Constants which represent the average kinetic energy per nucleon of the
        fission fragment (efl = light, efh = heavy)
    tm : openmc.data.Tabulated1D
        Parameter tabulated as a function of incident neutron energy

    """

    def __init__(self, efl, efh, tm):
        super().__init__()
        self.efl = efl
        self.efh = efh
        self.tm = tm

    @staticmethod
    def dict_from_endf(file_obj: TextIO, params: list) -> dict:
        """Parse Madland-Nix fission spectrum (LF=12)

        Parameters
        ----------
        file_obj : file-like object
            ENDF file positioned at the start of a section for an energy
            distribution.
        params : list
            List of parameters at the start of the energy distribution that
            includes the LF value indicating what type of energy distribution is
            present.

        Returns
        -------
        data
            Madland-Nix fission spectrum data

        """
        _, T_M = get_tab1_record(file_obj)
        return {'EFL': params[0], 'EFH': params[1], 'T_M': T_M}

    @classmethod
    def from_endf(cls, file_obj: TextIO, params: list):
        data = cls.dict_from_endf(file_obj, params)
        return cls(data['EFL'], data['EFH'], data['T_M'])

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data['EFL'], data['EFH'], data['T_M'])


class LevelInelastic:
    r"""Level inelastic scattering

    Parameters
    ----------
    threshold : float
        Energy threshold in the laboratory system, :math:`(A + 1)/A * |Q|`
    mass_ratio : float
        :math:`(A/(A + 1))^2`

    Attributes
    ----------
    threshold : float
        Energy threshold in the laboratory system, :math:`(A + 1)/A * |Q|`
    mass_ratio : float
        :math:`(A/(A + 1))^2`

    """

    def __init__(self, threshold, mass_ratio):
        self.threshold = threshold
        self.mass_ratio = mass_ratio

    @classmethod
    def from_ace(cls, table, idx):
        """Generate a level inelastic distribution from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Offset to read from in the XSS array

        """
        threshold = table.xss[idx] * EV_PER_MEV
        mass_ratio = table.xss[idx + 1]
        return cls(threshold, mass_ratio)


class DiscretePhoton(EnergyDistribution):
    """Discrete photon energy distribution.

    Parameters
    ----------
    primary_flag
        Indicator of whether the photon is a primary or non-primary photon.
    energy
        Photon energy (if primary) or binding energy (if non-primary) in eV
    atomic_weight_ratio
        Atomic weight ratio of the target nuclide responsible for the emitted
        particle

    Attributes
    ----------
    primary_flag : int
        Indicator of whether the photon is a primary or non-primary photon.
    energy : float
        Photon energy (if primary) or binding energy (if non-primary) in eV
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide responsible for the emitted
        particle

    """

    def __init__(self, primary_flag, energy, atomic_weight_ratio):
        super().__init__()
        self.primary_flag = primary_flag
        self.energy = energy
        self.atomic_weight_ratio = atomic_weight_ratio

    @classmethod
    def from_ace(cls, table, idx):
        """Generate a discrete photon energy distribution from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from
        idx
            Offset to read from in the XSS array

        """
        primary_flag = int(table.xss[idx])
        energy = table.xss[idx + 1] * EV_PER_MEV
        return cls(primary_flag, energy, table.atomic_weight_ratio)


class ContinuousTabular(EnergyDistribution):
    """Continuous tabular distribution.

    The outgoing energy distribution is tabulated at a set of incident
    energies, and interpolated between them.

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

    """

    def __init__(self, breakpoints, interpolation, energy, energy_out):
        super().__init__()
        self.breakpoints = breakpoints
        self.interpolation = interpolation
        self.energy = energy
        self.energy_out = energy_out

    @classmethod
    def from_ace(cls, table, idx, ldis):
        """Generate a continuous tabular energy distribution from ACE data.

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
        for i in range(len(energy)):
            idx = ldis + loc_dist[i] - 1
            energy_out.append(_ace_outgoing_energy(table, idx, n_cols=3)[0])

        return cls(breakpoints, interpolation, energy, energy_out)


def _ace_incident_grid(table, idx):
    """Read the incident energy grid shared by ACE laws 4, 44 and 61.

    Returns ``(breakpoints, interpolation, energy, loc_dist)``, where
    ``loc_dist`` locates each incident energy's outgoing distribution.
    """
    n_regions = int(table.xss[idx])
    n_energy_in = int(table.xss[idx + 1 + 2 * n_regions])

    idx += 1
    if n_regions > 0:
        breakpoints = table.xss[idx:idx + n_regions].astype(int)
        interpolation = table.xss[
            idx + n_regions:idx + 2 * n_regions].astype(int)
    else:
        # Zero regions implies lin-lin interpolation by default
        breakpoints = np.array([n_energy_in])
        interpolation = np.array([2])

    idx += 2 * n_regions + 1
    energy = table.xss[idx:idx + n_energy_in] * EV_PER_MEV

    idx += n_energy_in
    loc_dist = table.xss[idx:idx + n_energy_in].astype(int)

    return breakpoints, interpolation, energy, loc_dist


def _ace_outgoing_energy(table, idx, n_cols):
    """Read one tabulated outgoing energy distribution from an ACE table.

    Laws 4, 44 and 61 all store the outgoing energy the same way and differ only
    in how many extra columns follow: three columns (energy, PDF, CDF) for law
    4, five for Kalbach-Mann (precompound fraction and slope), four for
    correlated angle-energy (a locator for the angular distribution).

    Returns ``(distribution, data)``, where ``data`` has shape
    ``(n_cols, n_energy_out)`` so the caller can read the extra columns, and the
    outgoing energies in row 0 are already converted to eV.
    """
    # intt is the interpolation scheme (1 = histogram, 2 = lin-lin). When
    # discrete lines are present the stored value is 10*n_discrete_lines + intt.
    n_discrete_lines, intt = divmod(int(table.xss[idx]), 10)
    if intt not in (1, 2):
        warn("Interpolation scheme for continuous tabular distribution is not "
             "histogram or linear-linear.")
        intt = 2

    n_energy_out = int(table.xss[idx + 1])
    data = table.xss[idx + 2:idx + 2 + n_cols * n_energy_out].copy()
    data.shape = (n_cols, n_energy_out)
    data[0, :] *= EV_PER_MEV

    # Law 4 rejects negative probabilities; laws 44 and 61 tolerate them but
    # warn, because ACE files do contain small negative entries.
    ignore_negative = n_cols > 3
    eout_continuous = Tabular(
        data[0][n_discrete_lines:], data[1][n_discrete_lines:] / EV_PER_MEV,
        INTERPOLATION_SCHEME[intt], ignore_negative=ignore_negative)
    eout_continuous.c = data[2][n_discrete_lines:]
    if ignore_negative and np.any(data[1][n_discrete_lines:] < 0.0):
        warn("Energy distribution has negative probabilities.")

    if n_discrete_lines > 0:
        eout_discrete = Discrete(data[0][:n_discrete_lines],
                                 data[1][:n_discrete_lines])
        eout_discrete.c = data[2][:n_discrete_lines]
        if n_discrete_lines == n_energy_out:
            eout_i = eout_discrete
        else:
            p_discrete = min(sum(eout_discrete.p), 1.0)
            eout_i = Mixture([p_discrete, 1. - p_discrete],
                             [eout_discrete, eout_continuous])
    else:
        eout_i = eout_continuous

    return eout_i, data
