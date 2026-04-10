# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TextIO
from warnings import warn

import numpy as np
from numpy.polynomial import Legendre

from .function import Tabulated1D
from .records import get_head_record, get_cont_record, get_tab2_record, \
    get_tab1_record, get_list_record


def parse_mf4(file_obj: TextIO) -> dict:
    # Read first two records
    ZA, AWR, LVT, LTT, _, _ = get_head_record(file_obj)
    _, _, LI, LCT, NK, NM = get_cont_record(file_obj)

    # initialize dictionary for angular distribution
    data = {'ZA': ZA, 'AWR': AWR, 'LTT': LTT, 'LI': LI, 'LCT': LCT}

    # Check for obsolete energy transformation matrix. If present, just skip
    # it and keep reading
    if LVT > 0:
        warn('Obsolete energy transformation matrix in MF=4 angular distribution.')
        for _ in range((NK + 5)//6):
            file_obj.readline()

    def legendre_data(file_obj):
        data = {}
        params, data['E_int'] = get_tab2_record(file_obj)
        n_energy = params[5]

        energy = np.zeros(n_energy)
        a_l = []
        for i in range(n_energy):
            items, al = get_list_record(file_obj)
            data['T'] = items[0]
            energy[i] = items[1]
            data['LT'] = items[2]
            coefficients = np.array(al)
            a_l.append(coefficients)
        data['a_l'] = a_l
        data['E'] = energy
        return data

    def tabulated_data(file_obj):
        data = {}
        params, data['E_int'] = get_tab2_record(file_obj)
        n_energy = params[5]

        energy = np.zeros(n_energy)
        mu = []
        for i in range(n_energy):
            params, f = get_tab1_record(file_obj)
            data['T'] = params[0]
            energy[i] = params[1]
            data['LT'] = params[2]
            mu.append(f)
        data['E'] = energy
        data['mu'] = mu
        return data

    if LTT == 0 and LI == 1:
        # Purely isotropic
        pass

    elif LTT == 1 and LI == 0:
        # Legendre polynomial coefficients
        data['legendre'] = legendre_data(file_obj)

    elif LTT == 2 and LI == 0:
        # Tabulated probability distribution
        data['tabulated'] = tabulated_data(file_obj)

    elif LTT == 3 and LI == 0:
        # Legendre for low energies / tabulated for high energies
        data['legendre'] = legendre_data(file_obj)
        data['tabulated'] = tabulated_data(file_obj)

    return data


class AngleDistribution:
    """Angle distribution as a function of incoming energy

    Parameters
    ----------
    energy
        Incoming energies in eV at which distributions exist
    mu
        Distribution of scattering cosines corresponding to each incoming energy

    Attributes
    ----------
    energy
        Incoming energies in eV at which distributions exist
    mu
        Distribution of scattering cosines corresponding to each incoming energy

    """

    def __init__(self, energy, mu):
        self.energy = energy
        self.mu = mu

    @classmethod
    def from_dict(cls, data: dict) -> AngleDistribution:
        LTT = data['LTT']
        LI = data['LI']
        if LTT == 0 and LI == 1:
            # Purely isotropic
            # TODO: Use uniform here
            energy = []
            mu = []
        elif LTT == 1 and LI == 0:
            energy = data['legendre']['E']
            mu = []
            for a_l in data['legendre']['a_l']:
                coef = np.insert(a_l, 0, 1.0)
                mu.append(Legendre(coef))
        elif LTT == 2 and LI == 0:
            energy = data['tabulated']['E']
            mu = data['tabulated']['mu']
        elif LTT == 3 and LI == 0:
            # Get Legendre first
            energy_leg = data['legendre']['E']
            mu_leg = []
            for a_l in data['legendre']['a_l']:
                coef = np.insert(a_l, 0, 1.0)
                mu_leg.append(Legendre(coef))

            # Then get tabulated
            energy_tab = data['tabulated']['E']
            mu_tab = data['tabulated']['mu']

            # Combine
            energy = np.hstack((energy_leg, energy_tab))
            mu = mu_leg + mu_tab

        return cls(energy, mu)

    def forward_fraction(self, mu_cutoff: float = 0.0) -> np.ndarray:
        """Fraction of scattering into the forward cone at each energy.

        The forward cone is defined as scattering cosines in the range
        [mu_cutoff, 1]. This is useful for computing removal cross sections
        used in point kernel shielding codes.

        Parameters
        ----------
        mu_cutoff
            Cosine of the forward cone half-angle. Must be in [-1, 1].
            A value of 0.0 corresponds to the forward hemisphere (theta < 90
            degrees).

        Returns
        -------
        numpy.ndarray
            Forward-scattered fraction at each energy in :attr:`energy`

        """
        fractions = np.empty(len(self.energy))

        for i, mu_i in enumerate(self.mu):
            if isinstance(mu_i, Legendre):
                # Stored coefficients are [a_0, a_1, ...] (ENDF convention).
                # The actual PDF is p(mu) = sum_l (2l+1)/2 * a_l * P_l(mu).
                l_vals = np.arange(len(mu_i.coef))
                pdf_coeffs = (2 * l_vals + 1) / 2 * mu_i.coef
                pdf_leg = Legendre(pdf_coeffs)
                antideriv = pdf_leg.integ()
                fractions[i] = antideriv(1.0) - antideriv(mu_cutoff)

            elif isinstance(mu_i, Tabulated1D):
                # Build CDF from the integral of the tabulated PDF
                cdf = mu_i.integral()
                cdf_func = Tabulated1D(mu_i.x, cdf)
                cdf_at_cutoff = cdf_func(mu_cutoff)
                total = cdf[-1]
                fractions[i] = (total - cdf_at_cutoff) / total

        return fractions

