# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/urr.py.

"""Unresolved resonance region probability tables."""

from __future__ import annotations

import numpy as np

from .data import EV_PER_MEV


__all__ = ["ProbabilityTables"]


class ProbabilityTables:
    r"""Unresolved resonance region probability tables.

    In the unresolved resonance region the individual resonances are not
    resolved, so the cross section is described statistically: at each energy a
    set of equiprobable bands is given, and one is sampled.

    Parameters
    ----------
    energy
        Energies in eV at which probability tables exist
    table
        Probability tables for each energy, of shape ``(N, 6, M)`` where N is
        the number of energies and M is the number of bands. The second
        dimension indicates whether the value is for the cumulative probability
        (0), total (1), elastic (2), fission (3), :math:`(n,\gamma)` (4), or
        heating number (5).
    interpolation
        Interpolation scheme between tables, 2 (lin-lin) or 5 (log-log)
    inelastic_flag
        A value less than zero indicates that the inelastic cross section is
        zero within the unresolved energy range. A value greater than zero
        indicates the MT number for a reaction whose cross section is to be used
        in the unresolved range.
    absorption_flag
        A value less than zero indicates that the "other absorption" cross
        section is zero within the unresolved energy range. A value greater than
        zero indicates the MT number for a reaction whose cross section is to be
        used in the unresolved range.
    multiply_smooth
        Whether the values are cross sections (False) or factors to multiply the
        smooth background cross section by (True)

    Attributes
    ----------
    absorption_flag : int
        See the ``absorption_flag`` parameter.
    energy : numpy.ndarray
        Energies in eV at which probability tables exist
    inelastic_flag : int
        See the ``inelastic_flag`` parameter.
    interpolation : int
        Interpolation scheme between tables
    multiply_smooth : bool
        See the ``multiply_smooth`` parameter.
    table : numpy.ndarray
        Probability tables for each energy

    """

    def __init__(self, energy, table, interpolation, inelastic_flag=-1,
                 absorption_flag=-1, multiply_smooth=False):
        self.energy = energy
        self.table = table
        self.interpolation = interpolation
        self.inelastic_flag = inelastic_flag
        self.absorption_flag = absorption_flag
        self.multiply_smooth = multiply_smooth

    def __repr__(self) -> str:
        return (f"<ProbabilityTables: {len(self.energy)} energies, "
                f"{self.table.shape[2]} bands>")

    @classmethod
    def from_ace(cls, table) -> ProbabilityTables | None:
        """Generate probability tables from an ACE table.

        Parameters
        ----------
        table : endf.ace.Table
            ACE table to read from

        Returns
        -------
        Unresolved resonance region probability tables, or None when the ACE
        table has no URR block.

        """
        idx = table.jxs[23]
        if idx == 0:
            return None

        N = int(table.xss[idx])       # Number of incident energies
        M = int(table.xss[idx + 1])   # Length of probability table
        interpolation = int(table.xss[idx + 2])
        inelastic_flag = int(table.xss[idx + 3])
        absorption_flag = int(table.xss[idx + 4])
        multiply_smooth = (int(table.xss[idx + 5]) == 1)
        idx += 6

        # Energies at which tables exist
        energy = table.xss[idx:idx + N] * EV_PER_MEV
        idx += N

        # Probability tables
        prob_table = table.xss[idx:idx + N * 6 * M].copy()
        prob_table.shape = (N, 6, M)

        # Convert units on heating numbers
        prob_table[:, 5, :] *= EV_PER_MEV

        return cls(energy, prob_table, interpolation, inelastic_flag,
                   absorption_flag, multiply_smooth)
