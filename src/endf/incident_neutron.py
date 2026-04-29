# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import Union, List

import numpy as np

from .data import gnds_name, temperature_str, ATOMIC_SYMBOL, EV_PER_MEV, SUM_RULES
from .material import Material
from .fileutils import PathLike
from .function import Tabulated1D
from .reaction import Reaction, REACTION_MT
from . import ace



class IncidentNeutron:
    """Continuous-energy neutron interaction data.

    This class stores data derived from an ENDF-6 format neutron interaction
    sublibrary.

    Parameters
    ----------
    atomic_number : int
        Number of protons in the target nucleus
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates the
        ground state.

    Attributes
    ----------
    atomic_number : int
        Number of protons in the target nucleus
    atomic_symbol : str
        Atomic symbol of the nuclide, e.g., 'Zr'
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates the
        ground state.
    name : str
        Name of the nuclide using the GNDS naming convention
    reactions : dict
        Contains the cross sections, secondary angle and energy distributions,
        and other associated data for each reaction. The keys are the MT values
        and the values are Reaction objects.
    """

    def __init__(self, atomic_number: int, mass_number: int, metastable: int = 0):
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        self.metastable = metastable
        self.reactions = {}

    @classmethod
    def from_endf(cls, filename_or_mat: Union[PathLike, Material]) -> IncidentNeutron:
        """Generate incident neutron data from an ENDF file

        Parameters
        ----------
        filename_or_mat
            Path to ENDF-6 formatted file or material object

        Returns
        -------
        Incident neutron data

        """
        if not isinstance(filename_or_mat, Material):
            material = Material(filename_or_mat)
        else:
            material = filename_or_mat

        # Determine atomic number, mass number, and metastable state
        metadata = material[1, 451]
        Z, A = divmod(metadata['ZA'], 1000)
        data = cls(Z, A, metadata['LISO'])

        # Read each reaction
        for MF, MT in material.sections:
            if MF == 3:
                data.reactions[MT] = Reaction.from_endf(MT, material)
        return data

    @classmethod
    def from_ace(
        cls,
        filename_or_table: Union[PathLike, ace.Table],
        metastable_scheme: str = 'mcnp'
    ) -> IncidentNeutron:
        """Generate incident neutron continuous-energy data from an ACE table

        Parameters
        ----------
        ace_or_filename
            ACE table to read from. If the value is a string, it is assumed to
            be the filename for the ACE file.
        metastable_scheme : {'mcnp', 'nndc'}
            Determine how ZAID identifiers are to be interpreted in the case of
            a metastable nuclide. Because the normal ZAID (=1000*Z + A) does not
            encode metastable information, different conventions are used among
            different libraries. In MCNP libraries, the convention is to add 400
            for a metastable nuclide except for Am242m, for which 95242 is
            metastable and 95642 (or 1095242 in newer libraries) is the ground
            state. For NNDC libraries, ZAID is given as 1000*Z + A + 100*m.

        Returns
        -------
        Incident neutron continuous-energy data
        """
        # First obtain the data for the first provided ACE table/file
        if isinstance(filename_or_table, ace.Table):
            table = filename_or_table
        else:
            table = ace.get_table(filename_or_table)

        # If mass number hasn't been specified, make an educated guess
        zaid, xs = table.name.split('.')
        if not xs.endswith('c'):
            raise TypeError(f"{table} is not a continuous-energy neutron ACE table.")
        name, _, Z, mass_number, metastable = \
            ace.get_metadata(int(zaid), metastable_scheme)

        # Get string of temperature to use as a dictionary key
        strT = temperature_str(table.temperature)

        # Create IncidentNeutron object (reactions will be added after)
        data = cls(Z, mass_number, metastable)

        # Read energy grid
        n_energy = table.nxs[3]
        i = table.jxs[1]
        energy = table.xss[i : i + n_energy]*EV_PER_MEV
        total_xs = table.xss[i + n_energy : i + 2*n_energy]
        absorption_xs = table.xss[i + 2*n_energy : i + 3*n_energy]
        heating_number = table.xss[i + 4*n_energy : i + 5*n_energy]*EV_PER_MEV

        # Create redundant reaction for total (MT=1)
        xs = {strT: Tabulated1D(energy, total_xs)}
        data.reactions[1] = Reaction(1, xs, redundant=True)

        # Create redundant reaction for absorption (MT=101)
        if np.count_nonzero(absorption_xs) > 0:
            xs = {strT: Tabulated1D(energy, absorption_xs)}
            data.reactions[101] = Reaction(101, xs, redundant=True)

        # Create redundant reaction for heating (MT=301)
        xs = {strT: Tabulated1D(energy, heating_number*total_xs)}
        data.reactions[301] = Reaction(301, xs, redundant=True)

        # Read each reaction
        n_reaction = table.nxs[4] + 1
        for i in range(n_reaction):
            rx = Reaction.from_ace(table, i)
            data.reactions[rx.MT] = rx

        # Make sure redundant cross sections that are present in an ACE file get
        # marked as such
        for rx in data:
            mts = data._get_reaction_components(rx.MT)
            if mts != [rx.MT]:
                rx.redundant = True
            if rx.MT in (203, 204, 205, 206, 207, 444):
                rx.redundant = True

        return data

    def __contains__(self, MT: int):
        return MT in self.reactions

    def __getitem__(self, MT_or_name: int) -> Reaction:
        if isinstance(MT_or_name, str):
            if MT_or_name in REACTION_MT:
                MT = REACTION_MT[MT_or_name]
            elif f'({MT_or_name})' in REACTION_MT:
                MT = REACTION_MT[f'({MT_or_name})']
            else:
                raise ValueError(f"No reaction with label {MT_or_name}")
        else:
            MT = MT_or_name

        if MT in self.reactions:
            return self.reactions[MT]
        else:
            # TODO: Try to create a redundant cross section
            raise ValueError(f"No reaction with {MT=}")

    def __repr__(self) -> str:
        return f"<IncidentNeutron: {self.name}, {len(self.reactions)} reactions>"

    def __iter__(self):
        return iter(self.reactions.values())

    @property
    def name(self) -> str:
        return gnds_name(self.atomic_number, self.mass_number, self.metastable)

    @property
    def atomic_symbol(self) -> str:
        return ATOMIC_SYMBOL[self.atomic_number]

    def removal_xs(self, temperature: str = '0K', mu_cutoff: float = 0.0) -> Tabulated1D:
        """Compute the removal cross section.

        The removal cross section is defined as:

        .. math::

            \\sigma_r(E) = \\sigma_t(E) - f_{\\text{fwd}}(E) \\, \\sigma_{\\text{el}}(E)

        where :math:`f_{\\text{fwd}}(E)` is the fraction of elastic scattering
        into the forward cone :math:`[\\mu_0, 1]` and :math:`\\mu_0 =
        \\cos\\theta_0` is the cosine of the forward cone half-angle. This is
        used in point kernel shielding codes where forward-scattered neutrons
        are considered to remain in the uncollided beam.

        Parameters
        ----------
        temperature
            Temperature key for cross section lookup (e.g., ``'0K'``,
            ``'294K'``).
        mu_cutoff
            Cosine of the forward cone half-angle. Must be in [-1, 1].
            A value of 0.0 (the default) corresponds to the forward hemisphere.

        Returns
        -------
        Tabulated1D
            Removal cross section as a function of incident energy in eV.

        """
        if 1 not in self:
            raise ValueError("Total cross section (MT=1) not available.")
        if 2 not in self:
            raise ValueError("Elastic scattering cross section (MT=2) not available.")

        total_xs = self[1].xs[temperature]
        elastic_xs = self[2].xs[temperature]

        # Get elastic angular distribution
        elastic_rx = self[2]
        has_angle = (
            elastic_rx.products
            and elastic_rx.products[0].distribution
            and hasattr(elastic_rx.products[0].distribution[0], 'angle')
        )

        if has_angle:
            angle_dist = elastic_rx.products[0].distribution[0].angle
        else:
            angle_dist = None

        if angle_dist is not None and len(angle_dist.energy) > 0:
            # Use the angular distribution energy grid
            energies = angle_dist.energy
            fwd_frac = angle_dist.forward_fraction(mu_cutoff)
        else:
            # Isotropic: use elastic XS energy grid with constant fraction
            energies = elastic_xs.x
            fwd_frac = np.full(len(energies), (1.0 - mu_cutoff) / 2.0)

        total_vals = total_xs(energies)
        elastic_vals = elastic_xs(energies)
        removal_vals = total_vals - fwd_frac * elastic_vals

        return Tabulated1D(energies, removal_vals)

    def multiplication_factor(self, temperature: str = '0K') -> Tabulated1D:
        """Compute the average number of neutrons exiting per neutron-induced
        collision as a function of incident energy.

        .. math::

            M(E) = \\frac{\\sum_{\\text{MT}} n_{\\text{out}}^{(\\text{MT})}(E)
                   \\, \\sigma^{(\\text{MT})}(E)}{\\sigma_{\\text{tot}}(E)}

        where the sum runs over every non-redundant reaction MT and
        :math:`n_{\\text{out}}^{(\\text{MT})}(E)` is the total neutron yield of
        that reaction at energy :math:`E`. Yields are taken from each
        Reaction's neutron Products, so:

        - Elastic and inelastic give 1.
        - (n,2n)/(n,3n)/(n,4n)/... give 2/3/4/...
        - Multi-particle channels with one or more neutrons in the exit
          channel ((n,nα), (n,2np), etc.) contribute their neutron count.
        - Fission contributes ν̄(E) (energy-dependent), reading the 'total'
          yield when present to avoid prompt+delayed double-counting.
        - Pure absorption channels ((n,γ), (n,p), (n,α) without exit
          neutrons, ...) contribute 0 to the numerator and only via
          σ_total to the denominator.

        Useful as a physics-derived asymptotic-slope bound for shielding
        codes: in a multiplying medium the upper bound on neutron-count
        growth is ``d(ln N)/d(μt) ≤ M(E) - 1``.

        Notes
        -----
        For an :class:`IncidentNeutron` constructed via
        :meth:`from_endf`, the cross sections are smooth-pointwise only -
        resonance contributions are not reconstructed from MF=2 parameters.
        The result is therefore unreliable in the resolved- and
        unresolved-resonance regions (typically below a few keV up to a
        few hundred keV depending on the nuclide). For resonance-accurate
        :math:`M(E)`, load from an ACE file via :meth:`from_ace` (or load
        an OpenMC HDF5 file with :func:`openmc.data.IncidentNeutron.from_hdf5`
        and call the equivalent function there) - both paths consume data
        already reconstructed and Doppler-broadened by NJOY. At
        fast-neutron energies (above ~1 MeV) the smooth pointwise xs is
        sufficient and the result is accurate.

        .. todo::
           Implement MF=2 resonance reconstruction (Reich-Moore,
           Multi-Level Breit-Wigner, R-Matrix Limited) in endf-python so
           that :meth:`multiplication_factor` can be called on a
           :meth:`from_endf` object and return correct values at all
           energies including the resolved- and unresolved-resonance
           regions. Until then, callers wanting resonance-accurate
           :math:`M(E)` must use ACE/HDF5 sources.

        Parameters
        ----------
        temperature
            Temperature key for cross section lookup (e.g., ``'0K'``,
            ``'294K'``).

        Returns
        -------
        Tabulated1D
            Multiplication factor as a function of incident energy in eV,
            on the nuclide's native MT=1 (total xs) energy grid.

        """
        if 1 not in self:
            raise ValueError("Total cross section (MT=1) not available.")

        total_xs = self[1].xs[temperature]
        energies = total_xs.x
        sigma_total = total_xs(energies)

        numerator = np.zeros_like(energies)

        for rx in self.reactions.values():
            if rx.redundant:
                continue
            if rx.MT == 1:
                continue
            if not rx.products:
                continue
            if rx.xs is None or temperature not in rx.xs:
                continue

            sigma_mt = rx.xs[temperature](energies)

            neutron_products = [p for p in rx.products if p.name == 'neutron']
            if not neutron_products:
                continue

            # If a 'total' yield is present (fission with both MF=1/MT=452
            # and MT=456 split into prompt + delayed elsewhere), use only
            # the total to avoid double-counting prompt+delayed.
            total_yield_products = [
                p for p in neutron_products if p.emission_mode == 'total'
            ]
            if total_yield_products:
                relevant = total_yield_products
            else:
                relevant = neutron_products

            n_out = np.zeros_like(energies)
            for product in relevant:
                n_out = n_out + product.yield_(energies)

            numerator = numerator + n_out * sigma_mt

        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(sigma_total > 0, numerator / sigma_total, 0.0)

        return Tabulated1D(energies, M)

    def _get_reaction_components(self, MT: int) -> List[int]:
        """Determine what reactions make up redundant reaction.

        Parameters
        ----------
        mt : int
            ENDF MT number of the reaction to find components of.

        Returns
        -------
        mts : list of int
            ENDF MT numbers of reactions that make up the redundant reaction and
            have cross sections provided.

        """
        mts = []
        if MT in SUM_RULES:
            for MT_i in SUM_RULES[MT]:
                mts += self._get_reaction_components(MT_i)
        if mts:
            return mts
        else:
            return [MT] if MT in self else []
