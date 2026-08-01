# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import Union, List

import numpy as np

import os
import tempfile
from io import StringIO
from warnings import warn

from .data import gnds_name, temperature_str, ATOMIC_SYMBOL, EV_PER_MEV, \
    SUM_RULES, K_BOLTZMANN
from .material import Material, get_materials
from .fileutils import PathLike
from .function import Tabulated1D, Sum
from .reaction import Reaction, REACTION_MT, _get_photon_products_ace
from .records import get_head_record, get_tab1_record
from .urr import ProbabilityTables
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

    def __init__(self, atomic_number: int, mass_number: int, metastable: int = 0,
                 atomic_weight_ratio: float = None, kTs: List[float] = None):
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        self.metastable = metastable
        self.atomic_weight_ratio = atomic_weight_ratio
        # Temperatures as kT in [eV], the unit the arrow format stores. ACE
        # files give kT in MeV; see ace.Table.kT.
        self.kTs = [] if kTs is None else list(kTs)
        self.reactions = {}
        # Energy grid per temperature. ENDF evaluations are at a single
        # temperature; ACE tables carry one each and are merged with
        # add_temperature_from_ace.
        self.energy = {}
        self.urr = {}
        self._name = None

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
        filename_or_table
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
        if isinstance(filename_or_table, ace.Table):
            table = filename_or_table
        else:
            table = ace.get_table(filename_or_table)

        zaid, xs = table.name.split('.')
        if not xs.endswith('c'):
            raise TypeError(f"{table} is not a continuous-energy neutron ACE table.")
        name, _, Z, mass_number, metastable = \
            ace.get_metadata(int(zaid), metastable_scheme)

        # Table.kT is the raw value from the file in MeV; Table.temperature is
        # the same thing in Kelvin. kTs is kept in eV.
        data = cls(Z, mass_number, metastable, table.atomic_weight_ratio,
                   [table.kT * EV_PER_MEV])
        strT = data.temperatures[0]

        # Energy grid and the cross sections stored alongside it
        n_energy = table.nxs[3]
        i = table.jxs[1]
        energy = table.xss[i:i + n_energy] * EV_PER_MEV
        data.energy[strT] = energy
        total_xs = table.xss[i + n_energy:i + 2*n_energy]
        absorption_xs = table.xss[i + 2*n_energy:i + 3*n_energy]
        heating_number = table.xss[i + 4*n_energy:i + 5*n_energy] * EV_PER_MEV

        # Redundant reaction for total (MT=1)
        total = Reaction(1)
        total.xs[strT] = Tabulated1D(energy, total_xs)
        total.redundant = True
        data.reactions[1] = total

        # Redundant reaction for absorption (MT=101)
        if np.count_nonzero(absorption_xs) > 0:
            absorption = Reaction(101)
            absorption.xs[strT] = Tabulated1D(energy, absorption_xs)
            absorption.redundant = True
            data.reactions[101] = absorption

        # Redundant reaction for heating (MT=301)
        heating = Reaction(301)
        heating.xs[strT] = Tabulated1D(energy, heating_number * total_xs)
        heating.redundant = True
        data.reactions[301] = heating

        # Read each reaction
        for i in range(table.nxs[4] + 1):
            rx = Reaction.from_ace(table, i)
            data.reactions[rx.MT] = rx

        # Some photon production reactions are assigned to MTs that have no
        # cross section of their own, usually MT=4. Create a redundant reaction
        # from the components so the photons have somewhere to live.
        n_photon_reactions = table.nxs[6]
        photon_mts = table.xss[table.jxs[13]:
                               table.jxs[13] + n_photon_reactions].astype(int)

        for MT in np.unique(photon_mts // 1000):
            if MT not in data:
                if MT not in SUM_RULES:
                    warn(f"Photon production is present for MT={MT} but no "
                         "cross section is given.")
                    continue
                mts = data.get_reaction_components(MT)
                if len(mts) == 0:
                    warn(f"Photon production is present for MT={MT} but no "
                         "reaction components exist.")
                    continue
                rx = data._get_redundant_reaction(MT, mts)
                rx.products += _get_photon_products_ace(table, rx)
                data.reactions[MT] = rx

        # An ACE file sometimes gives only the individual levels of a
        # transmutation reaction, e.g. MT=600-649 rather than the MT=103
        # summation. Create the summation explicitly so it can be tallied.
        for MT in (16, 103, 104, 105, 106, 107):
            if MT not in data:
                mts = data.get_reaction_components(MT)
                if len(mts) == 0:
                    continue
                data.reactions[MT] = data._get_redundant_reaction(MT, mts)

        # Mark reactions that are redundant sums of others
        for rx in data:
            mts = data.get_reaction_components(rx.MT)
            if mts != [rx.MT]:
                rx.redundant = True
            if rx.MT in (203, 204, 205, 206, 207, 444):
                rx.redundant = True

        # Unresolved resonance probability tables
        urr = ProbabilityTables.from_ace(table)
        if urr is not None:
            data.urr[strT] = urr

        return data

    def add_temperature_from_ace(
        self,
        filename_or_table: Union[PathLike, ace.Table],
        metastable_scheme: str = 'mcnp'
    ):
        """Append data from an ACE table at a different temperature.

        Parameters
        ----------
        filename_or_table
            ACE table to read from. If the value is a string, it is assumed to
            be the filename for the ACE file.
        metastable_scheme : {'mcnp', 'nndc'}
            How ZAID identifiers are interpreted for a metastable nuclide; see
            :meth:`from_ace`.

        """
        data = IncidentNeutron.from_ace(filename_or_table, metastable_scheme)

        strT = data.temperatures[0]
        if strT in self.temperatures:
            warn(f"Cross sections at T={strT} already exist.")
            return

        if data.name != self.name:
            raise ValueError("Data provided for an incorrect nuclide.")

        self.kTs += data.kTs
        self.energy[strT] = data.energy[strT]

        for MT in data.reactions:
            if MT in self:
                self.reactions[MT].xs[strT] = data.reactions[MT].xs[strT]
            else:
                warn(f"Tried to add cross sections for MT={MT} at T={strT} but "
                     "this reaction doesn't exist.")

        if strT in data.urr:
            self.urr[strT] = data.urr[strT]

    @classmethod
    def from_njoy(
        cls,
        filename: PathLike,
        temperatures: List[float] = None,
        material: Material = None,
        **kwargs
    ) -> IncidentNeutron:
        """Generate incident neutron data by running NJOY.

        An ENDF evaluation gives cross sections at 0 K, with the resonance
        region described by resonance parameters rather than pointwise data, so
        NJOY is used to reconstruct and Doppler broaden it and write the result
        as ACE, which is then read back.

        Parameters
        ----------
        filename
            Path to the ENDF file
        temperatures
            Temperatures in Kelvin to produce data at. Defaults to room
            temperature (293.6 K).
        material
            Material to use, when the ENDF file holds more than one evaluation
        **kwargs
            Keyword arguments passed to :func:`endf.njoy.make_ace`

        Returns
        -------
        Incident neutron continuous-energy data

        """
        from .fission_energy import FissionEnergyRelease
        from .njoy import make_ace

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs.setdefault('output_dir', tmpdir)
            for key in ('acer', 'pendf', 'heatr', 'broadr', 'gaspr', 'purr'):
                kwargs.setdefault(key, os.path.join(kwargs['output_dir'], key))
            kwargs['material'] = material
            make_ace(filename, temperatures, **kwargs)

            # One ACE table per temperature
            tables = ace.get_tables(kwargs['acer'])
            data = cls.from_ace(tables[0])
            for table in tables[1:]:
                data.add_temperature_from_ace(table)

            mat = material if material is not None else Material(filename)
            metadata = mat.section_data[1, 451]

            # Identify the nuclide from the evaluation rather than the ACE table:
            # the ZAID does not encode higher metastable states, so from_ace
            # gets names like Hf178_m2 wrong.
            #
            # metastable has to be corrected too, not just the name. The ZAID is
            # ambiguous for Am242 in particular: under the MCNP convention 95242
            # denotes the *metastable* nuclide and 95642 the ground state, yet the
            # ground-state evaluation also carries ZA=95242, so from_ace reads it
            # as m1. Left uncorrected that mismatch makes
            # FissionEnergyRelease.from_endf reject its own evaluation, which took
            # out both Am242 and Am242_m1. The evaluation is authoritative here.
            Z, A = divmod(metadata['ZA'], 1000)
            data.metastable = metadata['LISO']
            data.name = gnds_name(Z, A, metadata['LISO'])

            # Add the 0 K elastic scattering cross section from the PENDF tape
            if '0K' not in data.energy:
                pendf = Material(kwargs['pendf'])
                file_obj = StringIO(pendf.section_text[3, 2])
                get_head_record(file_obj)
                params, xs = get_tab1_record(file_obj)
                data.energy['0K'] = xs.x
                data.reactions[2].xs['0K'] = xs

            # Fission energy release, needed for the heating correction below
            if (1, 458) in mat.section_data:
                data.fission_energy = f = FissionEnergyRelease.from_endf(
                    mat, data)
            else:
                f = None

            if not kwargs['heatr']:
                return data

            # NJOY computes the fission heating number as h = EFR, but two
            # different KERMAs are wanted: one where outgoing photons deposit
            # their energy locally and one where they carry it away. Correct
            # MT=301 for the non-local case and add MT=901 for the local one.
            def file3_xs(m, MT, E):
                return m.section_data[3, MT]['sigma'](E)

            heating_local = Reaction(901)
            heating_local.redundant = True

            heatr_mats = get_materials(kwargs['heatr'])
            heatr_local_mats = get_materials(str(kwargs['heatr']) + '_local')

            for m, m_local, temp in zip(heatr_mats, heatr_local_mats,
                                        data.temperatures):
                kerma = data.reactions[301].xs[temp]
                E = kerma.x

                if f is not None:
                    # Replace the fission KERMA with (EFR + EB)*sigma_f
                    fission = data.reactions[18].xs[temp]
                    kerma.y = kerma.y - file3_xs(m, 318, E) + (
                        f.fragments(E) + f.betas(E)) * fission(E)

                kerma_local = file3_xs(m_local, 301, E)
                if f is not None:
                    # With photons deposited locally the fission KERMA becomes
                    # (EFR + EGP + EGD + EB)*sigma_f
                    kerma_local = kerma_local - file3_xs(m_local, 318, E) + (
                        f.fragments(E) + f.prompt_photons(E)
                        + f.delayed_photons(E) + f.betas(E)) * fission(E)

                heating_local.xs[temp] = Tabulated1D(E, kerma_local)

            data.reactions[901] = heating_local

        return data

    def get_reaction_components(self, MT: int) -> List[int]:
        """Determine what reactions make up a redundant reaction.

        Parameters
        ----------
        MT
            ENDF MT number of the reaction to find components of.

        Returns
        -------
        ENDF MT numbers of the reactions that make up the redundant reaction and
        have cross sections provided.

        """
        mts = []
        if MT in SUM_RULES:
            for MT_i in SUM_RULES[MT]:
                mts += self.get_reaction_components(MT_i)
        if mts:
            return mts
        return [MT] if MT in self else []

    def _get_redundant_reaction(self, MT: int, mts: List[int]) -> Reaction:
        """Create a redundant reaction by summing its components."""
        rx = Reaction(MT)
        for strT in self.temperatures:
            energy = self.energy[strT]
            xss = [self.reactions[mt_i].xs[strT] for mt_i in mts]
            idx = min(getattr(xs, '_threshold_idx', 0) for xs in xss)
            rx.xs[strT] = Tabulated1D(energy[idx:], Sum(xss)(energy[idx:]))
            rx.xs[strT]._threshold_idx = idx
        rx.redundant = True
        return rx

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
        if self._name is not None:
            return self._name
        return gnds_name(self.atomic_number, self.mass_number, self.metastable)

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def temperatures(self) -> List[str]:
        """Temperatures at which data is available, as strings like '294K'.

        Derived from :attr:`kTs`, which is in [eV].
        """
        return [temperature_str(kT / K_BOLTZMANN) for kT in self.kTs]

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


