# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

from __future__ import annotations
from collections import defaultdict
from typing import Union, List, Dict, Tuple
from warnings import warn

import numpy as np

from .data import gnds_name, temperature_str, ATOMIC_SYMBOL, EV_PER_MEV, SUM_RULES
from .material import Material
from .fileutils import PathLike
from .function import Tabulated1D
from .reaction import Reaction, REACTION_MT
from . import ace


def _cascade_gammas(
    level_energy: float,
    level_transitions: Dict[float, List[Tuple[float, float, float]]],
    memo: dict,
) -> List[Tuple[float, float]]:
    """Compute all gamma lines emitted during a cascade from *level_energy*.

    The level transition map has the form
    ``{level_energy: [(target_energy, transition_prob, gamma_prob), ...]}``.
    *gamma_prob* (GP) accounts for internal conversion — the probability that
    the transition actually emits a photon rather than a conversion electron.

    Returns a list of ``(gamma_energy, yield)`` pairs.  The same gamma energy
    may appear more than once if it is reachable via different cascade paths;
    the caller should consolidate them.
    """
    if level_energy in memo:
        return memo[level_energy]

    trans = level_transitions.get(level_energy)
    if trans is None or level_energy <= 0.0:
        memo[level_energy] = []
        return []

    result: List[Tuple[float, float]] = []
    for target_e, tp, gp in trans:
        gamma_e = level_energy - target_e
        if gamma_e > 0.0:
            result.append((gamma_e, tp * gp))
        # Continue cascade from the target level
        if target_e > 0.0:
            for sub_gamma_e, sub_yield in _cascade_gammas(
                target_e, level_transitions, memo
            ):
                result.append((sub_gamma_e, tp * sub_yield))

    memo[level_energy] = result
    return result



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
        self._material = None

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

        # Store material for access to gamma production data (MF=12/13)
        data._material = material

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

    def gamma_production_xs(self, temperature: str = '0K') -> Tabulated1D:
        """Compute the total gamma production cross section.

        This sums the photon production from every reaction that has gamma
        data in the ENDF evaluation.  Two ENDF file types contribute:

        * **MF=12** (photon production multiplicities, ``LO=1``): the
          production cross section for reaction *MT* is
          ``Y(E) × σ_MT(E)`` where *Y* is the total photon yield and
          *σ_MT* comes from MF=3.
        * **MF=13** (photon production cross sections): the production
          cross section is given directly.

        Transition-probability data (MF=12, ``LO=2``) is not yet
        supported and will be skipped with a warning.

        Parameters
        ----------
        temperature
            Temperature key for the reaction cross section lookup (e.g.,
            ``'0K'``, ``'294K'``).  Only used for MF=12 contributions
            where the multiplicity must be multiplied by the MF=3 cross
            section.

        Returns
        -------
        Tabulated1D
            Total gamma production cross section as a function of
            incident neutron energy in eV.

        """
        if self._material is None:
            raise ValueError(
                "Gamma production data requires the ENDF material. "
                "Use IncidentNeutron.from_endf() to load data."
            )

        material = self._material
        contributions = []

        for MF, MT in material.sections:
            if MF == 12:
                mf12 = material[12, MT]
                if mf12['LO'] == 1:
                    # Total photon yield
                    if mf12['NK'] > 1:
                        total_yield = mf12['Y']
                    else:
                        total_yield = mf12['multiplicities'][0]['y']

                    # Multiply yield by reaction cross section from MF=3
                    if MT not in self.reactions:
                        continue
                    rxn_xs = self[MT].xs[temperature]
                    energies = total_yield.x
                    production = total_yield.y * rxn_xs(energies)
                    contributions.append(Tabulated1D(energies, production))
                else:
                    warn(
                        f"MF=12, MT={MT}: transition probability data "
                        f"(LO={mf12['LO']}) is not supported for gamma "
                        "production cross sections and will be skipped."
                    )

            elif MF == 13:
                mf13 = material[13, MT]
                if mf13['NK'] > 1:
                    contributions.append(mf13['sigma_total'])
                else:
                    contributions.append(mf13['photons'][0]['sigma'])

        if not contributions:
            raise ValueError(
                "No gamma production data (MF=12 or MF=13) found in "
                "this evaluation."
            )

        # Single contribution — return it directly
        if len(contributions) == 1:
            return contributions[0]

        # Multiple contributions — sum on a union energy grid
        all_energies = np.concatenate([c.x for c in contributions])
        energies = np.unique(all_energies)

        total = np.zeros_like(energies)
        for contrib in contributions:
            vals = np.zeros_like(energies)
            mask = (energies >= contrib.x[0]) & (energies <= contrib.x[-1])
            if np.any(mask):
                vals[mask] = contrib(energies[mask])
            total += vals

        return Tabulated1D(energies, total)

    def gamma_line_production_xs(
        self, temperature: str = '0K'
    ) -> List[dict]:
        """Compute production cross sections for each discrete gamma line.

        For every photon-producing reaction in the evaluation, this method
        determines the discrete gamma energies emitted and the corresponding
        production cross section as a function of incident neutron energy.
        Contributions from different reactions that produce the same gamma
        energy are summed.

        Three ENDF data representations are handled:

        * **MF=12, LO=2** — nuclear level transition probabilities.  A full
          cascade calculation is performed so that secondary gammas from
          intermediate levels are included.  The internal-conversion
          coefficient (GP) is applied when present.
        * **MF=12, LO=1** — photon multiplicities for discrete lines
          (those with ``Eg > 0``).
        * **MF=13** — photon production cross sections for discrete lines
          (those with ``EG > 0``).

        Continuous-spectrum contributions (``Eg = 0`` in MF=12 LO=1 or
        MF=13, and MF=15 data) are not included; use
        :meth:`gamma_production_xs` for the total (discrete + continuum)
        production rate.

        Parameters
        ----------
        temperature
            Temperature key for the reaction cross section lookup (e.g.,
            ``'0K'``, ``'294K'``).

        Returns
        -------
        list of dict
            Each element describes one discrete gamma line::

                {
                    'gamma_energy_eV': float,
                    'neutron_energy_eV': numpy.ndarray,
                    'production_xs_barns': numpy.ndarray,
                }

            The list is sorted by ascending gamma energy.

        """
        if self._material is None:
            raise ValueError(
                "Gamma production data requires the ENDF material. "
                "Use IncidentNeutron.from_endf() to load data."
            )

        material = self._material

        # --- Step 1: build level transition map from MF=12 LO=2 sections ---
        # {level_energy: [(target_energy, transition_prob, gamma_prob), ...]}
        level_transitions: Dict[float, List[Tuple[float, float, float]]] = {}
        for MF, MT in material.sections:
            if MF == 12:
                mf12 = material[12, MT]
                if mf12.get('LO') == 2:
                    level_e = mf12['ES_NS']
                    lg = mf12.get('LG', 1)
                    trans = []
                    for t in mf12['transitions']:
                        gp = t.get('GP', 1.0) if lg == 2 else 1.0
                        trans.append((t['ES'], t['TP'], gp))
                    level_transitions[level_e] = trans

        # --- Step 2: collect per-line contributions ---
        # gamma_energy -> list of (neutron_energy_array, production_xs_array)
        line_contribs: Dict[float, list] = defaultdict(list)

        # 2a: MF=12 LO=2 — discrete levels with cascade
        cascade_memo: dict = {}
        for MF, MT in material.sections:
            if MF == 12:
                mf12 = material[12, MT]
                if mf12.get('LO') != 2:
                    continue
                if MT not in self.reactions:
                    continue

                level_e = mf12['ES_NS']
                gammas = _cascade_gammas(level_e, level_transitions,
                                        cascade_memo)

                # Consolidate yields for the same gamma energy within
                # this reaction before multiplying by the cross section
                yields: Dict[float, float] = {}
                for gamma_e, y in gammas:
                    yields[gamma_e] = yields.get(gamma_e, 0.0) + y

                rxn_xs = self[MT].xs[temperature]
                for gamma_e, total_yield in yields.items():
                    line_contribs[gamma_e].append(
                        (rxn_xs.x.copy(), total_yield * rxn_xs.y)
                    )

        # 2b: MF=12 LO=1 — discrete multiplicities (Eg > 0)
        for MF, MT in material.sections:
            if MF == 12:
                mf12 = material[12, MT]
                if mf12.get('LO') != 1:
                    continue
                if MT not in self.reactions:
                    continue
                rxn_xs = self[MT].xs[temperature]
                for mult in mf12['multiplicities']:
                    eg = mult['Eg']
                    if eg > 0.0:
                        y = mult['y']
                        production = y.y * rxn_xs(y.x)
                        line_contribs[eg].append((y.x.copy(), production))

        # 2c: MF=13 — direct production cross sections (EG > 0)
        for MF, MT in material.sections:
            if MF == 13:
                mf13 = material[13, MT]
                for photon in mf13['photons']:
                    eg = photon['EG']
                    if eg > 0.0:
                        sigma = photon['sigma']
                        line_contribs[eg].append(
                            (sigma.x.copy(), sigma.y.copy())
                        )

        # --- Step 3: consolidate contributions per gamma energy ---
        result = []
        for gamma_e in sorted(line_contribs):
            contribs = line_contribs[gamma_e]

            if len(contribs) == 1:
                e_grid, xs = contribs[0]
            else:
                # Sum on union energy grid
                all_e = np.concatenate([c[0] for c in contribs])
                e_grid = np.unique(all_e)
                xs = np.zeros_like(e_grid)
                for e_arr, xs_arr in contribs:
                    tab = Tabulated1D(e_arr, xs_arr)
                    mask = (e_grid >= e_arr[0]) & (e_grid <= e_arr[-1])
                    if np.any(mask):
                        xs[mask] += tab(e_grid[mask])

            result.append({
                'gamma_energy_eV': gamma_e,
                'neutron_energy_eV': e_grid,
                'production_xs_barns': np.maximum(xs, 0.0),
            })

        return result

    def gamma_continuum_data(
        self, temperature: str = '0K'
    ) -> List[dict]:
        """Extract continuous gamma spectrum data from MF=15.

        For each reaction with continuous photon energy spectra (MF=15),
        this method returns the total continuum production cross section
        and the gamma energy probability density at each tabulated
        incident neutron energy.

        The production cross section is derived from the continuum
        multiplicity in MF=12 (entries with ``Eg = 0``) multiplied by the
        MF=3 reaction cross section, or from continuum entries in MF=13
        (``EG = 0``).

        The gamma spectrum at each neutron energy is the normalised
        probability density ``g(E_γ | E_n)`` in units of 1/eV.  The
        differential production cross section is:

        .. math::

            \\frac{d\\sigma}{dE_\\gamma}(E_n, E_\\gamma)
            = \\sigma_{\\text{prod}}(E_n)\\, g(E_\\gamma \\mid E_n)

        Parameters
        ----------
        temperature
            Temperature key for the reaction cross section lookup.

        Returns
        -------
        list of dict
            Each element describes one reaction's continuum::

                {
                    'mt': int,
                    'production_neutron_energy_eV': numpy.ndarray,
                    'production_xs_barns': numpy.ndarray,
                    'spectra': [
                        {
                            'neutron_energy_eV': float,
                            'gamma_energy_eV': numpy.ndarray,
                            'pdf_per_eV': numpy.ndarray,
                        },
                        ...
                    ],
                }

        """
        if self._material is None:
            raise ValueError(
                "Gamma production data requires the ENDF material. "
                "Use IncidentNeutron.from_endf() to load data."
            )

        material = self._material
        results = []

        for MF, MT in material.sections:
            if MF != 15:
                continue

            mf15 = material[15, MT]

            # --- Determine the continuum production cross section ---
            production_xs = None

            # Check MF=12 for continuum multiplicity (Eg = 0)
            if (12, MT) in material:
                mf12 = material[12, MT]
                if mf12.get('LO') == 1 and MT in self.reactions:
                    for mult in mf12['multiplicities']:
                        if mult['Eg'] == 0.0:
                            rxn_xs = self[MT].xs[temperature]
                            energies = mult['y'].x
                            prod = mult['y'].y * rxn_xs(energies)
                            production_xs = Tabulated1D(
                                energies, np.maximum(prod, 0.0))
                            break

            # Fallback: check MF=13 for continuum production XS (EG = 0)
            if production_xs is None and (13, MT) in material:
                mf13 = material[13, MT]
                for photon in mf13['photons']:
                    if photon['EG'] == 0.0:
                        production_xs = photon['sigma']
                        break

            if production_xs is None:
                continue

            # --- Extract spectra from MF=15 subsections ---
            spectra = []
            for sub in mf15['subsections']:
                for k in range(sub['NE']):
                    spectra.append({
                        'neutron_energy_eV': float(sub['E'][k]),
                        'gamma_energy_eV': sub['g'][k].x.copy(),
                        'pdf_per_eV': sub['g'][k].y.copy(),
                    })

            results.append({
                'mt': MT,
                'production_neutron_energy_eV': production_xs.x.copy(),
                'production_xs_barns': production_xs.y.copy(),
                'spectra': spectra,
            })

        return results

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
