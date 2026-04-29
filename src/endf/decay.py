# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/decay.py — the only changes are to load data through
# endf.Material (which already parses MF1/MT451 metadata) rather than
# openmc.data.endf.Evaluation, and to use endf-python's record helpers.

from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from math import log
import os
import re
from warnings import warn

import numpy as np
from uncertainties import ufloat, UFloat

from . import _checkvalue as cv
from ._univariate import Discrete, Tabular
from .data import ATOMIC_SYMBOL, ATOMIC_NUMBER, gnds_name
from .material import Material
from .records import get_head_record, get_list_record, get_tab1_record


__all__ = ["FissionProductYields", "DecayMode", "Decay", "get_decay_modes"]


# ENDF spectrum label (as parsed into Decay.spectra) -> OpenMC particle-type
# string used in chain XML ``<source particle="...">`` attributes.
_ENDF_TO_PARTICLE = {
    'gamma': 'photon',
    'beta-': 'electron',
    'ec/beta+': 'positron',
    'alpha': 'alpha',
    'n': 'neutron',
    'sf': 'fragment',
    'p': 'proton',
    'e-': 'electron',
    'xray': 'photon',
    'anti-neutrino': 'anti-neutrino',
    'neutrino': 'neutrino',
}

# MF=6 TAB1 interpolation scheme -> tabular scheme name.
_TAB1_INTERPOLATION = {
    1: 'histogram',
    2: 'linear-linear',
    3: 'linear-log',
    4: 'log-linear',
    5: 'log-log',
}


# Gives name and (change in A, change in Z) resulting from decay
_DECAY_MODES = {
    0: ('gamma', (0, 0)),
    1: ('beta-', (0, 1)),
    2: ('ec/beta+', (0, -1)),
    3: ('IT', (0, 0)),
    4: ('alpha', (-4, -2)),
    5: ('n', (-1, 0)),
    6: ('sf', None),
    7: ('p', (-1, -1)),
    8: ('e-', (0, 0)),
    9: ('xray', (0, 0)),
    10: ('unknown', None)
}

_RADIATION_TYPES = {
    0: 'gamma',
    1: 'beta-',
    2: 'ec/beta+',
    4: 'alpha',
    5: 'n',
    6: 'sf',
    7: 'p',
    8: 'e-',
    9: 'xray',
    10: 'anti-neutrino',
    11: 'neutrino'
}


def _as_material(material_or_filename) -> Material:
    """Coerce a filename/path or Material into a Material."""
    if isinstance(material_or_filename, Material):
        return material_or_filename
    return Material(material_or_filename)


def _nuclide_info(material: Material) -> dict:
    """Extract nuclide properties from MF=1/MT=451 of a Material."""
    info = material.section_data[1, 451]
    ZA = int(info['ZA'])
    Z, A = divmod(ZA, 1000)
    m = int(info.get('LISO', 0) or 0)
    return {
        'name': gnds_name(Z, A, m),
        'atomic_number': Z,
        'mass_number': A,
        'isomeric_state': m,
        'excited_state': int(info.get('LIS', 0) or 0),
    }


def get_decay_modes(value):
    """Return sequence of decay modes given an ENDF RTYP value.

    Parameters
    ----------
    value : float
        ENDF definition of sequence of decay modes

    Returns
    -------
    list of str
        List of successive decays, e.g. ('beta-', 'neutron')

    """
    return [_DECAY_MODES[int(x)][0] for x in
            str(value).strip('0').replace('.', '')]


class FissionProductYields:
    """Independent and cumulative fission product yields.

    Parameters
    ----------
    material_or_filename : str, os.PathLike, or endf.Material
        ENDF fission product yield evaluation to read from. If a string or path
        is given, it is treated as a filename.

    Attributes
    ----------
    cumulative : list of dict
        Cumulative yields for each tabulated energy.
    energies : numpy.ndarray or None
        Energies at which fission product yields are tabulated.
    independent : list of dict
        Independent yields for each tabulated energy.
    nuclide : dict
        Properties of the fissioning nuclide.
    """

    def __init__(self, material_or_filename):
        def get_yields(file_obj):
            n_energy = get_head_record(file_obj)[2]
            energies = np.zeros(n_energy)

            data = []
            for i in range(n_energy):
                items, values = get_list_record(file_obj)
                energies[i] = items[0]
                n_products = items[5]

                yields = {}
                for j in range(n_products):
                    Z, A = divmod(int(values[4*j]), 1000)
                    isomeric_state = int(values[4*j + 1])
                    name = ATOMIC_SYMBOL[Z] + str(A)
                    if isomeric_state > 0:
                        name += f'_m{isomeric_state}'
                    yields[name] = ufloat(values[4*j + 2], values[4*j + 3])

                data.append(yields)

            return energies, data

        mat = _as_material(material_or_filename)

        self.nuclide = _nuclide_info(mat)
        self.energies = None
        self.independent = []
        self.cumulative = []

        if (8, 454) in mat.section_text:
            file_obj = StringIO(mat.section_text[8, 454])
            self.energies, self.independent = get_yields(file_obj)

        if (8, 459) in mat.section_text:
            file_obj = StringIO(mat.section_text[8, 459])
            energies, self.cumulative = get_yields(file_obj)
            if self.energies is not None:
                assert np.all(energies == self.energies)
            else:
                self.energies = energies

    @classmethod
    def from_endf(cls, material_or_filename):
        """Generate fission product yield data from an ENDF evaluation."""
        return cls(material_or_filename)


class DecayMode:
    """Radioactive decay mode.

    Parameters
    ----------
    parent : str
        Parent decaying nuclide
    modes : list of str
        Successive decay modes
    daughter_state : int
        Metastable state of the daughter nuclide
    energy : uncertainties.UFloat
        Total decay energy in eV available in the decay process.
    branching_ratio : uncertainties.UFloat
        Fraction of the decay of the parent nuclide which proceeds by this mode.
    """

    def __init__(self, parent, modes, daughter_state, energy, branching_ratio):
        self._daughter_state = daughter_state
        self.parent = parent
        self.modes = modes
        self.energy = energy
        self.branching_ratio = branching_ratio

    def __repr__(self):
        return (f"<DecayMode: ({','.join(self.modes)}), "
                f"{self.parent} -> {self.daughter}, {self.branching_ratio}>")

    @property
    def branching_ratio(self):
        return self._branching_ratio

    @property
    def daughter(self):
        symbol, A = re.match(r'([A-Zn][a-z]*)(\d+)', self.parent).groups()
        A = int(A)
        Z = ATOMIC_NUMBER[symbol]

        for mode in self.modes:
            for name, changes in _DECAY_MODES.values():
                if name == mode:
                    if changes is not None:
                        delta_A, delta_Z = changes
                        A += delta_A
                        Z += delta_Z

        if self._daughter_state > 0:
            return f'{ATOMIC_SYMBOL[Z]}{A}_m{self._daughter_state}'
        return f'{ATOMIC_SYMBOL[Z]}{A}'

    @property
    def energy(self):
        return self._energy

    @property
    def modes(self):
        return self._modes

    @property
    def parent(self):
        return self._parent

    @branching_ratio.setter
    def branching_ratio(self, branching_ratio):
        cv.check_type('branching ratio', branching_ratio, UFloat)
        cv.check_greater_than(
            'branching ratio', branching_ratio.nominal_value, 0.0, True)
        if branching_ratio.nominal_value == 0.0:
            warn(f"Decay mode {self.modes} of parent {self.parent} has a "
                 "zero branching ratio.")
        cv.check_greater_than(
            'branching ratio uncertainty', branching_ratio.std_dev, 0.0, True)
        self._branching_ratio = branching_ratio

    @energy.setter
    def energy(self, energy):
        cv.check_type('decay energy', energy, UFloat)
        cv.check_greater_than('decay energy', energy.nominal_value, 0.0, True)
        cv.check_greater_than(
            'decay energy uncertainty', energy.std_dev, 0.0, True)
        self._energy = energy

    @modes.setter
    def modes(self, modes):
        cv.check_type('decay modes', modes, Iterable, str)
        self._modes = modes

    @parent.setter
    def parent(self, parent):
        cv.check_type('parent nuclide', parent, str)
        self._parent = parent


class Decay:
    """Radioactive decay data.

    Parameters
    ----------
    material_or_filename : str, os.PathLike, or endf.Material
        ENDF radioactive decay data evaluation to read from.

    Attributes
    ----------
    average_energies : dict
        Average decay energies in eV of each type of radiation for decay heat
        applications.
    decay_constant : uncertainties.UFloat
        Decay constant in inverse seconds.
    decay_energy : uncertainties.UFloat
        Average energy in [eV] per decay for decay heat applications.
    half_life : uncertainties.UFloat
        Half-life of the decay in seconds.
    modes : list
        Decay mode information for each mode of decay.
    nuclide : dict
        Dictionary describing decaying nuclide.
    spectra : dict
        Resulting radiation spectra for each radiation type.
    """

    def __init__(self, material_or_filename):
        mat = _as_material(material_or_filename)

        file_obj = StringIO(mat.section_text[8, 457])

        self.nuclide = {}
        self.modes = []
        self.spectra = {}
        self.average_energies = {}
        self._sources = None

        # Head record
        items = get_head_record(file_obj)
        Z, A = divmod(items[0], 1000)
        metastable = items[3]
        self.nuclide['atomic_number'] = Z
        self.nuclide['mass_number'] = A
        self.nuclide['isomeric_state'] = metastable
        self.nuclide['name'] = gnds_name(Z, A, metastable)
        self.nuclide['mass'] = items[1]             # AWR
        self.nuclide['excited_state'] = items[2]    # LIS
        self.nuclide['stable'] = (items[4] == 1)    # NST flag

        if not self.nuclide['stable']:
            NSP = items[5]

            # Half-life and average decay energies
            items, values = get_list_record(file_obj)
            self.half_life = ufloat(items[0], items[1])
            NC = items[4] // 2
            pairs = list(zip(values[::2], values[1::2]))
            ex = self.average_energies
            ex['light'] = ufloat(*pairs[0])
            ex['electromagnetic'] = ufloat(*pairs[1])
            ex['heavy'] = ufloat(*pairs[2])
            if NC == 17:
                ex['beta-'] = ufloat(*pairs[3])
                ex['beta+'] = ufloat(*pairs[4])
                ex['auger'] = ufloat(*pairs[5])
                ex['conversion'] = ufloat(*pairs[6])
                ex['gamma'] = ufloat(*pairs[7])
                ex['xray'] = ufloat(*pairs[8])
                ex['bremsstrahlung'] = ufloat(*pairs[9])
                ex['annihilation'] = ufloat(*pairs[10])
                ex['alpha'] = ufloat(*pairs[11])
                ex['recoil'] = ufloat(*pairs[12])
                ex['SF'] = ufloat(*pairs[13])
                ex['neutron'] = ufloat(*pairs[14])
                ex['proton'] = ufloat(*pairs[15])
                ex['neutrino'] = ufloat(*pairs[16])

            items, values = get_list_record(file_obj)
            spin = items[0]
            # ENDF-102: unknown spin is reported as -77.777
            self.nuclide['spin'] = None if spin == -77.777 else spin
            self.nuclide['parity'] = items[1]

            # Decay mode information
            n_modes = items[5]
            for i in range(n_modes):
                decay_type = get_decay_modes(values[6*i])
                isomeric_state = int(values[6*i + 1])
                energy = ufloat(*values[6*i + 2:6*i + 4])
                branching_ratio = ufloat(*values[6*i + 4:6*(i + 1)])

                mode = DecayMode(self.nuclide['name'], decay_type,
                                 isomeric_state, energy, branching_ratio)
                self.modes.append(mode)

            discrete_type = {0.0: None, 1.0: 'allowed', 2.0: 'first-forbidden',
                             3.0: 'second-forbidden', 4.0: 'third-forbidden',
                             5.0: 'fourth-forbidden', 6.0: 'fifth-forbidden'}

            # Spectra
            for i in range(NSP):
                spectrum = {}

                items, values = get_list_record(file_obj)
                spectrum['type'] = _RADIATION_TYPES[items[1]]
                spectrum['continuous_flag'] = {
                    0: 'discrete', 1: 'continuous', 2: 'both'}[items[2]]
                spectrum['discrete_normalization'] = ufloat(*values[0:2])
                spectrum['energy_average'] = ufloat(*values[2:4])
                spectrum['continuous_normalization'] = ufloat(*values[4:6])

                NER = items[5]

                if spectrum['continuous_flag'] != 'continuous':
                    spectrum['discrete'] = []
                    for j in range(NER):
                        items, values = get_list_record(file_obj)
                        di = {}
                        di['energy'] = ufloat(*items[0:2])
                        di['from_mode'] = get_decay_modes(values[0])
                        di['type'] = discrete_type[values[1]]
                        di['intensity'] = ufloat(*values[2:4])
                        if spectrum['type'] == 'ec/beta+':
                            di['positron_intensity'] = ufloat(*values[4:6])
                        elif spectrum['type'] == 'gamma':
                            if len(values) >= 6:
                                di['internal_pair'] = ufloat(*values[4:6])
                            if len(values) >= 8:
                                di['total_internal_conversion'] = ufloat(*values[6:8])
                            if len(values) == 12:
                                di['k_shell_conversion'] = ufloat(*values[8:10])
                                di['l_shell_conversion'] = ufloat(*values[10:12])
                        spectrum['discrete'].append(di)

                if spectrum['continuous_flag'] != 'discrete':
                    ci = {}
                    params, ci['probability'] = get_tab1_record(file_obj)
                    ci['type'] = get_decay_modes(params[0])

                    # Read covariance if present. endf-python's
                    # get_tab1_record returns params = [C1, C2, L1, L2]; the
                    # LCOV flag lives in L2.
                    LCOV = params[3]
                    if LCOV != 0:
                        items, values = get_list_record(file_obj)
                        ci['covariance_lb'] = items[3]
                        ci['covariance'] = list(zip(values[0::2], values[1::2]))

                    spectrum['continuous'] = ci

                self.spectra[spectrum['type']] = spectrum

        else:
            # Stable nuclide: two LIST records for spin/parity
            items, values = get_list_record(file_obj)
            items, values = get_list_record(file_obj)
            self.nuclide['spin'] = items[0]
            self.nuclide['parity'] = items[1]
            self.half_life = ufloat(float('inf'), float('inf'))

    @property
    def decay_constant(self):
        if hasattr(self.half_life, 'n'):
            return log(2.) / self.half_life
        mu, sigma = self.half_life
        return ufloat(log(2.) / mu, log(2.) / mu**2 * sigma)

    @property
    def decay_energy(self):
        energy = self.average_energies
        if energy:
            return energy['light'] + energy['electromagnetic'] + energy['heavy']
        return ufloat(0, 0)

    @property
    def sources(self):
        """Radioactive decay source distributions.

        Returns a dict mapping particle type (``"photon"``, ``"electron"``,
        ``"alpha"``, …) to a :class:`.Discrete` or :class:`.Tabular`
        distribution whose abscissae are particle energies in eV and whose
        ordinates are emission rates in decays⁻¹·s⁻¹ (i.e. already multiplied
        by the decay constant λ). This is the same representation OpenMC
        consumes when reading/writing ``<source>`` elements in chain XML.
        """
        if self._sources is not None:
            return self._sources

        # Stable nuclide (no decay) has no sources
        if self.nuclide.get('stable', False):
            self._sources = {}
            return self._sources

        decay_constant = self.decay_constant.n
        # Mirror openmc's bookkeeping: per-particle list, then combine. In
        # practice a single ENDF spectrum never populates both a discrete and
        # a continuous branch for the same particle (we scanned all of
        # ENDF/B-VIII.0 — zero 'both' cases), so each list ends up length 1
        # and combining is a no-op. We still merge multiple ENDF spectra
        # that map to the same OpenMC particle type (e.g. 'gamma' + 'xray'
        # → 'photon') by summing line spectra.
        raw = {}

        for particle, spectrum in self.spectra.items():
            particle_type = _ENDF_TO_PARTICLE.get(particle)
            if particle_type is None:
                continue

            flag = spectrum.get('continuous_flag')
            bucket = raw.setdefault(particle_type, [])

            # Discrete component — always emitted when flag indicates discrete,
            # even if the discrete list is empty (matches openmc: U235/sf,
            # U238/n, etc. emit <source><parameters> </parameters></source>).
            if flag in ('discrete', 'both'):
                energies = []
                intensities = []
                for di in spectrum.get('discrete', []):
                    energies.append(di['energy'].n)
                    intensities.append(di['intensity'].n)
                norm = spectrum['discrete_normalization'].n
                rates = [decay_constant * norm * i for i in intensities]
                bucket.append(Discrete(energies, rates))

            if flag in ('continuous', 'both'):
                cont = spectrum.get('continuous')
                if cont is not None and 'probability' in cont:
                    f = cont['probability']
                    interp_code = int(f.interpolation[0]) if len(f.interpolation) >= 1 else 2
                    interp = _TAB1_INTERPOLATION.get(interp_code, 'linear-linear')
                    norm = spectrum['continuous_normalization'].n
                    rates = (decay_constant * norm * f.y).tolist()
                    bucket.append(Tabular(list(f.x), rates, interp))

        # Reduce each per-particle list to a single distribution. Matches
        # openmc's combine_distributions -> Discrete.merge path: even for a
        # single Discrete the function deduplicates equal x values by summing
        # their rates and sorting, which collapses internal duplicates that
        # the raw ENDF discrete list can contain (e.g. K- and L-shell
        # conversion electrons tabulated at the same parent gamma energy).
        sources = {}
        for particle_type, dists in raw.items():
            if not dists:
                continue
            discretes = [d for d in dists if isinstance(d, Discrete)]
            tabulars = [d for d in dists if isinstance(d, Tabular)]

            merged_discrete = None
            if discretes:
                p_merged = defaultdict(float)
                for d in discretes:
                    for x, p in zip(d.x, d.p):
                        p_merged[x] += p
                x_arr = sorted(p_merged)
                p_arr = [p_merged[x] for x in x_arr]
                merged_discrete = Discrete(x_arr, p_arr)

            if tabulars and not discretes:
                # Single tabular → pass through; multiple would need Mixture
                # (never seen in ENDF/B-VIII, so pick the longest).
                sources[particle_type] = (
                    tabulars[0] if len(tabulars) == 1
                    else max(tabulars, key=len)
                )
            elif merged_discrete is not None and not tabulars:
                sources[particle_type] = merged_discrete
            elif merged_discrete is not None and tabulars:
                # Discrete + Tabular combination would need Mixture. Not
                # produced by ENDF/B-VIII; prefer the discrete (which
                # dominates the structure of real chain XML).
                sources[particle_type] = merged_discrete

        self._sources = sources
        return self._sources

    @classmethod
    def from_endf(cls, material_or_filename):
        """Generate radioactive decay data from an ENDF evaluation."""
        return cls(material_or_filename)
