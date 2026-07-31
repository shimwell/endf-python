# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/decay.py — the only changes are to load data through
# endf.Material (which already parses MF1/MT451 metadata) rather than
# openmc.data.endf.Evaluation, and to use endf-python's record helpers.

from collections.abc import Iterable
from io import StringIO
from math import log
import os
import re
from warnings import warn

import numpy as np
from uncertainties import ufloat, UFloat

from . import _checkvalue as cv
from .data import ATOMIC_SYMBOL, ATOMIC_NUMBER, gnds_name
from .function import INTERPOLATION_SCHEME
from .material import Material
from .records import get_head_record, get_list_record, get_tab1_record
from .univariate import Discrete, Tabular, combine_distributions


__all__ = ["FissionProductYields", "DecayMode", "Decay", "get_decay_modes"]


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

# Particle emitted for each ENDF radiation type, using the names a source
# distribution is keyed by. Several radiation types map onto the same particle
# (gammas and x-rays are both photons, betas and Auger/conversion electrons are
# both electrons), and their spectra are combined in Decay.sources.
_SOURCE_PARTICLES = {
    'gamma': 'photon',
    'xray': 'photon',
    'beta-': 'electron',
    'e-': 'electron',
    'ec/beta+': 'positron',
    'alpha': 'alpha',
    'n': 'neutron',
    'sf': 'fragment',
    'p': 'proton',
    'anti-neutrino': 'anti-neutrino',
    'neutrino': 'neutrino',
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

        The radiation spectra in :attr:`spectra` are given as intensities per
        decay; multiplying by the decay constant turns them into emission rates
        per second, which is what a source needs. Discrete lines and continuous
        spectra of the same particle type are combined into one distribution.

        Returns
        -------
        dict
            Mapping of particle type (``'photon'``, ``'electron'``,
            ``'positron'``, ``'alpha'``, ``'neutron'``, ``'fragment'``,
            ``'proton'``, ``'anti-neutrino'``, ``'neutrino'``) to the
            distribution of emitted particles in [/s].

        """
        sources = {}
        name = self.nuclide['name']
        decay_constant = self.decay_constant.n

        for particle, spectra in self.spectra.items():
            try:
                particle_type = _SOURCE_PARTICLES[particle]
            except KeyError:
                raise ValueError(
                    f"{name}: no source particle type known for radiation "
                    f"type {particle!r}"
                ) from None

            dists = sources.setdefault(particle_type, [])

            # Discrete lines
            if spectra['continuous_flag'] in ('discrete', 'both'):
                energies = np.array(
                    [d['energy'].n for d in spectra['discrete']])
                intensities = np.array(
                    [d['intensity'].n for d in spectra['discrete']])
                norm = spectra['discrete_normalization'].n
                dists.append(
                    Discrete(energies, decay_constant * norm * intensities))

            # Continuous spectrum
            if spectra['continuous_flag'] in ('continuous', 'both'):
                f = spectra['continuous']['probability']
                if len(f.interpolation) > 1:
                    raise NotImplementedError(
                        f"Multiple interpolation regions in the continuous "
                        f"spectrum for {name}, {particle}."
                    )
                interpolation = INTERPOLATION_SCHEME[f.interpolation[0]]
                if interpolation not in ('histogram', 'linear-linear'):
                    warn(f"Continuous spectra with {interpolation} "
                         f"interpolation ({name}, {particle}) encountered.")

                norm = spectra['continuous_normalization'].n
                dists.append(
                    Tabular(f.x, decay_constant * norm * f.y, interpolation))

        return {
            particle_type: combine_distributions(dists, [1.0] * len(dists))
            for particle_type, dists in sources.items()
        }

    @classmethod
    def from_endf(cls, material_or_filename):
        """Generate radioactive decay data from an ENDF evaluation."""
        return cls(material_or_filename)
