from __future__ import annotations
from pathlib import Path
from typing import Union, List, Optional

import numpy as np

from .data import ATOMIC_SYMBOL, EV_PER_MEV, SUM_RULES
from .material import Material
from .fileutils import PathLike
from .function import Tabulated1D
from . import ace


# Auxiliary data shipped with this package. These are not ENDF evaluations:
# they are the Compton profiles, bremsstrahlung cross sections and density
# effect parameters needed to describe photon interactions, which the
# photoatomic sublibrary does not carry. See datafiles/README.md for provenance.
_DATAFILES = Path(__file__).parent / 'datafiles'

# Populated on first use by _load_compton_profiles and _load_bremsstrahlung,
# which read the whole table once and cache it for every element.
_COMPTON_PROFILES = {}
_BREMSSTRAHLUNG = {}

# Highest atomic number covered by the auxiliary data
_MAX_Z = 100


def _load_compton_profiles():
    """Read the tabulated Compton profiles for every element."""
    if _COMPTON_PROFILES:
        return
    import h5py

    with h5py.File(_DATAFILES / 'compton_profiles.h5', 'r') as f:
        _COMPTON_PROFILES['pz'] = f['pz'][()]
        for i in range(1, _MAX_Z + 1):
            group = f[f'{i:03}']
            _COMPTON_PROFILES[i] = {
                'num_electrons': group['num_electrons'][()],
                'binding_energy': group['binding_energy'][()] * EV_PER_MEV,
                'J': group['J'][()],
            }


def _load_bremsstrahlung():
    """Read the bremsstrahlung cross sections and density effect data for every
    element, resampling the scaled cross sections onto a common energy grid."""
    if _BREMSSTRAHLUNG:
        return
    import h5py
    from scipy.interpolate import CubicSpline

    with h5py.File(_DATAFILES / 'density_effect.h5', 'r') as f:
        for i in range(1, _MAX_Z + 1):
            group = f[f'{i:03}']
            _BREMSSTRAHLUNG[i] = {
                'I': group.attrs['I'],
                'num_electrons': group['num_electrons'][()],
                'ionization_energy': group['ionization_energy'][()],
            }

    brem = (_DATAFILES / 'BREMX.DAT').read_text().split()

    # Incident electron kinetic energy grid in eV
    _BREMSSTRAHLUNG['electron_energy'] = np.logspace(3, 9, 200)
    log_energy = np.log(_BREMSSTRAHLUNG['electron_energy'])

    # Number of tabulated electron and photon energy values
    n = int(brem[37])
    k = int(brem[38])
    p = 39

    # Log of the tabulated incident electron kinetic energies, used for cubic
    # spline interpolation in log energy. Stored in MeV, so convert to eV.
    logx = np.log(np.fromiter(brem[p:p + n], float, n) * EV_PER_MEV)
    p += n

    # Reduced photon energy values
    _BREMSSTRAHLUNG['photon_energy'] = np.fromiter(brem[p:p + k], float, k)
    p += k

    for i in range(1, _MAX_Z + 1):
        dcs = np.empty([len(log_energy), k])

        # Scaled cross sections for each electron energy and reduced photon
        # energy for this Z. Stored in mb, so convert to b.
        y = np.reshape(np.fromiter(brem[p:p + n * k], float, n * k),
                       (n, k)) * 1.0e-3
        p += k * n

        for j in range(k):
            # Cubic spline interpolation in log energy, linear in the cross
            # section itself
            dcs[:, j] = CubicSpline(logx, y[:, j])(log_energy)

        _BREMSSTRAHLUNG[i]['dcs'] = dcs


# Electron subshell labels (index corresponds to ENDF designator)
_SUBSHELLS = (
    None, 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
    'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3',
    'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'P1', 'P2', 'P3', 'P4',
    'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'Q1', 'Q2', 'Q3'
)

# Mapping of MT values to short reaction names
PHOTON_REACTION_NAME = {
    501: 'total',
    502: 'coherent',
    504: 'incoherent',
    515: 'pair_production_electron',
    516: 'pair_production_total',
    517: 'pair_production_nuclear',
    522: 'photoelectric',
    525: 'heating',
    526: 'electro_atomic_scat',
    527: 'electro_atomic_brem',
    528: 'electro_atomic_excit',
}
for _i, _shell in enumerate(_SUBSHELLS[1:], 1):
    PHOTON_REACTION_NAME[533 + _i] = _shell

# Reverse mapping from name to MT
PHOTON_REACTION_MT = {name: mt for mt, name in PHOTON_REACTION_NAME.items()}


def compton_profile_cdfs(J: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Cumulative distributions for a set of Compton profiles.

    Integrates each profile over the electron momentum grid by the trapezoidal
    rule. The result is deliberately left un-normalised, so each row ends near
    0.5 rather than 1: a Compton profile is tabulated for positive pz only, and
    the missing half is symmetric, which is the convention samplers expect.

    Parameters
    ----------
    J
        Profile values, shape ``(n_shells, n_pz)``
    pz
        Electron momentum grid, shape ``(n_pz,)``

    Returns
    -------
    Cumulative values of the same shape as ``J``

    """
    J = np.asarray(J, dtype=float)
    pz = np.asarray(pz, dtype=float)
    cdf = np.zeros_like(J)
    # Trapezoidal cumulative integral along the momentum axis
    increments = 0.5 * (J[:, :-1] + J[:, 1:]) * np.diff(pz)
    cdf[:, 1:] = np.cumsum(increments, axis=1)
    return cdf


def compton_subshell_map(compton_num_electrons, subshell_num_electrons):
    """Associate each Compton-profile shell with its atomic-relaxation subshells.

    The Compton profiles are Biggs Hartree-Fock shells grouped by (n, l), while
    the atomic-relaxation subshells are EADL shells split by total angular
    momentum j, so the single 2p Compton shell corresponds to the L2 and L3
    subshells. The two tabulations use different binding-energy evaluations that
    disagree by roughly 0.5 to 1%, so they cannot be matched on energy. They do
    match on occupancy: a Compton (n, l) shell holds exactly the electrons of its
    constituent j-split subshells. Walking both lists in canonical
    most-bound-first order and greedily accumulating subshells until their
    electron count reaches the Compton shell's gives the association.

    Parameters
    ----------
    compton_num_electrons
        Electron occupancy per Compton-profile shell, canonical order
    subshell_num_electrons
        Electron occupancy per atomic-relaxation subshell, canonical order

    Returns
    -------
    (offsets, indices, weights)
        CSR arrays. Compton shell ``c`` maps to the subshells
        ``indices[offsets[c]:offsets[c + 1]]`` with the given occupancy weights,
        which sum to 1 within a group. An empty range means the shell has no
        clean counterpart, which happens for the outer and valence shells where
        the two orderings diverge.

    """
    sub_occ = [float(n) for n in subshell_num_electrons]
    offsets = [0]
    indices = []
    weights = []
    cursor = 0
    stopped = False
    for c_occ in compton_num_electrons:
        c_occ = float(c_occ)
        if not stopped and c_occ > 0.0:
            acc = 0.0
            group = []
            j = cursor
            # Accumulate consecutive subshells until their occupancy reaches
            # this Compton shell's. The small negative tolerance keeps the
            # subshell that completes the group.
            while j < len(sub_occ) and acc < c_occ - 1e-6:
                acc += sub_occ[j]
                group.append(j)
                j += 1
            if group and abs(acc - c_occ) <= 1e-3 * max(1.0, c_occ):
                for k in group:
                    indices.append(k)
                    weights.append(sub_occ[k] / c_occ)
                cursor = j
            else:
                # First occupancy mismatch means the orderings have diverged,
                # so drop this shell and every later one.
                stopped = True
        offsets.append(len(indices))
    return offsets, indices, weights


class PhotonReaction:
    """A photon interaction reaction.

    This class represents a single photon reaction channel with an associated
    cross section and, for coherent/incoherent scattering, form factors.

    Parameters
    ----------
    MT : int
        The ENDF MT number for this reaction.
    xs : Tabulated1D, optional
        Cross section as a function of incident photon energy.
    scattering_factor : Tabulated1D, optional
        Coherent or incoherent form factor.
    anomalous_real : Tabulated1D, optional
        Real part of the anomalous scattering factor.
    anomalous_imag : Tabulated1D, optional
        Imaginary part of the anomalous scattering factor.

    Attributes
    ----------
    MT : int
        The ENDF MT number for this reaction.
    xs : Tabulated1D or None
        Cross section as a function of incident photon energy.
    scattering_factor : Tabulated1D or None
        Coherent or incoherent form factor.
    anomalous_real : Tabulated1D or None
        Real part of the anomalous scattering factor (MT=502 only).
    anomalous_imag : Tabulated1D or None
        Imaginary part of the anomalous scattering factor (MT=502 only).
    subshell_binding_energy : float or None
        Subshell binding energy in [eV] for subshell photoelectric reactions
        (MT=534-572).
    fluorescence_yield : float or None
        Fluorescence yield for subshell photoelectric reactions (MT=534-572).

    """

    def __init__(self, MT: int, xs: Optional[Tabulated1D] = None,
                 scattering_factor: Optional[Tabulated1D] = None,
                 anomalous_real: Optional[Tabulated1D] = None,
                 anomalous_imag: Optional[Tabulated1D] = None):
        self.MT = MT
        self.xs = xs
        self.scattering_factor = scattering_factor
        self.anomalous_real = anomalous_real
        self.anomalous_imag = anomalous_imag
        self.subshell_binding_energy = None
        self.fluorescence_yield = None

    def __repr__(self):
        name = PHOTON_REACTION_NAME.get(self.MT)
        if name is not None:
            return f"<PhotonReaction: MT={self.MT} {name}>"
        else:
            return f"<PhotonReaction: MT={self.MT}>"


class AtomicRelaxation:
    """Atomic relaxation data.

    This class stores the binding energy, number of electrons, and electron
    transitions possible from ionization for each electron subshell of an atom.
    The data originates from an ENDF-6 atomic relaxation sub-library (NSUB=6)
    or from an ACE photoatomic table.

    Parameters
    ----------
    binding_energy : dict
        Dictionary mapping subshell names (e.g., 'K', 'L1') to binding
        energies in [eV].
    num_electrons : dict
        Dictionary mapping subshell names to number of electrons when the
        atom is neutral.
    transitions : dict
        Dictionary mapping subshell names to transition data. Each entry is a
        dict with keys 'secondary_subshell' (list of str),
        'tertiary_subshell' (list of str), 'energy' (ndarray in [eV]),
        and 'probability' (ndarray).

    Attributes
    ----------
    binding_energy : dict
        Dictionary mapping subshell names to binding energies in [eV].
    num_electrons : dict
        Dictionary mapping subshell names to number of electrons.
    transitions : dict
        Dictionary mapping subshell names to transition data.
    subshells : list
        Sorted list of subshell names present.

    """

    def __init__(self, binding_energy: dict, num_electrons: dict,
                 transitions: dict):
        self.binding_energy = binding_energy
        self.num_electrons = num_electrons
        self.transitions = transitions

    @property
    def subshells(self) -> list:
        return sorted(self.binding_energy.keys(),
                      key=lambda s: _SUBSHELLS.index(s))

    @classmethod
    def from_endf(cls, material: Material) -> AtomicRelaxation:
        """Generate atomic relaxation data from an ENDF material.

        Parameters
        ----------
        material
            ENDF material containing MF=28, MT=533 data

        Returns
        -------
        Atomic relaxation data

        """
        data = material[28, 533]

        binding_energy = {}
        num_electrons = {}
        transitions = {}

        for shell_data in data['shells']:
            subi = _SUBSHELLS[int(shell_data['SUBI'])]
            binding_energy[subi] = shell_data['EBI']
            num_electrons[subi] = shell_data['ELN']

            n_transitions = shell_data['NTR']
            if n_transitions > 0:
                transitions[subi] = {
                    'secondary_subshell': [
                        _SUBSHELLS[int(s)] for s in shell_data['SUBJ']
                    ],
                    'tertiary_subshell': [
                        _SUBSHELLS[int(s)] for s in shell_data['SUBK']
                    ],
                    'energy': np.asarray(shell_data['ETR'], dtype=float),
                    'probability': np.asarray(shell_data['FTR'], dtype=float),
                }

        return cls(binding_energy, num_electrons, transitions)

    @classmethod
    def from_ace(cls, table: ace.Table) -> AtomicRelaxation:
        """Generate atomic relaxation data from an ACE table.

        Parameters
        ----------
        table
            ACE photoatomic table

        Returns
        -------
        Atomic relaxation data

        """
        binding_energy = {}
        num_electrons = {}
        transitions = {}

        # Get shell designators
        n = table.nxs[7]
        idx = table.jxs[11]
        shells = [_SUBSHELLS[int(i)] for i in table.xss[idx:idx + n]]

        # Get number of electrons for each shell
        idx = table.jxs[12]
        for shell, num in zip(shells, table.xss[idx:idx + n]):
            num_electrons[shell] = num

        # Get binding energy for each shell
        idx = table.jxs[13]
        for shell, e in zip(shells, table.xss[idx:idx + n]):
            binding_energy[shell] = e * EV_PER_MEV

        # Get transition data
        idx = table.jxs[18]
        for i, subi in enumerate(shells):
            n_transitions = int(table.xss[table.jxs[15] + i])
            if n_transitions > 0:
                secondary = []
                tertiary = []
                energies = []
                probabilities = []
                for j in range(n_transitions):
                    secondary.append(_SUBSHELLS[int(table.xss[idx])])
                    tertiary.append(_SUBSHELLS[int(table.xss[idx + 1])])
                    energies.append(table.xss[idx + 2] * EV_PER_MEV)
                    # ACE stores cumulative probabilities
                    if j == 0:
                        probabilities.append(table.xss[idx + 3])
                    else:
                        probabilities.append(
                            table.xss[idx + 3] - table.xss[idx - 1]
                        )
                    idx += 4

                transitions[subi] = {
                    'secondary_subshell': secondary,
                    'tertiary_subshell': tertiary,
                    'energy': np.array(energies),
                    'probability': np.array(probabilities),
                }

        return cls(binding_energy, num_electrons, transitions)

    def __repr__(self):
        return f"<AtomicRelaxation: {len(self.binding_energy)} subshells>"


class IncidentPhoton:
    """Continuous-energy photon interaction data.

    This class stores data derived from an ENDF-6 format photo-atomic
    interaction sublibrary or an ACE photoatomic table.

    Parameters
    ----------
    atomic_number : int
        Number of protons in the target nucleus

    Attributes
    ----------
    atomic_number : int
        Number of protons in the target nucleus
    name : str
        Atomic symbol of the element, e.g., 'Fe'
    reactions : dict
        Contains the cross sections and associated data for each photon
        reaction. The keys are the MT values and the values are
        PhotonReaction objects.
    atomic_relaxation : AtomicRelaxation or None
        Atomic relaxation data
    compton_profiles : dict
        Dictionary of Compton profile data with keys 'num_electrons'
        (number of electrons in each subshell), 'binding_energy'
        (ionization potential of each subshell in [eV]), and 'J' (list of
        Tabulated1D Compton profiles).
    bremsstrahlung : dict
        Dictionary of bremsstrahlung data with keys 'I' (mean excitation energy
        in [eV]), 'num_electrons' (number of electrons in each subshell),
        'ionization_energy' (ionization potential of each subshell in [eV]),
        'electron_energy' (incident electron kinetic energy grid in [eV]),
        'photon_energy' (ratio of the photon energy to the incident electron
        kinetic energy), and 'dcs' (scaled bremsstrahlung differential cross
        sections in [mb]).

    """

    def __init__(self, atomic_number: int):
        self.atomic_number = atomic_number
        self.reactions = {}
        self.atomic_relaxation = None
        self.compton_profiles = {}
        self.bremsstrahlung = {}

    @classmethod
    def from_endf(
        cls,
        photoatomic: Union[PathLike, Material],
        relaxation: Union[PathLike, Material, None] = None
    ) -> IncidentPhoton:
        """Generate incident photon data from an ENDF file

        Parameters
        ----------
        photoatomic
            Path to ENDF-6 formatted photoatomic data file or Material object
        relaxation
            Path to ENDF-6 formatted atomic relaxation data file or Material
            object. Optional.

        Returns
        -------
        Incident photon data

        """
        if not isinstance(photoatomic, Material):
            photoatomic = Material(photoatomic)

        # Determine atomic number from ZA
        metadata = photoatomic[1, 451]
        Z = metadata['ZA'] // 1000
        data = cls(Z)

        # Read each photon reaction from MF=23
        for MF, MT in photoatomic.sections:
            if MF == 23:
                rx_data = photoatomic[23, MT]
                rx = PhotonReaction(MT, xs=rx_data['sigma'])

                # Set subshell binding energy and fluorescence yield
                if 534 <= MT <= 599:
                    rx.subshell_binding_energy = rx_data['EPE']
                if 534 <= MT <= 572:
                    rx.fluorescence_yield = rx_data['EFL']

                data.reactions[MT] = rx

        # Read form factors / scattering functions from MF=27
        for MF, MT in photoatomic.sections:
            if MF == 27:
                mf27_data = photoatomic[27, MT]
                if MT in (502, 504) and MT in data.reactions:
                    data.reactions[MT].scattering_factor = mf27_data['H']
                elif MT == 505 and 502 in data.reactions:
                    data.reactions[502].anomalous_imag = mf27_data['H']
                elif MT == 506 and 502 in data.reactions:
                    data.reactions[502].anomalous_real = mf27_data['H']

        # Read atomic relaxation data if present in the same material
        if (28, 533) in photoatomic:
            data.atomic_relaxation = AtomicRelaxation.from_endf(photoatomic)

        # Read atomic relaxation data from separate file if provided
        if relaxation is not None:
            if not isinstance(relaxation, Material):
                relaxation = Material(relaxation)
            data.atomic_relaxation = AtomicRelaxation.from_endf(relaxation)

        # The photoatomic sublibrary carries neither Compton profiles nor
        # bremsstrahlung cross sections, so add them from the auxiliary data
        # shipped with this package. (An ACE photoatomic table does carry
        # Compton profiles, which is why from_ace only adds bremsstrahlung.)
        data._add_compton_profiles()
        data._add_bremsstrahlung()

        return data

    @classmethod
    def from_ace(
        cls,
        filename_or_table: Union[PathLike, ace.Table]
    ) -> IncidentPhoton:
        """Generate incident photon data from an ACE table

        Parameters
        ----------
        filename_or_table
            ACE table to read from. If the value is a string or path, it is
            assumed to be the filename for the ACE file.

        Returns
        -------
        Incident photon continuous-energy data

        """
        if isinstance(filename_or_table, ace.Table):
            table = filename_or_table
        else:
            table = ace.get_table(filename_or_table)

        # Verify this is a photoatomic table
        zaid, xs = table.name.split('.')
        if not xs.endswith('p'):
            raise TypeError(
                f"{table} is not a photoatomic ACE table."
            )
        Z = ace.get_metadata(int(zaid))[2]

        data = cls(Z)

        # Read energy grid (stored as logarithms)
        n_energy = table.nxs[3]
        idx = table.jxs[1]
        energy = np.exp(table.xss[idx:idx + n_energy]) * EV_PER_MEV

        # Read cross sections for each main reaction
        for mt in (502, 504, 517, 522, 525):
            rx = PhotonReaction(mt)

            if mt == 502:
                xs_idx = table.jxs[1] + 2 * n_energy
            elif mt == 504:
                xs_idx = table.jxs[1] + n_energy
            elif mt == 517:
                xs_idx = table.jxs[1] + 4 * n_energy
            elif mt == 522:
                xs_idx = table.jxs[1] + 3 * n_energy
            elif mt == 525:
                xs_idx = table.jxs[5]

            xs_vals = table.xss[xs_idx:xs_idx + n_energy].copy()
            if mt == 525:
                # Heating factors in [MeV/collision] -> [eV/collision]
                xs_vals *= EV_PER_MEV
            else:
                nonzero = (xs_vals != 0.0)
                xs_vals[nonzero] = np.exp(xs_vals[nonzero])
                xs_vals[~nonzero] = np.exp(-500.0)
            rx.xs = Tabulated1D(energy, xs_vals, [n_energy], [5])

            data.reactions[mt] = rx

        # Convert heating from [eV/collision] to [eV-barn] by multiplying
        # with total cross section
        total_xs = sum(
            data.reactions[mt].xs.y for mt in (502, 504, 517, 522)
        )
        data.reactions[525].xs = Tabulated1D(
            energy, data.reactions[525].xs.y * total_xs, [n_energy], [5]
        )

        # Read form factors
        new_format = (table.nxs[6] > 0)

        # Coherent scattering factor
        idx = table.jxs[3]
        if new_format:
            n = (table.jxs[4] - table.jxs[3]) // 3
            x = table.xss[idx:idx + n]
            idx += n
        else:
            x = np.array([
                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12,
                0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
                3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6,
                5.8, 6.0])
            n = x.size
        ff = table.xss[idx + n:idx + 2 * n]
        data.reactions[502].scattering_factor = Tabulated1D(x, ff)

        # Incoherent scattering factor
        idx = table.jxs[2]
        if new_format:
            n = (table.jxs[3] - table.jxs[2]) // 2
            x = table.xss[idx:idx + n]
            idx += n
        else:
            x = np.array([
                0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
            n = x.size
        ff = table.xss[idx:idx + n]
        data.reactions[504].scattering_factor = Tabulated1D(x, ff)

        # Compton profiles
        n_shell = table.nxs[5]
        if n_shell != 0:
            idx = table.jxs[6]
            num_electrons = table.xss[idx:idx + n_shell]

            idx = table.jxs[7]
            binding_energy = table.xss[idx:idx + n_shell] * EV_PER_MEV

            profiles = []
            for k in range(n_shell):
                loca = int(table.xss[table.jxs[9] + k])
                jj = int(table.xss[table.jxs[10] + loca - 1])
                m = int(table.xss[table.jxs[10] + loca])

                idx = table.jxs[10] + loca + 1
                pz = table.xss[idx:idx + m]
                pdf = table.xss[idx + m:idx + 2 * m]

                profiles.append(Tabulated1D(pz, pdf, [m], [jj]))

            data.compton_profiles = {
                'num_electrons': num_electrons,
                'binding_energy': binding_energy,
                'J': profiles,
            }

        # Subshell photoelectric cross sections and atomic relaxation
        if table.nxs[7] > 0:
            data.atomic_relaxation = AtomicRelaxation.from_ace(table)

            # Get subshell designators
            n_subshells = table.nxs[7]
            idx = table.jxs[11]
            designators = [int(i) for i in table.xss[idx:idx + n_subshells]]

            # Get cross section for each subshell
            idx = table.jxs[16]
            for d in designators:
                mt = 533 + d
                rx = PhotonReaction(mt)
                data.reactions[mt] = rx

                # Store cross section, determining threshold
                xs_vals = table.xss[idx:idx + n_energy].copy()
                nonzero = (xs_vals != 0.0)
                xs_vals[nonzero] = np.exp(xs_vals[nonzero])
                threshold = np.where(xs_vals > 0.0)[0][0]
                rx.xs = Tabulated1D(
                    energy[threshold:], xs_vals[threshold:],
                    [n_energy - threshold], [5]
                )
                idx += n_energy

                # Copy binding energy from atomic relaxation
                shell = _SUBSHELLS[d]
                rx.subshell_binding_energy = \
                    data.atomic_relaxation.binding_energy[shell]

        # Compton profiles came from the table itself; bremsstrahlung data is
        # not in an ACE photoatomic table, so add it from the auxiliary data.
        data._add_bremsstrahlung()

        return data

    def _add_compton_profiles(self):
        """Attach the tabulated Compton profiles for this element."""
        _load_compton_profiles()
        profile = _COMPTON_PROFILES[self.atomic_number]
        pz = _COMPTON_PROFILES['pz']
        self.compton_profiles['num_electrons'] = profile['num_electrons']
        self.compton_profiles['binding_energy'] = profile['binding_energy']
        self.compton_profiles['J'] = [Tabulated1D(pz, J_k) for J_k in profile['J']]

    def _add_bremsstrahlung(self):
        """Attach the data used in the thick-target bremsstrahlung
        approximation for this element."""
        _load_bremsstrahlung()
        self.bremsstrahlung['electron_energy'] = _BREMSSTRAHLUNG['electron_energy']
        self.bremsstrahlung['photon_energy'] = _BREMSSTRAHLUNG['photon_energy']
        self.bremsstrahlung.update(_BREMSSTRAHLUNG[self.atomic_number])

    def __contains__(self, MT: int):
        return MT in self.reactions

    def __getitem__(self, MT_or_name) -> PhotonReaction:
        if isinstance(MT_or_name, str):
            if MT_or_name in PHOTON_REACTION_MT:
                MT = PHOTON_REACTION_MT[MT_or_name]
            else:
                raise ValueError(f"No reaction with label {MT_or_name}")
        else:
            MT = MT_or_name

        if MT in self.reactions:
            return self.reactions[MT]
        else:
            raise ValueError(f"No reaction with {MT=}")

    def __repr__(self) -> str:
        return (
            f"<IncidentPhoton: {self.name}, {len(self.reactions)} reactions>"
        )

    def __iter__(self):
        return iter(self.reactions.values())

    @property
    def name(self) -> str:
        return ATOMIC_SYMBOL[self.atomic_number]

    def _get_reaction_components(self, MT: int) -> List[int]:
        """Determine what reactions make up a redundant reaction.

        Parameters
        ----------
        MT : int
            ENDF MT number of the reaction to find components of.

        Returns
        -------
        mts : list of int
            ENDF MT numbers of reactions that make up the redundant reaction
            and have cross sections provided.

        """
        mts = []
        if MT in SUM_RULES:
            for MT_i in SUM_RULES[MT]:
                mts += self._get_reaction_components(MT_i)
        if mts:
            return mts
        else:
            return [MT] if MT in self else []
