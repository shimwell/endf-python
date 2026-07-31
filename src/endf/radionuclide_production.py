# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .fileutils import PathLike
from .function import Tabulated1D
from .material import Material

__all__ = ['RadionuclideProduction', 'radionuclide_production',
           'isomer_table', 'level_to_isomeric_state']


@dataclass
class RadionuclideProduction:
    """Production data for a single final state of one reaction.

    Joins the MF=8 identification of a radioactive product (its ZAP, LFS
    level number, and excitation energy) with the energy-dependent data
    evaluated for that state: the MF=9 yield multiplicity and/or the
    MF=10 production cross section. Note that LFS is a level index of
    the product nuclide, not an isomeric-state index; matching a level
    to a metastable state requires comparing :attr:`excitation_energy`
    against decay data.

    Attributes
    ----------
    ZAP : int
        1000*Z + A of the product nuclide
    LFS : int
        Level number of the final state (0 for the ground state)
    QM : float
        Mass-difference Q value in [eV]
    QI : float
        Reaction Q value for this state in [eV]
    ELFS : float or None
        Excitation energy of the final state in [eV] from MF=8, or None
        when the evaluation has no MF=8 subsection for this state
    yields : Tabulated1D or None
        MF=9 yield multiplicity of the reaction cross section as a
        function of incident energy in [eV]
    cross_section : Tabulated1D or None
        MF=10 production cross section in [b] as a function of incident
        energy in [eV]

    """

    ZAP: int
    LFS: int
    QM: float
    QI: float
    ELFS: Optional[float] = None
    yields: Optional[Tabulated1D] = None
    cross_section: Optional[Tabulated1D] = None

    @property
    def excitation_energy(self) -> float:
        """Excitation energy of the final state in [eV], taken from the
        MF=8 ELFS value when present and otherwise from QM minus QI."""
        if self.ELFS is not None:
            return self.ELFS
        return self.QM - self.QI


def radionuclide_production(
        material: Material) -> Dict[int, List[RadionuclideProduction]]:
    """Collect radionuclide production data from MF=8/9/10.

    For every reaction that has an MF=9 or MF=10 section, the final
    states are returned in the order they appear in the evaluation, with
    the MF=9 and MF=10 data for the same (ZAP, LFS) pair merged into one
    :class:`RadionuclideProduction` and the MF=8 excitation energy
    attached when available. The tabulated functions are returned
    exactly as evaluated.

    Parameters
    ----------
    material
        Material to read production data from

    Returns
    -------
    dict
        Mapping of MT numbers to lists of :class:`RadionuclideProduction`

    """
    by_mt: Dict[int, set] = {}
    for mf, mt in material.sections:
        if mf in (9, 10):
            by_mt.setdefault(mt, set()).add(mf)

    result = {}
    for mt in sorted(by_mt):
        # MF=8 links each (ZAP, LFS) pair to an excitation energy
        elfs = {}
        if (8, mt) in material.section_data:
            for subsection in material.section_data[8, mt]['subsections']:
                key = int(subsection['ZAP']), int(subsection['LFS'])
                elfs[key] = float(subsection['ELFS'])

        states: Dict[tuple, RadionuclideProduction] = {}
        ordered = []
        for mf in sorted(by_mt[mt]):
            for level in material.section_data[mf, mt]['levels']:
                key = int(level['IZAP']), int(level['LFS'])
                state = states.get(key)
                if state is None:
                    state = RadionuclideProduction(
                        ZAP=key[0],
                        LFS=key[1],
                        QM=level['QM'],
                        QI=level['QI'],
                        ELFS=elfs.get(key),
                    )
                    states[key] = state
                    ordered.append(state)
                if mf == 9:
                    state.yields = level['Y']
                else:
                    state.cross_section = level['sigma']
        result[mt] = ordered
    return result


def isomer_table(decay_files: Iterable[PathLike]) -> Dict[Tuple[int, int], dict]:
    """Build a table of isomeric states from decay data evaluations.

    MF=8 identifies a radioactive product by a *nuclear level* index (LFS), not
    by an isomeric-state ordinal, so relating production data to a named
    metastable nuclide needs the excitation energies of the isomers, which decay
    data provides.

    The absolute excitation energy of each isomer is recovered from its
    isomeric-transition decay mode (RTYP == 3) Q value, chained through that
    mode's final isomeric state (RFS) down to lower isomers. It is ``None`` for a
    pure-beta isomer, which has no isomeric-transition mode to measure.

    Parameters
    ----------
    decay_files
        Decay data evaluations. Only the metastable files are needed; ground
        states are implicit.

    Returns
    -------
    Mapping of ``(Z, A)`` to ``{LISO: {"LIS", "half_life", "E_iso"}}``, where
    LISO is the isomeric-state ordinal, LIS the nuclear level index, and E_iso
    the excitation energy in eV.

    """
    raw = {}
    for filename in decay_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            material = Material(str(filename))
        section = material.section_data.get((8, 457))
        if section is None:
            continue
        Z, A = divmod(int(section['ZA']), 1000)
        liso = int(section['LISO'])
        lis = int(section['LIS'])

        half_life = section.get('T1/2')
        if isinstance(half_life, (tuple, list)):
            half_life = float(half_life[0])
        elif half_life is not None:
            half_life = float(half_life)

        it_q = None
        it_rfs = 0
        for mode in section.get('modes', []) or []:
            if float(mode.get('RTYP', 0)) == 3.0:
                q = mode.get('Q')
                it_q = float(q[0] if isinstance(q, (tuple, list)) else q)
                it_rfs = int(float(mode.get('RFS', 0)))
                break

        raw.setdefault((Z, A), {})[liso] = {
            'LIS': lis, 'half_life': half_life, 'it_q': it_q, 'it_rfs': it_rfs,
        }

    table = {}
    for za, isomers in raw.items():
        resolved = {0: 0.0}
        out = {}
        for liso in sorted(isomers):
            info = isomers[liso]
            if liso == 0:
                energy = 0.0
            elif info['it_q'] is not None:
                energy = info['it_q'] + resolved.get(info['it_rfs'], 0.0)
            else:
                energy = None
            resolved[liso] = energy if energy is not None else 0.0
            out[liso] = {'LIS': info['LIS'], 'half_life': info['half_life'],
                         'E_iso': energy}
        table[za] = out
    return table


def level_to_isomeric_state(Z: int, A: int, lfs: int,
                            excitation_energy: Optional[float],
                            table: Dict[Tuple[int, int], dict], *,
                            tol_eV: float = 3000.0) -> int:
    """Map a production level to an isomeric-state ordinal (LISO).

    The ground state maps to 0. Otherwise the level's excitation energy is
    matched against the isomer energies in ``table``; failing that, the level
    index is compared against LIS; failing that, a nuclide with exactly one
    isomer maps to it. A level that resolves to none of these is treated as
    ground, on the basis that a short-lived level gamma-cascades down.

    Parameters
    ----------
    Z, A
        Atomic and mass number of the product
    lfs
        MF=8 level index of the product
    excitation_energy
        Excitation energy of the level in eV, or None if unknown
    table
        Isomer table from :func:`isomer_table`
    tol_eV
        Tolerance for the energy match

    Returns
    -------
    Isomeric-state ordinal, 0 for the ground state

    """
    isomers = table.get((Z, A))
    if not isomers:
        return 0
    metastable = {liso: d for liso, d in isomers.items() if liso > 0}
    if not metastable:
        return 0
    if lfs == 0 or excitation_energy is None or excitation_energy < 1000.0:
        return 0

    # 1. energy match against the decay isomer energies
    best = None
    for liso, d in metastable.items():
        if d['E_iso'] is not None:
            residual = abs(excitation_energy - d['E_iso'])
            if best is None or residual < best[1]:
                best = (liso, residual)
    if best is not None and best[1] <= tol_eV:
        return best[0]

    # 2. level index fallback
    for liso, d in metastable.items():
        if d['LIS'] == lfs:
            return liso

    # 3. single isomer fallback
    if len(metastable) == 1:
        return next(iter(metastable))

    # 4. unresolved, so cascade to ground
    return 0
