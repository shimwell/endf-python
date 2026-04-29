# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/deplete/chain.py. Chain.from_endf is adapted to use
# endf.Material (which already parses MF=3 Q values) instead of the openmc
# Evaluation object.

"""Depletion chain representation.

A depletion chain describes the nuclides involved in burnup — their decay
modes, transmutation reactions, and fission product yields — so OpenMC can
build the Bateman matrix and write a chain XML file.
"""

import math
import re
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Mapping, Iterable
from itertools import chain as ichain
from numbers import Real, Integral
from warnings import warn

try:
    import lxml.etree as ET
    _have_lxml = True
except ImportError:
    import xml.etree.ElementTree as ET
    _have_lxml = False

from ._checkvalue import check_type, check_greater_than
from ._deplete_nuclide import (
    Nuclide, DecayTuple, ReactionTuple, FissionYieldDistribution
)
from ._xml import clean_indentation
from .data import ATOMIC_SYMBOL, gnds_name, zam
from .decay import Decay, FissionProductYields
from .material import Material
from .reaction import FISSION_MTS


# (possible MT values, (dA, dZ), secondaries) where dA is the change in the
# mass number and dZ is the change in the atomic number.
ReactionInfo = namedtuple('ReactionInfo', ('mts', 'dadz', 'secondaries'))

REACTIONS = {
    '(n,2nd)': ReactionInfo({11}, (-3, -1), ('H2',)),
    '(n,2n)': ReactionInfo(set(ichain([16], range(875, 892))), (-1, 0), ()),
    '(n,3n)': ReactionInfo({17}, (-2, 0), ()),
    '(n,na)': ReactionInfo({22}, (-4, -2), ('He4',)),
    '(n,n3a)': ReactionInfo({23}, (-12, -6), ('He4', 'He4', 'He4')),
    '(n,2na)': ReactionInfo({24}, (-5, -2), ('He4',)),
    '(n,3na)': ReactionInfo({25}, (-6, -2), ('He4',)),
    '(n,np)': ReactionInfo({28}, (-1, -1), ('H1',)),
    '(n,n2a)': ReactionInfo({29}, (-8, -4), ('He4', 'He4')),
    '(n,2n2a)': ReactionInfo({30}, (-9, -4), ('He4', 'He4')),
    '(n,nd)': ReactionInfo({32}, (-2, -1), ('H2',)),
    '(n,nt)': ReactionInfo({33}, (-3, -1), ('H3',)),
    '(n,n3He)': ReactionInfo({34}, (-3, -2), ('He3',)),
    '(n,nd2a)': ReactionInfo({35}, (-10, -5), ('H2', 'He4', 'He4')),
    '(n,nt2a)': ReactionInfo({36}, (-11, -5), ('H3', 'He4', 'He4')),
    '(n,4n)': ReactionInfo({37}, (-3, 0), ()),
    '(n,2np)': ReactionInfo({41}, (-2, -1), ('H1',)),
    '(n,3np)': ReactionInfo({42}, (-3, -1), ('H1',)),
    '(n,n2p)': ReactionInfo({44}, (-2, -2), ('H1', 'H1')),
    '(n,npa)': ReactionInfo({45}, (-5, -3), ('H1', 'He4')),
    '(n,gamma)': ReactionInfo({102}, (1, 0), ()),
    '(n,p)': ReactionInfo(set(ichain([103], range(600, 650))), (0, -1), ('H1',)),
    '(n,d)': ReactionInfo(set(ichain([104], range(650, 700))), (-1, -1), ('H2',)),
    '(n,t)': ReactionInfo(set(ichain([105], range(700, 750))), (-2, -1), ('H3',)),
    '(n,3He)': ReactionInfo(set(ichain([106], range(750, 800))), (-2, -2), ('He3',)),
    '(n,a)': ReactionInfo(set(ichain([107], range(800, 850))), (-3, -2), ('He4',)),
    '(n,2a)': ReactionInfo({108}, (-7, -4), ('He4', 'He4')),
    '(n,3a)': ReactionInfo({109}, (-11, -6), ('He4', 'He4', 'He4')),
    '(n,2p)': ReactionInfo({111}, (-1, -2), ('H1', 'H1')),
    '(n,pa)': ReactionInfo({112}, (-4, -3), ('H1', 'He4')),
    '(n,t2a)': ReactionInfo({113}, (-10, -5), ('H3', 'He4', 'He4')),
    '(n,d2a)': ReactionInfo({114}, (-9, -5), ('H2', 'He4', 'He4')),
    '(n,pd)': ReactionInfo({115}, (-2, -2), ('H1', 'H2')),
    '(n,pt)': ReactionInfo({116}, (-3, -2), ('H1', 'H3')),
    '(n,da)': ReactionInfo({117}, (-5, -3), ('H2', 'He4')),
    '(n,5n)': ReactionInfo({152}, (-4, 0), ()),
    '(n,6n)': ReactionInfo({153}, (-5, 0), ()),
    '(n,2nt)': ReactionInfo({154}, (-4, -1), ('H3',)),
    '(n,ta)': ReactionInfo({155}, (-6, -3), ('H3', 'He4')),
    '(n,4np)': ReactionInfo({156}, (-4, -1), ('H1',)),
    '(n,3nd)': ReactionInfo({157}, (-4, -1), ('H2',)),
    '(n,nda)': ReactionInfo({158}, (-6, -3), ('H2', 'He4')),
    '(n,2npa)': ReactionInfo({159}, (-6, -3), ('H1', 'He4')),
    '(n,7n)': ReactionInfo({160}, (-6, 0), ()),
    '(n,8n)': ReactionInfo({161}, (-7, 0), ()),
    '(n,5np)': ReactionInfo({162}, (-5, -1), ('H1',)),
    '(n,6np)': ReactionInfo({163}, (-6, -1), ('H1',)),
    '(n,7np)': ReactionInfo({164}, (-7, -1), ('H1',)),
    '(n,4na)': ReactionInfo({165}, (-7, -2), ('He4',)),
    '(n,5na)': ReactionInfo({166}, (-8, -2), ('He4',)),
    '(n,6na)': ReactionInfo({167}, (-9, -2), ('He4',)),
    '(n,7na)': ReactionInfo({168}, (-10, -2), ('He4',)),
    '(n,4nd)': ReactionInfo({169}, (-5, -1), ('H2',)),
    '(n,5nd)': ReactionInfo({170}, (-6, -1), ('H2',)),
    '(n,6nd)': ReactionInfo({171}, (-7, -1), ('H2',)),
    '(n,3nt)': ReactionInfo({172}, (-5, -1), ('H3',)),
    '(n,4nt)': ReactionInfo({173}, (-6, -1), ('H3',)),
    '(n,5nt)': ReactionInfo({174}, (-7, -1), ('H3',)),
    '(n,6nt)': ReactionInfo({175}, (-8, -1), ('H3',)),
    '(n,2n3He)': ReactionInfo({176}, (-4, -2), ('He3',)),
    '(n,3n3He)': ReactionInfo({177}, (-5, -2), ('He3',)),
    '(n,4n3He)': ReactionInfo({178}, (-6, -2), ('He3',)),
    '(n,3n2p)': ReactionInfo({179}, (-4, -2), ('H1', 'H1')),
    '(n,3n2a)': ReactionInfo({180}, (-10, -4), ('He4', 'He4')),
    '(n,3npa)': ReactionInfo({181}, (-7, -3), ('H1', 'He4')),
    '(n,dt)': ReactionInfo({182}, (-4, -2), ('H2', 'H3')),
    '(n,npd)': ReactionInfo({183}, (-3, -2), ('H1', 'H2')),
    '(n,npt)': ReactionInfo({184}, (-4, -2), ('H1', 'H3')),
    '(n,ndt)': ReactionInfo({185}, (-5, -2), ('H2', 'H3')),
    '(n,np3He)': ReactionInfo({186}, (-4, -3), ('H1', 'He3')),
    '(n,nd3He)': ReactionInfo({187}, (-5, -3), ('H2', 'He3')),
    '(n,nt3He)': ReactionInfo({188}, (-6, -3), ('H3', 'He3')),
    '(n,nta)': ReactionInfo({189}, (-7, -3), ('H3', 'He4')),
    '(n,2n2p)': ReactionInfo({190}, (-3, -2), ('H1', 'H1')),
    '(n,p3He)': ReactionInfo({191}, (-4, -3), ('H1', 'He3')),
    '(n,d3He)': ReactionInfo({192}, (-5, -3), ('H2', 'He3')),
    '(n,3Hea)': ReactionInfo({193}, (-6, -4), ('He3', 'He4')),
    '(n,4n2p)': ReactionInfo({194}, (-5, -2), ('H1', 'H1')),
    '(n,4n2a)': ReactionInfo({195}, (-11, -4), ('He4', 'He4')),
    '(n,4npa)': ReactionInfo({196}, (-8, -3), ('H1', 'He4')),
    '(n,3p)': ReactionInfo({197}, (-2, -3), ('H1', 'H1', 'H1')),
    '(n,n3p)': ReactionInfo({198}, (-3, -3), ('H1', 'H1', 'H1')),
    '(n,3n2pa)': ReactionInfo({199}, (-8, -4), ('H1', 'H1', 'He4')),
    '(n,5n2p)': ReactionInfo({200}, (-6, -2), ('H1', 'H1')),
}

__all__ = ["Chain", "REACTIONS"]


def _as_material(x) -> Material:
    """Coerce a filename/path or Material to a Material."""
    if isinstance(x, Material):
        return x
    return Material(x)


def replace_missing(product, decay_data):
    """Replace missing product with a suitable decay daughter."""
    Z, A, state = zam(product)
    symbol = ATOMIC_SYMBOL[Z]

    # Replace neutron with nothing
    if Z == 0:
        return None

    # If ground state is available, prefer it
    if state:
        product = f'{symbol}{A}'

    # Find isotope of this element with the longest half-life
    half_life = 0.0
    mass_longest_lived = A
    for nuclide, data in decay_data.items():
        m = re.match(rf'{symbol}(\d+)(?:_m\d+)?', nuclide)
        if m:
            if data.nuclide['stable']:
                mass_longest_lived = int(m.group(1))
                break
            if data.half_life.nominal_value > half_life:
                mass_longest_lived = int(m.group(1))
                half_life = data.half_life.nominal_value

    # If mass number of longest-lived isotope is less than that of missing
    # product, assume it undergoes beta-. Otherwise assume beta+.
    beta_minus = (mass_longest_lived < A)

    while product not in decay_data:
        if Z > 98:
            Z -= 2
            A -= 4
        else:
            if beta_minus:
                Z += 1
            else:
                Z -= 1
        product = f'{ATOMIC_SYMBOL[Z]}{A}'

    return product


def replace_missing_fpy(actinide, fpy_data, decay_data):
    """Replace missing fission product yields."""
    Z, A, m = zam(actinide)
    if m == 0:
        metastable = gnds_name(Z, A, 1)
        if metastable in fpy_data:
            return metastable

    # Try increasing Z, holding N constant
    isotone = actinide
    while isotone in decay_data:
        Z += 1
        A += 1
        isotone = gnds_name(Z, A, 0)
        if isotone in fpy_data:
            return isotone

    # Try decreasing Z, holding N constant
    isotone = actinide
    while isotone in decay_data:
        Z -= 1
        A -= 1
        isotone = gnds_name(Z, A, 0)
        if isotone in fpy_data:
            return isotone

    # If all else fails, use U235 yields
    return 'U235'


class Chain:
    """Full representation of a depletion chain."""

    def __init__(self):
        self.nuclides = []
        self.reactions = []
        self.nuclide_dict = OrderedDict()
        self._fission_yields = None

    def __contains__(self, nuclide):
        return nuclide in self.nuclide_dict

    def __getitem__(self, name):
        return self.nuclides[self.nuclide_dict[name]]

    def __len__(self):
        return len(self.nuclides)

    def add_nuclide(self, nuclide):
        """Add a nuclide to the depletion chain."""
        self.nuclide_dict[nuclide.name] = len(self.nuclides)
        self.nuclides.append(nuclide)

        for rx in nuclide.reactions:
            if rx.type not in self.reactions:
                self.reactions.append(rx.type)

    @classmethod
    def from_endf(cls, decay_files, fpy_files, neutron_files,
                  reactions=('(n,2n)', '(n,3n)', '(n,4n)', '(n,gamma)',
                             '(n,p)', '(n,a)'),
                  progress=True):
        """Create a depletion chain from ENDF files.

        Parameters
        ----------
        decay_files : list of str, os.PathLike, or endf.Material
            ENDF decay sub-library files
        fpy_files : list of str, os.PathLike, or endf.Material
            ENDF neutron-induced fission product yield sub-library files
        neutron_files : list of str, os.PathLike, or endf.Material
            ENDF neutron reaction sub-library files
        reactions : iterable of str, optional
            Transmutation reactions to include. Fission is always included
            if present. See :data:`endf.chain.REACTIONS` for the complete
            listing.
        progress : bool, optional
            Print status messages during processing.

        Returns
        -------
        Chain
        """
        transmutation_reactions = reactions

        # Map target name -> {MT: Q value}
        if progress:
            print('Processing neutron sub-library files...')
        reactions = {}
        for f in neutron_files:
            mat = _as_material(f)
            meta = mat.section_data[1, 451]
            ZA = int(meta['ZA'])
            Z, A = divmod(ZA, 1000)
            m = int(meta.get('LISO', 0) or 0)
            name = gnds_name(Z, A, m)
            reactions[name] = {}
            for mf, mt, _nc, _mod in meta['section_list']:
                if mf == 3 and (3, mt) in mat.section_data:
                    reactions[name][mt] = mat.section_data[3, mt]['QM']

        # Build decay data
        if progress:
            print('Processing decay sub-library files...')
        decay_data = {}
        for f in decay_files:
            data = Decay(f)
            # Skip decay data for the neutron itself
            if data.nuclide['atomic_number'] == 0:
                continue
            decay_data[data.nuclide['name']] = data

        if progress:
            print('Processing fission product yield sub-library files...')
        fpy_data = {}
        for f in fpy_files:
            data = FissionProductYields(f)
            fpy_data[data.nuclide['name']] = data

        if progress:
            print('Creating depletion_chain...')
        missing_daughter = []
        missing_rx_product = []
        missing_fpy = []
        missing_fp = []

        chain = cls()
        for idx, parent in enumerate(sorted(decay_data, key=zam)):
            data = decay_data[parent]

            nuclide = Nuclide(parent)

            if not data.nuclide['stable'] and data.half_life.nominal_value != 0.0:
                nuclide.half_life = data.half_life.nominal_value
                nuclide.decay_energy = data.decay_energy.nominal_value
                sum_br = 0.0
                for i, mode in enumerate(data.modes):
                    type_ = ','.join(mode.modes)
                    if mode.daughter in decay_data:
                        target = mode.daughter
                    else:
                        print(f'missing {parent} {",".join(mode.modes)} '
                              f'{mode.daughter}')
                        target = replace_missing(mode.daughter, decay_data)

                    br = mode.branching_ratio.nominal_value
                    sum_br += br
                    if i == len(data.modes) - 1 and sum_br != 1.0:
                        br = 1.0 - sum(m.branching_ratio.nominal_value
                                       for m in data.modes[:-1])

                    nuclide.add_decay_mode(type_, target, br)

                # Attach decay radiation sources (photon, electron, etc.)
                try:
                    nuclide.sources = data.sources
                except Exception as exc:
                    warn(f"Failed to extract decay sources for {parent}: {exc}")

            fissionable = False
            if parent in reactions:
                reactions_available = set(reactions[parent].keys())
                for name in transmutation_reactions:
                    mts, changes, _ = REACTIONS[name]
                    if mts & reactions_available:
                        delta_A, delta_Z = changes
                        A = data.nuclide['mass_number'] + delta_A
                        Z = data.nuclide['atomic_number'] + delta_Z
                        daughter = f'{ATOMIC_SYMBOL[Z]}{A}'

                        if daughter not in decay_data:
                            daughter = replace_missing(daughter, decay_data)
                            if daughter is None:
                                missing_rx_product.append(
                                    (parent, name, daughter))

                        for mt in sorted(mts):
                            if mt in reactions[parent]:
                                q_value = reactions[parent][mt]
                                break
                        else:
                            q_value = 0.0

                        nuclide.add_reaction(name, daughter, q_value, 1.0)

                if any(mt in reactions_available for mt in FISSION_MTS):
                    q_value = reactions[parent][18]
                    nuclide.add_reaction('fission', None, q_value, 1.0)
                    fissionable = True

            if fissionable:
                if parent in fpy_data:
                    fpy = fpy_data[parent]

                    if fpy.energies is not None:
                        yield_energies = fpy.energies
                    else:
                        yield_energies = [0.0]

                    yield_data = {}
                    for E, yield_table in zip(yield_energies, fpy.independent):
                        yield_replace = 0.0
                        yields = defaultdict(float)
                        for product, y in yield_table.items():
                            if product not in decay_data:
                                daughter = replace_missing(product, decay_data)
                                product = daughter
                                yield_replace += y.nominal_value

                            yields[product] += y.nominal_value

                        if yield_replace > 0.0:
                            missing_fp.append((parent, E, yield_replace))
                        yield_data[E] = yields

                    nuclide.yield_data = FissionYieldDistribution(yield_data)
                else:
                    nuclide._fpy = replace_missing_fpy(
                        parent, fpy_data, decay_data)
                    missing_fpy.append((parent, nuclide._fpy))

            chain.add_nuclide(nuclide)

        # Replace missing FPY data
        for nuclide in chain.nuclides:
            if hasattr(nuclide, '_fpy'):
                nuclide.yield_data = chain[nuclide._fpy].yield_data

        if missing_daughter:
            print('The following decay modes have daughters with no decay data:')
            for mode in missing_daughter:
                print(f'  {mode}')
            print('')

        if missing_rx_product:
            print('The following reaction products have no decay data:')
            for vals in missing_rx_product:
                print('{} {} -> {}'.format(*vals))
            print('')

        if missing_fpy:
            print('The following fissionable nuclides have no fission '
                  'product yields:')
            for parent, replacement in missing_fpy:
                print(f'  {parent}, replaced with {replacement}')
            print('')

        if missing_fp:
            print('The following nuclides have fission products with no '
                  'decay data:')
            for vals in missing_fp:
                print('  {}, E={} eV (total yield={})'.format(*vals))

        return chain

    @classmethod
    def from_xml(cls, filename, fission_q=None):
        """Read a depletion chain XML file."""
        chain = cls()

        if fission_q is not None:
            check_type("fission_q", fission_q, Mapping)
        else:
            fission_q = {}

        root = ET.parse(str(filename))

        for nuclide_elem in root.findall('nuclide'):
            this_q = fission_q.get(nuclide_elem.get("name"))
            nuc = Nuclide.from_xml(nuclide_elem, root, this_q)
            chain.add_nuclide(nuc)

        return chain

    def export_to_xml(self, filename):
        """Write a depletion chain XML file."""
        root_elem = ET.Element('depletion_chain')
        for nuclide in self.nuclides:
            root_elem.append(nuclide.to_xml_element())

        tree = ET.ElementTree(root_elem)
        if _have_lxml:
            tree.write(str(filename), encoding='utf-8', pretty_print=True)
        else:
            clean_indentation(root_elem)
            tree.write(str(filename), encoding='utf-8')

    def get_default_fission_yields(self):
        """Return fission yields at the lowest incident neutron energy."""
        out = defaultdict(dict)
        for nuc in self.nuclides:
            if nuc.yield_data is None:
                continue
            yield_obj = nuc.yield_data[min(nuc.yield_energies)]
            out[nuc.name] = dict(yield_obj)
        return out

    def form_matrix(self, rates, fission_yields=None):
        """Form the depletion matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix representing depletion. Requires scipy.
        """
        import scipy.sparse as sp  # lazy import so chain XML I/O doesn't need scipy

        matrix = defaultdict(float)
        rxn_seen = set()

        if fission_yields is None:
            fission_yields = self.get_default_fission_yields()

        for i, nuc in enumerate(self.nuclides):
            # Loss from radioactive decay
            decay_constant = 0.0
            if nuc.half_life is not None:
                decay_constant = math.log(2) / nuc.half_life
                if decay_constant != 0.0:
                    matrix[i, i] -= decay_constant

            # Gain from radioactive decay
            if nuc.n_decay_modes != 0:
                for _, target, branching_ratio in nuc.decay_modes:
                    if target is not None:
                        branch_val = branching_ratio * decay_constant
                        if branch_val != 0.0:
                            k = self.nuclide_dict[target]
                            matrix[k, i] += branch_val

            if nuc.name in rates.index_nuc:
                nuc_ind = rates.index_nuc[nuc.name]
                nuc_rates = rates[nuc_ind, :]

                for r_type, target, _, br in nuc.reactions:
                    r_id = rates.index_rx[r_type]
                    path_rate = nuc_rates[r_id]

                    if r_type not in rxn_seen:
                        rxn_seen.add(r_type)
                        if path_rate != 0.0:
                            matrix[i, i] -= path_rate

                    if r_type != 'fission':
                        if target is not None and path_rate != 0.0:
                            k = self.nuclide_dict[target]
                            matrix[k, i] += path_rate * br

                        light_nucs = REACTIONS[r_type].secondaries
                        for light_nuc in light_nucs:
                            k = self.nuclide_dict.get(light_nuc)
                            if k is not None:
                                matrix[k, i] += path_rate * br
                    else:
                        for product, y in fission_yields[nuc.name].items():
                            yield_val = y * path_rate
                            if yield_val != 0.0:
                                k = self.nuclide_dict[product]
                                matrix[k, i] += yield_val

                rxn_seen.clear()

        n = len(self)
        matrix_dok = sp.dok_matrix((n, n))
        dict.update(matrix_dok, matrix)
        return matrix_dok.tocsr()

    def get_branch_ratios(self, reaction="(n,gamma)"):
        """Return reaction branching ratios keyed by parent nuclide."""
        capt = {}
        for nuclide in self.nuclides:
            nuc_capt = {}
            for rx in nuclide.reactions:
                if rx.type == reaction and rx.branching_ratio != 1.0:
                    nuc_capt[rx.target] = rx.branching_ratio
            if nuc_capt:
                capt[nuclide.name] = nuc_capt
        return capt

    def set_branch_ratios(self, branch_ratios, reaction="(n,gamma)",
                          strict=True, tolerance=1e-5):
        """Set the branching ratios for a given reaction."""
        sums = {}
        rxn_ix_map = {}
        grounds = {}

        tolerance = abs(tolerance)

        missing_parents = set()
        missing_products = {}
        missing_reaction = set()
        bad_sums = {}

        secondary = REACTIONS[reaction].secondaries

        for parent, sub in branch_ratios.items():
            if parent not in self:
                if strict:
                    raise KeyError(parent)
                missing_parents.add(parent)
                continue

            prod_flag = False
            for product in sub:
                if product not in self:
                    if strict:
                        raise KeyError(product)
                    missing_products[parent] = product
                    prod_flag = True
                    break

            if prod_flag:
                continue

            indexes = []
            for ix, rx in enumerate(self[parent].reactions):
                if rx.type == reaction and rx.target not in secondary:
                    indexes.append(ix)
                    if "_m" not in rx.target:
                        grounds[parent] = rx.target

            if len(indexes) == 0:
                if strict:
                    raise AttributeError(
                        f"Nuclide {parent} does not have {reaction} reactions")
                missing_reaction.add(parent)
                continue

            this_sum = sum(sub.values())
            if (this_sum >= 1 + tolerance or
                    (grounds[parent] in sub and this_sum <= 1 - tolerance)):
                if strict:
                    raise ValueError(
                        f"Sum of {reaction} branching ratios for {parent} "
                        f"({this_sum:7.3f}) outside tolerance of 1 +/- "
                        f"{tolerance:5.3e}")
                bad_sums[parent] = this_sum
            else:
                rxn_ix_map[parent] = indexes
                sums[parent] = this_sum

        if len(rxn_ix_map) == 0:
            raise IndexError(
                f"No {reaction} reactions found in this "
                f"{self.__class__.__name__}")

        if missing_parents:
            warn(f"The following nuclides were not found in "
                 f"{self.__class__.__name__}: "
                 f"{', '.join(sorted(missing_parents))}")

        if missing_reaction:
            warn(f"The following nuclides did not have {reaction} reactions: "
                 f"{', '.join(sorted(missing_reaction))}")

        if missing_products:
            tail = (f"{k} -> {v}" for k, v in sorted(missing_products.items()))
            warn("The following products were not found in the "
                 f"{self.__class__.__name__} and parents were unmodified: \n"
                 f"{', '.join(tail)}")

        if bad_sums:
            tail = (f"{k}: {s:5.3f}" for k, s in sorted(bad_sums.items()))
            warn(f"The following parent nuclides were given {reaction} branch "
                 f"ratios with a sum outside tolerance of 1 +/- "
                 f"{tolerance:5.3e}:\n{chr(10).join(tail)}")

        for parent_name, rxn_index in rxn_ix_map.items():
            parent = self[parent_name]
            new_ratios = branch_ratios[parent_name]
            rxn_index = rxn_ix_map[parent_name]

            rxn_Q = parent.reactions[rxn_index[0]].Q

            for ix in reversed(rxn_index):
                parent.reactions.pop(ix)

            all_meta = True
            for target, br in new_ratios.items():
                all_meta = all_meta and ("_m" in target)
                parent.add_reaction(reaction, target, rxn_Q, br)

            if all_meta and sums[parent_name] != 1.0:
                ground_br = 1.0 - sums[parent_name]
                ground_target = grounds.get(parent_name)
                if ground_target is None:
                    pz, pa, pm = zam(parent_name)
                    ground_target = gnds_name(pz, pa + 1, 0)
                new_ratios[ground_target] = ground_br
                parent.add_reaction(reaction, ground_target, rxn_Q, ground_br)

    @property
    def fission_yields(self):
        if self._fission_yields is None:
            self._fission_yields = [self.get_default_fission_yields()]
        return self._fission_yields

    @fission_yields.setter
    def fission_yields(self, yields):
        if yields is not None:
            if isinstance(yields, Mapping):
                yields = [yields]
            check_type("fission_yields", yields, Iterable, Mapping)
        self._fission_yields = yields

    def validate(self, strict=True, quiet=False, tolerance=1e-4):
        """Search for possible inconsistencies in branching ratios and yields."""
        check_type("tolerance", tolerance, Real)
        check_greater_than("tolerance", tolerance, 0.0, True)
        valid = True
        for name in sorted(self.nuclide_dict):
            stat = self[name].validate(strict, quiet, tolerance)
            if quiet and not stat:
                return stat
            valid = valid and stat
        return valid

    def reduce(self, initial_isotopes, level=None):
        """Reduce the chain by following transmutation paths from initial isotopes."""
        check_type("initial_isotopes", initial_isotopes, Iterable, str)
        if level is None:
            level = math.inf
        else:
            check_type("level", level, Integral)
            check_greater_than("level", level, 0, equality=True)

        all_isotopes = self._follow(set(initial_isotopes), level)

        name_sort = sorted(all_isotopes)

        new_chain = type(self)()

        for iso in sorted(all_isotopes, key=zam):
            previous = self[iso]
            new_nuclide = Nuclide(previous.name)
            new_nuclide.half_life = previous.half_life
            new_nuclide.decay_energy = previous.decay_energy
            new_nuclide.sources = dict(previous.sources)
            if hasattr(previous, '_fpy'):
                new_nuclide._fpy = previous._fpy

            for mode in previous.decay_modes:
                if mode.target in all_isotopes:
                    new_nuclide.add_decay_mode(*mode)
                else:
                    new_nuclide.add_decay_mode(
                        mode.type, None, mode.branching_ratio)

            for rx in previous.reactions:
                if rx.target in all_isotopes:
                    new_nuclide.add_reaction(*rx)
                elif rx.type == "fission":
                    new_yields = new_nuclide.yield_data = (
                        previous.yield_data.restrict_products(name_sort))
                    if new_yields is not None:
                        new_nuclide.add_reaction(*rx)
                else:
                    new_nuclide.add_reaction(
                        rx.type, None, rx.Q, rx.branching_ratio)

            new_chain.add_nuclide(new_nuclide)

        new_chain.reactions = sorted(new_chain.reactions)
        return new_chain

    def _follow(self, isotopes, level):
        """Return all isotopes present up to depth ``level``."""
        found = isotopes.copy()
        remaining = set(self.nuclide_dict)
        if not found.issubset(remaining):
            raise IndexError(
                "The following isotopes were not found in the chain: "
                f"{', '.join(found - remaining)}")

        if level == 0:
            return found

        remaining -= found

        depth = 0
        next_iso = set()

        while depth < level and remaining:
            while isotopes:
                iso = isotopes.pop()
                found.add(iso)
                nuclide = self[iso]

                for rxn in nuclide.reactions + nuclide.decay_modes:
                    if rxn.type == "fission":
                        continue

                    if rxn.type in REACTIONS:
                        secondaries = REACTIONS[rxn.type].secondaries
                    else:
                        secondaries = []

                    secondaries = [x for x in secondaries if x in self]

                    for product in ichain([rxn.target], secondaries):
                        if product is None:
                            continue
                        if (product in next_iso or product in found
                                or product in isotopes):
                            continue
                        next_iso.add(product)

                if nuclide.yield_data is not None:
                    for product in nuclide.yield_data.products:
                        if (product in next_iso or product in found
                                or product in isotopes):
                            continue
                        next_iso.add(product)

            if not next_iso:
                return found

            depth += 1
            isotopes |= next_iso
            remaining -= next_iso
            next_iso.clear()

        found.update(isotopes)
        return found
