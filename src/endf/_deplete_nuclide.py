# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/deplete/nuclide.py.

"""Per-nuclide components of a depletion chain."""

import bisect
from collections.abc import Mapping
from collections import namedtuple, defaultdict
from warnings import warn
from numbers import Real
try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from numpy import empty, searchsorted

from ._checkvalue import check_type
from .univariate import Univariate


__all__ = [
    "DecayTuple", "ReactionTuple", "Nuclide", "FissionYield",
    "FissionYieldDistribution"
]


DecayTuple = namedtuple('DecayTuple', 'type target branching_ratio')
DecayTuple.__doc__ = """Decay mode information."""

ReactionTuple = namedtuple('ReactionTuple', 'type target Q branching_ratio')
ReactionTuple.__doc__ = """Transmutation reaction information."""


class Nuclide:
    """Decay modes, reactions, and fission yields for a single nuclide.

    Parameters
    ----------
    name : str, optional
        GND name of this nuclide, e.g. ``"He4"``, ``"Am242_m1"``

    Attributes
    ----------
    sources : dict
        Radiation emitted by the decay of this nuclide, mapping particle type
        to a distribution of emission rates in [/s]. See
        :attr:`endf.Decay.sources`.
    """

    def __init__(self, name=None):
        self.name = name
        self.half_life = None
        self.decay_energy = 0.0

        self.decay_modes = []
        self.reactions = []
        self.sources = {}

        self._yield_data = None

    @property
    def n_decay_modes(self):
        return len(self.decay_modes)

    @property
    def n_reaction_paths(self):
        return len(self.reactions)

    @property
    def yield_data(self):
        return self._yield_data

    @yield_data.setter
    def yield_data(self, fission_yields):
        if fission_yields is None:
            self._yield_data = None
        else:
            check_type("fission_yields", fission_yields, Mapping)
            if isinstance(fission_yields, FissionYieldDistribution):
                self._yield_data = fission_yields
            else:
                self._yield_data = FissionYieldDistribution(fission_yields)

    @property
    def yield_energies(self):
        if self._yield_data is None:
            return None
        return self.yield_data.energies

    def add_decay_mode(self, type, target, branching_ratio):
        self.decay_modes.append(
            DecayTuple(type, target, branching_ratio)
        )

    def add_reaction(self, type, target, Q, branching_ratio):
        self.reactions.append(
            ReactionTuple(type, target, Q, branching_ratio)
        )

    @classmethod
    def from_xml(cls, element, root=None, fission_q=None):
        """Read nuclide from an XML element."""
        nuc = cls()
        nuc.name = element.get('name')

        if 'half_life' in element.attrib:
            nuc.half_life = float(element.get('half_life'))
            nuc.decay_energy = float(element.get('decay_energy', '0'))

        for decay_elem in element.iter('decay'):
            d_type = decay_elem.get('type')
            target = decay_elem.get('target')
            if target is not None and target.lower() == "nothing":
                target = None
            branching_ratio = float(decay_elem.get('branching_ratio'))
            nuc.decay_modes.append(DecayTuple(d_type, target, branching_ratio))

        for src_elem in element.iter('source'):
            particle = src_elem.get('particle')
            nuc.sources[particle] = Univariate.from_xml_element(src_elem)

        for reaction_elem in element.iter('reaction'):
            r_type = reaction_elem.get('type')
            Q = float(reaction_elem.get('Q', '0'))
            branching_ratio = float(reaction_elem.get('branching_ratio', '1'))

            if r_type != 'fission':
                target = reaction_elem.get('target')
                if target is not None and target.lower() == "nothing":
                    target = None
            else:
                target = None
                if fission_q is not None:
                    Q = fission_q

            nuc.reactions.append(ReactionTuple(
                r_type, target, Q, branching_ratio))

        fpy_elem = element.find('neutron_fission_yields')
        if fpy_elem is not None:
            parent = fpy_elem.get('parent')
            if parent is not None:
                assert root is not None
                fpy_elem = root.find(
                    f'.//nuclide[@name="{parent}"]/neutron_fission_yields'
                )
                if fpy_elem is None:
                    raise ValueError(
                        f"Fission product yields for {nuc.name} borrow from "
                        f"{parent}, but {parent} is not present in the chain "
                        "file or has no yields."
                    )
                nuc._fpy = parent

            nuc.yield_data = FissionYieldDistribution.from_xml_element(fpy_elem)

        return nuc

    def to_xml_element(self):
        """Write nuclide to XML element."""
        elem = ET.Element('nuclide')
        elem.set('name', self.name)

        if self.half_life is not None:
            elem.set('half_life', str(self.half_life))
            elem.set('decay_modes', str(len(self.decay_modes)))
            elem.set('decay_energy', str(self.decay_energy))
            for mode_type, daughter, br in self.decay_modes:
                mode_elem = ET.SubElement(elem, 'decay')
                mode_elem.set('type', mode_type)
                mode_elem.set('target', daughter or "Nothing")
                mode_elem.set('branching_ratio', str(br))

        for particle, source in self.sources.items():
            src_elem = source.to_xml_element('source')
            src_elem.set('particle', particle)
            elem.append(src_elem)

        elem.set('reactions', str(len(self.reactions)))
        for rx, daughter, Q, br in self.reactions:
            rx_elem = ET.SubElement(elem, 'reaction')
            rx_elem.set('type', rx)
            rx_elem.set('Q', str(Q))
            if daughter is not None:
                rx_elem.set('target', daughter)
            if br != 1.0:
                rx_elem.set('branching_ratio', str(br))

        if self.yield_data:
            fpy_elem = ET.SubElement(elem, 'neutron_fission_yields')

            if hasattr(self, '_fpy'):
                fpy_elem.set('parent', self._fpy)
            else:
                energy_elem = ET.SubElement(fpy_elem, 'energies')
                energy_elem.text = ' '.join(str(E) for E in self.yield_energies)
                self.yield_data.to_xml_element(fpy_elem)

        return elem

    def validate(self, strict=True, quiet=False, tolerance=1e-4):
        """Search for possible inconsistencies in branching ratios and yields."""
        msg_func = ("Nuclide {name} has {prop} that sum to {actual} "
                    "instead of {expected} +/- {tol:7.4e}").format
        valid = True

        if self.decay_modes:
            sum_br = sum(m.branching_ratio for m in self.decay_modes)
            if not (1.0 - tolerance <= sum_br <= 1.0 + tolerance):
                msg = msg_func(name=self.name, actual=sum_br, expected=1.0,
                               tol=tolerance, prop="decay mode branch ratios")
                if strict:
                    raise ValueError(msg)
                if quiet:
                    return False
                warn(msg)
                valid = False

        if self.reactions:
            type_map = defaultdict(set)
            for reaction in self.reactions:
                type_map[reaction.type].add(reaction)
            for rxn_type, reactions in type_map.items():
                sum_rxn = sum(rx.branching_ratio for rx in reactions)
                if 1.0 - tolerance <= sum_rxn <= 1.0 + tolerance:
                    continue
                msg = msg_func(name=self.name, actual=sum_rxn, expected=1.0,
                               tol=tolerance,
                               prop=f"{rxn_type} reaction branch ratios")
                if strict:
                    raise ValueError(msg)
                if quiet:
                    return False
                warn(msg)
                valid = False

        if self.yield_data:
            for energy, fission_yield in self.yield_data.items():
                sum_yield = fission_yield.yields.sum()
                if 2.0 - tolerance <= sum_yield <= 2.0 + tolerance:
                    continue
                msg = msg_func(
                    name=self.name, actual=sum_yield, expected=2.0,
                    tol=tolerance,
                    prop=f"fission yields (E = {energy:7.4e} eV)")
                if strict:
                    raise ValueError(msg)
                if quiet:
                    return False
                warn(msg)
                valid = False

        return valid


class FissionYieldDistribution(Mapping):
    """Energy-dependent fission product yields for a single nuclide."""

    def __init__(self, fission_yields):
        energies = sorted(fission_yields)

        shared_prod = set.union(*(set(x) for x in fission_yields.values()))
        ordered_prod = sorted(shared_prod)

        yield_matrix = empty((len(energies), len(shared_prod)))

        for g_index, energy in enumerate(energies):
            prod_map = fission_yields[energy]
            for prod_ix, product in enumerate(ordered_prod):
                yield_matrix[g_index, prod_ix] = prod_map.get(product, 0.0)
        self.energies = tuple(energies)
        self.products = tuple(ordered_prod)
        self.yield_matrix = yield_matrix

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, energy):
        if energy not in self.energies:
            raise KeyError(energy)
        return FissionYield(
            self.products, self.yield_matrix[self.energies.index(energy)])

    def __iter__(self):
        return iter(self.energies)

    def __repr__(self):
        return (f"<{self.__class__.__name__} with {self.yield_matrix.shape[1]} "
                f"products at {len(self.energies)} energies>")

    @classmethod
    def from_xml_element(cls, element):
        all_yields = {}
        for yield_elem in element.iter("fission_yields"):
            energy = float(yield_elem.get("energy"))
            products = yield_elem.find("products").text.split()
            yields = map(float, yield_elem.find("data").text.split())
            all_yields[energy] = dict(zip(products, yields))

        return cls(all_yields)

    def to_xml_element(self, root):
        for energy, yield_obj in self.items():
            yield_element = ET.SubElement(root, "fission_yields")
            yield_element.set("energy", str(energy))
            product_elem = ET.SubElement(yield_element, "products")
            product_elem.text = " ".join(map(str, yield_obj.products))
            data_elem = ET.SubElement(yield_element, "data")
            data_elem.text = " ".join(map(str, yield_obj.yields))

    def restrict_products(self, possible_products):
        overlap = set(self.products).intersection(possible_products)
        if not overlap:
            return None

        products = sorted(overlap)
        indices = searchsorted(self.products, products)

        new_yields = {}
        for ene, yields in zip(self.energies, self.yield_matrix.copy()):
            new_yields[ene] = dict(zip(products, yields[indices]))

        return type(self)(new_yields)


class FissionYield(Mapping):
    """Mapping for fission yields of a parent at a specific energy."""

    def __init__(self, products, yields):
        self.products = products
        self.yields = yields

    def __contains__(self, product):
        ix = bisect.bisect_left(self.products, product)
        return ix != len(self.products) and self.products[ix] == product

    def __getitem__(self, product):
        ix = bisect.bisect_left(self.products, product)
        if ix == len(self.products) or self.products[ix] != product:
            raise KeyError(product)
        return self.yields[ix]

    def __len__(self):
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    def items(self):
        return zip(self.products, self.yields)

    def __add__(self, other):
        if not isinstance(other, FissionYield):
            return NotImplemented
        new = FissionYield(self.products, self.yields.copy())
        new += other
        return new

    def __iadd__(self, other):
        if not isinstance(other, FissionYield):
            return NotImplemented
        self.yields += other.yields
        return self

    def __radd__(self, other):
        return self + other

    def __imul__(self, scalar):
        if not isinstance(scalar, Real):
            return NotImplemented
        self.yields *= scalar
        return self

    def __mul__(self, scalar):
        if not isinstance(scalar, Real):
            return NotImplemented
        new = FissionYield(self.products, self.yields.copy())
        new *= scalar
        return new

    def __rmul__(self, scalar):
        return self * scalar

    def __repr__(self):
        return (f"<{self.__class__.__name__} containing {len(self)} "
                "products and yields>")

    # Avoid greedy numpy operations on scalars (see openmc issue #1492)
    __array_ufunc__ = None
