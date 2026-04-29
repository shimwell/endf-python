# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Minimal port of the bits of openmc.stats.univariate needed to round-trip
# <source> elements in depletion chain XML — Discrete and Tabular only.

"""Minimal univariate distributions for chain XML ``<source>`` elements."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Real

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from ._checkvalue import check_type, check_greater_than


__all__ = ["Univariate", "Discrete", "Tabular"]


_INTERPOLATION_SCHEMES = {
    'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'
}


class Univariate(ABC):
    """Base class for 1-D probability distributions."""

    @abstractmethod
    def to_xml_element(self, element_name):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @classmethod
    def from_xml_element(cls, elem):
        """Dispatch on the ``type`` attribute to the right subclass."""
        dist_type = elem.get('type')
        if dist_type == 'discrete':
            return Discrete.from_xml_element(elem)
        if dist_type == 'tabular':
            return Tabular.from_xml_element(elem)
        raise ValueError(
            f"Unsupported <source type={dist_type!r}> in chain XML — "
            "endf-python only ports Discrete and Tabular."
        )


class Discrete(Univariate):
    """Distribution with probability mass at discrete points."""

    def __init__(self, x, p):
        self.x = x
        self.p = p

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def p(self):
        return self._p

    @x.setter
    def x(self, x):
        if isinstance(x, Real):
            x = [x]
        check_type('discrete values', x, Iterable, Real)
        self._x = x

    @p.setter
    def p(self, p):
        if isinstance(p, Real):
            p = [p]
        check_type('discrete probabilities', p, Iterable, Real)
        for pk in p:
            check_greater_than('discrete probability', pk, 0.0, True)
        self._p = p

    def to_xml_element(self, element_name):
        element = ET.Element(element_name)
        element.set("type", "discrete")
        params = ET.SubElement(element, "parameters")
        params.text = (' '.join(map(str, self.x)) + ' '
                       + ' '.join(map(str, self.p)))
        return element

    @classmethod
    def from_xml_element(cls, elem):
        parameters = elem.find('parameters')
        text = parameters.text if parameters is not None else elem.get('parameters', '')
        params = [float(v) for v in text.split()]
        half = len(params) // 2
        return cls(params[:half], params[half:])


class Tabular(Univariate):
    """Piecewise-continuous tabulated probability density."""

    def __init__(self, x, p, interpolation='linear-linear',
                 ignore_negative=False):
        self._ignore_negative = ignore_negative
        self.x = x
        self.p = p
        self.interpolation = interpolation

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def p(self):
        return self._p

    @property
    def interpolation(self):
        return self._interpolation

    @x.setter
    def x(self, x):
        check_type('tabulated values', x, Iterable, Real)
        self._x = x

    @p.setter
    def p(self, p):
        check_type('tabulated probabilities', p, Iterable, Real)
        if not self._ignore_negative:
            for pk in p:
                check_greater_than('tabulated probability', pk, 0.0, True)
        self._p = p

    @interpolation.setter
    def interpolation(self, interpolation):
        if interpolation not in _INTERPOLATION_SCHEMES:
            raise ValueError(
                f"interpolation must be one of {_INTERPOLATION_SCHEMES}, "
                f"got {interpolation!r}"
            )
        self._interpolation = interpolation

    def to_xml_element(self, element_name):
        element = ET.Element(element_name)
        element.set("type", "tabular")
        element.set("interpolation", self.interpolation)
        params = ET.SubElement(element, "parameters")
        params.text = (' '.join(map(str, self.x)) + ' '
                       + ' '.join(map(str, self.p)))
        return element

    @classmethod
    def from_xml_element(cls, elem):
        interpolation = elem.get('interpolation', 'linear-linear')
        parameters = elem.find('parameters')
        text = parameters.text if parameters is not None else elem.get('parameters', '')
        params = [float(v) for v in text.split()]
        half = len(params) // 2
        return cls(params[:half], params[half:], interpolation)
