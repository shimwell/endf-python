# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/stats/univariate.py, reduced to the distributions needed
# to represent secondary particle data. Sampling and biasing are omitted (this
# library reads data, it does not transport particles), as is the XML
# serialisation, which leaves numpy as the only dependency: openmc's version
# pulls exprel and hyp1f1 from scipy.

"""Probability distributions of a single random variable.

These are the containers used for tabulated secondary distributions: the
scattering cosines of an angular distribution, and the outgoing energies of an
energy distribution. ENDF and ACE both describe such data as a set of tabulated
points, either as a probability mass function over discrete values
(:class:`Discrete`), as a piecewise density function (:class:`Tabular`), or as a
weighted combination of the two (:class:`Mixture`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np

from ._checkvalue import check_type, check_greater_than


__all__ = [
    "Univariate", "Discrete", "Tabular", "Mixture", "combine_distributions"
]


INTERPOLATION_SCHEMES = {
    'histogram',
    'linear-linear',
    'linear-log',
    'log-linear',
    'log-log',
}


def _exprel(x):
    """Evaluate ``(exp(x) - 1)/x`` without loss of precision near zero.

    Equivalent to ``scipy.special.exprel``, reimplemented so that this module
    needs only numpy.
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-16
    # Substitute 1 in the denominator where x is (near) zero so the division
    # never warns; those entries are replaced by the limit value below.
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0, np.expm1(x) / safe)


class Univariate(ABC):
    """Probability distribution of a single random variable.

    Abstract base class for :class:`Discrete`, :class:`Tabular` and
    :class:`Mixture`.

    Attributes
    ----------
    c : numpy.ndarray or None
        Tabulated cumulative distribution, when one was supplied with the data.
        ACE tables store a CDF alongside the PDF, and it is kept verbatim
        rather than recomputed so that values round-trip exactly; see
        :meth:`cdf` for the computed equivalent. ``None`` when the source
        format did not provide one.

    """

    def __init__(self):
        self.c = None

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def cdf(self) -> np.ndarray:
        """Return the cumulative distribution computed from the tabulated
        probabilities.

        This is not the same as :attr:`c`, which is the CDF as stored in the
        source file when it provided one.
        """

    @abstractmethod
    def integral(self) -> float:
        """Return the integral of the distribution."""

    @abstractmethod
    def normalize(self):
        """Scale the stored probabilities so the distribution integrates to 1."""


class Discrete(Univariate):
    """Distribution characterized by a probability mass function.

    The Discrete distribution assigns probability values to discrete values of a
    random variable, rather than expressing the distribution as a continuous
    random variable.

    Parameters
    ----------
    x
        Values of the random variable
    p
        Discrete probability for each value

    Attributes
    ----------
    x : numpy.ndarray
        Values of the random variable
    p : numpy.ndarray
        Discrete probability for each value

    """

    def __init__(self, x, p):
        super().__init__()
        self.x = x
        self.p = p

    def __len__(self) -> int:
        return self.x.size

    def __repr__(self) -> str:
        return f"<Discrete: {len(self)} values>"

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, Real):
            x = [x]
        check_type('discrete values', x, Iterable, Real)
        self._x = np.array(x, dtype=float)

    @property
    def p(self) -> np.ndarray:
        return self._p

    @p.setter
    def p(self, p):
        if isinstance(p, Real):
            p = [p]
        check_type('discrete probabilities', p, Iterable, Real)
        for pk in p:
            check_greater_than('discrete probability', pk, 0.0, True)
        self._p = np.array(p, dtype=float)

    def cdf(self) -> np.ndarray:
        return np.insert(np.cumsum(self.p), 0, 0.0)

    def integral(self) -> float:
        return float(np.sum(self.p))

    def normalize(self):
        self._p = self._p / self._p.sum()

    @classmethod
    def merge(cls, dists: Sequence[Discrete], probs: Sequence[float]) -> Discrete:
        """Merge multiple discrete distributions into a single distribution.

        Values appearing in more than one distribution are combined into a
        single entry whose probability is the weighted sum.

        Parameters
        ----------
        dists
            Discrete distributions to combine
        probs
            Probability (or intensity) of each distribution

        Returns
        -------
        Combined discrete distribution

        """
        if len(dists) != len(probs):
            raise ValueError(
                "Number of distributions and probabilities must match.")

        x_merged = set()
        p_merged = defaultdict(float)
        for dist, p_dist in zip(dists, probs):
            for x, p in zip(dist.x, dist.p):
                x_merged.add(x)
                p_merged[x] += p * p_dist

        x_arr = np.array(sorted(x_merged))
        p_arr = np.array([p_merged[x] for x in x_arr])
        return cls(x_arr, p_arr)


class Tabular(Univariate):
    """Piecewise continuous probability distribution.

    This class is used to represent a probability distribution whose density
    function is tabulated at specific values with a specified interpolation
    scheme.

    Parameters
    ----------
    x
        Tabulated values of the random variable
    p
        Tabulated probabilities. For histogram interpolation, if the length of
        `p` is the same as `x`, the last value is ignored. Probabilities `p` are
        given per unit of `x`.
    interpolation
        Indicates how the density function is interpolated between tabulated
        points. One of ``'histogram'``, ``'linear-linear'``, ``'linear-log'``,
        ``'log-linear'`` or ``'log-log'``. Defaults to ``'linear-linear'``.
    ignore_negative
        Allow negative probabilities. ACE tables occasionally carry small
        negative entries which are kept verbatim rather than clipped.

    Attributes
    ----------
    x : numpy.ndarray
        Tabulated values of the random variable
    p : numpy.ndarray
        Tabulated probabilities
    interpolation : str
        Indicates how the density function is interpolated between tabulated
        points.

    Notes
    -----
    The probabilities `p` are interpreted per unit of the corresponding
    independent variable `x`, following the usual definition of a probability
    density function. If `x` is an energy in eV, `p` is a probability per eV.

    """

    def __init__(
        self,
        x: Sequence[float],
        p: Sequence[float],
        interpolation: str = 'linear-linear',
        ignore_negative: bool = False,
    ):
        super().__init__()
        self.interpolation = interpolation

        check_type('tabulated values', x, Iterable, Real)
        check_type('tabulated probabilities', p, Iterable, Real)

        x = np.array(x, dtype=float)
        p = np.array(p, dtype=float)

        if p.size > x.size:
            raise ValueError(
                'Number of probabilities exceeds number of table values.')
        if self.interpolation != 'histogram' and x.size != p.size:
            raise ValueError(f'Tabulated values ({x.size}) and probabilities '
                             f'({p.size}) should have the same length')

        if not ignore_negative:
            for pk in p:
                check_greater_than('tabulated probability', pk, 0.0, True)

        self._x = x
        self._p = p

    def __len__(self) -> int:
        return self.p.size

    def __repr__(self) -> str:
        return f"<Tabular: {len(self)} points, {self.interpolation}>"

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def interpolation(self) -> str:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation: str):
        if interpolation not in INTERPOLATION_SCHEMES:
            raise ValueError(
                f"Unable to set 'interpolation' to {interpolation!r}; it must "
                f"be one of {sorted(INTERPOLATION_SCHEMES)}"
            )
        self._interpolation = interpolation

    def cdf(self) -> np.ndarray:
        c = np.zeros_like(self.x)
        x = self.x
        p = self.p

        if self.interpolation == 'histogram':
            c[1:] = p[:x.size - 1] * np.diff(x)
        elif self.interpolation == 'linear-linear':
            c[1:] = 0.5 * (p[:-1] + p[1:]) * np.diff(x)
        elif self.interpolation == 'linear-log':
            m = np.diff(p) / np.diff(np.log(x))
            c[1:] = p[:-1] * np.diff(x) + m * (
                x[1:] * (np.diff(np.log(x)) - 1.0) + x[:-1]
            )
        elif self.interpolation == 'log-linear':
            m = np.diff(np.log(p)) / np.diff(x)
            c[1:] = p[:-1] * np.diff(x) * _exprel(m * np.diff(x))
        elif self.interpolation == 'log-log':
            m = np.diff(np.log(x * p)) / np.diff(np.log(x))
            c[1:] = (x * p)[:-1] * np.diff(np.log(x)) * _exprel(
                m * np.diff(np.log(x)))
        else:
            raise NotImplementedError(
                f"Cannot generate CDFs for tabular distributions using "
                f"{self.interpolation} interpolation"
            )

        return np.cumsum(c)

    def integral(self) -> float:
        return float(self.cdf()[-1])

    def normalize(self):
        self._p = self._p / self.cdf().max()


class Mixture(Univariate):
    """Probability distribution characterized by a mixture of random variables.

    Parameters
    ----------
    probability
        Probability of selecting a particular distribution
    distribution
        List of distributions with corresponding probabilities

    Attributes
    ----------
    probability : numpy.ndarray
        Probability of selecting a particular distribution
    distribution : list of Univariate
        List of distributions with corresponding probabilities

    """

    def __init__(
        self,
        probability: Sequence[float],
        distribution: Sequence[Univariate],
    ):
        super().__init__()
        self.probability = probability
        self.distribution = distribution

    def __len__(self) -> int:
        return sum(len(d) for d in self.distribution)

    def __repr__(self) -> str:
        return f"<Mixture: {len(self.distribution)} components>"

    @property
    def probability(self) -> np.ndarray:
        return self._probability

    @probability.setter
    def probability(self, probability):
        check_type('mixture distribution probabilities', probability,
                   Iterable, Real)
        for p in probability:
            check_greater_than('mixture distribution probabilities', p, 0.0,
                               True)
        self._probability = np.array(probability, dtype=float)

    @property
    def distribution(self) -> Sequence[Univariate]:
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        check_type('mixture distribution components', distribution,
                   Iterable, Univariate)
        self._distribution = distribution

    def cdf(self) -> np.ndarray:
        raise NotImplementedError(
            "Mixture distributions do not have a single tabulated CDF; take "
            "the CDF of each component in 'distribution' instead."
        )

    def integral(self) -> float:
        return float(sum(
            p * d.integral() for p, d in zip(self.probability, self.distribution)
        ))

    def normalize(self):
        self._probability = self._probability / self._probability.sum()


def combine_distributions(
    dists: Sequence[Union[Discrete, Tabular, Mixture]],
    probs: Sequence[float],
) -> Union[Discrete, Tabular, Mixture]:
    """Combine distributions with specified probabilities.

    Multiple :class:`Discrete` distributions are merged into a single
    distribution; the remainder are put into a :class:`Mixture`. Any
    :class:`Mixture` in the input is flattened first.

    Parameters
    ----------
    dists
        Distributions to combine
    probs
        Probability (or intensity) of each distribution

    Returns
    -------
    The combined distribution

    """
    new_probs = []
    new_dists = []
    for i, dist in enumerate(dists):
        check_type(f'dists[{i}]', dist, (Discrete, Tabular, Mixture))
        check_type(f'probs[{i}]', probs[i], Real)
        check_greater_than(f'probs[{i}]', probs[i], 0.0)
        if isinstance(dist, Mixture):
            for j, d in enumerate(dist.distribution):
                check_type(f'dists[{i}].distribution[{j}]', d,
                           (Discrete, Tabular))
                new_probs.append(probs[i] * dist.probability[j])
                new_dists.append(d)
        else:
            new_probs.append(probs[i])
            new_dists.append(dist)

    probs = new_probs
    dists = new_dists

    discrete_index = [i for i, d in enumerate(dists) if isinstance(d, Discrete)]
    cont_index = [i for i, d in enumerate(dists) if isinstance(d, Tabular)]

    cont_dists = [dists[i] for i in cont_index]
    cont_probs = [probs[i] for i in cont_index]

    if discrete_index:
        # Create combined discrete distribution
        dist_discrete = [dists[i] for i in discrete_index]
        discrete_probs = [probs[i] for i in discrete_index]
        combined_dist = Discrete.merge(dist_discrete, discrete_probs)
        if cont_index:
            return Mixture(cont_probs + [1.0], cont_dists + [combined_dist])
        return combined_dist

    if len(cont_dists) == 1:
        dist = cont_dists[0]
        return Tabular(dist.x, dist.p * cont_probs[0], dist.interpolation)
    return Mixture(cont_probs, cont_dists)
