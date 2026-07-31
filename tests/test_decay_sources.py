"""Tests for Decay.sources.

The spectra themselves are parsed by Decay.__init__ from an ENDF MF=8/MT=457
tape; `sources` is a pure transformation of the already-parsed `spectra` dict,
so it is exercised here on synthetic spectra rather than on a decay tape.
"""

from math import log

import numpy as np
import pytest
from uncertainties import ufloat

from endf.decay import Decay
from endf.function import Tabulated1D
from endf.univariate import Discrete, Tabular, Mixture


HALF_LIFE = 100.0
DECAY_CONSTANT = log(2.0) / HALF_LIFE

# Real ENDF decay data carries uncertainties. Using nonzero ones keeps the
# `uncertainties` library quiet and checks that sources takes nominal values.
UNC = 0.01


def make_decay(spectra, name='Co60'):
    """Build a Decay carrying the given spectra, bypassing the file parser."""
    d = object.__new__(Decay)
    d.nuclide = {'name': name}
    d.half_life = ufloat(HALF_LIFE, UNC)
    d.spectra = spectra
    return d


def discrete_spectrum(rad_type, energies, intensities, norm=1.0):
    return {
        'type': rad_type,
        'continuous_flag': 'discrete',
        'discrete_normalization': ufloat(norm, UNC),
        'discrete': [
            {'energy': ufloat(e, UNC), 'intensity': ufloat(i, UNC)}
            for e, i in zip(energies, intensities)
        ],
    }


def continuous_spectrum(rad_type, x, y, norm=1.0, law=2):
    return {
        'type': rad_type,
        'continuous_flag': 'continuous',
        'continuous_normalization': ufloat(norm, UNC),
        'continuous': {'probability': Tabulated1D(x, y, [len(x)], [law])},
    }


def test_discrete_gamma_becomes_photon_source():
    spectra = {'gamma': discrete_spectrum(
        'gamma', [1.17e6, 1.33e6], [0.999, 0.998], norm=2.0)}
    src = make_decay(spectra).sources

    assert set(src) == {'photon'}
    dist = src['photon']
    assert isinstance(dist, Discrete)
    np.testing.assert_allclose(dist.x, [1.17e6, 1.33e6])
    # Intensities are scaled by the decay constant and the normalisation.
    np.testing.assert_allclose(
        dist.p, DECAY_CONSTANT * 2.0 * np.array([0.999, 0.998]))


def test_continuous_beta_becomes_electron_source():
    spectra = {'beta-': continuous_spectrum(
        'beta-', [0.0, 1e5, 3e5], [0.0, 2.0, 0.0], norm=0.5)}
    dist = make_decay(spectra).sources['electron']

    assert isinstance(dist, Tabular)
    assert dist.interpolation == 'linear-linear'
    np.testing.assert_allclose(dist.x, [0.0, 1e5, 3e5])
    np.testing.assert_allclose(
        dist.p, DECAY_CONSTANT * 0.5 * np.array([0.0, 2.0, 0.0]))


def test_histogram_interpolation_is_carried_through():
    spectra = {'beta-': continuous_spectrum(
        'beta-', [0.0, 1e5, 3e5], [1.0, 2.0, 0.0], law=1)}
    assert make_decay(spectra).sources['electron'].interpolation == 'histogram'


def test_both_flag_makes_a_mixture():
    spectra = {'gamma': {
        'type': 'gamma',
        'continuous_flag': 'both',
        'discrete_normalization': ufloat(1.0, UNC),
        'discrete': [{'energy': ufloat(5e5, UNC), 'intensity': ufloat(0.4, UNC)}],
        'continuous_normalization': ufloat(1.0, UNC),
        'continuous': {'probability': Tabulated1D(
            [0.0, 1e6], [1e-6, 1e-6], [2], [2])},
    }}
    dist = make_decay(spectra).sources['photon']

    # combine_distributions puts the continuous parts first and appends the
    # merged discrete part with probability 1.
    assert isinstance(dist, Mixture)
    assert isinstance(dist.distribution[-1], Discrete)
    assert isinstance(dist.distribution[0], Tabular)


def test_gamma_and_xray_merge_into_one_photon_source():
    spectra = {
        'gamma': discrete_spectrum('gamma', [1e6], [1.0]),
        'xray': discrete_spectrum('xray', [1e4, 1e6], [2.0, 3.0]),
    }
    dist = make_decay(spectra).sources['photon']

    # Both radiation types are photons, so they merge; the shared 1e6 line sums.
    assert isinstance(dist, Discrete)
    np.testing.assert_allclose(dist.x, [1e4, 1e6])
    np.testing.assert_allclose(dist.p, DECAY_CONSTANT * np.array([2.0, 4.0]))


def test_beta_and_conversion_electrons_merge():
    spectra = {
        'beta-': discrete_spectrum('beta-', [1e5], [1.0]),
        'e-': discrete_spectrum('e-', [2e5], [1.0]),
    }
    dist = make_decay(spectra).sources['electron']
    np.testing.assert_allclose(dist.x, [1e5, 2e5])


@pytest.mark.parametrize('rad_type,particle', [
    ('gamma', 'photon'), ('xray', 'photon'),
    ('beta-', 'electron'), ('e-', 'electron'),
    ('ec/beta+', 'positron'), ('alpha', 'alpha'), ('n', 'neutron'),
    ('sf', 'fragment'), ('p', 'proton'),
    ('anti-neutrino', 'anti-neutrino'), ('neutrino', 'neutrino'),
])
def test_every_radiation_type_maps_to_a_particle(rad_type, particle):
    spectra = {rad_type: discrete_spectrum(rad_type, [1e6], [1.0])}
    assert set(make_decay(spectra).sources) == {particle}


def test_unknown_radiation_type_is_reported_with_the_nuclide():
    spectra = {'quark': discrete_spectrum('quark', [1e6], [1.0])}
    with pytest.raises(ValueError, match="Co60.*'quark'"):
        make_decay(spectra).sources


def test_multiple_interpolation_regions_rejected():
    spectra = {'beta-': {
        'type': 'beta-',
        'continuous_flag': 'continuous',
        'continuous_normalization': ufloat(1.0, UNC),
        'continuous': {'probability': Tabulated1D(
            [0.0, 1e5, 3e5], [1.0, 2.0, 0.0], [2, 3], [1, 2])},
    }}
    with pytest.raises(NotImplementedError, match="Multiple interpolation"):
        make_decay(spectra).sources


def test_unusual_interpolation_warns_but_works():
    spectra = {'beta-': continuous_spectrum(
        'beta-', [1.0, 1e5], [1.0, 2.0], law=5)}
    with pytest.warns(UserWarning, match="log-log"):
        dist = make_decay(spectra).sources['electron']
    assert dist.interpolation == 'log-log'


def test_stable_nuclide_has_no_sources():
    assert make_decay({}).sources == {}
