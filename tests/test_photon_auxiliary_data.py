"""Tests for the auxiliary photon data attached to IncidentPhoton.

Compton profiles and bremsstrahlung cross sections are not in the photoatomic
sublibrary; they come from the tables shipped in endf/datafiles.
"""

from pathlib import Path

import numpy as np
import pytest

import endf
from endf.incident_photon import (
    _COMPTON_PROFILES, _BREMSSTRAHLUNG, _MAX_Z,
    _load_compton_profiles, _load_bremsstrahlung,
)

TESTS = Path(__file__).parent


@pytest.fixture(scope='module')
def hydrogen():
    return endf.IncidentPhoton.from_endf(
        TESTS / 'photoat-001_H_000.endf', TESTS / 'atom-001_H_000.endf')


def test_compton_profiles_attached(hydrogen):
    profile = hydrogen.compton_profiles
    assert set(profile) == {'num_electrons', 'binding_energy', 'J'}
    # Hydrogen has a single occupied shell holding its one electron.
    np.testing.assert_allclose(profile['num_electrons'], [1.0])
    assert len(profile['J']) == 1
    # Binding energies are converted from MeV to eV on load.
    assert 0.0 < profile['binding_energy'][0] < 100.0


def test_compton_profile_is_tabulated_on_the_shared_pz_grid(hydrogen):
    _load_compton_profiles()
    J = hydrogen.compton_profiles['J'][0]
    np.testing.assert_array_equal(J.x, _COMPTON_PROFILES['pz'])
    # A Compton profile is peaked at pz = 0 and falls away monotonically.
    assert J.y[0] == J.y.max()
    assert np.all(np.diff(J.y) <= 0.0)


def test_bremsstrahlung_attached(hydrogen):
    brem = hydrogen.bremsstrahlung
    assert set(brem) == {'I', 'num_electrons', 'ionization_energy',
                         'electron_energy', 'photon_energy', 'dcs'}
    # The DCS is resampled onto 200 electron energies from 1 keV to 1 GeV.
    assert brem['dcs'].shape == (200, len(brem['photon_energy']))
    assert len(brem['electron_energy']) == 200
    np.testing.assert_allclose(brem['electron_energy'][[0, -1]], [1e3, 1e9])
    assert brem['I'] > 0.0


def test_reduced_photon_energy_grid_is_normalised():
    _load_bremsstrahlung()
    k = _BREMSSTRAHLUNG['photon_energy']
    # Photon energies are given as a fraction of the incident electron energy.
    assert k[0] == pytest.approx(0.0)
    assert k[-1] == pytest.approx(1.0)
    assert np.all(np.diff(k) > 0.0)


def test_tables_cover_every_element():
    _load_compton_profiles()
    _load_bremsstrahlung()
    for Z in range(1, _MAX_Z + 1):
        assert Z in _COMPTON_PROFILES, Z
        assert Z in _BREMSSTRAHLUNG, Z
        assert _BREMSSTRAHLUNG[Z]['dcs'].shape[0] == 200


def test_subshell_occupancy_sums_to_atomic_number():
    """Each element's Compton shells hold exactly Z electrons."""
    _load_compton_profiles()
    for Z in range(1, _MAX_Z + 1):
        total = _COMPTON_PROFILES[Z]['num_electrons'].sum()
        assert total == pytest.approx(Z), f"Z={Z} has {total} electrons"


def test_mean_excitation_energy_grows_with_z():
    """The mean excitation energy I rises roughly linearly with Z, at very
    close to the textbook 10*Z eV."""
    _load_bremsstrahlung()
    Z = np.arange(1, 99)
    I = np.array([_BREMSSTRAHLUNG[z]['I'] for z in Z])
    assert np.all(I > 0.0)
    # Measured values, so the rise is not strictly monotonic shell to shell.
    assert np.corrcoef(Z, I)[0, 1] > 0.99
    ratio = I[Z >= 10] / Z[Z >= 10]
    assert 8.0 < ratio.min() and ratio.max() < 14.0


def test_einsteinium_and_fermium_excitation_energies_are_outliers():
    """Z=99 and Z=100 carry I values around 65 eV, two orders of magnitude
    below the ~970 eV the trend implies, and below even hydrogen's 19.2 eV.

    This looks wrong, but it is what the source table says and OpenMC reads the
    same values, so it is recorded here rather than silently corrected. It only
    affects the density effect correction for einsteinium and fermium.
    """
    _load_bremsstrahlung()
    assert _BREMSSTRAHLUNG[98]['I'] == pytest.approx(966.0)
    assert _BREMSSTRAHLUNG[99]['I'] == pytest.approx(65.1)
    assert _BREMSSTRAHLUNG[100]['I'] == pytest.approx(64.2)


def test_loaders_are_idempotent():
    _load_compton_profiles()
    before = _COMPTON_PROFILES[26]['J']
    _load_compton_profiles()
    assert _COMPTON_PROFILES[26]['J'] is before
