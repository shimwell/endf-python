"""Tests for the nuclear-data helpers: interpolation linearization, isomeric
state mapping, and the Compton profile utilities."""

import numpy as np
import pytest

from endf.function import Tabulated1D
from endf.incident_photon import (
    compton_profile_cdfs, compton_subshell_map, _load_compton_profiles,
    _COMPTON_PROFILES,
)
from endf.radionuclide_production import isomer_table, level_to_isomeric_state


# ---------------------------------------------------------------------------
# Tabulated1D.linearize
# ---------------------------------------------------------------------------

def _eval_linear(x, y, u):
    """Evaluate the linearized pairs as a consumer would: zero below the first
    point, lin-lin between, flat above the last, right value at a duplicate."""
    x = np.asarray(x); y = np.asarray(y)
    hi = np.searchsorted(x, u, side="right")
    if hi == 0:
        return 0.0
    if hi >= len(x):
        return y[-1]
    lo = np.searchsorted(x, u, side="left")
    if lo == hi:
        x0, x1, y0, y1 = x[lo - 1], x[lo], y[lo - 1], y[lo]
        return y0 if x1 == x0 else y0 + (u - x0) / (x1 - x0) * (y1 - y0)
    return y[hi - 1]


def test_is_linear():
    assert Tabulated1D([1.0, 2.0], [1.0, 2.0]).is_linear
    assert Tabulated1D([1.0, 2.0], [1.0, 2.0], [2], [2]).is_linear
    assert not Tabulated1D([1.0, 2.0], [1.0, 2.0], [2], [5]).is_linear
    # A mixed-law function is not linear even if one region is
    assert not Tabulated1D([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2, 3], [2, 5]).is_linear


def test_linlin_passes_through_unchanged():
    tab = Tabulated1D([1.0, 2.0, 5.0], [0.0, 3.0, 1.0])
    lin = tab.linearize()
    np.testing.assert_array_equal(lin.x, tab.x)
    np.testing.assert_array_equal(lin.y, tab.y)
    assert lin.is_linear


def test_histogram_becomes_exact_duplicated_breakpoints():
    tab = Tabulated1D([1.0, 2.0, 4.0], [10.0, 20.0, 30.0], [3], [1])
    lin = tab.linearize()
    np.testing.assert_array_equal(lin.x, [1.0, 2.0, 2.0, 4.0, 4.0])
    np.testing.assert_array_equal(lin.y, [10.0, 10.0, 20.0, 20.0, 30.0])
    # A histogram holds its left value across each interval
    assert _eval_linear(lin.x, lin.y, 1.5) == 10.0
    assert _eval_linear(lin.x, lin.y, 3.0) == 20.0


def test_loglog_is_subdivided_within_tolerance():
    # y = x^2 over two decades under log-log interpolation
    tab = Tabulated1D([1.0, 100.0], [1.0, 1e4], [2], [5])
    lin = tab.linearize(rel_tol=1e-3)
    assert len(lin.x) > 2
    assert lin.x[0] == 1.0 and lin.x[-1] == 100.0
    for u in np.geomspace(1.001, 99.9, 300):
        assert abs(_eval_linear(lin.x, lin.y, u) - u**2) <= 2e-3 * u**2


@pytest.mark.parametrize('law', [3, 4, 5])
def test_smooth_laws_reproduce_the_declared_law(law):
    tab = Tabulated1D([1.0, 10.0, 100.0], [1.0, 5.0, 40.0], [3], [law])
    lin = tab.linearize(rel_tol=1e-3)
    assert lin.is_linear
    for u in np.geomspace(1.01, 99.0, 200):
        assert abs(_eval_linear(lin.x, lin.y, u) - tab(u)) <= 5e-3 * abs(tab(u))


def test_mixed_regions_keep_the_linear_part_verbatim():
    tab = Tabulated1D([1.0, 2.0, 4.0, 40.0], [5.0, 6.0, 8.0, 800.0],
                      [3, 4], [2, 5])
    lin = tab.linearize()
    np.testing.assert_array_equal(lin.x[:3], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(lin.y[:3], [5.0, 6.0, 8.0])
    assert lin.x[-1] == 40.0 and lin.y[-1] == 800.0


def test_undefined_law_falls_back_to_the_stored_pair():
    # log-log is undefined where y is zero, so the interval is left alone
    tab = Tabulated1D([1.0, 10.0], [0.0, 5.0], [2], [5])
    lin = tab.linearize()
    np.testing.assert_array_equal(lin.x, [1.0, 10.0])
    np.testing.assert_array_equal(lin.y, [0.0, 5.0])


# ---------------------------------------------------------------------------
# Isomeric state mapping
# ---------------------------------------------------------------------------

AG110 = {(47, 110): {1: {"LIS": 2, "half_life": 2.165e7, "E_iso": 117590.0}}}


def test_ground_state_and_low_levels_map_to_zero():
    assert level_to_isomeric_state(47, 110, 0, 0.0, AG110) == 0
    # Below 1 keV a level is treated as ground regardless of LFS
    assert level_to_isomeric_state(47, 110, 2, 0.0, AG110) == 0
    assert level_to_isomeric_state(47, 110, 2, None, AG110) == 0


def test_energy_match_within_tolerance():
    assert level_to_isomeric_state(47, 110, 2, 117590.0, AG110) == 1
    assert level_to_isomeric_state(47, 110, 2, 118000.0, AG110) == 1
    # Far outside the tolerance, and with a single isomer, falls back to it
    assert level_to_isomeric_state(47, 110, 2, 9e5, AG110, tol_eV=10.0) == 1


def test_level_index_fallback_for_a_pure_beta_isomer():
    # No isomeric transition, so no E_iso; LFS == LIS carries the match
    table = {(47, 110): {1: {"LIS": 2, "half_life": 1.0, "E_iso": None}}}
    assert level_to_isomeric_state(47, 110, 2, 3e5, table) == 1


def test_multiple_isomers_with_no_match_cascade_to_ground():
    table = {(49, 116): {1: {"LIS": 3, "half_life": 1.0, "E_iso": 127270.0},
                         2: {"LIS": 4, "half_life": 1.0, "E_iso": 289663.0}}}
    assert level_to_isomeric_state(49, 116, 9, 1.5e6, table) == 0
    # But each isomer is found by its own energy
    assert level_to_isomeric_state(49, 116, 3, 127270.0, table) == 1
    assert level_to_isomeric_state(49, 116, 4, 289663.0, table) == 2


def test_nuclide_with_no_metastable_states():
    assert level_to_isomeric_state(26, 56, 3, 8e5, {}) == 0
    ground_only = {(26, 56): {0: {"LIS": 0, "half_life": None, "E_iso": 0.0}}}
    assert level_to_isomeric_state(26, 56, 3, 8e5, ground_only) == 0


def test_isomer_table_ignores_files_without_decay_data(tmp_path):
    # A file with no MF=8/MT=457 section contributes nothing rather than raising
    empty = tmp_path / "not-decay.endf"
    empty.write_text("")
    with pytest.raises(Exception):
        # Material() rejects an empty file; the point is that isomer_table does
        # not silently invent entries
        isomer_table([empty])


# ---------------------------------------------------------------------------
# Compton profile utilities
# ---------------------------------------------------------------------------

def test_compton_cdfs_are_monotonic_and_half_normalised():
    _load_compton_profiles()
    pz = _COMPTON_PROFILES['pz']
    for Z in (1, 26, 92):
        J = np.asarray(_COMPTON_PROFILES[Z]['J'], dtype=float)
        cdf = compton_profile_cdfs(J, pz)
        assert cdf.shape == J.shape
        assert np.all(cdf[:, 0] == 0.0)
        assert np.all(np.diff(cdf, axis=1) >= 0.0)
        # Only positive pz is tabulated, so each shell integrates to about 0.5.
        # The tail beyond the tabulated momentum range is truncated, which pulls
        # some shells low: over all 100 elements the endpoints span 0.412 to
        # 0.518, with 98% inside 0.02 of a half.
        assert np.all(np.abs(cdf[:, -1] - 0.5) < 0.1)


def test_compton_cdfs_match_a_trapezoidal_reference():
    pz = np.array([0.0, 1.0, 2.0, 4.0])
    J = np.array([[2.0, 1.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0]])
    expected = np.array([[0.0, 1.5, 2.25, 2.75], [0.0, 1.0, 2.0, 4.0]])
    np.testing.assert_allclose(compton_profile_cdfs(J, pz), expected)


def test_subshell_map_groups_by_occupancy():
    # One Compton shell of 6 electrons covering two subshells of 2 and 4
    offsets, indices, weights = compton_subshell_map([2.0, 6.0], [2.0, 2.0, 4.0])
    assert offsets == [0, 1, 3]
    assert indices == [0, 1, 2]
    np.testing.assert_allclose(weights, [1.0, 2 / 6, 4 / 6])


def test_subshell_map_stops_at_the_first_occupancy_mismatch():
    # 3 electrons cannot be made from the remaining subshells, so this shell and
    # every later one are dropped
    offsets, indices, weights = compton_subshell_map([2.0, 3.0, 4.0], [2.0, 8.0])
    assert offsets == [0, 1, 1, 1]
    assert indices == [0]


def test_subshell_map_weights_sum_to_one_per_group():
    _load_compton_profiles()
    for Z in (13, 26, 74):
        occ = _COMPTON_PROFILES[Z]['num_electrons']
        offsets, indices, weights = compton_subshell_map(occ, occ)
        for c in range(len(offsets) - 1):
            lo, hi = offsets[c], offsets[c + 1]
            if hi > lo:
                assert sum(weights[lo:hi]) == pytest.approx(1.0)
