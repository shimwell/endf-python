import numpy as np
import pytest

from endf.function import Tabulated1D
from endf.incident_neutron import IncidentNeutron, _cascade_gammas
from endf.reaction import Reaction


class _MockMaterial:
    """Minimal Material-like object for testing gamma production."""

    def __init__(self):
        self.section_data = {}

    @property
    def sections(self):
        return list(self.section_data.keys())

    def __contains__(self, mf_mt):
        return mf_mt in self.section_data

    def __getitem__(self, mf_mt):
        return self.section_data[mf_mt]


def _make_neutron(reactions, material):
    """Create an IncidentNeutron with given reactions and mock material."""
    data = IncidentNeutron(26, 56)
    for mt, xs_arr in reactions.items():
        energies, values = xs_arr
        xs = Tabulated1D(np.array(energies), np.array(values))
        data.reactions[mt] = Reaction(mt, {'0K': xs})
    data._material = material
    return data


# =====================================================================
# gamma_production_xs (total) tests
# =====================================================================

class TestGammaProductionMF12:
    def test_single_photon(self):
        """MF=12, NK=1: production = multiplicity x reaction XS."""
        energies = [1e4, 1e5, 1e6, 1e7]
        rxn_xs = [0.0, 1.0, 2.0, 3.0]
        mult_y = [2.0, 2.0, 2.0, 2.0]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array(mult_y)),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({102: (energies, rxn_xs)}, mat)
        result = data.gamma_production_xs()

        expected = np.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(result.y, expected)

    def test_multiple_photons_uses_total_yield(self):
        """MF=12, NK>1: total yield Y is provided and used."""
        energies = [1e5, 1e6, 1e7]
        rxn_xs = [1.0, 2.0, 3.0]
        total_yield = [3.0, 3.0, 3.0]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 2,
            'Y': Tabulated1D(np.array(energies), np.array(total_yield)),
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([1.0, 1.0, 1.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0},
                {'y': Tabulated1D(np.array(energies), np.array([2.0, 2.0, 2.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0},
            ],
        }

        data = _make_neutron({102: (energies, rxn_xs)}, mat)
        result = data.gamma_production_xs()

        expected = np.array([3.0, 6.0, 9.0])
        np.testing.assert_allclose(result.y, expected)

    def test_energy_dependent_multiplicity(self):
        """Multiplicity that varies with energy."""
        energies = [1e5, 1e6, 1e7]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([1.0, 2.0, 4.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({102: (energies, [10.0, 5.0, 1.0])}, mat)
        result = data.gamma_production_xs()

        expected = np.array([10.0, 10.0, 4.0])
        np.testing.assert_allclose(result.y, expected)

    def test_lo2_skipped_with_warning(self):
        """MF=12 with LO=2 is skipped by gamma_production_xs (total)."""
        energies = [1e5, 1e6]

        mat = _MockMaterial()
        mat.section_data[(12, 51)] = {
            'LO': 2, 'NK': 0, 'transitions': [],
        }
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([0.5, 1.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({51: (energies, [1.0, 2.0]),
                              102: (energies, [1.0, 2.0])}, mat)
        with pytest.warns(UserWarning, match="transition probability"):
            result = data.gamma_production_xs()

        np.testing.assert_allclose(result.y, [0.5, 1.0])


class TestGammaProductionMF13:
    def test_single_photon(self):
        energies = [1e4, 1e5, 1e6]
        sigma = [0.5, 1.5, 3.0]

        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array(sigma)),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({102: (energies, [1.0, 2.0, 4.0])}, mat)
        result = data.gamma_production_xs()
        np.testing.assert_allclose(result.y, sigma)

    def test_multiple_photons_uses_sigma_total(self):
        energies = [1e5, 1e6]
        sigma_total = [2.0, 4.0]

        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 2,
            'sigma_total': Tabulated1D(np.array(energies), np.array(sigma_total)),
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([0.5, 1.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0},
                {'sigma': Tabulated1D(np.array(energies), np.array([1.5, 3.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0},
            ],
        }

        data = _make_neutron({102: (energies, [10.0, 20.0])}, mat)
        result = data.gamma_production_xs()
        np.testing.assert_allclose(result.y, sigma_total)


class TestGammaProductionCombined:
    def test_sum_mf12_and_mf13(self):
        energies = [1e5, 1e6, 1e7]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([2.0, 2.0, 2.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        mat.section_data[(13, 4)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([0.5, 1.0, 1.5])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron(
            {102: (energies, [1.0, 2.0, 3.0]),
             4: (energies, [5.0, 5.0, 5.0])}, mat)
        result = data.gamma_production_xs()

        expected = np.array([2.5, 5.0, 7.5])
        np.testing.assert_allclose(result.y, expected)

    def test_different_energy_grids(self):
        e1 = [1e5, 1e6, 1e7]
        e2 = [5e5, 5e6, 2e7]

        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(e1), np.array([1.0, 2.0, 3.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        mat.section_data[(13, 4)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(e2), np.array([10.0, 20.0, 30.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron(
            {102: (e1, [1.0, 1.0, 1.0]),
             4: (e2, [1.0, 1.0, 1.0])}, mat)
        result = data.gamma_production_xs()

        assert len(result.x) == 6
        np.testing.assert_allclose(result(np.array([1e5])), [1.0], atol=1e-10)


class TestGammaProductionErrors:
    def test_no_material_raises(self):
        data = IncidentNeutron(26, 56)
        with pytest.raises(ValueError, match="ENDF material"):
            data.gamma_production_xs()

    def test_no_gamma_data_raises(self):
        mat = _MockMaterial()
        mat.section_data[(3, 1)] = {}
        mat.section_data[(3, 2)] = {}
        data = _make_neutron(
            {1: ([1e5, 1e7], [10.0, 10.0]),
             2: ([1e5, 1e7], [5.0, 5.0])}, mat)
        with pytest.raises(ValueError, match="No gamma production data"):
            data.gamma_production_xs()

    def test_missing_mf3_for_mf12_skipped(self):
        energies = [1e5, 1e6]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([2.0, 2.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        mat.section_data[(13, 4)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([1.0, 1.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        data = _make_neutron({4: (energies, [5.0, 5.0])}, mat)
        result = data.gamma_production_xs()
        np.testing.assert_allclose(result.y, [1.0, 1.0])


class TestGammaProductionOutput:
    def test_returns_tabulated1d(self):
        energies = [1e5, 1e6]
        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([1.0, 2.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        data = _make_neutron({102: (energies, [10.0, 20.0])}, mat)
        result = data.gamma_production_xs()
        assert isinstance(result, Tabulated1D)

    def test_callable_at_arbitrary_energy(self):
        energies = [1e5, 1e6, 1e7]
        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([1.0, 2.0, 3.0])),
                 'EG': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        data = _make_neutron({102: (energies, [10.0, 20.0, 30.0])}, mat)
        result = data.gamma_production_xs()
        val = result(5e5)
        assert np.isfinite(val)
        assert val > 0

    def test_non_negative(self):
        energies = [1e4, 1e5, 1e6, 1e7]
        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([0.0, 1.0, 2.0, 3.0])),
                 'Eg': 0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        data = _make_neutron({102: (energies, [0.0, 1.0, 2.0, 3.0])}, mat)
        result = data.gamma_production_xs()
        assert np.all(result.y >= 0)


# =====================================================================
# _cascade_gammas unit tests
# =====================================================================

class TestCascadeGammas:
    def test_single_level_to_ground(self):
        """Level 1 -> ground: one gamma at the level energy."""
        transitions = {
            1e6: [(0.0, 1.0, 1.0)],  # 1 MeV level -> ground, TP=1, GP=1
        }
        result = _cascade_gammas(1e6, transitions, {})
        assert len(result) == 1
        assert result[0] == (1e6, 1.0)

    def test_two_step_cascade(self):
        """Level 2 -> Level 1 -> ground: two gammas.

        Level 2 (2 MeV) -> Level 1 (0.8 MeV): gamma at 1.2 MeV
        Level 1 (0.8 MeV) -> ground: gamma at 0.8 MeV
        """
        transitions = {
            2e6: [(0.8e6, 1.0, 1.0)],  # 2 MeV -> 0.8 MeV
            0.8e6: [(0.0, 1.0, 1.0)],  # 0.8 MeV -> ground
        }
        result = _cascade_gammas(2e6, transitions, {})
        gammas = dict(result)
        np.testing.assert_allclose(gammas[1.2e6], 1.0)
        np.testing.assert_allclose(gammas[0.8e6], 1.0)

    def test_branching_cascade(self):
        """Level 2 branches to level 1 (80%) and ground (20%).

        Level 2 (3 MeV):
          -> Level 1 (1 MeV): TP=0.8, gamma=2 MeV
          -> Ground (0 MeV):  TP=0.2, gamma=3 MeV
        Level 1 (1 MeV):
          -> Ground: TP=1.0, gamma=1 MeV
        """
        transitions = {
            3e6: [(1e6, 0.8, 1.0), (0.0, 0.2, 1.0)],
            1e6: [(0.0, 1.0, 1.0)],
        }
        result = _cascade_gammas(3e6, transitions, {})

        # Consolidate
        yields = {}
        for gamma_e, y in result:
            yields[gamma_e] = yields.get(gamma_e, 0.0) + y

        np.testing.assert_allclose(yields[2e6], 0.8)   # 3->1
        np.testing.assert_allclose(yields[3e6], 0.2)   # 3->ground
        np.testing.assert_allclose(yields[1e6], 0.8)   # 1->ground (via cascade)
        # Total photon yield = 0.8 + 0.2 + 0.8 = 1.8
        total = sum(yields.values())
        np.testing.assert_allclose(total, 1.8)

    def test_internal_conversion_reduces_gamma_yield(self):
        """GP < 1 means some transitions produce conversion electrons, not gammas."""
        transitions = {
            1e6: [(0.0, 1.0, 0.7)],  # TP=1, GP=0.7 -> only 70% emit a gamma
        }
        result = _cascade_gammas(1e6, transitions, {})
        assert len(result) == 1
        np.testing.assert_allclose(result[0][1], 0.7)

    def test_no_transitions_returns_empty(self):
        """Level with no transition data gives empty list."""
        result = _cascade_gammas(1e6, {}, {})
        assert result == []

    def test_ground_state_returns_empty(self):
        result = _cascade_gammas(0.0, {0.0: [(0.0, 1.0, 1.0)]}, {})
        assert result == []

    def test_memo_is_reused(self):
        """Memo from a previous call is reused for shared sub-cascades."""
        transitions = {
            3e6: [(1e6, 1.0, 1.0)],
            2e6: [(1e6, 1.0, 1.0)],
            1e6: [(0.0, 1.0, 1.0)],
        }
        memo = {}
        # Cascade from 3 MeV computes and caches 1 MeV cascade
        _cascade_gammas(3e6, transitions, memo)
        assert 1e6 in memo

        # Cascade from 2 MeV reuses the cached 1 MeV cascade
        result = _cascade_gammas(2e6, transitions, memo)
        gammas = dict(result)
        np.testing.assert_allclose(gammas[1e6], 1.0)  # 2->1
        np.testing.assert_allclose(gammas[1e6], 1.0)   # 1->ground (from cache)


# =====================================================================
# gamma_line_production_xs tests
# =====================================================================

class TestGammaLineProductionLO2:
    def test_single_level_single_line(self):
        """One inelastic level -> one gamma line."""
        energies = [1e6, 5e6, 1e7]
        rxn_xs = [0.0, 1.0, 2.0]

        mat = _MockMaterial()
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 846778.0,
            'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }

        data = _make_neutron({51: (energies, rxn_xs)}, mat)
        result = data.gamma_line_production_xs()

        assert len(result) == 1
        assert result[0]['gamma_energy_eV'] == 846778.0
        np.testing.assert_allclose(result[0]['production_xs_barns'], rxn_xs)

    def test_cascade_produces_two_lines(self):
        """Level 2 -> Level 1 -> ground produces two gamma lines."""
        energies = [2e6, 5e6, 1e7]
        xs_51 = [0.5, 0.3, 0.1]  # cross section for MT=51
        xs_52 = [0.0, 0.8, 1.5]  # cross section for MT=52

        mat = _MockMaterial()
        # Level 1 at 0.8 MeV -> ground
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 0.8e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }
        # Level 2 at 2.0 MeV -> Level 1
        mat.section_data[(12, 52)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 2.0e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.8e6, 'TP': 1.0}],
        }

        data = _make_neutron({51: (energies, xs_51),
                              52: (energies, xs_52)}, mat)
        result = data.gamma_line_production_xs()

        # Should have two lines: 0.8 MeV and 1.2 MeV
        energies_out = [r['gamma_energy_eV'] for r in result]
        assert 0.8e6 in energies_out
        assert 1.2e6 in energies_out

        # 0.8 MeV gamma: from MT=51 (yield=1) + cascade of MT=52 (yield=1)
        line_08 = [r for r in result if r['gamma_energy_eV'] == 0.8e6][0]
        # Total production = 1.0 * xs_51 + 1.0 * xs_52
        expected_08 = np.array(xs_51) + np.array(xs_52)
        np.testing.assert_allclose(line_08['production_xs_barns'], expected_08)

        # 1.2 MeV gamma: only from MT=52 (yield=1)
        line_12 = [r for r in result if r['gamma_energy_eV'] == 1.2e6][0]
        np.testing.assert_allclose(line_12['production_xs_barns'], xs_52)

    def test_branching_with_gp(self):
        """Branching transitions with GP < 1."""
        energies = [1e6, 5e6]
        rxn_xs = [1.0, 2.0]

        mat = _MockMaterial()
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 2, 'NK': 1,
            'ES_NS': 1e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0, 'GP': 0.8}],
        }

        data = _make_neutron({51: (energies, rxn_xs)}, mat)
        result = data.gamma_line_production_xs()

        assert len(result) == 1
        # yield = TP * GP = 1.0 * 0.8 = 0.8
        np.testing.assert_allclose(
            result[0]['production_xs_barns'], np.array(rxn_xs) * 0.8)


class TestGammaLineProductionLO1:
    def test_discrete_line(self):
        """MF=12 LO=1 with Eg > 0 produces a discrete line."""
        energies = [1e5, 1e6, 1e7]
        rxn_xs = [1.0, 2.0, 3.0]
        mult = [0.5, 0.5, 0.5]

        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array(mult)),
                 'Eg': 6e6, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({102: (energies, rxn_xs)}, mat)
        result = data.gamma_line_production_xs()

        assert len(result) == 1
        assert result[0]['gamma_energy_eV'] == 6e6
        expected = np.array(mult) * np.array(rxn_xs)
        np.testing.assert_allclose(result[0]['production_xs_barns'], expected)

    def test_eg_zero_is_skipped(self):
        """MF=12 LO=1 with Eg=0 (continuum) is not included in line data."""
        energies = [1e5, 1e6]
        mat = _MockMaterial()
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 2,
            'Y': Tabulated1D(np.array(energies), np.array([3.0, 3.0])),
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array([1.0, 1.0])),
                 'Eg': 0.0, 'ES': 0, 'LP': 0, 'LF': 0},  # continuum
                {'y': Tabulated1D(np.array(energies), np.array([2.0, 2.0])),
                 'Eg': 5e6, 'ES': 0, 'LP': 0, 'LF': 0},  # discrete
            ],
        }

        data = _make_neutron({102: (energies, [1.0, 1.0])}, mat)
        result = data.gamma_line_production_xs()

        # Only the Eg=5 MeV line should appear
        assert len(result) == 1
        assert result[0]['gamma_energy_eV'] == 5e6


class TestGammaLineProductionMF13:
    def test_discrete_line(self):
        """MF=13 with EG > 0 is a discrete line."""
        energies = [1e5, 1e6]
        sigma = [0.5, 1.0]

        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array(sigma)),
                 'EG': 7.6e6, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }

        data = _make_neutron({102: (energies, [10.0, 20.0])}, mat)
        result = data.gamma_line_production_xs()

        assert len(result) == 1
        assert result[0]['gamma_energy_eV'] == 7.6e6
        np.testing.assert_allclose(result[0]['production_xs_barns'], sigma)

    def test_eg_zero_is_skipped(self):
        """MF=13 with EG=0 (continuum) is not included in line data."""
        energies = [1e5, 1e6]
        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array([0.5, 1.0])),
                 'EG': 0.0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        # Need at least one LO=2 reaction so the method doesn't raise
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 1e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }
        data = _make_neutron({51: (energies, [1.0, 1.0]),
                              102: (energies, [1.0, 1.0])}, mat)
        result = data.gamma_line_production_xs()

        # Only the LO=2 line at 1 MeV should appear, not the MF=13 EG=0
        gamma_energies = [r['gamma_energy_eV'] for r in result]
        assert 0.0 not in gamma_energies
        assert 1e6 in gamma_energies


class TestGammaLineConsolidation:
    def test_same_gamma_from_two_reactions(self):
        """Same gamma energy from two reactions is summed."""
        energies = [1e6, 5e6, 1e7]
        xs_51 = [1.0, 2.0, 3.0]
        xs_52 = [0.0, 0.5, 1.0]

        mat = _MockMaterial()
        # Level 1 (0.8 MeV) -> ground: gamma at 0.8 MeV
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 0.8e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }
        # Level 2 (1.6 MeV) -> Level 1: gamma at 0.8 MeV (same!)
        mat.section_data[(12, 52)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 1.6e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.8e6, 'TP': 1.0}],
        }

        data = _make_neutron({51: (energies, xs_51),
                              52: (energies, xs_52)}, mat)
        result = data.gamma_line_production_xs()

        # 0.8 MeV line: from MT=51 (direct, yield=1) + MT=52 (2->1, yield=1)
        #               + MT=52 cascade (1->ground, yield=1)
        gamma_energies = [r['gamma_energy_eV'] for r in result]
        assert 0.8e6 in gamma_energies

        line_08 = [r for r in result if r['gamma_energy_eV'] == 0.8e6][0]
        # MT=51 contributes 1.0 * xs_51, MT=52 cascade contributes 1.0 * xs_52
        # (both from the level 1->ground transition)
        # PLUS MT=52 direct (level 2->level 1) also emits at 0.8 MeV
        # So total = xs_51 + 2 * xs_52 (one from 2->1, one from 1->0 cascade)
        expected = np.array(xs_51) + 2.0 * np.array(xs_52)
        np.testing.assert_allclose(line_08['production_xs_barns'], expected)


class TestGammaLineErrors:
    def test_no_material_raises(self):
        data = IncidentNeutron(26, 56)
        with pytest.raises(ValueError, match="ENDF material"):
            data.gamma_line_production_xs()

    def test_no_gamma_data_returns_empty(self):
        """Nuclide with no MF=12/13 data returns empty list."""
        mat = _MockMaterial()
        mat.section_data[(3, 1)] = {}
        data = _make_neutron({1: ([1e5, 1e7], [10.0, 10.0])}, mat)
        result = data.gamma_line_production_xs()
        assert result == []


class TestGammaLineOutput:
    def test_sorted_by_gamma_energy(self):
        """Lines are sorted by ascending gamma energy."""
        energies = [1e6, 5e6]

        mat = _MockMaterial()
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 2e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }
        mat.section_data[(12, 52)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 1e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }

        data = _make_neutron({51: (energies, [1.0, 1.0]),
                              52: (energies, [1.0, 1.0])}, mat)
        result = data.gamma_line_production_xs()

        gamma_es = [r['gamma_energy_eV'] for r in result]
        assert gamma_es == sorted(gamma_es)

    def test_non_negative_production(self):
        energies = [1e6, 5e6]
        mat = _MockMaterial()
        mat.section_data[(12, 51)] = {
            'LO': 2, 'LG': 1, 'NK': 1,
            'ES_NS': 1e6, 'LP': 0, 'NT': 1,
            'transitions': [{'ES': 0.0, 'TP': 1.0}],
        }
        data = _make_neutron({51: (energies, [1.0, 2.0])}, mat)
        result = data.gamma_line_production_xs()
        for line in result:
            assert np.all(line['production_xs_barns'] >= 0)


# =====================================================================
# Integration test with real Fe-56 ENDF data (if available)
# =====================================================================

FE56_PATH = 'tests/n-026_Fe_056.endf'


@pytest.fixture
def fe56():
    """Load Fe-56 from the ENDF file if available."""
    import os
    # Try the test directory first, then the nuclear data directory
    for path in [FE56_PATH,
                 os.path.expanduser('~/nuclear_data/endfb-viii.0-endf/neutron/n-026_Fe_056.endf')]:
        if os.path.exists(path):
            import endf
            return endf.IncidentNeutron.from_endf(path)
    pytest.skip("Fe-56 ENDF file not available")


class TestFe56Integration:
    def test_has_gamma_lines(self, fe56):
        """Fe-56 should produce gamma lines from its 39 inelastic levels."""
        lines = fe56.gamma_line_production_xs()
        assert len(lines) > 0

    def test_847_kev_line_present(self, fe56):
        """The 847 keV line (first excited level) should be present."""
        lines = fe56.gamma_line_production_xs()
        gamma_es = [r['gamma_energy_eV'] for r in lines]
        # Find the 847 keV line (846778 eV)
        close_to_847 = [e for e in gamma_es if abs(e - 846778.0) < 100]
        assert len(close_to_847) > 0

    def test_847_kev_is_strongest(self, fe56):
        """The 847 keV line should have the largest peak production XS."""
        lines = fe56.gamma_line_production_xs()
        peak_xs = [(r['gamma_energy_eV'], r['production_xs_barns'].max())
                   for r in lines]
        strongest = max(peak_xs, key=lambda x: x[1])
        # The 847 keV line at ~846778 eV should be the strongest
        assert abs(strongest[0] - 846778.0) < 100

    def test_all_production_xs_finite(self, fe56):
        lines = fe56.gamma_line_production_xs()
        for line in lines:
            assert np.all(np.isfinite(line['production_xs_barns']))
            assert np.all(line['production_xs_barns'] >= 0)

    def test_no_continuum(self, fe56):
        """Fe-56 has no MF=15, so continuum data should be empty."""
        continuum = fe56.gamma_continuum_data()
        assert continuum == []


# =====================================================================
# gamma_continuum_data tests
# =====================================================================

class TestGammaContinuumMock:
    def test_mf12_continuum_with_mf15(self):
        """MF=12 LO=1 Eg=0 (continuum multiplicity) + MF=15 spectrum."""
        energies = [1e5, 1e6, 1e7]
        rxn_xs = [1.0, 2.0, 3.0]
        mult_y = [5.0, 5.0, 5.0]

        gamma_e_grid = np.array([0.0, 1e6, 2e6, 3e6])
        pdf = np.array([0.0, 0.5e-6, 0.5e-6, 0.0])  # 1/eV

        mat = _MockMaterial()
        mat.section_data[(3, 102)] = {}
        mat.section_data[(12, 102)] = {
            'LO': 1, 'NK': 1,
            'multiplicities': [
                {'y': Tabulated1D(np.array(energies), np.array(mult_y)),
                 'Eg': 0.0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        mat.section_data[(15, 102)] = {
            'ZA': 26056, 'AWR': 55.0,
            'NC': 1,
            'subsections': [{
                'LF': 1,
                'p': Tabulated1D(np.array(energies), np.array([1.0, 1.0, 1.0])),
                'NE': 2,
                'E_int': None,
                'E': np.array([1e5, 1e7]),
                'g': [
                    Tabulated1D(gamma_e_grid, pdf),
                    Tabulated1D(gamma_e_grid, pdf),
                ],
            }],
        }

        data = _make_neutron({102: (energies, rxn_xs)}, mat)
        result = data.gamma_continuum_data()

        assert len(result) == 1
        assert result[0]['mt'] == 102

        # Production XS = multiplicity * reaction XS = 5 * [1, 2, 3]
        np.testing.assert_allclose(
            result[0]['production_xs_barns'], [5.0, 10.0, 15.0])

        # Two spectra (at E_n = 1e5 and 1e7)
        assert len(result[0]['spectra']) == 2
        assert result[0]['spectra'][0]['neutron_energy_eV'] == 1e5
        np.testing.assert_array_equal(
            result[0]['spectra'][0]['gamma_energy_eV'], gamma_e_grid)
        np.testing.assert_array_equal(
            result[0]['spectra'][0]['pdf_per_eV'], pdf)

    def test_mf13_continuum_with_mf15(self):
        """MF=13 EG=0 (continuum production XS) + MF=15 spectrum."""
        energies = [1e5, 1e6]
        sigma = [0.5, 1.0]

        gamma_e_grid = np.array([0.0, 5e6, 1e7])
        pdf = np.array([0.0, 2e-7, 0.0])

        mat = _MockMaterial()
        mat.section_data[(13, 102)] = {
            'NK': 1,
            'photons': [
                {'sigma': Tabulated1D(np.array(energies), np.array(sigma)),
                 'EG': 0.0, 'ES': 0, 'LP': 0, 'LF': 0}
            ],
        }
        mat.section_data[(15, 102)] = {
            'ZA': 26056, 'AWR': 55.0,
            'NC': 1,
            'subsections': [{
                'LF': 1,
                'p': Tabulated1D(np.array(energies), np.array([1.0, 1.0])),
                'NE': 1,
                'E_int': None,
                'E': np.array([0.0]),
                'g': [Tabulated1D(gamma_e_grid, pdf)],
            }],
        }

        data = _make_neutron({102: (energies, [10.0, 20.0])}, mat)
        result = data.gamma_continuum_data()

        assert len(result) == 1
        np.testing.assert_allclose(
            result[0]['production_xs_barns'], sigma)

    def test_no_mf15_returns_empty(self):
        """No MF=15 sections gives empty list."""
        mat = _MockMaterial()
        mat.section_data[(3, 1)] = {}
        data = _make_neutron({1: ([1e5, 1e7], [10.0, 10.0])}, mat)
        result = data.gamma_continuum_data()
        assert result == []

    def test_no_material_raises(self):
        data = IncidentNeutron(26, 56)
        with pytest.raises(ValueError, match="ENDF material"):
            data.gamma_continuum_data()


@pytest.fixture
def cr52():
    """Load Cr-52 (has MF=15 for MT=102 capture) if available."""
    import os
    for path in [os.path.expanduser(
            '~/nuclear_data/endfb-viii.0-endf/neutron/n-024_Cr_052.endf')]:
        if os.path.exists(path):
            import endf
            return endf.IncidentNeutron.from_endf(path)
    pytest.skip("Cr-52 ENDF file not available")


class TestCr52ContinuumIntegration:
    def test_has_continuum(self, cr52):
        """Cr-52 should have continuum data for capture (MT=102)."""
        continuum = cr52.gamma_continuum_data()
        assert len(continuum) > 0
        mts = [c['mt'] for c in continuum]
        assert 102 in mts

    def test_spectra_are_valid_pdfs(self, cr52):
        """Each spectrum should have non-negative values."""
        continuum = cr52.gamma_continuum_data()
        for entry in continuum:
            for spec in entry['spectra']:
                assert np.all(np.isfinite(spec['pdf_per_eV']))
                assert np.all(spec['pdf_per_eV'] >= 0)
                assert len(spec['gamma_energy_eV']) == len(spec['pdf_per_eV'])

    def test_production_xs_non_negative(self, cr52):
        continuum = cr52.gamma_continuum_data()
        for entry in continuum:
            assert np.all(entry['production_xs_barns'] >= 0)
