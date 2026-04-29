import numpy as np
import pytest

import endf
from endf.function import Tabulated1D


@pytest.fixture
def am244():
    """Am-244 - fissile, exercises the fission ν̄ path."""
    return endf.IncidentNeutron.from_endf('tests/n-095_Am_244.endf')


class TestMultiplicationFactor:
    def test_returns_tabulated1d(self, am244):
        M = am244.multiplication_factor()
        assert isinstance(M, Tabulated1D)

    def test_energy_grid_matches_total_xs(self, am244):
        """Output grid is the nuclide's native MT=1 (total) grid."""
        M = am244.multiplication_factor()
        total_xs = am244[1].xs['0K']
        np.testing.assert_array_equal(M.x, total_xs.x)

    def test_callable(self, am244):
        """Returned Tabulated1D is callable at arbitrary energies."""
        M = am244.multiplication_factor()
        val = M(1e6)
        assert np.isfinite(val)

    def test_fissile_thermal(self, am244):
        """Am-244 is a thermal fissile nuclide; M at thermal should reflect
        ν̄_thermal (~2.5 for Am-244)."""
        M = am244.multiplication_factor()
        assert 2.0 < M(0.025) < 3.5

    def test_fissile_fast(self, am244):
        """At 14 MeV, fission ν̄ rises to ~3 and (n,2n)/(n,3n) add more."""
        M = am244.multiplication_factor()
        assert M(14e6) > 2.5

    def test_non_negative(self, am244):
        """M(E) should never be negative on the grid."""
        M = am244.multiplication_factor()
        assert np.all(M.y >= 0.0)

    def test_temperature_lookup(self, am244):
        """Default temperature key '0K' is what's available."""
        M0 = am244.multiplication_factor(temperature='0K')
        assert isinstance(M0, Tabulated1D)
        with pytest.raises(KeyError):
            am244.multiplication_factor(temperature='999K')

    def test_no_total_raises(self):
        """Without MT=1 the method must raise."""
        # Construct an IncidentNeutron with no MT=1 reaction
        nuc = endf.IncidentNeutron(82, 208, 0)
        with pytest.raises(ValueError, match="Total cross section"):
            nuc.multiplication_factor()
