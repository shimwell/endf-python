import numpy as np
import pytest
from numpy.polynomial import Legendre

import endf
from endf.function import Tabulated1D
from endf.mf4 import AngleDistribution


# --- AngleDistribution.forward_fraction tests ---

class TestForwardFraction:
    def test_isotropic_legendre(self):
        """Isotropic distribution (a_0=1 only) gives (1 - mu_cutoff)/2."""
        energy = np.array([1e6, 10e6])
        mu = [Legendre([1.0]), Legendre([1.0])]
        ad = AngleDistribution(energy, mu)

        for cutoff in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            frac = ad.forward_fraction(cutoff)
            expected = (1.0 - cutoff) / 2.0
            np.testing.assert_allclose(frac, expected, atol=1e-14)

    def test_legendre_known_coefficients(self):
        """Forward-peaked Legendre with a_1=0.5 at a single energy.

        PDF: p(mu) = 1/2 + 3/2 * 0.5 * mu = 0.5 + 0.75*mu
        Integral from 0 to 1: 0.5*1 + 0.75*0.5 = 0.875
        """
        energy = np.array([1e6])
        mu = [Legendre([1.0, 0.5])]
        ad = AngleDistribution(energy, mu)
        frac = ad.forward_fraction(mu_cutoff=0.0)
        np.testing.assert_allclose(frac, [0.875], atol=1e-14)

    def test_legendre_higher_order(self):
        """Verify with a_1=0.3, a_2=0.1 at a single energy.

        PDF: p(mu) = 1/2*P_0 + 3/2*0.3*P_1 + 5/2*0.1*P_2
             = 0.5 + 0.45*mu + 0.25*(3*mu^2 - 1)/2
             = 0.5 + 0.45*mu + 0.375*mu^2 - 0.125
             = 0.375 + 0.45*mu + 0.375*mu^2
        Integral from 0 to 1: 0.375 + 0.45/2 + 0.375/3 = 0.375 + 0.225 + 0.125 = 0.725
        """
        energy = np.array([1e6])
        mu = [Legendre([1.0, 0.3, 0.1])]
        ad = AngleDistribution(energy, mu)
        frac = ad.forward_fraction(mu_cutoff=0.0)
        np.testing.assert_allclose(frac, [0.725], atol=1e-12)

    def test_tabulated(self):
        """Tabulated PDF matching the a_1=0.5 Legendre case."""
        mu_vals = np.linspace(-1, 1, 1001)
        pdf_vals = 0.5 + 0.75 * mu_vals
        energy = np.array([1e6])
        mu = [Tabulated1D(mu_vals, pdf_vals)]
        ad = AngleDistribution(energy, mu)
        frac = ad.forward_fraction(mu_cutoff=0.0)
        np.testing.assert_allclose(frac, [0.875], atol=1e-4)

    def test_bounds(self):
        """mu_cutoff=-1 gives 1.0, mu_cutoff=1 gives 0.0."""
        energy = np.array([1e6])
        mu = [Legendre([1.0, 0.5, 0.2])]
        ad = AngleDistribution(energy, mu)

        frac_all = ad.forward_fraction(mu_cutoff=-1.0)
        np.testing.assert_allclose(frac_all, [1.0], atol=1e-14)

        frac_none = ad.forward_fraction(mu_cutoff=1.0)
        np.testing.assert_allclose(frac_none, [0.0], atol=1e-14)

    def test_empty_isotropic(self):
        """Empty energy/mu (purely isotropic LTT=0) returns empty array."""
        ad = AngleDistribution([], [])
        frac = ad.forward_fraction(mu_cutoff=0.0)
        assert len(frac) == 0


# --- IncidentNeutron.removal_xs tests ---

@pytest.fixture
def am244():
    return endf.IncidentNeutron.from_endf('tests/n-095_Am_244.endf')


class TestRemovalXS:
    def test_returns_tabulated1d(self, am244):
        removal = am244.removal_xs()
        assert isinstance(removal, Tabulated1D)

    def test_energy_range(self, am244):
        """Energy range should match the elastic angular distribution."""
        removal = am244.removal_xs()
        angle_dist = am244[2].products[0].distribution[0].angle
        np.testing.assert_array_equal(removal.x, angle_dist.energy)

    def test_less_than_total(self, am244):
        """Removal XS should be less than or equal to total XS."""
        removal = am244.removal_xs(mu_cutoff=0.0)
        total_xs = am244[1].xs['0K']
        total_vals = total_xs(removal.x)
        assert np.all(removal.y <= total_vals + 1e-10)

    def test_mu_cutoff_minus1(self, am244):
        """With mu_cutoff=-1, all elastic is 'forward', so removal = total - elastic."""
        removal = am244.removal_xs(mu_cutoff=-1.0)
        total_xs = am244[1].xs['0K']
        elastic_xs = am244[2].xs['0K']
        expected = total_xs(removal.x) - elastic_xs(removal.x)
        np.testing.assert_allclose(removal.y, expected, atol=1e-10)

    def test_mu_cutoff_plus1(self, am244):
        """With mu_cutoff=1, no elastic is 'forward', so removal = total."""
        removal = am244.removal_xs(mu_cutoff=1.0)
        total_xs = am244[1].xs['0K']
        expected = total_xs(removal.x)
        np.testing.assert_allclose(removal.y, expected, atol=1e-10)

    def test_callable(self, am244):
        """Returned Tabulated1D should be callable at arbitrary energies."""
        removal = am244.removal_xs()
        val = removal(1e6)
        assert np.isfinite(val)
        assert val > 0
