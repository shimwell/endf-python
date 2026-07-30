import numpy as np
import pytest

from endf.univariate import (
    Discrete, Tabular, Mixture, combine_distributions, _exprel,
)


def test_discrete_basics():
    d = Discrete([1.0, 2.0, 3.0], [0.2, 0.3, 0.5])
    assert len(d) == 3
    np.testing.assert_array_equal(d.x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(d.p, [0.2, 0.3, 0.5])
    # The CDF is prepended with zero and has one more entry than x.
    np.testing.assert_allclose(d.cdf(), [0.0, 0.2, 0.5, 1.0])
    assert d.integral() == pytest.approx(1.0)


def test_discrete_scalar_promotion():
    d = Discrete(1.5, 2.0)
    np.testing.assert_array_equal(d.x, [1.5])
    np.testing.assert_array_equal(d.p, [2.0])


def test_discrete_normalize():
    d = Discrete([1.0, 2.0], [1.0, 3.0])
    d.normalize()
    np.testing.assert_allclose(d.p, [0.25, 0.75])
    assert d.integral() == pytest.approx(1.0)


def test_discrete_rejects_negative_probability():
    with pytest.raises(ValueError):
        Discrete([1.0], [-0.1])


def test_discrete_merge_combines_duplicate_values():
    a = Discrete([1.0, 2.0], [1.0, 1.0])
    b = Discrete([2.0, 3.0], [1.0, 1.0])
    m = Discrete.merge([a, b], [1.0, 2.0])
    # x=2.0 appears in both: 1*1 + 1*2 = 3
    np.testing.assert_array_equal(m.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(m.p, [1.0, 3.0, 2.0])


def test_discrete_merge_length_mismatch():
    with pytest.raises(ValueError):
        Discrete.merge([Discrete([1.0], [1.0])], [1.0, 2.0])


def test_tabular_linlin_cdf():
    # Triangle on [0, 2] with peak 1 at x=1: total area is 1.
    t = Tabular([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 'linear-linear')
    assert len(t) == 3
    np.testing.assert_allclose(t.cdf(), [0.0, 0.5, 1.0])
    assert t.integral() == pytest.approx(1.0)


def test_tabular_histogram_cdf():
    # Two bins of width 1 with densities 0.25 and 0.75.
    t = Tabular([0.0, 1.0, 2.0], [0.25, 0.75, 0.0], 'histogram')
    np.testing.assert_allclose(t.cdf(), [0.0, 0.25, 1.0])


def test_tabular_histogram_allows_short_p():
    # Histogram interpolation may omit the trailing probability.
    t = Tabular([0.0, 1.0, 2.0], [0.25, 0.75], 'histogram')
    np.testing.assert_allclose(t.cdf(), [0.0, 0.25, 1.0])


def test_tabular_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Tabular([0.0, 1.0, 2.0], [1.0, 1.0], 'linear-linear')
    with pytest.raises(ValueError):
        Tabular([0.0, 1.0], [1.0, 1.0, 1.0], 'histogram')


def test_tabular_bad_interpolation_rejected():
    with pytest.raises(ValueError):
        Tabular([0.0, 1.0], [1.0, 1.0], 'quadratic')


def test_tabular_ignore_negative():
    with pytest.raises(ValueError):
        Tabular([0.0, 1.0], [1.0, -1e-12])
    t = Tabular([0.0, 1.0], [1.0, -1e-12], ignore_negative=True)
    assert t.p[-1] == -1e-12


def test_tabular_normalize():
    t = Tabular([0.0, 1.0, 2.0], [0.0, 2.0, 0.0], 'linear-linear')
    t.normalize()
    assert t.integral() == pytest.approx(1.0)


@pytest.mark.parametrize('scheme', ['linear-log', 'log-linear', 'log-log'])
def test_tabular_log_schemes_are_monotonic(scheme):
    t = Tabular([1.0, 2.0, 4.0], [1.0, 2.0, 4.0], scheme)
    c = t.cdf()
    assert c[0] == 0.0
    assert np.all(np.diff(c) > 0.0)


def test_exprel_limit_at_zero():
    # (exp(x) - 1)/x tends to 1 as x -> 0 and must not divide by zero.
    np.testing.assert_allclose(_exprel(0.0), 1.0)
    np.testing.assert_allclose(_exprel(np.array([0.0, 1e-20])), [1.0, 1.0])
    # Away from zero it matches the direct evaluation.
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    np.testing.assert_allclose(_exprel(x), (np.exp(x) - 1.0) / x)


def test_c_defaults_to_none_and_is_settable():
    # ACE stores a CDF alongside the PDF; it is kept verbatim rather than
    # recomputed, so `c` is a plain attribute separate from cdf().
    t = Tabular([0.0, 1.0], [1.0, 1.0])
    assert t.c is None
    t.c = np.array([0.0, 1.0])
    np.testing.assert_array_equal(t.c, [0.0, 1.0])
    d = Discrete([1.0], [1.0])
    assert d.c is None


def test_mixture_basics():
    a = Discrete([1.0], [1.0])
    b = Tabular([0.0, 1.0], [1.0, 1.0])
    m = Mixture([0.3, 0.7], [a, b])
    assert len(m) == 3           # 1 discrete point + 2 tabulated points
    assert m.integral() == pytest.approx(0.3 * 1.0 + 0.7 * 1.0)
    with pytest.raises(NotImplementedError):
        m.cdf()


def test_mixture_normalize():
    m = Mixture([1.0, 3.0], [Discrete([1.0], [1.0]), Discrete([2.0], [1.0])])
    m.normalize()
    np.testing.assert_allclose(m.probability, [0.25, 0.75])


def test_combine_all_discrete_merges():
    a = Discrete([1.0, 2.0], [1.0, 1.0])
    b = Discrete([2.0], [1.0])
    out = combine_distributions([a, b], [1.0, 2.0])
    assert isinstance(out, Discrete)
    np.testing.assert_array_equal(out.x, [1.0, 2.0])
    np.testing.assert_allclose(out.p, [1.0, 3.0])


def test_combine_single_tabular_scales_in_place():
    t = Tabular([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    out = combine_distributions([t], [2.0])
    assert isinstance(out, Tabular)
    np.testing.assert_allclose(out.p, [0.0, 2.0, 0.0])
    assert out.interpolation == 'linear-linear'


def test_combine_mixed_puts_discrete_last():
    d = Discrete([1.0], [1.0])
    t = Tabular([0.0, 1.0], [1.0, 1.0])
    out = combine_distributions([d, t], [0.5, 0.5])
    assert isinstance(out, Mixture)
    # Continuous components first, then the merged discrete with probability 1.
    assert isinstance(out.distribution[-1], Discrete)
    np.testing.assert_allclose(out.probability, [0.5, 1.0])


def test_combine_flattens_nested_mixture():
    inner = Mixture([0.25, 0.75], [Discrete([1.0], [1.0]), Discrete([2.0], [1.0])])
    out = combine_distributions([inner], [2.0])
    assert isinstance(out, Discrete)
    np.testing.assert_array_equal(out.x, [1.0, 2.0])
    np.testing.assert_allclose(out.p, [0.5, 1.5])


def test_combine_two_tabulars_makes_mixture():
    t1 = Tabular([0.0, 1.0], [1.0, 1.0])
    t2 = Tabular([0.0, 2.0], [0.5, 0.5])
    out = combine_distributions([t1, t2], [0.4, 0.6])
    assert isinstance(out, Mixture)
    assert len(out.distribution) == 2
    np.testing.assert_allclose(out.probability, [0.4, 0.6])


def test_exported_from_package_root():
    import endf
    assert endf.Discrete is Discrete
    assert endf.Tabular is Tabular
    assert endf.Mixture is Mixture
