"""Tests for depletion chain construction.

Building a whole chain needs the decay sub-library closed over its daughters, so
the two things that were wrong here are tested at the level they operate on: the
branching-ratio normalisation directly, and the reaction Q value against the one
evaluation whose QM and QI disagree in sign.
"""

import math
from pathlib import Path

import pytest

import endf
from endf.chain import normalise_branch_ratios

TESTS = Path(__file__).parent


# ---------------------------------------------------------------------------
# Branching ratio normalisation
# ---------------------------------------------------------------------------

def test_already_normalised_ratios_are_left_untouched():
    """Ratios summing to one must be returned exactly as evaluated, not
    re-derived, so no floating-point noise is introduced."""
    br = [0.6, 0.3, 0.1]
    original = list(br)
    normalise_branch_ratios(br)
    assert br == original


def test_ratios_within_floating_point_noise_are_left_untouched():
    """Yb158 as evaluated in ENDF/B-VIII.1: an alpha branch and an electron
    capture branch summing to 0.99999999955, which is one to within 4.5e-10.
    Correcting that would introduce noise rather than remove it."""
    br = [2.099955e-05, 0.999979]
    original = list(br)
    assert sum(br) != 1.0 and math.isclose(sum(br), 1.0)
    normalise_branch_ratios(br)
    assert br == original


def test_residual_goes_into_the_largest_branch():
    br = [0.5, 0.4, 0.05]           # sums to 0.95
    normalise_branch_ratios(br)
    assert sum(br) == pytest.approx(1.0)
    assert br[0] == pytest.approx(0.55)   # the largest absorbed it
    assert br[1] == 0.4                   # others untouched
    assert br[2] == 0.05


def test_tiny_branches_are_not_distorted():
    """The reason the largest branch absorbs the residual.

    Zn82 as evaluated in ENDF/B-VIII.1: a dominant beta- branch, a percent-level
    delayed-neutron branch, and a 6.4e-09 two-neutron branch. Absorbing the
    residual into the last branch rather than the largest would inflate that
    6.4e-09 to 1e-07, a factor of 14.
    """
    br = [0.65883, 0.3411699, 6.4102e-09]
    normalise_branch_ratios(br)
    assert sum(br) == pytest.approx(1.0)
    # The tiny branch survives intact
    assert br[2] == 6.4102e-09
    # The dominant branch took the correction
    assert br[0] != 0.65883


def test_a_branch_is_never_erased():
    """Absorbing a positive residual into the last branch can drive a small one
    to zero. The largest branch is always big enough to absorb it."""
    br = [0.9, 0.1000001, 1.3968e-09]
    normalise_branch_ratios(br)
    assert sum(br) == pytest.approx(1.0)
    assert br[2] > 0.0


def test_single_branch_is_normalised_to_one():
    br = [0.97]
    normalise_branch_ratios(br)
    assert br == [pytest.approx(1.0)]


def test_empty_list_is_accepted():
    br = []
    assert normalise_branch_ratios(br) == []


def test_ratios_summing_above_one_are_reduced():
    br = [0.7, 0.4]
    normalise_branch_ratios(br)
    assert sum(br) == pytest.approx(1.0)
    assert br[0] == pytest.approx(0.6)
    assert br[1] == 0.4


def test_returns_the_same_list_it_was_given():
    br = [0.5, 0.4]
    assert normalise_branch_ratios(br) is br


# ---------------------------------------------------------------------------
# Reaction Q value
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def xe136():
    # Trimmed to MF=1 and MF=3, which is all the chain builder reads from a
    # neutron evaluation.
    return endf.Material(TESTS / 'n-054_Xe_136_trimmed.endf')


@pytest.mark.parametrize('mt,q', [(103, -6096700.0), (104, -7672500.0),
                                  (105, -9317900.0)])
def test_xe136_qm_and_qi_disagree_in_sign(xe136, mt, q):
    """The evaluation this pins down.

    Xe136 is neutron rich, so knocking out a charged particle costs energy and
    the reaction Q must be negative. ENDF/B-VIII.1 gives QI negative as expected
    but QM with the opposite sign, so a chain builder reading QM reports these
    endothermic channels as exothermic. Of the 558 neutron evaluations in
    ENDF/B-VIII.1 only Xe136 (MT 103, 104, 105) and Np236m1 (MT 55) do this.
    """
    section = xe136.section_data[3, mt]
    assert section['QI'] == q
    assert section['QI'] < 0.0, "an endothermic channel must have negative Q"
    assert section['QM'] == -q, "QM has the opposite sign in this evaluation"


def test_reaction_q_is_taken_from_qi(xe136):
    """Chain.from_endf reads QI, so these channels come out endothermic."""
    for mt in (103, 104, 105):
        section = xe136.section_data[3, mt]
        assert section['QI'] < 0.0 < section['QM']


def test_qm_equals_qi_for_the_transmutation_channels_of_a_normal_evaluation():
    """For the channels a chain is built from, the two normally agree, which is
    why reading the wrong one goes unnoticed."""
    material = endf.Material(TESTS / 'n-095_Am_244.endf')
    for mt in (16, 17, 102):
        data = material.section_data[3, mt]
        assert data['QM'] == data['QI'], mt


def test_qm_is_zero_for_inelastic_levels():
    """A second reason QM is the wrong field. For inelastic scattering to a
    discrete level there is no mass change, so QM is zero while QI carries the
    level excitation energy. Those channels do not transmute and so never reach a
    depletion chain, but it shows the two quantities are not interchangeable."""
    material = endf.Material(TESTS / 'n-095_Am_244.endf')
    for mt in (51, 52, 53):
        data = material.section_data[3, mt]
        assert data['QM'] == 0.0
        assert data['QI'] < 0.0
