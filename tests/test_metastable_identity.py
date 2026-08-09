"""The evaluation, not the ZAID, decides which metastable state a nuclide is in.

A ZAID cannot express metastable state reliably. Under the MCNP convention 95242
means Am242**m1** and 95642 the ground state, an exception carried for that one
nuclide; the NNDC convention reverses it. Neither can express Ta-180m at all,
since 73180 reads as the ground state in both. So a nuclide built from an ACE
table can disagree with the evaluation it was generated from.

That mattered in practice: ``IncidentNeutron.from_njoy`` used to take ``metastable``
from the ACE table while correcting only ``name``, so the ground-state Am242
evaluation (ZA=95242, LISO=0) produced an object claiming to be m1.
``FissionEnergyRelease.from_endf`` compares the two and rejected its own
evaluation, which silently dropped both Am242 and Am242_m1 from any ENDF-route
build of ENDF/B-VIII.1.

The end-to-end path needs NJOY, so what is checked here is the premise the fix
rests on: the ZAID is ambiguous, the evaluation is not, and the two disagree for
exactly the nuclides that caused the failure.
"""

from pathlib import Path

import pytest

import endf
from endf.ace import get_metadata
from endf.fission_energy import FissionEnergyRelease

AM242_GROUND = Path(__file__).with_name("n-095_Am_242_trimmed.endf.xz")


@pytest.mark.parametrize("zaid, scheme, name, metastable", [
    # The Am242 exception, in both directions.
    (95242, "mcnp", "Am242_m1", 1),
    (95642, "mcnp", "Am242", 0),
    (95242, "nndc", "Am242", 0),
    (95642, "nndc", "Am242_m1", 1),
    # Ta-180m is not expressible: both schemes read 73180 as the ground state,
    # even though the only Ta-180 evaluation FENDL ships is the isomer.
    (73180, "mcnp", "Ta180", 0),
    (73180, "nndc", "Ta180", 0),
])
def test_zaid_metastable_is_convention_dependent(zaid, scheme, name, metastable):
    got_name, _, _, _, got_metastable = get_metadata(zaid, scheme)
    assert got_name == name
    assert got_metastable == metastable


def test_ground_state_am242_evaluation_disagrees_with_its_zaid():
    """The evaluation says ground state; its ZAID under MCNP says m1.

    This is the disagreement that made from_njoy build a mislabelled nuclide.
    """
    metadata = endf.Material(AM242_GROUND).section_data[1, 451]
    assert metadata["ZA"] == 95242
    assert metadata["LISO"] == 0, "this fixture is the ground-state evaluation"

    _, _, _, _, from_zaid = get_metadata(metadata["ZA"], "mcnp")
    assert from_zaid == 1, "the MCNP convention reads 95242 as metastable"
    assert from_zaid != metadata["LISO"], "so the ZAID contradicts the evaluation"


def test_fission_energy_release_rejects_a_mismatched_metastable():
    """Guard the check that turned the mislabelling into a hard failure.

    It is correct for it to reject a mismatch; the fix is to stop creating the
    mismatch, by taking metastable from the evaluation in from_njoy.
    """
    material = endf.Material(AM242_GROUND)

    class FakeNeutron:
        atomic_number = 95
        mass_number = 242
        metastable = 1          # what the ZAID would have produced

    with pytest.raises(ValueError, match="metastable state"):
        FissionEnergyRelease.from_endf(material, FakeNeutron())


def test_fission_energy_release_accepts_the_evaluations_own_state():
    """With metastable taken from the evaluation the release data reads back.

    This is the case that used to raise. Am242 is fissionable, so getting past the
    identity check yields real MF=1/458 data rather than merely a different error.
    """
    material = endf.Material(AM242_GROUND)

    class FakeNeutron:
        atomic_number = 95
        mass_number = 242
        metastable = 0          # what from_njoy now sets, from LISO

    release = FissionEnergyRelease.from_endf(material, FakeNeutron())
    assert release.fragments(0.0253) > 0, "fission fragment energy should be positive"
