"""Tests for the NJOY driver.

Running NJOY needs the executable and a full evaluation, so the end-to-end test
skips when either is missing. The input-deck tests need neither and always run,
since a wrong deck is the most likely way to break this quietly.
"""

import shutil
from pathlib import Path

import pytest

import endf
from endf.njoy import (
    make_ace, _TEMPLATE_RECONR, _TEMPLATE_BROADR, _TEMPLATE_ACER,
    _TEMPLATE_HEATR, _TEMPLATE_HEATR_LOCAL, _TEMPLATE_GASPR, _TEMPLATE_PURR,
)

TESTS = Path(__file__).parent
HAVE_NJOY = shutil.which('njoy') is not None


def test_templates_have_the_expected_placeholders():
    """The tape numbers are threaded between modules, so a missing or renamed
    placeholder silently sends a module the wrong input."""
    assert '{nendf}' in _TEMPLATE_RECONR and '{npendf}' in _TEMPLATE_RECONR
    assert '{library}' in _TEMPLATE_RECONR and '{zsymam}' in _TEMPLATE_RECONR
    assert '{nbroadr}' in _TEMPLATE_BROADR and '{temps}' in _TEMPLATE_BROADR
    assert '{nheatr}' in _TEMPLATE_HEATR
    assert '{nheatr_local}' in _TEMPLATE_HEATR_LOCAL
    assert '{ngaspr}' in _TEMPLATE_GASPR
    assert '{npurr}' in _TEMPLATE_PURR
    for field in ('{nace}', '{ndir}', '{ext}', '{temperature}', '{ismooth}'):
        assert field in _TEMPLATE_ACER, field


def test_heatr_asks_for_the_partial_kermas():
    """HEATR is asked for MTs 302, 318, 402 and 444, and the local run differs
    from the non-local one only in the photon deposition flag."""
    for template in (_TEMPLATE_HEATR, _TEMPLATE_HEATR_LOCAL):
        assert '302 318 402 444 /' in template
    assert '{mat} 4 0 0 0 /' in _TEMPLATE_HEATR
    assert '{mat} 4 0 0 1 /' in _TEMPLATE_HEATR_LOCAL


def test_output_dir_must_be_a_directory(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(IOError, match="not a directory"):
        make_ace(TESTS / 'n-095_Am_244.endf.xz', output_dir=missing)


def test_library_name_comes_from_the_evaluation():
    """The library string appears in NJOY's comment cards and so in the ACE
    file, and is built from NLIB, NVER and LREL."""
    from endf.material import _LIBRARY
    mat = endf.Material(TESTS / 'n-095_Am_244.endf.xz')
    metadata = mat.section_data[1, 451]
    library = '{}-{}.{}'.format(_LIBRARY.get(metadata['NLIB'], 'Unknown'),
                                metadata['NVER'], metadata['LREL'])
    assert library.count('-') >= 1 and '.' in library
    # NLIB=0 is ENDF/B, which is what this test file is.
    assert library.startswith('ENDF/B-')


@pytest.mark.skipif(not HAVE_NJOY, reason="njoy executable not found")
def test_make_ace_writes_a_readable_table(tmp_path):
    """A full run, checked by reading the ACE back."""
    ace_path = tmp_path / "ace"
    make_ace(TESTS / 'n-095_Am_244.endf.xz', temperatures=[293.6],
             acer=str(ace_path), output_dir=str(tmp_path))
    assert ace_path.is_file()

    tables = endf.ace.get_tables(ace_path)
    assert len(tables) == 1
    assert tables[0].temperature == pytest.approx(293.6, abs=0.5)

    data = endf.IncidentNeutron.from_ace(tables[0])
    assert data.atomic_number == 95
    assert data.mass_number == 244
    # Am244 is fissile, so MT=18 must be present with a nu value
    assert 18 in data.reactions
    assert data.temperatures == ['294K']


@pytest.mark.skipif(not HAVE_NJOY, reason="njoy executable not found")
def test_from_njoy_multiple_temperatures(tmp_path):
    data = endf.IncidentNeutron.from_njoy(
        TESTS / 'n-095_Am_244.endf.xz', temperatures=[293.6, 900.0],
        output_dir=str(tmp_path))

    assert data.temperatures == ['294K', '900K']
    assert len(data.kTs) == 2
    # The 0 K elastic cross section is lifted from the PENDF tape
    assert '0K' in data.energy
    assert '0K' in data.reactions[2].xs
    # HEATR runs produce both the non-local (MT=301) and local (MT=901) kermas
    assert 301 in data.reactions
    assert 901 in data.reactions
    assert data.reactions[901].redundant
    for temp in data.temperatures:
        assert temp in data.reactions[901].xs
    # The name comes from the evaluation, not the ACE ZAID
    assert data.name == 'Am244'
