"""Regression tests for parser bugs found while porting the reader to Rust.

Each test names the issue it closes and asserts the *corrected* behaviour, so
a revert fails here rather than in a golden file where the cause is harder to
read off. The behaviour was checked against OpenMC, ENDFtk and endf-parserpy
before being changed; see the issue threads.
"""

from pathlib import Path

import pytest

import endf

TESTS = Path(__file__).parent


def fixture(name):
    return str(TESTS / name)


# ---------------------------------------------------------------------------
# Issue #15 -- unresolved ranges with LRF=1 were dropped, and desynced the read
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def urr_cases():
    return endf.Material(fixture("synthetic-urr-cases.endf.xz"))


def test_unresolved_case_a_is_read(urr_cases):
    """LRU=2 with LRF=1 and LFW=0. Previously matched no branch at all."""
    rrange = urr_cases.section_data[2, 151]["isotopes"][0]["ranges"][0]
    assert rrange["LRU"] == 2
    assert rrange["LRF"] == 1

    # Before the fix the range carried only its six header fields.
    assert rrange["SPI"] == pytest.approx(0.5)
    assert rrange["AP"] == pytest.approx(0.94)
    assert rrange["LSSF"] == 0
    assert rrange["NLS"] == 2

    # Two L values with different J counts, so reusing NJS would be caught.
    by_l = rrange["ranges"]
    assert [r["L"] for r in by_l] == [0, 1]
    assert [r["NJS"] for r in by_l] == [2, 3]
    # Case A gives energy-independent parameters directly, not a 'parameters'
    # list -- that is the shape Cases B and C use.
    assert list(by_l[0]["D"]) == pytest.approx([8.9, 4.4])
    assert list(by_l[1]["GNO"]) == pytest.approx([0.003, 0.004, 0.005])


def test_unresolved_case_b_is_read(urr_cases):
    """LRU=2 with LRF=1 and LFW=1: only the fission widths vary with energy."""
    rrange = urr_cases.section_data[2, 151]["isotopes"][1]["ranges"][0]
    assert rrange["LRU"] == 2 and rrange["LRF"] == 1

    # NE and ES exist only in Case B.
    assert rrange["NE"] == 3
    assert list(rrange["ES"]) == pytest.approx([1.0e3, 1.0e4, 3.0e4])

    parameters = rrange["ranges"][0]["parameters"]
    assert [p["MUF"] for p in parameters] == [1, 2]
    assert list(parameters[0]["GF"]) == pytest.approx([0.010, 0.012, 0.015])


def test_the_range_after_an_unresolved_one_is_still_aligned(urr_cases):
    """The dangerous half of #15: a skipped range left its records unread.

    The second range is resolved multi-level Breit-Wigner. Before the fix it
    came back as LRU=0 with EL and EH holding the *first* range's SPI and AP,
    because parsing resumed in the middle of the range that was skipped.
    """
    rrange = urr_cases.section_data[2, 151]["isotopes"][0]["ranges"][1]
    assert rrange["LRU"] == 1
    assert rrange["LRF"] == 2
    assert rrange["EL"] == pytest.approx(3.0e4)
    assert rrange["EH"] == pytest.approx(1.0e5)
    assert list(rrange["sections"][0]["ER"]) == pytest.approx([5.5e4])


def test_case_c_still_works(urr_cases):
    """LRU=2 with LRF=2 worked before only because the two flags coincided."""
    u235 = endf.Material(fixture("n-092_U_235_trimmed.endf.xz"))
    ranges = u235.section_data[2, 151]["isotopes"][0]["ranges"]
    unresolved = [r for r in ranges if r["LRU"] == 2]
    assert unresolved, "U235 should still have an unresolved range"
    assert unresolved[0]["LRF"] == 2
    assert unresolved[0]["ranges"][0]["parameters"][0]["INT"] == 2


# ---------------------------------------------------------------------------
# Issue #23 -- a zero half-life divided by zero
# ---------------------------------------------------------------------------


def test_zero_half_life_has_no_decay_constant():
    """Xe136 is flagged unstable and evaluated with T1/2 = 0.

    Zero means "not evaluated", not "decays instantly", so there is no decay
    constant to report. This used to raise ZeroDivisionError.
    """
    decay = endf.Decay(fixture("dec-054_Xe_136.endf.xz"))
    assert not decay.nuclide["stable"]
    assert decay.half_life.nominal_value == 0.0
    assert decay.decay_constant is None


def test_zero_half_life_gives_no_sources():
    """`sources` scales every rate by the decay constant, so it follows."""
    decay = endf.Decay(fixture("dec-054_Xe_136.endf.xz"))
    assert decay.sources == {}


def test_an_ordinary_nuclide_still_has_a_decay_constant():
    from math import log

    decay = endf.Decay(fixture("dec-055_Cs_137.endf.xz"))
    assert decay.half_life.nominal_value > 0.0
    expected = log(2.0) / decay.half_life.nominal_value
    assert decay.decay_constant.nominal_value == pytest.approx(expected, rel=1e-12)


def test_a_whole_decay_sublibrary_can_be_iterated():
    """The point of #23: one nuclide with a zero half-life broke the lot."""
    constants = {}
    for path in sorted(TESTS.glob("dec-*.endf.xz")):
        decay = endf.Decay(str(path))
        constants[decay.nuclide["name"]] = decay.decay_constant
    assert len(constants) >= 10
    assert constants["Xe136"] is None
    assert any(c is not None for c in constants.values())


# ---------------------------------------------------------------------------
# Issue #12 -- MF=33 NC subsections with LTY=0 were appended twice
# ---------------------------------------------------------------------------


def test_mf33_nc_subsections_appear_once():
    """One entry per subsection, and NC says how many there should be."""
    from io import StringIO

    from endf.mf33 import parse_mf33_subsection

    def line(fields):
        return f"{''.join(f'{v:>11}' for v in fields):<66}9228331\n"

    # NC=2, NI=0: one LTY=0 subsection and one LTY=3, so both branches run.
    text = (
        line([0.0, 0.0, 0, 2, 2, 0])
        + line([0.0, 0.0, 0, 0, 0, 0])  # LTY=0
        + line([1.0, 2.0e7, 0, 0, 2, 1])
        + line([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        + line([0.0, 0.0, 0, 3, 0, 0])  # LTY=3
        + line([1.0, 2.0e7, 9228, 102, 4, 1])
        + line([3.0, 0.0, 1.0, 0.5, 0.0, 0.0])
    )
    sub = parse_mf33_subsection(StringIO(text))

    assert sub["NC"] == 2
    assert len(sub["nc_subsections"]) == 2, "one entry per subsection"
    assert [s["LTY"] for s in sub["nc_subsections"]] == [0, 3]


# ---------------------------------------------------------------------------
# Issue #18 -- MF=34 discarded its subsections, and LB was filled with LS
# ---------------------------------------------------------------------------


def test_mf34_keeps_its_subsections():
    """U235 has a real MF=34 section; it used to come back empty."""
    u235 = endf.Material(fixture("n-092_U_235_trimmed.endf.xz"))
    section = u235.section_data[34, 51]

    assert section["NMT1"] == 1
    assert len(section["subsections"]) == section["NMT1"]

    subsection = section["subsections"][0]
    assert subsection["MT1"] == 51
    assert subsection["NSS"] == 3
    assert len(subsection["subsubsections"]) == subsection["NSS"]


def test_mf34_reads_lb_rather_than_copying_ls():
    u235 = endf.Material(fixture("n-092_U_235_trimmed.endf.xz"))
    subsection = u235.section_data[34, 51]["subsections"][0]

    # Every block in this section is LB=5, a covariance matrix, while LS --
    # the symmetry flag -- varies. That is what makes the section good
    # evidence: copying LS into LB reported the blocks as LB = 1, 0, 1, i.e.
    # two of them as an absolute covariance in (E, F) pairs and one as
    # something else again, when all three are matrices. A consumer switching
    # on LB to unpack `Data` would have unpacked all three the wrong way.
    lb = [list(ss["LB"]) for ss in subsection["subsubsections"]]
    ls = [list(ss["LS"]) for ss in subsection["subsubsections"]]
    assert lb == [[5.0], [5.0], [5.0]]
    assert ls == [[1.0], [0.0], [1.0]]
    assert lb != ls, "the two fields must not track each other"


# ---------------------------------------------------------------------------
# Issue #19 -- ACE law 5 died with AttributeError instead of saying it is a gap
# ---------------------------------------------------------------------------


def test_ace_law_5_raises_not_implemented():
    """Neither this reader, OpenMC nor the Rust port implements law 5.

    What matters is that it says so: the dispatch used to reach a `from_ace`
    that did not exist and fail with AttributeError, which reads like an
    internal error rather than an unsupported format.
    """
    from endf.mf5 import GeneralEvaporation

    with pytest.raises(NotImplementedError, match="law 5"):
        GeneralEvaporation.from_ace(None, 0)


# ---------------------------------------------------------------------------
# Issue #25 -- MF=33 LB=0 to 4 split its E-tables at NT - NP, not 2*(NP - LT)
# ---------------------------------------------------------------------------


def _lb_list(lt, lb, pairs):
    """An NI subsection holding one LB=0 to 4 LIST record of `pairs`."""
    from io import StringIO

    def line(fields):
        return f"{''.join(f'{v:>11}' for v in fields):<66}9228331\n"

    values = [v for pair in pairs for v in pair]
    values += [0.0] * (-len(values) % 6)
    text = line([0.0, 0.0, 0, 0, 0, 1]) + line(
        [0.0, 0.0, lt, lb, 2 * len(pairs), len(pairs)]
    )
    for i in range(0, len(values), 6):
        text += line(values[i:i + 6])
    return StringIO(text)


def test_mf33_lb1_keeps_its_one_table_whole():
    """LT=0 means one table. NP is odd, so the old split also started El on an
    F value, swapping El and Fl as well as truncating Ek."""
    from endf.mf33 import parse_mf33_subsection

    pairs = [(1.0e-5, 0.0), (8.0e5, 0.01125), (2.0e7, 0.0)]
    ni = parse_mf33_subsection(_lb_list(0, 1, pairs))["ni_subsections"][0]
    assert list(ni["Ek"]) == [1.0e-5, 8.0e5, 2.0e7]
    assert list(ni["Fk"]) == [0.0, 0.01125, 0.0]
    assert list(ni["El"]) == [] and list(ni["Fl"]) == []


def test_mf33_lb3_splits_at_the_second_tables_own_length():
    """NP - LT pairs, then LT: three and two here, where NT - NP would have
    split two and a half pairs in."""
    from endf.mf33 import parse_mf33_subsection

    pairs = [(1.0, 0.1), (2.0, 0.2), (3.0, 0.0), (1.0, 0.3), (3.0, 0.0)]
    ni = parse_mf33_subsection(_lb_list(2, 3, pairs))["ni_subsections"][0]
    assert list(ni["Ek"]) == [1.0, 2.0, 3.0]
    assert list(ni["Fk"]) == [0.1, 0.2, 0.0]
    assert list(ni["El"]) == [1.0, 3.0]
    assert list(ni["Fl"]) == [0.3, 0.0]


@pytest.fixture(scope="module")
def ni58_fendl():
    return endf.Material(fixture("n_2825_28-Ni-58_trimmed.fendl32d.endf.xz"))


def test_a_real_lb1_table_runs_to_20_mev(ni58_fendl):
    """FENDL-3.2d Ni58 (n,a), the block that gives it 16.8% at 13 to 16 MeV.

    Split at NT - NP it ended at 4 MeV, and everything above went to El."""
    blocks = ni58_fendl.section_data[33, 107]["subsections"][0]["ni_subsections"]
    ni = blocks[2]
    assert (ni["LB"], ni["LT"], ni["NP"]) == (1, 0, 12)
    assert len(ni["Ek"]) == ni["NP"]
    assert ni["Ek"][-1] == 2.0e7
    assert ni["Fk"][list(ni["Ek"]).index(1.3e7)] == pytest.approx(2.8125e-2)
    assert list(ni["El"]) == [] and list(ni["Fl"]) == []


def test_every_lb0_to_2_block_has_one_table(ni58_fendl):
    """Every LB=0 to 2 block on the tape has LT=0, whatever NP's parity."""
    seen = 0
    for mt in (28, 103, 107):
        for sub in ni58_fendl.section_data[33, mt]["subsections"]:
            for ni in sub["ni_subsections"]:
                if ni["LB"] > 2:
                    continue
                assert ni["LT"] == 0
                assert len(ni["Ek"]) == len(ni["Fk"]) == ni["NP"]
                assert list(ni["El"]) == []
                seen += 1
    assert seen == 6


def test_the_lb4_block_still_splits_in_half(ni58_fendl):
    """The one LB=4 block on any tape has LT = NP/2, where the old split
    happened to agree, so this pins that the fix did not move it."""
    blocks = ni58_fendl.section_data[33, 103]["subsections"][0]["ni_subsections"]
    ni = blocks[1]
    assert (ni["LB"], ni["LT"], ni["NP"]) == (4, 21, 42)
    assert len(ni["Ek"]) == len(ni["El"]) == 21
    assert list(ni["Ek"]) == list(ni["El"])
    assert ni["Fk"][0] == -1.0


# ---------------------------------------------------------------------------
# LRF=7 resonances are padded to whole six-value lines
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cl35_rml():
    return endf.Material(fixture("n-017_Cl_035_mf2.endf.xz"))


def test_rml_resonances_step_over_their_padding(cl35_rml):
    """ENDF/B-VIII.1 Cl35, NCH = 3: each resonance is ER, three widths and two
    zeros of padding, six values a line (ENDF-102 section 2.2.1.6). Striding by
    NCH + 1 = 4 read widths and padding as resonance energies."""
    groups = cl35_rml.section_data[2, 151]["isotopes"][0]["ranges"][0]["spin_groups"]
    assert len(groups) == 8
    for group in groups:
        assert group["NCH"] == 3
        assert len(group["ER"]) == group["NRS"]
        assert group["GAM"].shape == (group["NCH"], group["NRS"])
    first = groups[0]
    assert list(first["ER"][:3]) == [54932.0, 68236.16, 115098.0]
    assert list(first["GAM"][:, 0]) == [0.36726, 46.4424, 0.0]
