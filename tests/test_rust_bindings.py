"""The Rust bindings, held to the pure-Python reader.

Every assertion here compares `_endf` against `endf` on the same fixture rather
than against a value written down by hand, so the two cannot drift apart
without a failure. The Rust crate's own test suite already compares 37,000
values; what is checked here is the *binding* — that the conversion at the
Python boundary keeps what the Rust side computed.

Skipped whole when the extension module is not built, since it is optional:

    maturin develop -m crates/endf-py/Cargo.toml
"""

from pathlib import Path

import pytest

import endf

_endf = pytest.importorskip(
    "_endf", reason="the Rust extension module is not built; see the module docstring"
)

TESTS = Path(__file__).parent


def fixture(name):
    return str(TESTS / name)


@pytest.fixture(scope="module")
def am244():
    return endf.Material(fixture("n-095_Am_244.endf.xz"))


@pytest.fixture(scope="module")
def rust_am244():
    return _endf.Material(fixture("n-095_Am_244.endf.xz"))


# ---------------------------------------------------------------------------
# Records and functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        " 1.234567+8",
        "-1.23481+10",
        " 3.14159   ",
        "1.234567-120",
        "         ",
        " 0.0000000",
    ],
)
def test_float_endf_matches(text):
    assert _endf.float_endf(text) == endf.records.float_endf(text)


@pytest.mark.parametrize("text", ["   9552", "     -1", "       ", "  12345"])
def test_int_endf_matches(text):
    assert _endf.int_endf(text) == endf.records.int_endf(text)


def test_tabulated1d_evaluates_the_same(am244):
    reference = am244.section_data[3, 1]["sigma"]
    tabulated = _endf.Tabulated1D(
        list(reference.x),
        list(reference.y),
        list(reference.breakpoints),
        list(reference.interpolation),
    )
    assert len(tabulated) == len(reference)
    assert tabulated.breakpoints == list(reference.breakpoints)
    for energy in (1.0e-5, 0.0253, 1.0e3, 1.0e6, 1.9e7):
        assert tabulated(energy) == pytest.approx(reference(energy), rel=1e-12)
    # Called on a sequence it comes back as a list of the same length.
    grid = [1.0, 10.0, 100.0]
    assert tabulated(grid) == pytest.approx([reference(e) for e in grid], rel=1e-12)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_material_sees_the_same_sections(am244, rust_am244):
    assert rust_am244.MAT == am244.MAT
    assert rust_am244.sections == sorted(am244.sections)
    assert rust_am244.section_text == am244.section_text
    assert (3, 1) in rust_am244
    assert (3, 999) not in rust_am244


def test_material_reads_an_uncompressed_file(tmp_path, am244):
    # The binding decompresses `.xz` the way `endf.fileutils` does, and leaves
    # anything else alone.
    plain = tmp_path / "am244.endf"
    with endf.fileutils.open_text(fixture("n-095_Am_244.endf.xz")) as fh:
        plain.write_text(fh.read())
    assert _endf.Material(str(plain)).MAT == am244.MAT


def test_cross_section_fields_match(am244, rust_am244):
    for mt in (1, 2, 18, 102):
        reference = am244.section_data[3, mt]
        section = rust_am244.mf3(mt)
        assert section.ZA == reference["ZA"]
        assert section.AWR == reference["AWR"]
        assert section.QM == reference["QM"]
        assert section.QI == reference["QI"]
        assert section.LR == reference["LR"]
        assert list(section.sigma.x) == list(reference["sigma"].x)
        assert list(section.sigma.y) == list(reference["sigma"].y)
    assert rust_am244.mf3(999) is None


def test_get_materials_matches():
    path = fixture("n-095_Am_244.endf.xz")
    assert [m.MAT for m in _endf.get_materials(path)] == [
        m.MAT for m in endf.get_materials(path)
    ]


# ---------------------------------------------------------------------------
# Reactions and nuclides
# ---------------------------------------------------------------------------


def test_reaction_names_match():
    for mt in range(1, 900):
        assert _endf.reaction_name(mt) == endf.reaction.REACTION_NAME.get(mt)
    for name, mt in endf.reaction.REACTION_MT.items():
        assert _endf.reaction_mt(name) == mt


def test_gnds_names_match():
    for z, a, m in [(1, 1, 0), (95, 242, 1), (49, 116, 2), (26, 56, 0)]:
        assert _endf.gnds_name(z, a, m) == endf.data.gnds_name(z, a, m)


def test_incident_neutron_from_endf(am244, rust_am244):
    reference = endf.IncidentNeutron.from_endf(am244)
    nuclide = _endf.IncidentNeutron.from_endf(rust_am244)

    assert nuclide.name == reference.name
    assert nuclide.atomic_number == reference.atomic_number
    assert nuclide.mass_number == reference.mass_number
    assert nuclide.metastable == reference.metastable
    assert nuclide.atomic_symbol == reference.atomic_symbol
    assert sorted(nuclide.reactions) == sorted(reference.reactions)


def test_reaction_matches_the_python_one(am244, rust_am244):
    reference = endf.IncidentNeutron.from_endf(am244)
    nuclide = _endf.IncidentNeutron.from_endf(rust_am244)

    for mt in sorted(reference.reactions):
        want = reference[mt]
        got = nuclide[mt]
        assert got.MT == want.MT
        assert got.q_reaction == want.q_reaction
        assert got.q_massdiff == want.q_massdiff
        assert sorted(got.xs) == sorted(want.xs)
        assert list(got.xs["0K"].x) == list(want.xs["0K"].x)
        assert list(got.xs["0K"].y) == list(want.xs["0K"].y)
        assert [p.name for p in got.products] == [p.name for p in want.products]


def test_reactions_are_reachable_by_name(rust_am244):
    nuclide = _endf.IncidentNeutron.from_endf(rust_am244)
    assert nuclide["elastic"].MT == 2
    assert nuclide["(n,gamma)"].MT == 102
    assert 2 in nuclide
    assert 999 not in nuclide
    with pytest.raises(ValueError):
        nuclide[999]


def test_fission_products_carry_their_emission_modes(rust_am244):
    nuclide = _endf.IncidentNeutron.from_endf(rust_am244)
    modes = [p.emission_mode for p in nuclide[18].products]
    assert modes[0] == "prompt"
    precursors = endf.Material(fixture("n-095_Am_244.endf.xz"))
    assert modes.count("delayed") == len(precursors.section_data[1, 455]["lambda"])
    # The total neutron is derived, not a product.
    assert [p.emission_mode for p in nuclide[18].derived_products] == ["total"]


def test_distributions_come_across_tagged(rust_am244):
    nuclide = _endf.IncidentNeutron.from_endf(rust_am244)
    # Level inelastic scattering: an angle from MF=4, an energy from the
    # kinematics.
    distribution = nuclide[51].products[0].distribution[0]
    assert distribution["kind"] == "uncorrelated"
    assert distribution["energy"]["kind"] == "level-inelastic"
    assert distribution["energy"]["threshold"] > 0.0
    assert {mu["kind"] for mu in distribution["angle"]["mu"]} == {"legendre"}

    # The fission spectrum is a Maxwellian in this evaluation.
    fission = nuclide[18].products[0].distribution[0]
    assert fission["energy"]["kind"] == "maxwell"
    assert set(fission["energy"]["theta"]) == {"x", "y", "breakpoints", "interpolation"}


def test_removal_xs_matches(am244, rust_am244):
    reference = endf.IncidentNeutron.from_endf(am244).removal_xs("0K", 0.0)
    got = _endf.IncidentNeutron.from_endf(rust_am244).removal_xs("0K", 0.0)
    assert list(got.x) == list(reference.x)
    assert got.y == pytest.approx(list(reference.y), rel=1e-9)


# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def li6_ace():
    return endf.ace.get_tables(fixture("Li6.ace.xz"))[0]


@pytest.fixture(scope="module")
def rust_li6_ace():
    return _endf.get_tables(fixture("Li6.ace.xz"))[0]


def test_ace_table_header_matches(li6_ace, rust_li6_ace):
    assert rust_li6_ace.name == li6_ace.name
    assert rust_li6_ace.atomic_weight_ratio == li6_ace.atomic_weight_ratio
    assert rust_li6_ace.kT == li6_ace.kT
    assert rust_li6_ace.temperature == li6_ace.temperature
    assert rust_li6_ace.zaid == li6_ace.zaid
    assert rust_li6_ace.data_type == li6_ace.data_type.value
    assert rust_li6_ace.nxs == list(li6_ace.nxs)
    assert rust_li6_ace.jxs == list(li6_ace.jxs)
    assert len(rust_li6_ace.xss) == len(li6_ace.xss)


def test_incident_neutron_from_ace_matches(li6_ace, rust_li6_ace):
    reference = endf.IncidentNeutron.from_ace(li6_ace)
    nuclide = _endf.IncidentNeutron.from_ace(rust_li6_ace)

    assert nuclide.name == reference.name
    assert nuclide.temperatures == reference.temperatures
    assert nuclide.kTs == list(reference.kTs)
    assert sorted(nuclide.reactions) == sorted(reference.reactions)
    for mt in sorted(reference.reactions):
        assert nuclide[mt].redundant == reference[mt].redundant
        assert nuclide.reaction_components(mt) == reference.get_reaction_components(mt)

    temperature = nuclide.temperatures[0]
    assert nuclide.energy[temperature] == pytest.approx(
        list(reference.energy[temperature]), rel=1e-15
    )


def test_an_unknown_metastable_scheme_is_refused(rust_li6_ace):
    with pytest.raises(ValueError, match="mcnp"):
        _endf.IncidentNeutron.from_ace(rust_li6_ace, "nonsense")


# ---------------------------------------------------------------------------
# Photons, decay and chains
# ---------------------------------------------------------------------------


def test_incident_photon_matches():
    photoatomic = fixture("photoat-001_H_000.endf.xz")
    relaxation = fixture("atom-001_H_000.endf.xz")
    reference = endf.IncidentPhoton.from_endf(
        endf.Material(photoatomic), endf.Material(relaxation)
    )
    element = _endf.IncidentPhoton.from_endf(
        _endf.Material(photoatomic), _endf.Material(relaxation)
    )

    assert element.atomic_number == reference.atomic_number
    assert element.name == reference.name
    assert sorted(element.reactions) == sorted(reference.reactions)
    relaxation = reference.atomic_relaxation
    assert element.atomic_relaxation["subshells"] == relaxation.subshells
    assert element.atomic_relaxation["binding_energy"] == pytest.approx(
        {k: float(v) for k, v in relaxation.binding_energy.items()}
    )
    coherent = element.reactions[502]
    assert coherent["name"] == "coherent"
    assert "scattering_factor" in coherent
    assert "anomalous_real" in coherent


def test_decay_matches():
    path = fixture("dec-049_In_116m1.endf.xz")
    reference = endf.decay.Decay(path)
    decay = _endf.Decay.from_endf(_endf.Material(path))

    assert decay.nuclide["name"] == reference.nuclide["name"]
    assert decay.nuclide["stable"] == reference.nuclide["stable"]
    assert decay.half_life[0] == reference.half_life.nominal_value
    assert decay.decay_constant[0] == pytest.approx(
        reference.decay_constant.nominal_value, rel=1e-12
    )
    assert decay.decay_energy[0] == pytest.approx(
        reference.decay_energy.nominal_value, rel=1e-12
    )
    assert [m["modes"] for m in decay.modes] == [m.modes for m in reference.modes]
    assert [m["daughter"] for m in decay.modes] == [m.daughter for m in reference.modes]

    # Gammas and x-rays are both photons, so their lines merge into one source.
    sources = decay.sources
    assert sorted(sources) == sorted(reference.sources)
    for particle, dist in sources.items():
        assert dist["x"] == pytest.approx(
            list(reference.sources[particle].x), rel=1e-12
        )
        assert dist["p"] == pytest.approx(
            list(reference.sources[particle].p), rel=1e-12
        )


def test_a_material_with_no_decay_section_is_refused(rust_am244):
    with pytest.raises(ValueError):
        _endf.Decay.from_endf(rust_am244)


DECAY_FILES = [
    "dec-048_Cd_116.endf.xz",
    "dec-049_In_115.endf.xz",
    "dec-049_In_116.endf.xz",
    "dec-049_In_116m1.endf.xz",
    "dec-049_In_116m2.endf.xz",
    "dec-050_Sn_115.endf.xz",
    "dec-050_Sn_116.endf.xz",
    "dec-054_Xe_136.endf.xz",
    "dec-054_Xe_137.endf.xz",
    "dec-055_Cs_137.endf.xz",
]

NEUTRON_FILES = [
    "n-049_In-115_trimmed.endf.xz",
    "n-054_Xe_136_trimmed.endf.xz",
]


def test_chain_matches():
    reference = endf.chain.Chain.from_endf(
        [endf.Material(fixture(n)) for n in DECAY_FILES],
        [],
        [endf.Material(fixture(n)) for n in NEUTRON_FILES],
        reactions=("(n,gamma)",),
        progress=False,
    )
    chain = _endf.Chain.from_endf(
        [_endf.Material(fixture(n)) for n in DECAY_FILES],
        [],
        [_endf.Material(fixture(n)) for n in NEUTRON_FILES],
        ["(n,gamma)"],
    )

    assert len(chain) == len(reference)
    assert [n["name"] for n in chain.nuclides] == [n.name for n in reference.nuclides]
    for got, want in zip(chain.nuclides, reference.nuclides):
        assert got["half_life"] == want.half_life
        assert got["decay_energy"] == pytest.approx(want.decay_energy, rel=1e-12)
        assert [m["type"] for m in got["decay_modes"]] == [
            m.type for m in want.decay_modes
        ]
        assert [m["target"] for m in got["decay_modes"]] == [
            m.target for m in want.decay_modes
        ]
        assert [r["Q"] for r in got["reactions"]] == [r.Q for r in want.reactions]

    assert "In116_m1" in chain
    assert chain["In116_m1"]["name"] == "In116_m1"
    # One step from In115 reaches both its beta- daughter and its capture
    # product, so three nuclides in all.
    assert sorted(n["name"] for n in chain.reduce(["In115"], 1).nuclides) == [
        "In115",
        "In116",
        "Sn115",
    ]
    # The evaluated branching ratios are consistent.
    assert chain.validate(1e-4) == []


# ---------------------------------------------------------------------------
# section_data
#
# The dictionaries are compared whole and recursively against the Python
# reader's, on every fixture, so a renamed or missing key fails rather than
# going unnoticed.
# ---------------------------------------------------------------------------


def compare_values(got, want, where):
    """Assert two section-dictionary values are the same thing."""
    import numpy as np

    from endf.function import Tabulated1D, Tabulated2D

    if isinstance(want, Tabulated2D):
        assert list(got.breakpoints) == list(want.breakpoints), f"{where}.breakpoints"
        assert list(got.interpolation) == list(want.interpolation), f"{where}.int"
    elif isinstance(want, Tabulated1D):
        assert list(got.x) == list(want.x), f"{where}.x"
        assert list(got.y) == list(want.y), f"{where}.y"
        assert list(got.breakpoints) == list(want.breakpoints), f"{where}.breakpoints"
        assert list(got.interpolation) == list(want.interpolation), f"{where}.int"
    elif isinstance(want, dict):
        assert sorted(got) == sorted(want), f"{where}: keys"
        for key in want:
            compare_values(got[key], want[key], f"{where}[{key!r}]")
    elif isinstance(want, (list, tuple, np.ndarray)):
        want = list(want)
        got = list(got)
        assert len(got) == len(want), f"{where}: length"
        for i, (g, w) in enumerate(zip(got, want)):
            compare_values(g, w, f"{where}[{i}]")
    elif isinstance(want, (float, np.floating)):
        assert got == pytest.approx(float(want), rel=1e-15, abs=0.0), where
    else:
        assert got == want, where


ENDF_FIXTURES = sorted(p.name for p in TESTS.glob("*.endf.xz"))


@pytest.mark.parametrize("name", ENDF_FIXTURES)
def test_section_data_matches(name):
    reference = endf.Material(fixture(name))
    material = _endf.Material(fixture(name))

    section_data = material.section_data
    assert section_data, f"{name}: no section has a dictionary form"

    for key, got in section_data.items():
        compare_values(got, reference.section_data[key], f"{name} {key}")

    # And the same through the item lookup.
    for key in section_data:
        compare_values(material[key], reference.section_data[key], f"{name} {key}")


def test_section_data_leaves_out_what_it_cannot_build(am244, rust_am244):
    # MF=1 MT=458 is parsed but has no dictionary form here, so it is absent
    # rather than half-built, and asking for it says so.
    assert (1, 458) in am244.section_data
    assert (1, 458) not in rust_am244.section_data
    with pytest.raises(ValueError, match="no dictionary form"):
        rust_am244[1, 458]
    # A section that does not exist at all is a different error.
    with pytest.raises(ValueError, match="no section"):
        rust_am244[3, 999]


#: The files whose sections have no dictionary form in the extension yet.
#:
#: MF=8 MT=457 is here by choice: decay data is reached through `Decay`, which
#: is a better shape than the dictionary. The rest are simply not written yet.
#: Pinned so the list cannot shrink or grow without saying so.
SECTIONS_WITHOUT_A_DICT = {
    (1, 458),
    (2, 151),
    (6, 102),
    (6, 105),
    (7, 2),
    (7, 4),
    (8, 457),
    (26, 525),
    (26, 527),
    (26, 528),
    (26, 534),
    (33, 103),
    (33, 105),
    (34, 51),
}


def test_the_uncovered_section_list_is_accurate():
    missing = set()
    for name in ENDF_FIXTURES:
        reference = endf.Material(fixture(name))
        material = _endf.Material(fixture(name))
        have = set(material.section_data)
        missing |= {key for key in reference.section_data if key not in have}

    assert missing == SECTIONS_WITHOUT_A_DICT, (
        "the set of sections with no dictionary form has changed. If one now "
        "has one, delete it from SECTIONS_WITHOUT_A_DICT; if a projection "
        "broke, that is the bug."
    )
