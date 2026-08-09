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


def test_zam_matches():
    for name in ["H1", "Am242_m1", "In116_m2", "Fe56", "n1"]:
        assert _endf.zam(name) == endf.data.zam(name)


def test_temperature_str_matches():
    for t in [0.0, 293.6, 250.0, 900.0, 1200.4, 2500.0]:
        assert _endf.temperature_str(t) == endf.data.temperature_str(t)


def test_photon_reaction_names_match():
    for mt in range(500, 600):
        assert _endf.photon_reaction_name(
            mt
        ) == endf.incident_photon.PHOTON_REACTION_NAME.get(mt)
    for name, mt in endf.incident_photon.PHOTON_REACTION_MT.items():
        assert _endf.photon_reaction_mt(name) == mt


def test_decay_modes_match():
    # RTYP packs a chain of modes as the digits of a decimal.
    for rtyp in [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.0, 1.4, 2.4, 1.55]:
        assert _endf.decay_modes(rtyp) == endf.decay.get_decay_modes(rtyp)


def test_normalise_branch_ratios_matches():
    for ratios in [[1.0], [0.5, 0.5], [2.0, 3.0, 5.0], [0.0, 0.0]]:
        reference = list(ratios)
        endf.chain.normalise_branch_ratios(reference)
        assert _endf.normalise_branch_ratios(ratios) == pytest.approx(reference)


def test_module_tables_match():
    assert _endf.ATOMIC_SYMBOL == {
        z: s for z, s in endf.data.ATOMIC_SYMBOL.items() if isinstance(z, int)
    }
    assert _endf.SUM_RULES == endf.data.SUM_RULES
    assert _endf.INTERPOLATION_SCHEME == endf.function.INTERPOLATION_SCHEME
    assert tuple(_endf.FISSION_MTS) == tuple(endf.reaction.FISSION_MTS)
    assert _endf.EV_PER_MEV == endf.data.EV_PER_MEV
    assert _endf.K_BOLTZMANN == endf.data.K_BOLTZMANN


#: The metastable decay evaluations the fixtures carry, which is what an isomer
#: table is built from — ground states are implicit.
METASTABLE_DECAY = [
    "dec-049_In_116m1.endf.xz",
    "dec-049_In_116m2.endf.xz",
]


def test_radionuclide_production_matches():
    # In-115 is the fixture with MF=9 and MF=10 production data.
    name = "n-049_In-115_trimmed.endf.xz"
    reference = endf.radionuclide_production(endf.Material(fixture(name)))
    got = _endf.radionuclide_production(_endf.Material(fixture(name)))

    assert sorted(got) == sorted(reference)
    for mt, states in reference.items():
        assert len(got[mt]) == len(states)
        for have, want in zip(got[mt], states):
            assert have.ZAP == want.ZAP
            assert have.LFS == want.LFS
            assert have.QM == pytest.approx(want.QM, rel=1e-15)
            assert have.QI == pytest.approx(want.QI, rel=1e-15)
            assert have.ELFS == want.ELFS
            assert have.excitation_energy == pytest.approx(
                want.excitation_energy, rel=1e-15
            )
            for got_tab, want_tab in [
                (have.yields, want.yields),
                (have.cross_section, want.cross_section),
            ]:
                assert (got_tab is None) == (want_tab is None)
                if want_tab is not None:
                    assert list(got_tab.x) == list(want_tab.x)
                    assert list(got_tab.y) == list(want_tab.y)


def test_isomer_table_matches():
    files = [fixture(name) for name in METASTABLE_DECAY]
    assert _endf.isomer_table(files) == endf.isomer_table(files)


def test_level_to_isomeric_state_matches():
    table = endf.isomer_table([fixture(name) for name in METASTABLE_DECAY])
    cases = [
        (49, 116, 0, 0.0, 3000.0),
        (49, 116, 1, 0.0, 3000.0),
        (49, 116, 1, 127300.0, 3000.0),
        (49, 116, 4, 162393.0, 3000.0),
        (49, 116, 4, 9.0e5, 10.0),
        # A nuclide with no isomers in the table maps to ground.
        (26, 56, 1, 1.0e5, 3000.0),
    ]
    for z, a, lfs, energy, tol in cases:
        assert _endf.level_to_isomeric_state(
            z, a, lfs, energy, table, tol_eV=tol
        ) == endf.level_to_isomeric_state(z, a, lfs, energy, table, tol_eV=tol)


def test_interpret_picks_the_class_by_sublibrary(am244, rust_am244):
    # NSUB=10, an incident-neutron evaluation.
    assert isinstance(am244.interpret(), endf.IncidentNeutron)
    assert isinstance(rust_am244.interpret(), _endf.IncidentNeutron)
    assert rust_am244.interpret().name == am244.interpret().name

    # NSUB=3, photoatomic.
    reference = endf.Material(fixture("photoat-001_H_000.endf.xz"))
    material = _endf.Material(fixture("photoat-001_H_000.endf.xz"))
    assert isinstance(reference.interpret(), endf.IncidentPhoton)
    assert isinstance(material.interpret(), _endf.IncidentPhoton)
    assert material.interpret().atomic_number == reference.interpret().atomic_number

    # NSUB=12, thermal scattering, which neither reader has a class for.
    tsl = _endf.Material(fixture("tsl-s-CH4.endf.xz"))
    with pytest.raises(ValueError):
        tsl.interpret()
    with pytest.raises(NotImplementedError):
        endf.Material(fixture("tsl-s-CH4.endf.xz")).interpret()


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


def test_asking_for_a_section_that_is_not_there(rust_am244):
    with pytest.raises(ValueError, match="no section"):
        rust_am244[3, 999]


#: The files whose sections have no dictionary form in the extension.
#:
#: Empty: every section the fixtures contain has one. Kept, and asserted
#: against, so a projection that stops being built shows up as a failure here
#: rather than as a section quietly missing from `section_data`.
SECTIONS_WITHOUT_A_DICT = set()


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
