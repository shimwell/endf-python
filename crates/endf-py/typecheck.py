# SPDX-License-Identifier: MIT
"""A type checker's view of `_endf`, exercised under `mypy --strict`.

`tests/test_rust_stub.py` compares the names the stub declares against the
names the module exports, which catches a method that was added and never
written down. It cannot catch a name that is present with the wrong type —
that needs a type checker, and a type checker needs something to check.

So this is that something: ordinary use of the module with every result bound
to an explicit annotation. Under `--strict` a wrong return type in the stub
makes the assignment fail. Nothing here runs; the paths are not opened.

    mypy --strict crates/endf-py/typecheck.py
"""

from __future__ import annotations

from typing import Any

import _endf


def records() -> None:
    value: float = _endf.float_endf(" 1.234567+8")
    count: int = _endf.int_endf("   9552")
    assert value or count

    table = _endf.Tabulated1D([1.0, 2.0], [3.0, 4.0])
    x: list[float] = table.x
    y: list[float] = table.y
    breakpoints: list[int] = table.breakpoints
    regions: int = table.n_regions
    running: list[float] = table.integral()
    assert x and y and breakpoints and regions and running


def materials() -> None:
    material = _endf.Material("n-095_Am_244.endf.xz")
    mat: int = material.MAT
    sections: list[tuple[int, int]] = material.sections
    text: dict[tuple[int, int], str] = material.section_text
    data: dict[tuple[int, int], dict[str, Any]] = material.section_data
    one: dict[str, Any] = material[3, 1]
    present: bool = (3, 1) in material
    assert mat and sections and text and data and one and present

    # `mf3` is optional, and a checker should make the caller say so.
    cross_section = material.mf3(102)
    if cross_section is not None:
        qm: float = cross_section.QM
        sigma: _endf.Tabulated1D = cross_section.sigma
        assert qm and sigma

    everything: list[_endf.Material] = _endf.get_materials("file.endf")
    assert everything


def nuclides() -> None:
    material = _endf.Material("n-095_Am_244.endf.xz")
    neutron = _endf.IncidentNeutron.from_endf(material)

    name: str = neutron.name
    z: int = neutron.atomic_number
    a: int = neutron.mass_number
    symbol: str = neutron.atomic_symbol
    reactions: dict[int, _endf.Reaction] = neutron.reactions
    temperatures: list[str] = neutron.temperatures
    grids: dict[str, list[float]] = neutron.energy
    assert name and z and a and symbol and reactions and temperatures and grids

    # Optional, because an ACE table has it and an ENDF evaluation may not.
    ratio = neutron.atomic_weight_ratio
    if ratio is not None:
        scaled: float = ratio * 2.0
        assert scaled

    reaction: _endf.Reaction = neutron[102]
    mt: int = reaction.MT
    q: float = reaction.q_reaction
    cm: bool = reaction.center_of_mass
    by_temperature: dict[str, _endf.Tabulated1D] = reaction.xs
    products: list[_endf.Product] = reaction.products
    assert mt and q and cm is not None and by_temperature and products

    for product in products:
        product_name: str = product.name
        mode: str = product.emission_mode
        at: float = product.yield_at(1.0e6)
        # The sum types are `kind`-tagged dictionaries, not classes.
        distributions: list[dict[str, Any]] = product.distribution
        assert product_name and mode and at and distributions


def ace() -> None:
    tables: list[_endf.AceTable] = _endf.get_tables("Li6.ace.xz")
    table = tables[0]
    zaid: int = table.zaid
    kt: float = table.kT
    nxs: list[int] = table.nxs
    xss: list[float] = table.xss
    assert zaid and kt and nxs and xss

    neutron = _endf.IncidentNeutron.from_ace(table)
    neutron.add_temperature_from_ace(table, "nndc")
    removal: _endf.Tabulated1D = neutron.removal_xs("294K", 0.0)
    components: list[int] = neutron.reaction_components(1)
    assert removal and components


def depletion() -> None:
    material = _endf.Material("dec-049_In_116m1.endf.xz")
    decay = _endf.Decay.from_endf(material)
    nuclide: dict[str, Any] = decay.nuclide
    modes: list[dict[str, Any]] = decay.modes
    assert nuclide and modes

    half_life = decay.half_life
    if half_life is not None:
        value, uncertainty = half_life
        total: float = value + uncertainty
        assert total

    yields = _endf.FissionProductYields("synthetic-nfy.endf.xz")
    independent: list[dict[str, tuple[float, float]]] = yields.independent
    assert independent
    energies = yields.energies
    if energies is not None:
        first: float = energies[0]
        assert first

    production: dict[int, list[_endf.RadionuclideProduction]] = (
        _endf.radionuclide_production(material)
    )
    for states in production.values():
        for state in states:
            zap: int = state.ZAP
            excitation: float = state.excitation_energy
            assert zap and excitation


def names_and_tables() -> None:
    gnds: str = _endf.gnds_name(95, 242, 1)
    z, a, m = _endf.zam("Am242_m1")
    temperature: str = _endf.temperature_str(293.6)
    assert gnds and z and a and m is not None and temperature

    # Optional: an unknown MT has no name.
    reaction = _endf.reaction_name(16)
    if reaction is not None:
        upper: str = reaction.upper()
        assert upper

    symbols: dict[int, str] = _endf.ATOMIC_SYMBOL
    rules: dict[int, list[int]] = _endf.SUM_RULES
    schemes: dict[int, str] = _endf.INTERPOLATION_SCHEME
    fission: list[int] = _endf.FISSION_MTS
    ev: float = _endf.EV_PER_MEV
    k: float = _endf.K_BOLTZMANN
    assert symbols and rules and schemes and fission and ev and k
