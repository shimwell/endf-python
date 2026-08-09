# SPDX-License-Identifier: MIT
"""Type stubs for the `_endf` extension module.

The module is compiled, so without this file an editor and a type checker see
nothing at all in it. That matters more here than for most extensions, because
`_endf` is meant to be substitutable for the pure-Python `endf` package — and
swapping one for the other should not cost you every type you had.

The sum types deliberately come across as `kind`-tagged dictionaries rather
than as a class per variant, so they are typed as `dict[str, Any]`. The `kind`
key says which shape it is; `crates/README.md` lists them.

Kept in step with the module by `tests/test_rust_stub.py`, which compares this
file against what the built module actually exports.
"""

from typing import Any, Final

# ---------------------------------------------------------------------------
# Records and functions
# ---------------------------------------------------------------------------

def float_endf(s: str) -> float:
    """Read an ENDF float, including the `e`-less form `-1.23481+10`."""

def int_endf(s: str) -> int:
    """Read an ENDF integer, treating an all-blank field as zero."""

class Tabulated1D:
    """A tabulated function of one variable, with interpolation regions."""

    def __init__(
        self,
        x: list[float],
        y: list[float],
        breakpoints: list[int] | None = ...,
        interpolation: list[int] | None = ...,
    ) -> None: ...
    @property
    def x(self) -> list[float]: ...
    @property
    def y(self) -> list[float]: ...
    @property
    def breakpoints(self) -> list[int]: ...
    @property
    def interpolation(self) -> list[int]: ...
    @property
    def n_pairs(self) -> int: ...
    @property
    def n_regions(self) -> int: ...
    def integral(self) -> list[float]:
        """The running integral at each tabulated point."""

    def __call__(self, x: float | list[float]) -> Any:
        """Evaluate at a point, or at every point of a sequence."""

    def __len__(self) -> int: ...

class Tabulated2D:
    """The interpolation across the outer variable of a two-dimensional table."""

    @property
    def breakpoints(self) -> list[int]: ...
    @property
    def interpolation(self) -> list[int]: ...

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class Material:
    """One material of an ENDF-6 file, split into its (MF, MT) sections."""

    def __init__(self, filename: str) -> None:
        """Read the first material of a file. A `.xz` path is decompressed."""

    @staticmethod
    def from_string(text: str) -> Material: ...
    @property
    def MAT(self) -> int: ...
    @property
    def sections(self) -> list[tuple[int, int]]: ...
    @property
    def section_text(self) -> dict[tuple[int, int], str]: ...
    @property
    def section_data(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Every section as the dictionary the Python reader builds."""

    def interpret(self) -> IncidentNeutron | IncidentPhoton:
        """The high-level class this material's sublibrary calls for."""

    def mf3(self, mt: int) -> CrossSection | None: ...
    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]: ...
    def __contains__(self, key: tuple[int, int]) -> bool: ...

def get_materials(filename: str) -> list[Material]:
    """Every material in an ENDF-6 file."""

class CrossSection:
    """An MF=3 reaction cross section."""

    @property
    def ZA(self) -> int: ...
    @property
    def AWR(self) -> float: ...
    @property
    def QM(self) -> float: ...
    @property
    def QI(self) -> float: ...
    @property
    def LR(self) -> int: ...
    @property
    def sigma(self) -> Tabulated1D: ...

# ---------------------------------------------------------------------------
# Reactions and nuclides
# ---------------------------------------------------------------------------

class Product:
    """One product of a reaction, with its yield and distribution."""

    @property
    def name(self) -> str: ...
    @property
    def emission_mode(self) -> str: ...
    @property
    def decay_rate(self) -> float: ...
    @property
    def yield_(self) -> dict[str, Any]: ...
    @property
    def applicability(self) -> list[Tabulated1D]: ...
    @property
    def distribution(self) -> list[dict[str, Any]]:
        """The angle-energy distributions, each tagged with a `kind` key."""

    def yield_at(self, energy: float) -> float: ...

class Reaction:
    """One reaction, gathered from every file that describes it."""

    @property
    def MT(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def q_reaction(self) -> float: ...
    @property
    def q_massdiff(self) -> float: ...
    @property
    def center_of_mass(self) -> bool: ...
    @property
    def redundant(self) -> bool: ...
    @property
    def xs(self) -> dict[str, Tabulated1D]:
        """Cross sections by temperature, e.g. `"294K"`."""

    @property
    def products(self) -> list[Product]: ...
    @property
    def derived_products(self) -> list[Product]: ...

class IncidentNeutron:
    """Incident-neutron data for one nuclide."""

    @staticmethod
    def from_endf(material: Material) -> IncidentNeutron: ...
    @staticmethod
    def from_ace(
        table: AceTable, metastable_scheme: str = "mcnp"
    ) -> IncidentNeutron: ...
    def add_temperature_from_ace(
        self, table: AceTable, metastable_scheme: str = "mcnp"
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def atomic_number(self) -> int: ...
    @property
    def atomic_symbol(self) -> str: ...
    @property
    def mass_number(self) -> int: ...
    @property
    def metastable(self) -> int: ...
    @property
    def atomic_weight_ratio(self) -> float | None: ...
    @property
    def reactions(self) -> dict[int, Reaction]: ...
    @property
    def energy(self) -> dict[str, list[float]]:
        """Unionised energy grids by temperature. Empty for ENDF data, which
        gives each cross section its own grid."""

    @property
    def temperatures(self) -> list[str]: ...
    @property
    def kTs(self) -> list[float]: ...
    @property
    def urr(self) -> dict[str, dict[str, Any]]:
        """Unresolved resonance probability tables by temperature."""

    def reaction_components(self, mt: int) -> list[int]: ...
    def removal_xs(
        self, temperature: str = "0K", mu_cutoff: float = 0.0
    ) -> Tabulated1D: ...
    def __getitem__(self, mt: int) -> Reaction: ...
    def __contains__(self, mt: int) -> bool: ...

class IncidentPhoton:
    """Photoatomic data for one element."""

    @staticmethod
    def from_endf(
        photoatomic: Material, relaxation: Material | None = ...
    ) -> IncidentPhoton: ...
    @staticmethod
    def from_ace(table: AceTable) -> IncidentPhoton: ...
    @property
    def name(self) -> str: ...
    @property
    def atomic_number(self) -> int: ...
    @property
    def reactions(self) -> dict[int, dict[str, Any]]: ...
    @property
    def atomic_relaxation(self) -> dict[str, Any] | None: ...
    def reaction_components(self, mt: int) -> list[int]: ...

def reaction_name(mt: int) -> str | None:
    """The name of a reaction, e.g. `"(n,2n)"` for MT=16."""

def reaction_mt(name: str) -> int | None:
    """The MT of a named reaction, by its own name or an alias."""

def photon_reaction_name(mt: int) -> str | None: ...
def photon_reaction_mt(name: str) -> int | None: ...

# ---------------------------------------------------------------------------
# Decay, yields and production
# ---------------------------------------------------------------------------

class Decay:
    """Radioactive decay data for one nuclide."""

    @staticmethod
    def from_endf(material: Material) -> Decay: ...
    @property
    def nuclide(self) -> dict[str, Any]: ...
    @property
    def half_life(self) -> tuple[float, float] | None: ...
    @property
    def decay_constant(self) -> tuple[float, float] | None: ...
    @property
    def decay_energy(self) -> tuple[float, float]: ...
    @property
    def average_energies(self) -> dict[str, tuple[float, float]]: ...
    @property
    def modes(self) -> list[dict[str, Any]]: ...
    @property
    def sources(self) -> dict[str, dict[str, Any]]:
        """Emission spectra by radiation type, each a `kind`-tagged dict."""

class FissionProductYields:
    """Independent and cumulative fission product yields."""

    def __init__(self, filename: str) -> None: ...
    @staticmethod
    def from_material(material: Material) -> FissionProductYields: ...
    @property
    def nuclide(self) -> dict[str, Any]: ...
    @property
    def energies(self) -> list[float] | None: ...
    @property
    def independent(self) -> list[dict[str, tuple[float, float]]]: ...
    @property
    def cumulative(self) -> list[dict[str, tuple[float, float]]]: ...

class RadionuclideProduction:
    """Production data for a single final state of one reaction."""

    @property
    def ZAP(self) -> int: ...
    @property
    def LFS(self) -> int: ...
    @property
    def QM(self) -> float: ...
    @property
    def QI(self) -> float: ...
    @property
    def ELFS(self) -> float | None: ...
    @property
    def yields(self) -> Tabulated1D | None: ...
    @property
    def cross_section(self) -> Tabulated1D | None: ...
    @property
    def excitation_energy(self) -> float: ...

def radionuclide_production(
    material: Material,
) -> dict[int, list[RadionuclideProduction]]:
    """A material's MF=8/9/10 production data, keyed by MT."""

def isomer_table(decay_files: list[str]) -> dict[tuple[int, int], dict[int, dict[str, Any]]]:
    """Isomeric states by (Z, A), then by isomeric-state ordinal."""

def level_to_isomeric_state(
    Z: int,
    A: int,
    lfs: int,
    excitation_energy: float | None,
    table: dict[tuple[int, int], dict[int, dict[str, Any]]],
    *,
    tol_eV: float = 3000.0,
) -> int:
    """Map an MF=8 production level to an isomeric-state ordinal (LISO)."""

class Chain:
    """A depletion chain: decay, fission yields and transmutation joined."""

    @staticmethod
    def from_endf(
        decay: list[Material],
        fpy: list[Material],
        neutron: list[Material],
        reactions: list[str] | None = ...,
    ) -> Chain: ...
    @property
    def nuclides(self) -> list[dict[str, Any]]: ...
    def reduce(self, initial: list[str], level: int | None = ...) -> Chain: ...
    def validate(self, tolerance: float = 1e-4) -> bool: ...

def decay_modes(rtyp: float) -> list[str]:
    """The decay modes an ENDF RTYP names, in order."""

def normalise_branch_ratios(ratios: list[float]) -> list[float]: ...

# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------

class AceTable:
    """One ACE Type 1 table."""

    @property
    def name(self) -> str: ...
    @property
    def zaid(self) -> int: ...
    @property
    def data_type(self) -> str: ...
    @property
    def atomic_weight_ratio(self) -> float: ...
    @property
    def temperature(self) -> float: ...
    @property
    def kT(self) -> float: ...
    @property
    def nxs(self) -> list[int]: ...
    @property
    def jxs(self) -> list[int]: ...
    @property
    def xss(self) -> list[float]:
        """The data array, padded at index 0 so the format's 1-based
        offsets in `nxs` and `jxs` index it directly."""

def get_tables(filename: str) -> list[AceTable]:
    """Every table in an ACE file. A `.xz` path is decompressed."""

def ace_tables_from_string(text: str) -> list[AceTable]: ...

# ---------------------------------------------------------------------------
# Names, constants and conversions
# ---------------------------------------------------------------------------

def gnds_name(z: int, a: int, m: int = 0) -> str:
    """A nuclide's GNDS name, e.g. `gnds_name(95, 242, 1)` is `"Am242_m1"`."""

def zam(name: str) -> tuple[int, int, int]:
    """The (Z, A, metastable state) a GNDS name denotes."""

def temperature_str(t: float) -> str:
    """A temperature in kelvin as the string ACE and HDF5 libraries key on."""

ATOMIC_SYMBOL: Final[dict[int, str]]
SUM_RULES: Final[dict[int, list[int]]]
INTERPOLATION_SCHEME: Final[dict[int, str]]
FISSION_MTS: Final[list[int]]
EV_PER_MEV: Final[float]
K_BOLTZMANN: Final[float]
