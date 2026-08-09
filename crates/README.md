# Rust layer

An in-progress port of the reader to Rust, so that the same parser serves
consumers that never load Python — in particular a converter emitting Arrow
tables directly.

```
crates/
├── endf/      the parser. No Arrow, no Python, no dependencies at all.
└── endf-py/   PyO3 bindings. Thin: every type forwards to the Rust one.
```

The `endf` crate has no dependencies. Its tests have one: the fixtures and the
golden dumps are stored xz-compressed, and `lzma-rs` (pure Rust) reads them.
That is a `[dev-dependencies]` entry, so it is not built for anything that uses
the crate.

## Why two crates

`endf` describes the ENDF-6 format and nothing else. A simulation-ready
projection of the data — reconstructed resonances, summed reactions, unionised
grids, an Arrow schema — is a consumer's concern and belongs in the consumer,
which is free to depend on `arrow-rs` without that cost reaching everyone who
just wants to read a file.

`endf-py` exists so the Python package keeps its API while the parser moves
underneath it.

## The Python surface

The concrete types come across as classes — `Material`, `Tabulated1D`,
`Tabulated2D`, `CrossSection`, `Product`, `Reaction`, `IncidentNeutron`,
`IncidentPhoton`, `Decay`, `Chain`, `AceTable`, `RadionuclideProduction` — with
the free functions beside them: `float_endf`, `int_endf`, `get_materials`,
`get_tables`, `ace_tables_from_string`, `reaction_name`, `reaction_mt`,
`photon_reaction_name`, `photon_reaction_mt`, `gnds_name`, `zam`,
`temperature_str`, `decay_modes`, `normalise_branch_ratios`,
`radionuclide_production`, `isomer_table` and `level_to_isomeric_state`, and the
constant tables `ATOMIC_SYMBOL`, `SUM_RULES`, `INTERPOLATION_SCHEME`,
`FISSION_MTS`, `EV_PER_MEV` and `K_BOLTZMANN`.

`Material.interpret()` picks the class the material's sublibrary calls for, as
it does upstream: an `IncidentNeutron` for NSUB=10, an `IncidentPhoton` for
NSUB=3, and an error for anything else.

The sum types do not. An angle-energy distribution, an outgoing energy law and
a univariate density come across as dicts tagged with a `kind` key:

```python
>>> rx = neutron[51]
>>> rx.products[0].distribution[0]
{'kind': 'uncorrelated',
 'angle': {'energy': [...], 'mu': [{'kind': 'legendre', 'coefficients': [...]}, ...]},
 'energy': {'kind': 'level-inelastic', 'threshold': 1.4e6, 'mass_ratio': 0.98}}
```

That is the shape a consumer wants anyway — `kind` is exactly the discriminant
an Arrow union column needs — and it saves a wrapper class per variant for no
gain in what can be expressed.

`Material.section_data` and `material[3, 1]` are there too, giving back the
same dictionaries the Python reader does, keyed by the same ENDF field names.
Every one of the 400 sections across the fixtures has one — MF 1, 2, 3, 4, 5, 6,
7, 8, 9, 10, 12, 13, 14, 15, 23, 26, 27, 28, 33, 34 and 40 — so code written
against `Material.section_data` runs unchanged against either reader. That is
held by `tests/test_rust_bindings.py`, which compares the two dictionaries key
by key and pins the set of sections without one, currently empty, so a
projection that stops being built fails rather than silently disappearing.

The upstream quirks come with them, because matching means matching: MT=458
reports `ZA` as a float since it is read from a CONT record, MF=7 MT=4 stores
the outer `LT` on each additional temperature rather than the `LI` it read, an
unresolved range with LRF=1 is dispatched past unread, and a decay record too
short for its internal conversion coefficients yields an empty tuple rather
than a zero. Each is commented at its site with the issue that tracks it.

A section with no dictionary form is left out of `section_data` rather than
half-built, and asking for it by key says so.

Paths ending in `.xz` are decompressed, matching `endf.fileutils.open_text`, so
a path that works in one reader works in the other.

Build it with:

```sh
maturin develop -m crates/endf-py/Cargo.toml
```

`tests/test_rust_bindings.py` then runs; it compares the extension against the
pure-Python reader on the same fixtures rather than against values written down
by hand, and skips itself when the module is not built.

## State

| | |
|---|---|
| Records | TEXT, CONT, HEAD, LIST, TAB1, TAB2, INTG |
| Functions | `Tabulated1D` (all five interpolation laws, and integrals), `Tabulated2D`, `Polynomial` |
| Materials | Section splitting for every MF/MT, multi-material files |
| Files | MF3 |

Every other MF splits correctly and keeps its text as
`Section::Unparsed`, so the Python reader can still handle it. That is what
makes the port incremental: the two readers run side by side, file by file,
rather than the Rust one having to be finished before it is useful.

## Porting a file

1. Add `crates/endf/src/mf/mfN.rs` with a struct and
   `parse_mfN(&mut Reader) -> Result<MfN>`.
2. Add a variant to `Section` and an arm to `parse_section` in `material.rs`.
3. Add a dumper to `DUMPERS` in `tools/dump_golden.py`.
4. `python tools/dump_golden.py && cargo test`.

Step 3 is the point. The Rust reader is not being written against the format
manual alone — it is held to what the Python reader already produces, value for
value. See `crates/endf/tests/golden/README.md`.

## Building

```sh
cargo test                 # parser and parity tests
cargo clippy --all-targets
maturin develop -m crates/endf-py/Cargo.toml   # the extension module
```
