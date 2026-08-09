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
`CrossSection`, `Product`, `Reaction`, `IncidentNeutron`, `IncidentPhoton`,
`Decay`, `Chain`, `AceTable` — with `float_endf`, `int_endf`, `get_materials`,
`get_tables`, `reaction_name`, `reaction_mt` and `gnds_name` beside them.

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
93% of the sections in the fixtures have one — MF 1, 3, 4, 5, 8, 9, 10, 12, 13,
14, 15, 23, 27 and 28. A section with no dictionary form is left out of
`section_data` rather than half-built, and asking for it by key says so.

What is left: MF 2, 6, 7, 26, 33 and 34 have no dictionary yet, and MF=8
MT=457 does not have one on purpose — decay data is reached through `Decay`,
which is a better shape. The set is pinned in
`tests/test_rust_bindings.py::SECTIONS_WITHOUT_A_DICT` and asserted, so it
cannot drift in either direction.

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
