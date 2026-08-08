# Rust layer

An in-progress port of the reader to Rust, so that the same parser serves
consumers that never load Python — in particular a converter emitting Arrow
tables directly.

```
crates/
├── endf/      the parser. No Arrow, no Python, no dependencies at all.
└── endf-py/   PyO3 bindings. Thin: every type forwards to the Rust one.
```

## Why two crates

`endf` describes the ENDF-6 format and nothing else. A simulation-ready
projection of the data — reconstructed resonances, summed reactions, unionised
grids, an Arrow schema — is a consumer's concern and belongs in the consumer,
which is free to depend on `arrow-rs` without that cost reaching everyone who
just wants to read a file.

`endf-py` exists so the Python package keeps its API while the parser moves
underneath it.

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
