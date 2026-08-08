# Golden files

Each `.txt` here is a reference dump of what the **Python** reader produces for
one ENDF evaluation. `tests/golden.rs` reads every one of them, runs the Rust
reader over the evaluation the `SOURCE` line names, and compares.

This is what makes the port safe: the Rust crate is not being written against
the ENDF-102 manual alone, it is being held to a reader that already works.

## ACE fixtures

A `.ace` fixture is dumped differently — ACE has tables, not materials — so its
golden opens with `KIND ace` and the Rust side reads it through
`endf::ace` instead. The XSS array runs to hundreds of thousands of numbers, so
the dump records a spread across it plus both ends and every JXS entry point,
which is where a consumer actually looks. Everything else is recorded in full.

## Adding an evaluation

1. Drop the file in `tests/data/`.
2. `python tools/dump_golden.py` (or pass the one path to regenerate just it).
3. `cargo test -p endf`.

Nothing in the Rust test needs changing — it discovers golden files and follows
`SOURCE`.

## What is compared

| Record | Covers |
|---|---|
| `MATERIALS`, `MAT` | Multi-material files, material numbers |
| `SECTION mf mt n` | Section splitting, for **every** MF including unported ones |
| `MF3 …`, `BP`, `INT`, `X`, `Y` | Parsed values, compared **exactly** |
| `EVALX` / `EVALY` | Interpolation, compared to 1e-12 relative |

Values are written as the shortest round-tripping decimal and both readers parse
decimals with correct rounding, so parsed values are compared bit-for-bit. Only
the interpolation samples use a tolerance, because logs and exps need not round
identically in the two languages.

The `SECTION` lines matter more than they look: they hold the section splitter
to the Python reader across files that have no Rust parser yet, so a new
evaluation is useful coverage the day it is added, long before every MF is
ported.

## Coverage still wanted

Every ENDF file the Python package parses now has a Rust parser, and all but two
are exercised by a fixture. The exceptions are

    MF 13 (photon production cross sections) and MF 40 (radionuclide
    production covariances)

which are structurally complete and entirely unverified. The list is pinned in
`golden.rs` as `UNCOVERED_BY_ANY_FIXTURE` and checked, so it cannot drift in
either direction: the test fails both when a fixture starts covering one, and
when a new parser arrives without coverage.

`MF2` is worth a line of its own. It now has real resonance parameters —
Reich-Moore from Fe56 and U235, and a Case C unresolved region from U235 — but
single-level and multi-level Breit-Wigner, Adler-Adler, R-matrix limited (LRF=7)
and unresolved Cases A and B are still untested. Cases A and B are additionally
unreachable through the current dispatch; see issue #15.

Trimming a fixture down to the sections that matter is what `tools/trim_endf.py`
is for. A full evaluation runs to tens of megabytes, most of it covariance
data — U235 is 36 MB whole and 451 KB with ten sections kept.

The remaining gaps below are what the Arrow conversion depends on, in roughly
the order it needs them.

### By format feature

- [ ] **MF1** nu-bar in both forms — polynomial (LNU=1) and tabulated (LNU=2);
      delayed nu and decay constants (MT455); fission energy release (MT458) in
      both polynomial and tabulated form. U235, U238 and Pu239 carry a tabulated
      prompt term with a polynomial delayed one in the same evaluation.
- [ ] **MF2** every resonance formalism: SLBW, MLBW, Reich-Moore, Adler-Adler,
      R-matrix limited (LRF=7), and unresolved resonances (LRU=2) both
      energy-dependent and not.
- [ ] **MF4** LTT=1 Legendre, LTT=2 tabulated, LTT=3 mixed.
- [ ] **MF5** the energy-distribution laws: LF=1, 5, 7, 9, 11, 12.
- [ ] **MF6** LAW=1 with LANG=1 (Legendre), LANG=2 (Kalbach-Mann) and LANG=11–15
      (tabulated), plus LAW=2, 3, 4, 6 (n-body) and 7. These are the four
      distribution shapes the Arrow `distributions` table has columns for.
- [ ] **MF7** thermal scattering: MT2 coherent and incoherent elastic, MT4
      inelastic S(a,b). Needs a `tsl-` file.
- [ ] **MF8** MT457 decay data, MT454/459 fission yields. Needs `decay-` and
      `nfy-` sublibrary files — these feed the transmutation tables.
- [ ] **MF9/MF10** isomer production, which is where isomeric branching ratios
      come from.
- [ ] **MF23/MF27** photo-atomic — needs a `photoat-` file. Fe is the one the
      converter's own tests use.
- [ ] **MF26/MF28** electro-atomic and atomic relaxation.
- [ ] **MF33/34/40** covariances, which are the only user of the INTG record.

### By nuclide

- [ ] H1 — trivial, no resonances; the degenerate case.
- [ ] Li6 — light, with MT=105; already a fixture in the converter.
- [ ] Fe56 — dense resolved resonances and large covariances.
- [ ] U235 / U238 / Pu239 — fission, delayed neutrons, yields, energy release.
- [ ] A metastable target (Am242m) — the `_m1` naming path.

### By library

The same nuclide from each of ENDF/B-VIII.1, JEFF-4.0, JENDL-5 and TENDL-2025.
Libraries differ in which optional records they write and in how strictly they
follow the format, and that is exactly what a format reader gets wrong.

## A note on size

Full evaluations with covariances run to tens of megabytes, and a golden file
that lists every value would be no smaller. Before adding those, the dump format
should grow a digest record — a hash over a table rather than its values — so
large evaluations cost a line instead of a megabyte. Small evaluations should
stay fully enumerated: an exact diff points at the value that broke, a hash only
says something did.
