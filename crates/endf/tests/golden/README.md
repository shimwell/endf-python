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

On top of the raw arrays, the ACE dump walks the blocks a transport code
reads: every locator in AND (the angular distributions), the whole of DLW (the
joint angle-energy distributions, following the linked list each reaction
carries), and every reaction the table holds, elastic scattering included.
That turns the ACE fixtures into a check on the interpretation, not only on the
numbers.

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
| `production/…`, `reaction/…` | The derived views: the MF=8/9/10 join, and every reaction gathered from its files |
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

The distribution shapes are tracked the same way, in `DISTRIBUTION_SHAPES`:
every angular, energy and joint angle-energy shape the dumpers can write has to
appear in some golden file, or the test names the one that does not. Where no
real table small enough to keep as a fixture holds a shape, one is built —
`tools/make_urr_ace.py` and `tools/make_laws_ace.py` write synthetic tables
whose values are invented but whose layout is the format's.

Trimming a fixture down to the sections that matter is what `tools/trim_endf.py`
is for. A full evaluation runs to tens of megabytes, most of it covariance
data — U235 is 36 MB whole and 451 KB with ten sections kept.

### Fixtures present

| Fixture | Covers |
|---|---|
| `n-095_Am_244` | MF1 (incl. MT458), MF2 LRF=0, MF3, MF4, MF5 LF=7 |
| `n-095_Am_242_trimmed` | MF1, a metastable target |
| `n-049_In-115_trimmed` | MF3, MF8, MF9, MF10 — isomer production |
| `n-054_Xe_136_trimmed` | MF1, MF3 |
| `n-003_Li_006_trimmed` | MF6 LAW=2 and LAW=4, MF12, MF14, MF33 |
| `n-026_Fe_056_trimmed` | MF2 Reich-Moore, MF6 LAW=1, MF12/14, MF33 |
| `n-092_U_235_trimmed` | MF2 Reich-Moore + Case C URR, MF5 LF=5, MF8, MF10, MF15, MF34, delayed neutron groups |
| `photoat-001_H_000` | MF23, MF27 |
| `atom-001_H_000` | MF28 |
| `e-001_H_000` | MF23, MF26 in all three laws |
| `tsl-s-CH4` | MF7 MT=2 and MT=4 |
| `dec-049_In_116m1` | MF8 MT=457 decay data: four spectra, beta- only |
| `dec-049_In_116m2` | MF8 MT=457 decay data: an isomeric transition down to m1 |
| `Li6.ace` | An ACE Type 1 table; AND in all three shapes, DLW laws 3, 33 and 44, 15 reactions with photon production |
| `synthetic-urr.ace` | The unresolved resonance block, which no small real table has |
| `synthetic-laws.ace` | DLW laws 2, 4, 7, 9, 11, 61 and 66 |

### Fixtures still wanted

- **MF13 and MF40**, the two parsers nothing exercises. MF13 appears in
  evaluations that give photon production as a cross section rather than a
  multiplicity; MF40 needs an evaluation with radionuclide production
  covariances.
- **MF6 with LANG=2 (Kalbach-Mann) and LAW=6 (n-body)**. LAW=1/LANG=1, LAW=2 and
  LAW=4 are covered, but these two are not, and they are among the shapes an
  Arrow projection needs columns for.
- **MF2 formalisms other than Reich-Moore**: single-level and multi-level
  Breit-Wigner, R-matrix limited (LRF=7), and unresolved Cases A and B. Note
  that Cases A and B cannot be reached at all through the current dispatch —
  see issue #15 — so a fixture alone will not cover them.
- **Adler-Adler (LRF=4)**, which both readers reject rather than parse. A
  fixture would only pin that rejection.
- **MF8 MT=454 and MT=459**, the fission product yields. MT=457 is now
  covered by the two In116 decay fixtures; the yields are not.
- **MF5 LF=12 (Madland-Nix)**. LF=1, LF=5, LF=7 and LF=9 are covered by Li6,
  Am244 and U235; Madland-Nix has no fixture. The ACE side is complete apart
  from law 5, which the Python reader cannot read at all — see issue #19.
- **A fissile ACE table.** Li6 has no NU block, so the ACE fission path —
  prompt and total nu, the delayed groups and their probabilities — is
  unexercised. So is the URR block on a real table, and MFTYPE=13 photon
  production.
- **A delayed neutron group whose applicability varies with energy.** U235
  gives each group a constant share, which is the usual case; the branch that
  takes the product on the union of two grids is unexercised.
- **Other libraries.** Everything here is ENDF/B-VIII.0 except the ACE table,
  which is TENDL-2023.1. JEFF-4.0, JENDL-5 and TENDL-2025 differ in which
  optional records they write and how strictly they follow the format, which is
  exactly what a format reader gets wrong.

## A note on size

Fixtures are trimmed with `tools/trim_endf.py` rather than added whole; a full
evaluation with covariances runs to tens of megabytes. If one ever has to be
added whole, the dump format should grow a digest record — a hash over a table
rather than its values — so that a large evaluation costs a line instead of a
megabyte. Small fixtures should stay fully enumerated: an exact diff points at
the value that broke, a hash only says something did.
