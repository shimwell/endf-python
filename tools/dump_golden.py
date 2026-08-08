# SPDX-License-Identifier: MIT
"""Dump golden files from the Python reader for the Rust port to check against.

The Rust crate is being ported file by file, and the only thing that makes that
safe is holding it to what the Python reader already produces. This writes that
reference out in a format the Rust test reads without pulling in a JSON
dependency: whitespace-separated tokens, one record per line.

Floats are written with ``repr``, the shortest string that round-trips to the
same double. Rust's float parser is correctly rounded, so it recovers
bit-identical values and the comparison is exact rather than approximate.

Every section is recorded with its line count, so the section splitter is
checked over the whole file even for the files that have no Rust parser yet.
Sections that do have one are recorded in full.

Regenerate every golden file:

    python tools/dump_golden.py

Add a fixture and regenerate only it:

    python tools/dump_golden.py tests/data/n-092_U_235.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

import endf

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "crates" / "endf" / "tests" / "golden"

#: Where ENDF fixtures are looked for when no argument is given. `tests/` holds
#: the original fixture; `tests/data/` is where the wider library and format
#: coverage goes.
FIXTURE_DIRS = [ROOT / "tests", ROOT / "tests" / "data"]

FIXTURE_SUFFIXES = {".endf", ".dat"}


def fmt(x) -> str:
    return repr(float(x))


#: How many interior sample points to record per tabulated function, on top of
#: the ones the region boundaries force. Enough to exercise each interpolation
#: law without the golden file growing to the size of the evaluation.
MAX_SAMPLES = 24


def sample_points(table) -> list[float]:
    """Points at which to pin the interpolation.

    Bin midpoints, because a tabulated point returns its own value whatever the
    interpolation law is and so proves nothing. The bins on either side of a
    region boundary are always included: picking the scheme for the bin a point
    falls in is the part of the evaluation most likely to be got wrong.
    """
    x = table.x
    if len(x) < 2:
        return list(map(float, x))

    def midpoint(i: int) -> float:
        return float(0.5 * (x[i] + x[i + 1]))

    wanted = set()
    # Either side of every region boundary.
    for b in table.breakpoints:
        for i in (int(b) - 2, int(b) - 1):
            if 0 <= i < len(x) - 1:
                wanted.add(i)
    # An even spread over the rest.
    n_bins = len(x) - 1
    step = max(1, n_bins // MAX_SAMPLES)
    wanted.update(range(0, n_bins, step))

    points = [midpoint(i) for i in sorted(wanted)]
    # Both ends, and just outside them, to pin the clamping behaviour.
    return [float(x[0]) * 0.5, float(x[0]), *points, float(x[-1]), float(x[-1]) * 2.0]


def dump_mf3(out, mt: int, section: dict) -> None:
    sigma = section["sigma"]
    out.write(
        "MF3 {} {} {} {} {} {} {} {}\n".format(
            mt,
            section["ZA"],
            fmt(section["AWR"]),
            fmt(section["QM"]),
            fmt(section["QI"]),
            section["LR"],
            len(sigma.x),
            len(sigma.breakpoints),
        )
    )
    out.write("BP " + " ".join(str(int(b)) for b in sigma.breakpoints) + "\n")
    out.write("INT " + " ".join(str(int(i)) for i in sigma.interpolation) + "\n")
    out.write("X " + " ".join(fmt(v) for v in sigma.x) + "\n")
    out.write("Y " + " ".join(fmt(v) for v in sigma.y) + "\n")

    points = sample_points(sigma)
    out.write("EVALX " + " ".join(fmt(p) for p in points) + "\n")
    out.write("EVALY " + " ".join(fmt(sigma(p)) for p in points) + "\n")


#: One entry per ENDF file the Rust crate can parse. Extend this as the port
#: proceeds so the golden files grow with it.
DUMPERS = {3: dump_mf3}


def dump(path: Path, out) -> None:
    materials = endf.get_materials(path)
    source = path.relative_to(ROOT).as_posix()

    out.write(f"# golden reference generated from {source} by the Python reader\n")
    out.write("# regenerate with: python tools/dump_golden.py\n")
    out.write(f"SOURCE {source}\n")
    out.write(f"MATERIALS {len(materials)}\n")

    for material in materials:
        out.write(f"MAT {material.MAT}\n")

        # Every (MF, MT) with the number of lines its body occupies. This holds
        # the section splitter to the Python reader across the whole file,
        # including the files nothing parses yet.
        for mf, mt in sorted(material.sections):
            n_lines = len(material.section_text[mf, mt].splitlines())
            out.write(f"SECTION {mf} {mt} {n_lines}\n")

        for mf, mt in sorted(material.sections):
            dumper = DUMPERS.get(mf)
            if dumper is not None:
                dumper(out, mt, material[mf, mt])


def fixtures() -> list[Path]:
    found = []
    for directory in FIXTURE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in FIXTURE_SUFFIXES:
                found.append(path)
    return found


def main() -> None:
    paths = [Path(a).resolve() for a in sys.argv[1:]] or fixtures()
    if not paths:
        sys.exit("no ENDF fixtures found; pass one explicitly")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = GOLDEN_DIR / f"{path.stem}.txt"
        with open(target, "w") as out:
            dump(path, out)
        print(f"{target.relative_to(ROOT)}  <-  {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
