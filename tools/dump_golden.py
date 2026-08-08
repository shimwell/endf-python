# SPDX-License-Identifier: MIT
"""Dump golden files from the Python reader for the Rust port to check against.

The Rust crate is being ported file by file, and the only thing that makes that
safe is holding it to what the Python reader already produces. This writes that
reference out in a format the Rust test reads without pulling in a JSON
dependency.

Every value is emitted as one line::

    V <path> <F|I|T> <values...>

where the path names the value — ``0/3/1/sigma/x`` is the MF=3 MT=1 cross
section abscissae of the first material — and the tag says how to read the rest:
``F`` floats, ``I`` integers, ``T`` a hex-encoded string. Strings are hex
because ENDF text fields are fixed-width and carry significant spaces, which a
whitespace-separated format would eat.

The Rust test builds the same map from its own parse and compares the two whole.
A path present on one side and not the other fails, so a field that is renamed,
dropped or added is caught as loudly as one whose value is wrong.

Floats are written with ``repr``, the shortest string that round-trips to the
same double. Rust's float parser is correctly rounded, so it recovers
bit-identical values and the comparison is exact rather than approximate.

Regenerate every golden file::

    python tools/dump_golden.py

Add a fixture and regenerate only it::

    python tools/dump_golden.py tests/data/n-092_U_235.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

import endf

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "crates" / "endf" / "tests" / "golden"

#: Where ENDF fixtures are looked for when no argument is given. `tests/` holds
#: the original fixture; `tests/data/` is where wider library and format
#: coverage goes.
FIXTURE_DIRS = [ROOT / "tests", ROOT / "tests" / "data"]

FIXTURE_SUFFIXES = {".endf", ".dat"}

#: How many interior sample points to record per tabulated function, on top of
#: the ones the region boundaries force. Enough to exercise each interpolation
#: law without the golden file growing to the size of the evaluation.
MAX_SAMPLES = 24


class Dump:
    """Collects `path -> value` lines."""

    def __init__(self, out):
        self.out = out

    def floats(self, path: str, values) -> None:
        self.out.write(f"V {path} F " + " ".join(repr(float(v)) for v in values) + "\n")

    def float(self, path: str, value) -> None:
        self.floats(path, [value])

    def ints(self, path: str, values) -> None:
        self.out.write(f"V {path} I " + " ".join(str(int(v)) for v in values) + "\n")

    def int(self, path: str, value) -> None:
        self.ints(path, [value])

    def text(self, path: str, value: str) -> None:
        self.out.write(f"V {path} T {value.encode().hex()}\n")

    def tab1(self, path: str, table) -> None:
        """A TAB1: its tabulation, and the interpolation sampled."""
        self.floats(f"{path}/x", table.x)
        self.floats(f"{path}/y", table.y)
        self.ints(f"{path}/bp", table.breakpoints)
        self.ints(f"{path}/int", table.interpolation)
        points = sample_points(table)
        if points:
            self.floats(f"{path}/evalx", points)
            self.floats(f"{path}/evaly", [table(p) for p in points])

    def tab2(self, path: str, table) -> None:
        self.ints(f"{path}/bp", table.breakpoints)
        self.ints(f"{path}/int", table.interpolation)


def sample_points(table) -> list[float]:
    """Points at which to pin the interpolation.

    Bin midpoints, because a tabulated point returns its own value whatever the
    interpolation law is and so proves nothing. The bins on either side of a
    region boundary are always included: picking the scheme for the bin a point
    falls in is the part of the evaluation most likely to be got wrong.
    """
    x = table.x
    if len(x) < 2:
        return [float(v) for v in x]

    def midpoint(i: int) -> float:
        return float(0.5 * (x[i] + x[i + 1]))

    wanted = set()
    for b in table.breakpoints:
        for i in (int(b) - 2, int(b) - 1):
            if 0 <= i < len(x) - 1:
                wanted.add(i)
    n_bins = len(x) - 1
    step = max(1, n_bins // MAX_SAMPLES)
    wanted.update(range(0, n_bins, step))

    points = [midpoint(i) for i in sorted(wanted)]
    # Both ends, and outside them, to pin the clamping behaviour.
    return [float(x[0]) * 0.5, float(x[0]), *points, float(x[-1]), float(x[-1]) * 2.0]


# --------------------------------------------------------------------------
# One dumper per ENDF file the Rust crate can parse. Extend as the port goes.
# --------------------------------------------------------------------------


def dump_nu(d: Dump, path: str, section: dict) -> None:
    """A nu-bar, in whichever of the two forms the evaluation uses."""
    if "C" in section:
        d.floats(f"{path}/poly", section["C"])
    if "nu" in section:
        d.tab1(f"{path}/tab", section["nu"])


def dump_mf1(d: Dump, path: str, mt: int, section: dict) -> None:
    if mt == 451:
        for key in (
            "ZA", "LRP", "LFI", "NLIB", "NMOD", "LIS", "LISO", "NFOR",
            "LREL", "NSUB", "NVER", "LDRV", "NWD", "NXC",
        ):
            d.int(f"{path}/{key}", section[key])
        for key in ("AWR", "ELIS", "STA", "AWI", "EMAX", "TEMP"):
            d.float(f"{path}/{key}", section[key])
        if section.get("ZSYMAM") is not None:
            for key in ("ZSYMAM", "ALAB", "EDATE", "AUTH", "REF", "DDATE",
                        "RDATE", "ENDATE"):
                d.text(f"{path}/{key}", section[key])
            for i, line in enumerate(section["HSUB"]):
                d.text(f"{path}/HSUB/{i}", line)
            for i, line in enumerate(section["description"]):
                d.text(f"{path}/description/{i}", line)
        for i, entry in enumerate(section["section_list"]):
            d.ints(f"{path}/section_list/{i}", entry)

    elif mt in (452, 456):
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LNU", section["LNU"])
        dump_nu(d, path, section)

    elif mt == 455:
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LDG", section["LDG"])
        d.int(f"{path}/LNU", section["LNU"])
        if "lambda" in section:
            d.floats(f"{path}/lambda", section["lambda"])
        if "E_int" in section:
            d.tab2(f"{path}/E_int", section["E_int"])
        for i, c in enumerate(section.get("constants", [])):
            d.float(f"{path}/constants/{i}/E", c["E"])
            d.floats(f"{path}/constants/{i}/lambda", c["lambda"])
            d.floats(f"{path}/constants/{i}/alpha", c["alpha"])
        dump_nu(d, path, section)

    elif mt == 458:
        # ZA comes from a CONT rather than a HEAD record here, so it is a float
        # in this section and an int everywhere else. See issue #14.
        d.float(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LFC", section["LFC"])
        d.int(f"{path}/NPLY", section["NPLY"])
        if "NFC" in section:
            d.int(f"{path}/NFC", section["NFC"])
        for name in ("EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET"):
            value = section[name]
            if isinstance(value, dict):
                d.int(f"{path}/{name}/LDRV", value["LDRV"])
                d.tab1(f"{path}/{name}/EIFC", value["EIFC"])
            else:
                # list of (coefficient, uncertainty) pairs
                d.floats(f"{path}/{name}/poly", [v for pair in value for v in pair])

    elif mt == 460:
        d.int(f"{path}/ZA", section["ZA"])
        d.float(f"{path}/AWR", section["AWR"])
        d.int(f"{path}/LO", section["LO"])
        if "NG" in section:
            d.int(f"{path}/NG", section["NG"])
        if "E" in section:
            d.floats(f"{path}/E", section["E"])
        for i, t in enumerate(section.get("T", [])):
            d.tab1(f"{path}/T/{i}", t)
        if "lambda" in section:
            d.floats(f"{path}/lambda", section["lambda"])


def dump_mf2(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NIS", section["NIS"])
    for a, iso in enumerate(section["isotopes"]):
        ip = f"{path}/isotopes/{a}"
        d.float(f"{ip}/ZAI", iso["ZAI"])
        d.float(f"{ip}/ABN", iso["ABN"])
        d.int(f"{ip}/LFW", iso["LFW"])
        d.int(f"{ip}/NER", iso["NER"])
        for b, r in enumerate(iso["ranges"]):
            rp = f"{ip}/ranges/{b}"
            d.float(f"{rp}/EL", r["EL"])
            d.float(f"{rp}/EH", r["EH"])
            for key in ("LRU", "LRF", "NRO", "NAPS"):
                d.int(f"{rp}/{key}", r[key])
            dump_mf2_parameters(d, rp, r)


def dump_mf2_parameters(d: Dump, rp: str, r: dict) -> None:
    """Whatever representation the range turned out to use.

    Driven by which keys are present rather than by LRU/LRF, so that a range
    the dispatch skips (issue #15) emits nothing and the Rust side, which
    reproduces the same skip, emits nothing either.
    """
    if "APE" in r:
        d.tab1(f"{rp}/APE", r["APE"])
    for key in ("SPI", "AP"):
        if key in r:
            d.float(f"{rp}/{key}", r[key])
    for key in ("NLS", "LAD", "NLSC", "LSSF", "IFG", "KRM", "NJS", "KRL", "NPP", "NE"):
        if key in r:
            d.int(f"{rp}/{key}", r[key])
    if "ES" in r:
        d.floats(f"{rp}/ES", r["ES"])

    # Resolved: Breit-Wigner and Reich-Moore both key off 'sections'.
    for i, s in enumerate(r.get("sections", [])):
        sp = f"{rp}/sections/{i}"
        for key in ("AWRI", "QX", "APL"):
            if key in s:
                d.float(f"{sp}/{key}", s[key])
        for key in ("L", "LRX", "NRS"):
            if key in s:
                d.int(f"{sp}/{key}", s[key])
        for key in ("ER", "AJ", "GT", "GN", "GG", "GF", "GFA", "GFB"):
            if key in s:
                d.floats(f"{sp}/{key}", s[key])

    # R-matrix limited.
    if "particle_pairs" in r:
        for key, values in r["particle_pairs"].items():
            d.floats(f"{rp}/particle_pairs/{key}", values)
    for i, g in enumerate(r.get("spin_groups", [])):
        gp = f"{rp}/spin_groups/{i}"
        for key in ("AJ", "PJ"):
            d.float(f"{gp}/{key}", g[key])
        for key in ("KBK", "KPS", "NCH", "NRS", "NX", "LCH", "LBK", "LPS"):
            if key in g:
                d.int(f"{gp}/{key}", g[key])
        for key, values in g["channels"].items():
            d.floats(f"{gp}/channels/{key}", values)
        d.floats(f"{gp}/ER", g["ER"])
        for c, row in enumerate(g["GAM"]):
            d.floats(f"{gp}/GAM/{c}", row)
        for key in ("ED", "EU"):
            if key in g:
                d.float(f"{gp}/{key}", g[key])
        for key in ("RBR", "RBI", "PSR", "PSI"):
            if key in g:
                d.tab1(f"{gp}/{key}", g[key])

    # Unresolved.
    for i, u in enumerate(r.get("ranges", [])):
        up = f"{rp}/ranges/{i}"
        d.float(f"{up}/AWRI", u["AWRI"])
        d.int(f"{up}/L", u["L"])
        d.int(f"{up}/NJS", u["NJS"])
        for key in ("D", "AJ", "AMUN", "GNO", "GG"):
            if key in u:
                d.floats(f"{up}/{key}", u[key])
        for j, p in enumerate(u.get("parameters", [])):
            pp = f"{up}/parameters/{j}"
            for key in ("MUF", "INT", "NE"):
                if key in p:
                    d.int(f"{pp}/{key}", p[key])
            for key in ("D", "AJ", "AMUN", "AMUX", "AMUF", "GN0", "GG"):
                if key in p:
                    value = p[key]
                    if hasattr(value, "__len__"):
                        d.floats(f"{pp}/{key}", value)
                    else:
                        d.float(f"{pp}/{key}", value)
            for key in ("E", "GX", "GF"):
                if key in p:
                    d.floats(f"{pp}/{key}", p[key])


def dump_mf3(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.float(f"{path}/QM", section["QM"])
    d.float(f"{path}/QI", section["QI"])
    d.int(f"{path}/LR", section["LR"])
    d.tab1(f"{path}/sigma", section["sigma"])


def dump_mf4(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    for key in ("LTT", "LI", "LCT"):
        d.int(f"{path}/{key}", section[key])
    for kind in ("legendre", "tabulated"):
        if kind not in section:
            continue
        sub = section[kind]
        sp = f"{path}/{kind}"
        d.tab2(f"{sp}/E_int", sub["E_int"])
        d.float(f"{sp}/T", sub["T"])
        d.int(f"{sp}/LT", sub["LT"])
        d.floats(f"{sp}/E", sub["E"])
        for i, a in enumerate(sub.get("a_l", [])):
            d.floats(f"{sp}/a_l/{i}", a)
        for i, mu in enumerate(sub.get("mu", [])):
            d.tab1(f"{sp}/mu/{i}", mu)


def dump_mf5(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    d.int(f"{path}/NK", section["NK"])
    for i, sub in enumerate(section["subsections"]):
        sp = f"{path}/subsections/{i}"
        d.int(f"{sp}/LF", sub["LF"])
        d.tab1(f"{sp}/p", sub["p"])
        dist = sub["distribution"]
        dp = f"{sp}/distribution"
        for key in ("U", "EFL", "EFH"):
            if key in dist:
                d.float(f"{dp}/{key}", dist[key])
        if "E_int" in dist:
            d.tab2(f"{dp}/E_int", dist["E_int"])
        if "E" in dist:
            d.floats(f"{dp}/E", dist["E"])
        # LF=1 stores a list under 'g'; LF=5 stores a single table there.
        if isinstance(dist.get("g"), list):
            for j, g in enumerate(dist["g"]):
                d.tab1(f"{dp}/g/{j}", g)
        elif "g" in dist:
            d.tab1(f"{dp}/g", dist["g"])
        for key in ("theta", "a", "b", "T_M"):
            if key in dist:
                d.tab1(f"{dp}/{key}", dist[key])


def dump_mf6(d: Dump, path: str, mt: int, section: dict) -> None:
    d.int(f"{path}/ZA", section["ZA"])
    d.float(f"{path}/AWR", section["AWR"])
    for key in ("JP", "LCT", "NK"):
        d.int(f"{path}/{key}", section[key])
    for i, p in enumerate(section["products"]):
        pp = f"{path}/products/{i}"
        d.int(f"{pp}/ZAP", p["ZAP"])
        d.float(f"{pp}/AWP", p["AWP"])
        d.int(f"{pp}/LIP", p["LIP"])
        d.int(f"{pp}/LAW", p["LAW"])
        d.tab1(f"{pp}/y_i", p["y_i"])
        if "distribution" not in p:
            continue
        dist = p["distribution"]
        dp = f"{pp}/distribution"
        for key in ("LANG", "LEP", "NR", "NE", "LIDP", "NPSX"):
            if key in dist:
                d.int(f"{dp}/{key}", dist[key])
        for key in ("SPI", "APSX"):
            if key in dist:
                d.float(f"{dp}/{key}", dist[key])
        if "E_int" in dist:
            d.tab2(f"{dp}/E_int", dist["E_int"])
        if "E" in dist:
            d.floats(f"{dp}/E", dist["E"])
        for j, s in enumerate(dist.get("distribution", [])):
            sp = f"{dp}/distribution/{j}"
            for key in ("ND", "NA", "NW", "NEP", "LANG", "NL", "LTP", "NRM", "NMU"):
                if key in s:
                    d.int(f"{sp}/{key}", s[key])
            if "E" in s:
                d.float(f"{sp}/E", s["E"])
            if "E'" in s:
                d.floats(f"{sp}/Eout", s["E'"])
            for key in ("A_l", "A"):
                if key in s:
                    d.floats(f"{sp}/{key}", s[key])
            if "b" in s:
                for r, row in enumerate(s["b"]):
                    d.floats(f"{sp}/b/{r}", row)
            if "mu_int" in s:
                d.tab2(f"{sp}/mu_int", s["mu_int"])
            for k, entry in enumerate(s.get("mu", [])):
                d.float(f"{sp}/mu/{k}/mu", entry["mu"])
                d.tab1(f"{sp}/mu/{k}/f", entry["f"])


DUMPERS = {
    1: dump_mf1,
    2: dump_mf2,
    3: dump_mf3,
    4: dump_mf4,
    5: dump_mf5,
    6: dump_mf6,
}


def dump(path: Path, out) -> None:
    materials = endf.get_materials(path)
    source = path.relative_to(ROOT).as_posix()
    d = Dump(out)

    out.write(f"# golden reference generated from {source} by the Python reader\n")
    out.write("# regenerate with: python tools/dump_golden.py\n")
    out.write(f"SOURCE {source}\n")
    out.write(f"MATERIALS {len(materials)}\n")

    for m, material in enumerate(materials):
        out.write(f"MAT {m} {material.MAT}\n")

        # Every (MF, MT) with the number of lines its body occupies. This holds
        # the section splitter to the Python reader across the whole file,
        # including the files nothing parses yet.
        for mf, mt in sorted(material.sections):
            n_lines = len(material.section_text[mf, mt].splitlines())
            out.write(f"SECTION {m} {mf} {mt} {n_lines}\n")

        for mf, mt in sorted(material.sections):
            dumper = DUMPERS.get(mf)
            if dumper is not None:
                dumper(d, f"{m}/{mf}/{mt}", mt, material[mf, mt])


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
