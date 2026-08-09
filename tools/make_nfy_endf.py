# SPDX-License-Identifier: MIT
"""Write a small synthetic ENDF evaluation carrying fission product yields.

MF=8 MT=454 and MT=459 are the only files with no fixture: a real neutron-
induced fission yield evaluation lists a thousand or so products at each of
several incident energies, which is megabytes, and it cannot be trimmed the way
the other ENDF fixtures were because the yields *are* the file — cutting
products would only test a shorter list of the same thing.

So a small one is built here instead. The values are invented, but the layout
is the format's, and that is what both readers are held to: a HEAD giving LE+1,
then one LIST per incident energy whose L1 field is overloaded — LE+1 on the
first, the interpolation scheme on the rest — and four values per product.

The independent (MT=454) and cumulative (MT=459) yields differ, so a reader
that returns one for the other is caught. The products include an isomer
(FPS=1) so the naming path is exercised, and the second energy has a different
product count from the first so a reader that reuses NFP is caught too.

    python tools/make_nfy_endf.py tests/synthetic-nfy.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

#: MAT number of the synthetic evaluation. 9999 is unassigned.
MAT = 9999

#: Z*1000 + A of the fissioning nuclide, and its mass in neutron masses.
ZA = 92235
AWR = 233.0248

#: Incident energies in eV, thermal and fast.
ENERGIES = [0.0253, 500000.0]

#: Interpolation scheme for the energies after the first. 2 is linear-linear.
INTERPOLATION = 2

#: (ZAFP, FPS, yield, uncertainty) per energy, for MT=454 and MT=459. The
#: cumulative yields are larger than the independent ones, as they are in a
#: real evaluation, and the second energy carries one more product.
INDEPENDENT = [
    [
        (40095.0, 0.0, 0.0201, 0.0008),  # Zr95
        (54135.0, 1.0, 0.0134, 0.0006),  # Xe135_m1
        (55137.0, 0.0, 0.0619, 0.0021),  # Cs137
    ],
    [
        (40095.0, 0.0, 0.0188, 0.0009),
        (54135.0, 1.0, 0.0142, 0.0007),
        (55137.0, 0.0, 0.0575, 0.0024),
        (58144.0, 0.0, 0.0043, 0.0002),  # Ce144, fast only
    ],
]

CUMULATIVE = [
    [
        (40095.0, 0.0, 0.0605, 0.0018),
        (54135.0, 1.0, 0.0139, 0.0006),
        (55137.0, 0.0, 0.0632, 0.0022),
    ],
    [
        (40095.0, 0.0, 0.0588, 0.0020),
        (54135.0, 1.0, 0.0147, 0.0008),
        (55137.0, 0.0, 0.0601, 0.0025),
        (58144.0, 0.0, 0.0049, 0.0003),
    ],
]


def endf_float(value: float) -> str:
    """An 11-character ENDF float, e.g. ``' 9.223500+4'``."""
    if value == 0.0:
        return " 0.000000+0"
    mantissa = f"{value:.6E}"
    digits, exponent = mantissa.split("E")
    power = int(exponent)
    sign = "+" if power >= 0 else "-"
    text = f"{digits}{sign}{abs(power)}"
    # Six decimal places leaves room for a two-digit exponent and no more; drop
    # one when the exponent needs three characters, as the format requires.
    if len(text) > 11:
        digits = f"{value:.5E}".split("E")[0]
        text = f"{digits}{sign}{abs(power)}"
    return text.rjust(11)


def control(mf: int, mt: int, line_number: int) -> str:
    """The MAT/MF/MT/NS trailer every record carries."""
    return f"{MAT:>4}{mf:>2}{mt:>3}{line_number % 100000:>5}"


class Section:
    """Accumulates the records of one section, numbering them as it goes."""

    def __init__(self, mf: int, mt: int) -> None:
        self.mf = mf
        self.mt = mt
        self.lines: list[str] = []

    def record(self, body: str) -> None:
        self.lines.append(f"{body:<66}{control(self.mf, self.mt, len(self.lines) + 1)}")

    def cont(self, c1: float, c2: float, l1: int, l2: int, n1: int, n2: int) -> None:
        self.record(f"{endf_float(c1)}{endf_float(c2)}{l1:>11}{l2:>11}{n1:>11}{n2:>11}")

    def values(self, values: list[float]) -> None:
        for i in range(0, len(values), 6):
            self.record("".join(endf_float(v) for v in values[i : i + 6]))

    def text(self, body: str) -> None:
        self.record(f"{body:<66}")

    def finish(self) -> list[str]:
        # SEND: MT=0, sequence number 99999.
        return self.lines + [f"{'':<66}{MAT:>4}{self.mf:>2}{0:>3}{99999:>5}"]


def mf1_mt451() -> list[str]:
    """The descriptive section, the minimum a Material needs to be read."""
    s = Section(1, 451)
    # LRP=-1 (no resonance data), LFI=1 (fissile), NLIB=0, NMOD=0.
    s.cont(ZA, AWR, -1, 1, 0, 0)
    # ELIS, STA, LIS, LISO, 0, NFOR=6.
    s.cont(0.0, 1.0, 0, 0, 0, 6)
    # AWI=1 (neutron), EMAX, LREL, 0, NSUB=11 (neutron-induced fission yields),
    # NVER.
    s.cont(1.0, 2.0e7, 0, 0, 11, 8)
    # TEMP, 0, LDRV, 0, NWD, NXC.
    s.cont(0.0, 0.0, 0, 0, 5, 3)
    # The first two text records are read by column, not by whitespace:
    # ZSYMAM, ALAB, EDATE, AUTH on the first, then REF, DDATE, RDATE, ENDATE.
    s.text(
        f"{' 92-U -235':<11}{'SYNTH':<11}{'EVAL-JAN26':<10} "
        f"{'tools/make_nfy_endf.py':<33}"
    )
    s.text(
        f" {'synthetic, not real':<21}{'DIST-JAN26':<10} {'':<10}"
        f"{'':<12}{'20260101':<8}"
    )
    s.text("----SYNTHETIC         MATERIAL 9999")
    s.text("-----NEUTRON-INDUCED FISSION PRODUCT YIELDS")
    s.text("------ENDF-6 FORMAT")
    # The directory: MF, MT, number of records, MOD.
    for mf, mt, count in [(1, 451, 9), (8, 454, 5), (8, 459, 5)]:
        s.record(f"{'':>22}{mf:>11}{mt:>11}{count:>11}{0:>11}")
    return s.finish()


def yields_section(mt: int, sets: list[list[tuple[float, float, float, float]]]):
    """MF=8 MT=454 or MT=459: the yields at each incident energy."""
    s = Section(8, mt)
    le_plus_one = len(sets)
    s.cont(ZA, AWR, le_plus_one, 0, 0, 0)
    for i, (energy, products) in enumerate(zip(ENERGIES, sets)):
        # L1 is LE+1 on the first energy and the interpolation scheme after,
        # which is the overload both readers have to reproduce.
        l1 = le_plus_one if i == 0 else INTERPOLATION
        s.cont(energy, 0.0, l1, 0, 4 * len(products), len(products))
        flat = [v for product in products for v in product]
        s.values(flat)
    return s.finish()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])

    def terminator(mat: int) -> str:
        return f"{'':<66}{mat:>4}{0:>2}{0:>3}{0:>5}"

    lines = [terminator(1)]  # TPID
    lines += mf1_mt451()
    lines.append(terminator(MAT))  # FEND, closing MF=1
    lines += yields_section(454, INDEPENDENT)
    lines += yields_section(459, CUMULATIVE)
    lines.append(terminator(MAT))  # FEND, closing MF=8
    lines.append(terminator(0))  # MEND
    lines.append(terminator(-1))  # TEND

    target.write_text("\n".join(lines) + "\n")
    products = sum(len(s) for s in INDEPENDENT) + sum(len(s) for s in CUMULATIVE)
    print(f"{target}: {len(ENERGIES)} energies, {products} yields", file=sys.stderr)


if __name__ == "__main__":
    main()
