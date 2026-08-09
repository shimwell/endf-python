# SPDX-License-Identifier: MIT
"""Convert the auxiliary photon data out of HDF5 into a plain text form.

`compton_profiles.h5` and `density_effect.h5` are the only reason this package
needs `h5py`, and HDF5 is the only reason the Rust crate could not read them --
it has no dependencies at all, and an HDF5 reader would mean a C library. The
data itself is a few thousand floats; the container is the problem, not the
contents.

So both are rewritten as one whitespace-separated text file. It is left
uncompressed, unlike the fixtures: the Rust crate reads this one at *runtime*,
and xz would mean a decompressor in a crate that deliberately has no
dependencies. Git packs it well enough. The format is deliberately dull,
because two readers have to agree on it:

    COMPTON
    pz <count> <values...>
    Z <z> <nss> <num_electrons...> <binding_energy...> <J row-major...>
    ...
    DENSITY
    Z <z> <nss> <I> <num_electrons...> <ionization_energy...>
    ...

`binding_energy` is written in eV, already multiplied by EV_PER_MEV, since both
readers want it that way and doing it once removes a chance to disagree. `J` is
row-major, `nss` rows of `len(pz)`.

Floats are written with `repr`, the shortest string that round-trips, so the
conversion is exact rather than nearly exact.

    python tools/convert_photon_data.py

Writes `src/endf/datafiles/photon_aux.txt`. The HDF5 files can be deleted
once both readers have moved over; that is what drops `h5py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATAFILES = ROOT / "src" / "endf" / "datafiles"

#: The data covers Z = 1 to 100.
MAX_Z = 100

#: eV per MeV, matching `endf.data.EV_PER_MEV`.
EV_PER_MEV = 1.0e6


def floats(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def main() -> None:
    import h5py

    out = []

    with h5py.File(DATAFILES / "compton_profiles.h5", "r") as f:
        pz = f["pz"][()]
        out.append("COMPTON")
        out.append(f"pz {len(pz)} {floats(pz)}")
        for z in range(1, MAX_Z + 1):
            group = f[f"{z:03}"]
            num_electrons = group["num_electrons"][()]
            # Converted here so neither reader has to remember to.
            binding = group["binding_energy"][()] * EV_PER_MEV
            j = group["J"][()]
            nss = len(num_electrons)
            assert j.shape == (nss, len(pz)), f"Z={z}: unexpected J shape {j.shape}"
            out.append(
                f"Z {z} {nss} {floats(num_electrons)} {floats(binding)} "
                f"{floats(j.reshape(-1))}"
            )

    with h5py.File(DATAFILES / "density_effect.h5", "r") as f:
        out.append("DENSITY")
        for z in range(1, MAX_Z + 1):
            group = f[f"{z:03}"]
            num_electrons = group["num_electrons"][()]
            ionization = group["ionization_energy"][()]
            nss = len(num_electrons)
            assert len(ionization) == nss, f"Z={z}: ragged density effect data"
            out.append(
                f"Z {z} {nss} {float(group.attrs['I'])!r} "
                f"{floats(num_electrons)} {floats(ionization)}"
            )

    target = DATAFILES / "photon_aux.txt"
    text = "\n".join(out) + "\n"
    target.write_text(text)

    print(
        f"{target.relative_to(ROOT)}: {target.stat().st_size / 1e3:.0f} kB, "
        f"replacing 805 kB of HDF5",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
