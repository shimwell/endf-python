# SPDX-License-Identifier: MIT
"""Build `photon_aux.txt`, the auxiliary photon data both readers use.

Two sections, from two sources:

**COMPTON** — Hartree-Fock Compton profiles, taken from the **Geant4 G4EMLOW**
data set, which is the primary distribution of Biggs, Mendelsohn and Mann,
*Atomic Data and Nuclear Data Tables* **16** (1975) 201. The archive is
downloaded and verified against a pinned SHA-256, so the provenance is
first-hand and the build is reproducible rather than a copy of somebody else's
copy. Three files inside it are read:

* `doppler/p-biggs.dat` — the 31 electron momenta, in atomic units
* `doppler/profile-<Z>.dat` — J(pz), one row per subshell
* `doppler/shell-doppler.dat` — occupancy and ionisation potential per
  subshell, per Z, each block ended by a `-1` line

**DENSITY** — mean excitation energies from the NIST ESTAR database and
subshell ionization energies for the density effect correction, after
Sternheimer, Berger and Seltzer, *ADNDT* **30** (1984) 261. No primary
distribution of this in a machine-readable form is known to this project, so it
still comes from `density_effect.h5`, which was copied from OpenMC (MIT). If a
primary source turns up, only `density_section` below has to change.

The output is plain text and uncompressed: the Rust crate reads it at runtime
and has no dependencies, so xz would mean adding a decompressor. Floats are
written with `repr`, the shortest string that round-trips, so nothing is lost.

    python tools/make_photon_aux.py                 # downloads G4EMLOW
    python tools/make_photon_aux.py --g4emlow DIR   # uses an extracted copy

Writes `src/endf/datafiles/photon_aux.txt`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATAFILES = ROOT / "src" / "endf" / "datafiles"

#: The data covers Z = 1 to 100.
MAX_Z = 100

#: eV per MeV, matching `endf.data.EV_PER_MEV`.
EV_PER_MEV = 1.0e6

#: Compton profiles come from this release. 6.48 is pinned rather than the
#: newest because the whole `doppler` directory is byte-identical in every
#: release from 6.48 to 8.7 — the data has not changed — and 6.48 is 24 MB
#: against 333 MB. Checked file by file, not assumed.
G4EMLOW_VERSION = "6.48"
G4EMLOW_URL = (
    f"https://geant4-data.web.cern.ch/datasets/G4EMLOW.{G4EMLOW_VERSION}.tar.gz"
)
G4EMLOW_SHA256 = "9815be88cbbcc4e8855b20244d586552a8b1819b8bf4e538c342b27c17dff1c7"


def floats(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def download_g4emlow(cache: Path) -> Path:
    """Fetch and unpack the `doppler` directory, verifying the archive."""
    archive = cache / f"G4EMLOW.{G4EMLOW_VERSION}.tar.gz"
    if not archive.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {G4EMLOW_URL} ...", file=sys.stderr)
        with urllib.request.urlopen(G4EMLOW_URL) as response:
            archive.write_bytes(response.read())

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != G4EMLOW_SHA256:
        raise SystemExit(
            f"{archive} does not match the pinned checksum.\n"
            f"  expected {G4EMLOW_SHA256}\n  got      {digest}\n"
            "Refusing to build data files from an archive that is not the one "
            "this script was written against."
        )

    extracted = cache / f"G4EMLOW{G4EMLOW_VERSION}"
    if not extracted.is_dir():
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if "/doppler/" in m.name]
            tar.extractall(cache, members=members)
    return extracted


def compton_section(doppler: Path) -> list[str]:
    """The Compton profiles, straight out of the Geant4 data files."""
    pz = [float(v) for v in (doppler / "p-biggs.dat").read_text().split()]
    out = ["COMPTON", f"pz {len(pz)} {floats(pz)}"]

    # One stream for every element in turn, each block ended by a -1 line.
    shells = (doppler / "shell-doppler.dat").read_text().splitlines()
    line = iter(shells)

    for z in range(1, MAX_Z + 1):
        j = [float(v) for v in (doppler / f"profile-{z}.dat").read_text().split()]
        nss, remainder = divmod(len(j), len(pz))
        if remainder:
            raise SystemExit(f"Z={z}: profile is not a whole number of subshells")

        num_electrons, binding = [], []
        for text in line:
            fields = text.split()
            if not fields or fields[0].startswith("-1"):
                break
            num_electrons.append(float(fields[0]))
            # Ionisation potentials are in MeV; both readers want eV, and
            # converting once here removes a chance for them to disagree.
            binding.append(float(fields[1]) * EV_PER_MEV)

        if len(num_electrons) != nss:
            raise SystemExit(
                f"Z={z}: {nss} subshells in the profile but "
                f"{len(num_electrons)} in shell-doppler.dat"
            )
        out.append(f"Z {z} {nss} {floats(num_electrons)} {floats(binding)} {floats(j)}")
    return out


def density_section() -> list[str]:
    """The density effect data, still from the vendored HDF5."""
    import h5py

    out = ["DENSITY"]
    with h5py.File(DATAFILES / "density_effect.h5", "r") as f:
        for z in range(1, MAX_Z + 1):
            group = f[f"{z:03}"]
            num_electrons = group["num_electrons"][()]
            ionization = group["ionization_energy"][()]
            nss = len(num_electrons)
            if len(ionization) != nss:
                raise SystemExit(f"Z={z}: ragged density effect data")
            out.append(
                f"Z {z} {nss} {float(group.attrs['I'])!r} "
                f"{floats(num_electrons)} {floats(ionization)}"
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g4emlow",
        type=Path,
        help="an already-extracted G4EMLOW directory, instead of downloading",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "endf-photon-data",
        help="where to keep the downloaded archive",
    )
    args = parser.parse_args()

    root = args.g4emlow or download_g4emlow(args.cache)
    doppler = root / "doppler" if (root / "doppler").is_dir() else root
    if not (doppler / "p-biggs.dat").is_file():
        raise SystemExit(f"no p-biggs.dat under {doppler}")

    lines = [
        "# Auxiliary photon data. Generated by tools/make_photon_aux.py.",
        f"# COMPTON: Geant4 G4EMLOW {G4EMLOW_VERSION} doppler/, which distributes",
        "#   Biggs, Mendelsohn and Mann, At. Data Nucl. Data Tables 16 (1975) 201.",
        "# DENSITY: NIST ESTAR mean excitation energies, and Sternheimer, Berger",
        "#   and Seltzer, At. Data Nucl. Data Tables 30 (1984) 261.",
        "# Binding and ionisation energies are in eV.",
    ]
    lines += compton_section(doppler)
    lines += density_section()

    target = DATAFILES / "photon_aux.txt"
    target.write_text("\n".join(lines) + "\n")
    print(
        f"{target.relative_to(ROOT)}: {target.stat().st_size / 1e3:.0f} kB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
