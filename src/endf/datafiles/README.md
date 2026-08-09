# Auxiliary photon data

These files are not ENDF evaluations. They hold data needed to describe photon
and electron interactions that the photoatomic sublibrary does not carry, and
they are attached to an `IncidentPhoton` by `_add_compton_profiles` and
`_add_bremsstrahlung`.

All three were copied verbatim from OpenMC's `openmc/data/` directory, which is
MIT licensed. The underlying evaluations are older than OpenMC and are credited
below.

| File | Contents | Source |
| --- | --- | --- |
| `compton_profiles.h5` | Hartree-Fock Compton profiles J(pz) per subshell for Z = 1 to 100 | Biggs, Mendelsohn and Mann, *Atomic Data and Nuclear Data Tables* **16** (1975) 201 |
| `BREMX.DAT` | Scaled bremsstrahlung differential cross sections, 57 electron energies by 30 reduced photon energies, for Z = 1 to 100 | Seltzer and Berger, *Atomic Data and Nuclear Data Tables* **35** (1986) 345 |
| `density_effect.h5` | Mean excitation energies and subshell ionization energies for the density effect correction, Z = 1 to 100 | Sternheimer, Berger and Seltzer, *Atomic Data and Nuclear Data Tables* **30** (1984) 261 |
| `photon_aux.txt` | Compton profiles and density effect data as plain text, built by `tools/make_photon_aux.py` | Geant4 G4EMLOW for the profiles; see below |

## photon_aux.txt

The Rust crate has no dependencies, so it cannot open an HDF5 file — that would
mean a C library. `tools/make_photon_aux.py` writes one whitespace-separated
text file that both readers parse with nothing but a float parser, and it takes
each half from the best source available.

**The Compton profiles come from the primary source.** The script downloads
Geant4's **G4EMLOW** data set — verified against a pinned SHA-256 — and reads
`doppler/p-biggs.dat`, `doppler/profile-<Z>.dat` and `doppler/shell-doppler.dat`
directly. That is the distribution of the Biggs, Mendelsohn and Mann tables, so
this data is no longer a copy of OpenMC's copy. The result was checked against
the HDF5 it replaced and is **bit-identical for all 100 elements**.

G4EMLOW 6.48 is pinned rather than the newest release. The whole `doppler`
directory is byte-identical in every version from 6.48 through 8.7 — checked
file by file, not assumed — and 6.48 is a 24 MB download against 333 MB.

**The density effect data is still vendored** from `density_effect.h5`, because
no primary machine-readable distribution of it is known here. NIST ESTAR
publishes the mean excitation energies through a web form rather than as a
download. If a primary source turns up, only `density_section` in the script
has to change.

It is left uncompressed, unlike the fixtures and goldens. Those are read by
tests, which already have `lzma-rs` as a dev-dependency; this one is read at
*runtime*, and compressing it would put a decompressor in the crate's real
dependencies. 359 kB of text replaces 805 kB of HDF5, and git packs it.

`BREMX.DAT` needs no conversion — it was already whitespace-separated text.

## Known follow-up

`h5py` is now only needed to *regenerate* `photon_aux.txt`, not to read it, so
the Python loaders could move over and the dependency could go. `scipy` likewise:
`crates/endf/src/spline.rs` reimplements the not-a-knot cubic spline that
`_load_bremsstrahlung` uses, and it agrees with SciPy to 3.8e-15 over all
600,000 resampled values, so the Python side could use the same algorithm. Both
deferred deliberately — keeping SciPy on the Python side means the parity
harness compares two independent spline implementations. See
fusion-neutronics/nuclear_data_to_yamc_format#19.
