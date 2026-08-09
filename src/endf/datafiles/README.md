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
| `photon_aux.txt` | The two HDF5 files above, rewritten as plain text by `tools/convert_photon_data.py` | derived, see below |

## photon_aux.txt

The Rust crate has no dependencies, so it cannot open an HDF5 file — that would
mean a C library. `tools/convert_photon_data.py` rewrites both HDF5 files as one
whitespace-separated text file, which both readers can parse with nothing but a
float parser. The conversion is bit-exact: it writes `repr` of each float, the
shortest string that round-trips, and this was checked against every value in
both files.

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
